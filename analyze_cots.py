import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform
import anthropic
import time

# Set up paths
cots_dir = Path("cots")
output_dir = Path("analysis")
output_dir.mkdir(exist_ok=True)

# Model to use for getting activations
model_name = 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'

def load_problem_and_solutions(problem_dir: Path) -> Tuple[Dict, List[Dict]]:
    """
    Load problem and its solutions from a problem directory.
    
    Args:
        problem_dir: Path to the problem directory
        
    Returns:
        Tuple of (problem, solutions)
    """
    problem_path = problem_dir / "problem.json"
    solutions_path = problem_dir / "solutions.json"
    
    with open(problem_path, 'r', encoding='utf-8') as f:
        problem = json.load(f)
    
    with open(solutions_path, 'r', encoding='utf-8') as f:
        solutions = json.load(f)
    
    return problem, solutions

def get_problem_dirs(cots_dir: Path) -> List[Path]:
    """
    Get all problem directories in the CoTs directory.
    
    Args:
        cots_dir: Path to the CoTs directory
        
    Returns:
        List of problem directory paths
    """
    return [d for d in cots_dir.iterdir() if d.is_dir() and d.name.startswith("problem_")]

def load_model_and_tokenizer(model_name: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load the model and tokenizer for activation analysis.
    
    Args:
        model_name: Name of the model to load
        
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device
    )
    
    # Set model to evaluation mode
    model.eval()
    
    print('Number of layers:', model.config.num_hidden_layers)
    print('Hidden size:', model.config.hidden_size)
    
    return model, tokenizer

def split_text_into_chunks(text: str, tokenizer, min_words: int = 2) -> List[str]:
    """
    Split text into chunks based on sentence boundaries, avoiding splitting numbered lists.
    
    Args:
        text: Text to split
        tokenizer: Tokenizer to use for tokenization
        min_words: Minimum number of words required for a valid chunk
        
    Returns:
        List of text chunks
    """
    # First, split by sentence endings (periods followed by space or newline)
    initial_chunks = re.split(r'(?<=\.)\s+|(?<=\.)\n+', text)
    
    # Process chunks to avoid splitting numbered lists
    valid_chunks = []
    current_chunk = ""
    
    for chunk in initial_chunks:
        chunk = chunk.strip()
        if not chunk or chunk == "DONE.":
            continue
        
        # Check if this chunk looks like a numbered list item (e.g., "1.", "a.")
        is_list_item = re.match(r'^[A-Za-z0-9]\.(\s|$)', chunk)
        
        # Count words in the chunk
        word_count = len(chunk.split())
        
        # If it's a list item or too short, merge with the next chunk
        if is_list_item or word_count < min_words:
            if current_chunk:
                current_chunk += " " + chunk
            else:
                current_chunk = chunk
        else:
            # If we have accumulated content, add it to the chunk
            if current_chunk:
                current_chunk += " " + chunk
                valid_chunks.append(current_chunk)
                current_chunk = ""
            else:
                valid_chunks.append(chunk)
    
    # Add any remaining content
    if current_chunk:
        valid_chunks.append(current_chunk)
    
    return valid_chunks

def get_residual_stream_activations(
    model: AutoModelForCausalLM, 
    tokenizer: AutoTokenizer,
    text: str, 
    layers: List[int]
) -> Dict[int, torch.Tensor]:
    """
    Get residual stream activations for a given text using Hugging Face Transformers.
    
    Args:
        model: Hugging Face model
        tokenizer: Tokenizer
        text: Input text
        layers: List of layers to get activations for
        
    Returns:
        Dictionary mapping layer indices to activation tensors
    """
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # Set up hooks to capture residual stream activations
    activations = {}
    handles = []
    
    def get_activation_hook(layer_idx):
        def hook(module, input, output):
            # For LLaMA models, the residual stream is the first element of the output tuple
            # This is model-specific and may need adjustment for other architectures
            if isinstance(output, tuple):
                activations[layer_idx] = output[0].detach().cpu()
            else:
                activations[layer_idx] = output.detach().cpu()
        return hook
    
    # Register hooks for each layer
    for layer_idx in layers:
        # The exact module path may vary depending on the model architecture
        # For LLaMA models, it's typically model.model.layers[layer_idx]
        layer = model.model.layers[layer_idx]
        handle = layer.register_forward_hook(get_activation_hook(layer_idx))
        handles.append(handle)
    
    # Forward pass
    with torch.no_grad():
        model(**inputs)
    
    # Remove hooks
    for handle in handles:
        handle.remove()
    
    return activations

# Function to analyze chunk with Claude
def analyze_chunk_with_claude(chunk_text, previous_chunk=None, previous_category=None, client=None):
    """
    Use Claude to analyze a chunk of text and classify it into a reasoning category.
    
    Args:
        chunk_text: The chunk of text to analyze
        previous_chunk: The previous chunk of text (optional)
        previous_category: The category of the previous chunk (optional)
        client: Anthropic client
        
    Returns:
        Category name as a string and its abbreviation
    """
    # If no client is provided, return unknown
    if client is None:
        return "Unknown", "NA"
    
    # Construct context about the previous chunk if available
    previous_context = ""
    if previous_chunk and previous_category:
        previous_context = f"""
        For context, the previous chunk of text was classified as "{previous_category}" and contained:
        
        [PREVIOUS CHUNK]
        {previous_chunk}
        
        Now analyze the following new chunk:
        """
    
    prompt = f"""You are an expert in analyzing language model reasoning patterns. Given a chunk of text from a language model's Chain of Thought reasoning, classify it into exactly one of the following categories:

    1. Initializing: Rephrasing the given task and stating initial thoughts at the beginning of reasoning.
    2. Deduction: Deriving conclusions based on current approach and assumptions.
    3. Adding Knowledge: Incorporating external knowledge to refine reasoning.
    4. Example Testing: Generating examples or scenarios to test hypotheses.
    5. Uncertainty Estimation: Explicitly stating confidence or uncertainty regarding reasoning.
    6. Backtracking: Abandoning the current approach and exploring alternative strategies.
    7. Comparative Analysis: Comparing multiple approaches or options side by side.
    8. Question Posing: Asking self-directed questions to guide reasoning.
    9. Summary: Summarizing or recapping thinking, often near the end of reasoning.
    10. Metaphorical Thinking: Using metaphors or analogies to understand the problem.
    11. Final Answer: Providing the final answer to the problem.

    {previous_context}
    
    For the following text, provide only the category name that best describes the reasoning pattern shown:

    [TEXT]
    {chunk_text}
    """
    
    try:
        response = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=50,
            system="You are an expert in analyzing reasoning patterns. Respond with only the category name.",
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Extract category from response
        category = response.content[0].text.strip()
        
        # Get abbreviation (first letters of each word)
        abbreviation = ''.join([word[0] for word in category.split()])
        
        if len(abbreviation) <= 2:
            return category, abbreviation
        
        return "Unknown", "NA"
            
    except Exception as e:
        print(f"API error: {e}")
        return "Unknown", "NA"

def compute_chunk_correlations(
    chunks: List[str], 
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    layers: List[int]
) -> np.ndarray:
    """
    Compute correlations between residual stream activations of text chunks.
    
    Args:
        chunks: List of text chunks
        model: Hugging Face model
        tokenizer: Tokenizer
        layers: List of layers to analyze
        
    Returns:
        Correlation matrix
    """
    # Get activations for each chunk
    chunk_activations = []
    
    for chunk in tqdm(chunks, desc="Processing chunks"):
        # Get activations
        activations = get_residual_stream_activations(model, tokenizer, chunk, layers)
        
        # Average activations across tokens (sequence dimension)
        layer_activations = []
        for layer in layers:
            # Check if layer exists in activations
            if layer in activations:
                # Average across sequence dimension (dim=1)
                layer_act = activations[layer].mean(dim=1).squeeze(0) # Remove batch dimension
                layer_activations.append(layer_act)
        
        if layer_activations:
            # Stack and average across layers
            avg_activation = torch.stack(layer_activations).mean(dim=0)
            chunk_activations.append(avg_activation)
    
    # Ensure all activations have the same dimension
    max_dim = max(act.shape[0] for act in chunk_activations)
    padded_activations = []
    
    for act in chunk_activations:
        if act.shape[0] < max_dim:
            # Create a new tensor with the maximum dimension
            padded = torch.zeros(max_dim)
            # Only copy as many elements as we have
            padded[:act.shape[0]] = act
            padded_activations.append(padded)
        else:
            padded_activations.append(act)
    
    # Convert to numpy array for vectorized operations
    n_chunks = len(padded_activations)
    if n_chunks == 0:
        print("Error: No valid chunks found to analyze.")
        return np.zeros((0, 0))
    
    # Convert list of tensors to 2D numpy array
    chunk_means_array = np.array([act.numpy() for act in padded_activations])
    
    # Calculate standard deviations for all chunks at once
    stds = np.std(chunk_means_array, axis=1, keepdims=True)
    
    # Find chunks with zero standard deviation
    zero_std_mask = (stds == 0).squeeze()
    
    # Center each chunk (subtract its mean)
    centered_chunks = chunk_means_array - np.mean(chunk_means_array, axis=1, keepdims=True)
    
    # Prepare normalized chunks
    # To avoid division by zero, temporarily set zero stds to 1 (will handle these cases later)
    stds[stds == 0] = 1
    normalized_chunks = centered_chunks / stds
    
    # Compute dot product of all normalized chunks
    # This gives us numerators of Pearson correlation
    dot_products = np.dot(normalized_chunks, normalized_chunks.T)
    
    # Divide by lengths to get correlations
    # n is the length of each chunk vector (number of dimensions)
    n = chunk_means_array.shape[1]
    correlation_matrix = dot_products / (n - 1)
    
    # Set correlations involving zero std chunks to 0
    correlation_matrix[zero_std_mask, :] = 0
    correlation_matrix[:, zero_std_mask] = 0
    
    # Set diagonal to 1.0 (self-correlation)
    np.fill_diagonal(correlation_matrix, 1.0)
    
    # Replace any remaining NaN values with 0
    correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
    
    return correlation_matrix

def plot_correlation_heatmap(
    corr_matrix: np.ndarray, 
    output_path: Path,
    title: str = "Chunk Residual Stream Correlation",
    chunk_labels: List[str] = None,
    layers: List[int] = None
):
    """
    Plot correlation heatmap.
    
    Args:
        corr_matrix: Correlation matrix
        output_path: Path to save the plot
        title: Plot title
        chunk_labels: Labels for chunks
        layers: List of layers used
    """
    plt.figure(figsize=(12, 10))
    
    # Create a copy of the matrix to avoid modifying the original
    matrix_for_plot = corr_matrix.copy()
    
    # Set diagonal to NaN to exclude it from color scaling
    # matrix_for_plot[np.diag_indices_from(matrix_for_plot)] = np.nan
    
    # Mask out the upper triangle (including diagonal)
    mask = np.triu(np.ones_like(matrix_for_plot, dtype=bool), k=1)
    
    # Create heatmap with quantile-based color scaling and mask
    ax = sns.heatmap(
        matrix_for_plot, 
        cmap="coolwarm", 
        vmin=np.nanquantile(matrix_for_plot, 0.1), 
        vmax=np.nanquantile(matrix_for_plot, 0.9), 
        square=True,
        mask=mask
    )
    
    # Set tick labels if provided, ensuring they match the matrix dimensions
    if chunk_labels:
        # Make sure we have the right number of labels
        n_matrix = corr_matrix.shape[0]
        if len(chunk_labels) != n_matrix:
            print(f"Warning: Number of labels ({len(chunk_labels)}) doesn't match matrix dimensions ({n_matrix}). Adjusting labels.")
            # Truncate or extend labels as needed
            if len(chunk_labels) > n_matrix:
                chunk_labels = chunk_labels[:n_matrix]
            else:
                chunk_labels = chunk_labels + [f"C-{i}" for i in range(len(chunk_labels), n_matrix)]
        
        # Get current tick positions
        ax.set_xticks(np.arange(n_matrix) + 0.5)
        ax.set_yticks(np.arange(n_matrix) + 0.5)
        
        # Set the labels
        ax.set_xticklabels(chunk_labels, rotation=90)
        ax.set_yticklabels(chunk_labels, rotation=0)
    
    # Add labels and title
    plt.xlabel("Chunk Index")
    plt.ylabel("Chunk Index")
    if layers:
        plt.title(f"{title} (Layers {'-'.join(map(str, layers))} Avg)")
    else:
        plt.title(title)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

# Function to compute cosine similarity matrix
def compute_cosine_similarity(chunk_activations: List[torch.Tensor]) -> np.ndarray:
    """
    Compute cosine similarity matrix between chunk activations.
    
    Args:
        chunk_activations: List of chunk activation tensors
        
    Returns:
        Cosine similarity matrix
    """
    # Convert tensors to numpy arrays
    activations_np = [act.numpy() for act in chunk_activations]
    
    # Compute cosine similarity matrix
    n_chunks = len(activations_np)
    cosine_sim_matrix = np.zeros((n_chunks, n_chunks))
    
    for i in range(n_chunks):
        for j in range(n_chunks):
            if i == j:
                cosine_sim_matrix[i, j] = 1.0
            else:
                # Reshape to 2D arrays for sklearn's cosine_similarity
                act_i = activations_np[i].reshape(1, -1)
                act_j = activations_np[j].reshape(1, -1)
                cosine_sim_matrix[i, j] = cosine_similarity(act_i, act_j)[0, 0]
    
    return cosine_sim_matrix

# Function to plot cosine similarity heatmap
def plot_cosine_similarity_heatmap(
    cosine_sim_matrix: np.ndarray, 
    output_path: Path,
    title: str = "Chunk Cosine Similarity",
    chunk_labels: List[str] = None,
    layers: List[int] = None
):
    """
    Plot cosine similarity heatmap.
    
    Args:
        cosine_sim_matrix: Cosine similarity matrix
        output_path: Path to save the plot
        title: Plot title
        chunk_labels: Labels for chunks
        layers: List of layers used
    """
    plt.figure(figsize=(12, 10))
    
    # Create a copy of the matrix to avoid modifying the original
    matrix_for_plot = cosine_sim_matrix.copy()
    
    # Set diagonal to NaN to exclude it from color scaling
    # matrix_for_plot[np.diag_indices_from(matrix_for_plot)] = np.nan
    
    # Mask out the upper triangle (including diagonal)
    mask = np.triu(np.ones_like(matrix_for_plot, dtype=bool), k=1)
    
    # Create heatmap with quantile-based color scaling and mask
    ax = sns.heatmap(
        matrix_for_plot, 
        cmap="coolwarm", 
        vmin=np.nanquantile(matrix_for_plot, 0.1), 
        vmax=np.nanquantile(matrix_for_plot, 0.9), 
        square=True,
        mask=mask
    )
    
    # Set tick labels if provided, ensuring they match the matrix dimensions
    if chunk_labels:
        # Make sure we have the right number of labels
        n_matrix = cosine_sim_matrix.shape[0]
        if len(chunk_labels) != n_matrix:
            print(f"Warning: Number of labels ({len(chunk_labels)}) doesn't match matrix dimensions ({n_matrix}). Adjusting labels.")
            # Truncate or extend labels as needed
            if len(chunk_labels) > n_matrix:
                chunk_labels = chunk_labels[:n_matrix]
            else:
                chunk_labels = chunk_labels + [f"C-{i}" for i in range(len(chunk_labels), n_matrix)]
        
        # Get current tick positions
        ax.set_xticks(np.arange(n_matrix) + 0.5)
        ax.set_yticks(np.arange(n_matrix) + 0.5)
        
        # Set the labels
        ax.set_xticklabels(chunk_labels, rotation=90)
        ax.set_yticklabels(chunk_labels, rotation=0)
    
    # Add labels and title
    plt.xlabel("Chunk Index")
    plt.ylabel("Chunk Index")
    if layers:
        plt.title(f"{title} (Layers {'-'.join(map(str, layers))} Avg)")
    else:
        plt.title(title)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

# Function to compute L2 distance matrix
def compute_l2_distance(chunk_activations: List[torch.Tensor]) -> np.ndarray:
    """
    Compute L2 distance matrix between chunk activations.
    
    Args:
        chunk_activations: List of chunk activation tensors
        
    Returns:
        L2 distance matrix
    """
    # Convert tensors to numpy arrays
    activations_np = np.array([act.numpy() for act in chunk_activations])
    
    # Compute pairwise L2 distances
    l2_distances = squareform(pdist(activations_np, 'euclidean'))
    
    return l2_distances

# Function to plot L2 distance heatmap
def plot_l2_distance_heatmap(
    l2_distance_matrix: np.ndarray, 
    output_path: Path,
    title: str = "Chunk L2 Distance",
    chunk_labels: List[str] = None,
    layers: List[int] = None
):
    """
    Plot L2 distance heatmap.
    
    Args:
        l2_distance_matrix: L2 distance matrix
        output_path: Path to save the plot
        title: Plot title
        chunk_labels: Labels for chunks
        layers: List of layers used
    """
    plt.figure(figsize=(12, 10))
    
    # Create a copy of the matrix to avoid modifying the original
    matrix_for_plot = l2_distance_matrix.copy()
    
    # Set diagonal to NaN to exclude it from color scaling
    # matrix_for_plot[np.diag_indices_from(matrix_for_plot)] = np.nan
    
    # Mask out the upper triangle (including diagonal)
    mask = np.triu(np.ones_like(matrix_for_plot, dtype=bool), k=1)
    
    # For L2 distance, we want to normalize differently
    # Lower values (blue) indicate similarity
    vmin = np.nanquantile(matrix_for_plot, 0.1)
    vmax = np.nanquantile(matrix_for_plot, 0.9)
    
    # Create heatmap
    ax = sns.heatmap(
        matrix_for_plot, 
        cmap="coolwarm_r",  # Reversed colormap so blue = similar
        vmin=vmin,
        vmax=vmax,
        square=True,
        mask=mask
    )
    
    # Set tick labels if provided, ensuring they match the matrix dimensions
    if chunk_labels:
        # Make sure we have the right number of labels
        n_matrix = l2_distance_matrix.shape[0]
        if len(chunk_labels) != n_matrix:
            print(f"Warning: Number of labels ({len(chunk_labels)}) doesn't match matrix dimensions ({n_matrix}). Adjusting labels.")
            # Truncate or extend labels as needed
            if len(chunk_labels) > n_matrix:
                chunk_labels = chunk_labels[:n_matrix]
            else:
                chunk_labels = chunk_labels + [f"C-{i}" for i in range(len(chunk_labels), n_matrix)]
        
        # Get current tick positions
        ax.set_xticks(np.arange(n_matrix) + 0.5)
        ax.set_yticks(np.arange(n_matrix) + 0.5)
        
        # Set the labels
        ax.set_xticklabels(chunk_labels, rotation=90)
        ax.set_yticklabels(chunk_labels, rotation=0)
    
    # Add labels and title
    plt.xlabel("Chunk Index")
    plt.ylabel("Chunk Index")
    if layers:
        plt.title(f"{title} (Layers {'-'.join(map(str, layers))} Avg)")
    else:
        plt.title(title)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def get_attention_weights(
    model: AutoModelForCausalLM, 
    tokenizer: AutoTokenizer,
    text: str, 
    layers: List[int]
) -> Dict[int, torch.Tensor]:
    """
    Get attention weights for a given text.
    
    Args:
        model: Hugging Face model
        tokenizer: Tokenizer
        text: Input text
        layers: List of layers to get attention weights for
        
    Returns:
        Dictionary mapping layer indices to attention weight tensors
    """
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # Set up hooks to capture attention weights
    attention_weights = {}
    handles = []
    
    def get_attention_hook(layer_idx):
        def hook(module, input, output):
            # Capture attention weights
            # For most models, attention weights are in output[0]
            attention_weights[layer_idx] = output[0].detach().cpu()
        return hook
    
    # Register hooks for each layer
    for layer_idx in layers:
        # The exact module path may vary depending on the model architecture
        # For most models, it's typically model.model.layers[layer_idx].self_attn
        layer = model.model.layers[layer_idx].self_attn
        handle = layer.register_forward_hook(get_attention_hook(layer_idx))
        handles.append(handle)
    
    # Forward pass
    with torch.no_grad():
        model(**inputs)
    
    # Remove hooks
    for handle in handles:
        handle.remove()
    
    return attention_weights

def compute_chunk_attention(
    chunks: List[str], 
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    layers: List[int]
) -> np.ndarray:
    """
    Compute average attention between chunks.
    
    Args:
        chunks: List of text chunks
        model: Hugging Face model
        tokenizer: Tokenizer
        layers: List of layers to analyze
        
    Returns:
        Attention matrix
    """
    n_chunks = len(chunks)
    attention_matrix = np.zeros((n_chunks, n_chunks))
    
    # First, tokenize all chunks to get token boundaries
    chunk_tokens = []
    
    for chunk in chunks:
        tokens = tokenizer.encode(chunk, add_special_tokens=False)
        chunk_tokens.append(tokens)
    
    # Process each chunk as a query
    for i, query_chunk in enumerate(tqdm(chunks, desc="Processing query chunks")):
        # Get full context up to and including this chunk
        context = "".join(chunks[:i+1])
        
        # Tokenize the full context
        context_tokens = tokenizer.encode(context, add_special_tokens=False)
        
        # Get attention weights for the full context
        try:
            attn_weights = get_attention_weights(model, tokenizer, context, layers)
            
            # Find token indices for the current (query) chunk
            # These are the last len(chunk_tokens[i]) tokens in the context
            query_start_idx = len(context_tokens) - len(chunk_tokens[i])
            query_end_idx = len(context_tokens)
            query_indices = list(range(query_start_idx, query_end_idx))
            
            # Average attention weights across layers and heads
            avg_attn = None
            count = 0
            
            for layer in layers:
                if layer in attn_weights:
                    # Shape should be [batch, heads, query_seq_len, key_seq_len]
                    layer_attn = attn_weights[layer]
                    
                    # Check tensor dimensions and reshape if needed
                    if len(layer_attn.shape) == 4:  # [batch, heads, query_seq_len, key_seq_len]
                        # Average across heads and batch dimensions
                        layer_attn = layer_attn.mean(dim=(0, 1))  # Now shape: [query_seq_len, key_seq_len]
                    elif len(layer_attn.shape) == 3:  # [heads, query_seq_len, key_seq_len]
                        # Average across heads dimension
                        layer_attn = layer_attn.mean(dim=0)  # Now shape: [query_seq_len, key_seq_len]
                    elif len(layer_attn.shape) == 2:  # [query_seq_len, key_seq_len]
                        # Already in the right shape
                        pass
                    else:
                        print(f"Warning: Unexpected attention tensor shape: {layer_attn.shape}")
                        continue
                    
                    if avg_attn is None:
                        avg_attn = layer_attn
                    else:
                        avg_attn += layer_attn
                    count += 1
            
            if count > 0 and avg_attn is not None:
                avg_attn = avg_attn / count
            else:
                # Skip if no valid attention weights
                print(f"Warning: No valid attention weights for chunk {i}")
                continue
            
            # Check if avg_attn has the expected shape
            if len(avg_attn.shape) != 2:
                print(f"Warning: Unexpected averaged attention shape: {avg_attn.shape}")
                continue
                
            # Extract attention from query chunk tokens to all previous tokens
            query_attn = avg_attn[query_indices, :]
            
            # For each key chunk, compute average attention from query chunk
            for j in range(i+1):  # Only process chunks up to the current one (causal attention)
                if i == j:  # Self-attention (diagonal)
                    attention_matrix[i, j] = 1.0
                    continue
                
                # Find token indices for the key chunk in the context
                key_tokens = chunk_tokens[j]
                
                # Find the start position of this chunk in the context
                key_start_idx = 0
                for k in range(j):
                    key_start_idx += len(chunk_tokens[k])
                key_end_idx = key_start_idx + len(key_tokens)
                
                # Ensure indices are within bounds
                if key_end_idx > query_attn.shape[1]:
                    print(f"Warning: Key chunk indices out of bounds. Adjusting.")
                    key_end_idx = query_attn.shape[1]
                
                if key_start_idx >= key_end_idx:
                    print(f"Warning: Invalid key chunk indices: {key_start_idx}-{key_end_idx}")
                    continue
                
                key_indices = list(range(key_start_idx, key_end_idx))
                
                # Calculate average attention from query tokens to key tokens
                chunk_attention = query_attn[:, key_indices].mean().item()
                attention_matrix[i, j] = chunk_attention
                
        except Exception as e:
            print(f"Error processing chunk {i}: {str(e)}")
            # Set diagonal to 1.0 (self-attention)
            attention_matrix[i, i] = 1.0
    
    return attention_matrix

def plot_attention_heatmap(
    attention_matrix: np.ndarray, 
    output_path: Path,
    title: str = "Chunk Attention",
    chunk_labels: List[str] = None,
    layers: List[int] = None
):
    """
    Plot attention heatmap.
    
    Args:
        attention_matrix: Attention matrix
        output_path: Path to save the plot
        title: Plot title
        chunk_labels: Labels for chunks
        layers: List of layers used
    """
    plt.figure(figsize=(12, 10))
    
    # Create a copy of the matrix to avoid modifying the original
    matrix_for_plot = attention_matrix.copy()
    
    # Set diagonal to NaN to exclude it from color scaling
    # matrix_for_plot[np.diag_indices_from(matrix_for_plot)] = np.nan
    
    # Mask out the upper triangle (excluding diagonal)
    mask = np.triu(np.ones_like(matrix_for_plot, dtype=bool), k=1)
    
    # Create heatmap with quantile-based color scaling and mask
    ax = sns.heatmap(
        matrix_for_plot, 
        cmap="viridis", 
        vmin=np.nanquantile(matrix_for_plot, 0.1), 
        vmax=np.nanquantile(matrix_for_plot, 0.9), 
        square=True,
        mask=mask
    )
    
    # Set tick labels if provided, ensuring they match the matrix dimensions
    if chunk_labels:
        # Make sure we have the right number of labels
        n_matrix = attention_matrix.shape[0]
        if len(chunk_labels) != n_matrix:
            print(f"Warning: Number of labels ({len(chunk_labels)}) doesn't match matrix dimensions ({n_matrix}). Adjusting labels.")
            # Truncate or extend labels as needed
            if len(chunk_labels) > n_matrix:
                chunk_labels = chunk_labels[:n_matrix]
            else:
                chunk_labels = chunk_labels + [f"C-{i}" for i in range(len(chunk_labels), n_matrix)]
        
        # Get current tick positions
        ax.set_xticks(np.arange(n_matrix) + 0.5)
        ax.set_yticks(np.arange(n_matrix) + 0.5)
        
        # Set the labels
        ax.set_xticklabels(chunk_labels, rotation=90)
        ax.set_yticklabels(chunk_labels, rotation=0)
    
    # Add labels and title
    plt.xlabel("Key Chunk Index")
    plt.ylabel("Query Chunk Index")
    if layers:
        plt.title(f"{title} (Layers {'-'.join(map(str, layers))} Avg)")
    else:
        plt.title(title)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def get_chunk_activations_from_full_context(
    full_text: str,
    chunks: List[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    layers: List[int]
) -> List[torch.Tensor]:
    """
    Get activations for each chunk from the full context.
    
    Args:
        full_text: The full solution text
        chunks: List of text chunks
        model: Hugging Face model
        tokenizer: Tokenizer
        layers: List of layers to analyze
        
    Returns:
        List of activation tensors for each chunk
    """
    # Tokenize the full text
    full_tokens = tokenizer.encode(full_text, return_tensors="pt").to(model.device)
    
    # Get the full text as a string for finding chunk positions
    full_text_str = tokenizer.decode(full_tokens[0], skip_special_tokens=True)
    
    # Get activations for the full text
    activations = get_residual_stream_activations(model, tokenizer, full_text, layers)
    
    # Extract activations for each chunk
    chunk_activations = []
    
    # Find the position of each chunk in the full text
    current_pos = 0
    for chunk in chunks:
        # Find the chunk in the full text, starting from current_pos
        chunk_start = full_text_str.find(chunk, current_pos)
        
        if chunk_start == -1:
            print(f"Warning: Chunk not found in full text: {chunk[:50]}...")
            # Skip this chunk
            chunk_activations.append(None)
            continue
        
        chunk_end = chunk_start + len(chunk)
        current_pos = chunk_end  # Update for next search
        
        # Convert character positions to token indices
        # This is an approximation - for exact mapping, we'd need alignment tools
        chunk_start_token = tokenizer.encode(full_text_str[:chunk_start], add_special_tokens=False)
        chunk_start_token_idx = len(chunk_start_token)
        
        chunk_end_token = tokenizer.encode(full_text_str[:chunk_end], add_special_tokens=False)
        chunk_end_token_idx = len(chunk_end_token)
        
        # Extract activations for this chunk's tokens
        chunk_layer_activations = []
        for layer in layers:
            if layer in activations:
                # Extract activations for this chunk's tokens
                # Shape: [batch, seq_len, hidden_size]
                layer_act = activations[layer][0, chunk_start_token_idx:chunk_end_token_idx, :]
                
                # Average across tokens
                avg_act = layer_act.mean(dim=0)
                chunk_layer_activations.append(avg_act)
        
        if chunk_layer_activations:
            # Stack and average across layers
            avg_activation = torch.stack(chunk_layer_activations).mean(dim=0)
            chunk_activations.append(avg_activation)
        else:
            chunk_activations.append(None)
    
    # Filter out None values
    chunk_activations = [act for act in chunk_activations if act is not None]
    
    return chunk_activations

def analyze_problem_solution(
    problem_dir: Path,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    claude_client,
    layers: List[int],
):
    """
    Analyze a problem solution and generate correlation heatmap.
    
    Args:
        problem_dir: Path to the problem directory
        model: Hugging Face model
        tokenizer: Tokenizer
        claude_client: Anthropic Claude client
        layers: List of layers to analyze
    """
    problem_id = problem_dir.name
    
    # Load problem and solutions
    problem, solutions = load_problem_and_solutions(problem_dir)
    
    # Use the first solution for analysis
    if not solutions:
        print(f"No solutions found for {problem_id}")
        return
    
    for solution_dict in tqdm(solutions[:1], desc="Iterating over seeds"):  # Just analyze the first solution
        seed = solution_dict["seed"]
        solution = solution_dict["solution"]
        
        # Create output directory for this problem
        problem_output_dir = output_dir / problem_id / f"seed_{seed}"
        problem_output_dir.mkdir(exist_ok=True, parents=True)
        
        # Check if chunks analysis already exists
        chunks_file = problem_output_dir / "chunks.json"
        
        if chunks_file.exists():
            with open(chunks_file, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
                
            # Extract chunks and their labels
            chunks = [item["text"] for item in chunks_data]
            chunk_categories = [item["category"] for item in chunks_data]
            chunk_abbreviations = [item["abbreviation"] for item in chunks_data]
        else:
            # Split solution into chunks
            chunks = split_text_into_chunks(solution, tokenizer)
            
            if len(chunks) < 5:
                print(f"Too few chunks ({len(chunks)}) for {problem_id}, skipping")
                return
            
            # Analyze chunks with Claude
            chunk_categories = []
            chunk_abbreviations = []
            
            prev_chunk = None
            prev_category = None
            
            for i, chunk in enumerate(tqdm(chunks, desc="Analyzing chunks with Claude")):
                if claude_client:
                    category, abbreviation = analyze_chunk_with_claude(
                        chunk, 
                        previous_chunk=prev_chunk, 
                        previous_category=prev_category, 
                        client=claude_client
                    )
                    chunk_categories.append(category)
                    chunk_abbreviations.append(f"{abbreviation}-{i}")
                    
                    # Update previous chunk and category for next iteration
                    prev_chunk = chunk
                    prev_category = category
                    
                    time.sleep(0.5)  # Avoid rate limiting
                else:
                    # If no Claude client, just use indices
                    chunk_categories.append("Unknown")
                    chunk_abbreviations.append(f"C-{i}")
            
            # Save chunks with their categories for reference
            chunks_with_categories = []
            for i, (chunk, category) in enumerate(zip(chunks, chunk_categories)):
                chunks_with_categories.append({
                    "index": i,
                    "text": chunk,
                    "category": category,
                    "abbreviation": chunk_abbreviations[i]
                })
            
            with open(chunks_file, 'w', encoding='utf-8') as f:
                json.dump(chunks_with_categories, f, indent=2)
        
        # Get activations for each chunk from the full context
        chunk_activations = get_chunk_activations_from_full_context(
            solution, chunks, model, tokenizer, layers
        )
        
        # Compute Pearson correlation
        corr_matrix = compute_chunk_correlations(chunks, model, tokenizer, layers)
        
        # Save correlation matrix as numpy array
        np.save(problem_output_dir / f"chunk_correlation_layer_{layers[0]}.npy", corr_matrix)
        
        # Plot correlation heatmap
        plot_correlation_heatmap(
            corr_matrix, 
            problem_output_dir / f"chunk_correlation_layer_{layers[0]}.png", 
            f"{problem_id} - Seed {seed} - Chunk Residual Stream Correlation",
            chunk_labels=chunk_abbreviations,
            layers=layers
        )
        
        # Compute cosine similarity
        cosine_sim_matrix = compute_cosine_similarity(chunk_activations)
        
        # Save cosine similarity matrix as numpy array
        np.save(problem_output_dir / f"chunk_cosine_similarity_layer_{layers[0]}.npy", cosine_sim_matrix)
        
        # Plot cosine similarity heatmap
        plot_cosine_similarity_heatmap(
            cosine_sim_matrix, 
            problem_output_dir / f"chunk_cosine_similarity_layer_{layers[0]}.png", 
            f"{problem_id} - Seed {seed} - Chunk Cosine Similarity",
            chunk_labels=chunk_abbreviations,
            layers=layers
        )
        
        # Compute L2 distance
        l2_distance_matrix = compute_l2_distance(chunk_activations)
        
        # Save L2 distance matrix as numpy array
        np.save(problem_output_dir / f"chunk_l2_distance_layer_{layers[0]}.npy", l2_distance_matrix)
        
        # Plot L2 distance heatmap
        plot_l2_distance_heatmap(
            l2_distance_matrix, 
            problem_output_dir / f"chunk_l2_distance_layer_{layers[0]}.png", 
            f"{problem_id} - Seed {seed} - Chunk L2 Distance",
            chunk_labels=chunk_abbreviations,
            layers=layers
        )
        
        # Compute attention weights
        attention_matrix = compute_chunk_attention(chunks, model, tokenizer, layers)
        
        # Save attention matrix as numpy array
        np.save(problem_output_dir / f"chunk_attention_layer_{layers[0]}.npy", attention_matrix)
        
        # Plot attention heatmap
        plot_attention_heatmap(
            attention_matrix, 
            problem_output_dir / f"chunk_attention_layer_{layers[0]}.png", 
            f"{problem_id} - Seed {seed} - Chunk Attention",
            chunk_labels=chunk_abbreviations,
            layers=layers
        )

# Set layers to analyze
layers = [11]

# Initialize Anthropic client
api_key = os.environ.get('ANTHROPIC_API_KEY')
if not api_key:
    print("Warning: ANTHROPIC_API_KEY not set. Claude analysis will not work.")
    claude_client = None
else:
    claude_client = anthropic.Anthropic(api_key=api_key)

# Load model and tokenizer
model, tokenizer = load_model_and_tokenizer(model_name)

# Get problem directories
problem_dirs = get_problem_dirs(cots_dir)
print(f"Found {len(problem_dirs)} problem directories")

# Analyze problems
for problem_dir in tqdm(problem_dirs, desc="Analyzing problems"):
    analyze_problem_solution(problem_dir, model, tokenizer, claude_client, layers)