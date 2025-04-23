import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
from sklearn.metrics import silhouette_score
import argparse
import pandas as pd
from collections import Counter, defaultdict
from utils import get_chunk_ranges

# Set up paths
cots_dir = Path("cots")
chunks_dir = Path("analysis")
output_dir = Path("clustering_analysis")
output_dir.mkdir(exist_ok=True)

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

def get_problem_dirs(cots_dir: Path, limit: Optional[int] = None) -> List[Path]:
    """
    Get all problem directories in the CoTs directory.
    
    Args:
        cots_dir: Path to the CoTs directory
        limit: Optional limit on number of directories to return
        
    Returns:
        List of problem directory paths
    """
    problem_dirs = [d for d in cots_dir.iterdir() if d.is_dir() and d.name.startswith("problem_")]
    
    if limit:
        return problem_dirs[:limit]
    
    return problem_dirs

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
        device_map=device,
    )
    
    # Set model to evaluation mode
    model.eval()
    
    print('Number of layers:', model.config.num_hidden_layers)
    print('Number of attention heads:', model.config.num_attention_heads)
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

def load_chunks_data(problem_dir: Path, seed_dir: Path) -> List[Dict]:
    """
    Load chunks data from a problem directory and seed directory.
    
    Args:
        problem_dir: Path to the problem directory
        seed_dir: Path to the seed directory
        
    Returns:
        List of chunk dictionaries
    """
    chunks_path = seed_dir / "chunks.json"
    if not chunks_path.exists():
        return []
    
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    return chunks

def extract_chunk_activations(
    problem_dirs: List[Path],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    layer: int,
    max_problems: int = 100,
    pooling_method: str = "attention_pooling",
    multi_layer: bool = False,
    layer_weights: Optional[List[float]] = None
) -> Tuple[List[torch.Tensor], List[Dict]]:
    """
    Extract residual stream activations for chunks across multiple problems.
    
    Args:
        problem_dirs: List of problem directory paths
        model: Hugging Face model
        tokenizer: Tokenizer
        layer: Layer to extract activations from (or base layer if multi_layer=True)
        max_problems: Maximum number of problems to process
        pooling_method: Method to use for pooling ('attention_pooling', 'cls_pooling', 
                        'first_last_pooling', 'max_pooling', 'svd_pooling', 'attention_weighted',
                        'stats_pooling', or 'mean')
        multi_layer: Whether to use multiple layers
        layer_weights: Weights for each layer if multi_layer=True
        
    Returns:
        Tuple of (activations, chunk_metadata)
    """
    all_activations = []
    all_chunk_metadata = []
    
    # Determine which layers to extract
    if multi_layer:
        # Use multiple layers
        num_layers = model.config.num_hidden_layers
        if layer_weights is None:
            # Default to using last 4 layers with equal weights
            layers_to_extract = [num_layers - 4, num_layers - 3, num_layers - 2, num_layers - 1]
            layer_weights = [0.25, 0.25, 0.25, 0.25]
        else:
            # Use specified weights, centered around the base layer
            half_window = len(layer_weights) // 2
            base_idx = min(layer, num_layers - 1)
            layers_to_extract = [max(0, base_idx - half_window + i) for i in range(len(layer_weights))]
    else:
        # Use single layer
        layers_to_extract = [layer]
        layer_weights = [1.0]
    
    # Process a limited number of problems
    for problem_dir in tqdm(problem_dirs[:max_problems], desc="Processing problems"):
        # Load problem and solutions
        problem, solutions = load_problem_and_solutions(problem_dir)
        if not solutions:
            continue
        
        # Process all seeds, not just seed_0
        analysis_dir = Path("analysis") / problem_dir.name
        seed_dirs = [d for d in analysis_dir.iterdir() if d.is_dir() and d.name.startswith("seed_")]
        
        for seed_dir in seed_dirs:
            seed_id = seed_dir.name
            seed_num = int(seed_id.replace("seed_", ""))
            
            # Find the corresponding solution for this seed
            solution_dict = None
            for sol in solutions:
                if sol.get("seed") == seed_num:
                    solution_dict = sol
                    break
            
            if not solution_dict:
                continue
                
            full_text = solution_dict["solution"]
            
            # Load chunks data
            chunks_data = load_chunks_data(problem_dir, seed_dir)
            
            if not chunks_data:
                continue
                
            # Extract chunks
            chunks = [item["text"] for item in chunks_data]
            
            # Special case for gradient projection pooling
            if pooling_method == "gradient_projection":
                try:
                    # Get token indices for each chunk in the full text
                    chunk_ranges = get_chunk_ranges(full_text, chunks)
                    chunk_token_ranges = []

                    for (chunk_start, chunk_end) in chunk_ranges:
                        # Convert character positions to token indices
                        chunk_tokens = tokenizer(full_text[:chunk_start], return_tensors="pt")
                        start_idx = len(chunk_tokens.input_ids[0]) - 1
                        
                        chunk_tokens = tokenizer(full_text[:chunk_end], return_tensors="pt")
                        end_idx = len(chunk_tokens.input_ids[0])
                        
                        chunk_token_ranges.append((max(0, start_idx), end_idx))
                    
                    # Get gradient-based representations for all chunks
                    chunk_reps = get_gradient_projection_pooling(model, tokenizer, full_text, layer, chunk_token_ranges)
                    
                    # Add activations and metadata
                    for i, chunk_rep in enumerate(chunk_reps):
                        if i >= len(chunks_data):
                            continue
                        
                        all_activations.append(chunk_rep)
                        
                        # Store metadata about this chunk
                        metadata = {
                            "problem_id": problem_dir.name,
                            "seed_id": seed_id,
                            "chunk_idx": i,
                            "chunk_text": chunks[i][:100] + "..." if len(chunks[i]) > 100 else chunks[i],
                            "category": chunks_data[i].get("category", "Unknown"),
                            "full_chunk": chunks_data[i]
                        }
                        all_chunk_metadata.append(metadata)
                except Exception as e:
                    print(f"Error processing chunk {i}: {e}")
                    continue
                
                # Skip the regular processing for this seed
                continue
            
            # Get token indices for each chunk in the full text
            chunk_ranges = get_chunk_ranges(full_text, chunks)
            chunk_token_ranges = []

            for (chunk_start, chunk_end) in chunk_ranges:
                # Convert character positions to token indices
                chunk_tokens = tokenizer(full_text[:chunk_start], return_tensors="pt")
                start_idx = len(chunk_tokens.input_ids[0]) - 1
                
                chunk_tokens = tokenizer(full_text[:chunk_end], return_tensors="pt")
                end_idx = len(chunk_tokens.input_ids[0])
                
                chunk_token_ranges.append((max(0, start_idx), end_idx))
            
            # Get activations for the full text
            with torch.no_grad():
                activations = get_residual_stream_activations(model, tokenizer, full_text, layers_to_extract)
                
                # If using attention_weighted pooling, we also need attention matrices
                if pooling_method == "attention_weighted":
                    # Prepare inputs
                    inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
                    
                    # Forward pass with output_attentions=True to get attention matrices
                    outputs = model(**inputs, output_attentions=True)
                    
                    # Get attention matrices for the specified layers
                    # Shape: [batch_size, num_heads, seq_len, seq_len]
                    attention_matrices = {}
                    for layer_idx in layers_to_extract:
                        attention_matrices[layer_idx] = outputs.attentions[layer_idx].detach().cpu()
            
            # Check if we have activations for all required layers
            if not all(layer in activations for layer in layers_to_extract):
                continue
                
            # Process each chunk
            for i, (start_idx, end_idx) in enumerate(chunk_token_ranges):
                if i >= len(chunks_data):
                    continue
                
                if multi_layer:
                    # Combine activations from multiple layers
                    chunk_reps = []
                    for layer_idx, weight in zip(layers_to_extract, layer_weights):
                        layer_activations = activations[layer_idx]
                        chunk_act = layer_activations[0, start_idx:end_idx, :]
                        chunk_act = chunk_act.to(torch.float32)  # Convert to float32
                        
                        # Apply pooling method
                        if pooling_method == "positional_aware":
                            chunk_rep = get_positional_aware_pooling(
                                chunk_act, 
                                hidden_size=chunk_act.size(1),
                                kernel_size=3, 
                                fixed_kernel=True
                            )
                        elif pooling_method == "frequency_domain":
                            chunk_rep = get_frequency_domain_pooling(chunk_act, k=10)
                        elif pooling_method == "attention_pooling":
                            weights = chunk_act.norm(dim=-1)
                            if weights.sum() > 0:
                                chunk_rep = (chunk_act * weights.unsqueeze(-1)).sum(dim=0) / weights.sum()
                            else:
                                chunk_rep = chunk_act.mean(dim=0)
                        elif pooling_method == "cls_pooling":
                            chunk_rep = chunk_act[0] if chunk_act.size(0) > 0 else torch.zeros_like(chunk_act[0])
                        elif pooling_method == "first_last_pooling":
                            if chunk_act.size(0) > 1:
                                chunk_rep = torch.cat([chunk_act[0], chunk_act[-1]])
                            else:
                                chunk_rep = chunk_act[0] if chunk_act.size(0) > 0 else torch.zeros_like(chunk_act[0])
                        elif pooling_method == "max_pooling":
                            chunk_rep = torch.max(chunk_act, dim=0)[0] if chunk_act.size(0) > 0 else torch.zeros_like(chunk_act[0])
                        elif pooling_method == "svd_pooling":
                            if chunk_act.size(0) > 1:
                                try:
                                    U, S, V = torch.svd(chunk_act)
                                    chunk_rep = V[:, 0]
                                except:
                                    # Fallback if SVD fails
                                    chunk_rep = chunk_act.mean(dim=0)
                            else:
                                chunk_rep = chunk_act[0] if chunk_act.size(0) > 0 else torch.zeros_like(chunk_act[0])
                        elif pooling_method == "attention_weighted":
                            # Get attention matrix for this layer
                            attn_matrix = attention_matrices[layer_idx]
                            
                            # Average across attention heads
                            avg_attn = attn_matrix.mean(dim=1)[0]  # Shape: [seq_len, seq_len]
                            
                            # Get attention weights for tokens in this chunk
                            # For each token in the chunk, look at what it attends to
                            chunk_attn = avg_attn[start_idx:end_idx, :]  # Shape: [chunk_len, seq_len]
                            
                            # Normalize attention weights within the chunk
                            chunk_attn_sum = chunk_attn.sum(dim=1, keepdim=True)
                            chunk_attn_norm = chunk_attn / chunk_attn_sum.clamp(min=1e-10)
                            
                            # Get weighted average of all token representations based on attention
                            all_token_reps = layer_activations[0]  # Shape: [seq_len, hidden_size]
                            
                            # For each token in the chunk, compute its representation as a weighted average
                            # of all tokens it attends to
                            weighted_reps = []
                            for token_idx in range(chunk_attn.size(0)):
                                token_attn = chunk_attn_norm[token_idx]  # Shape: [seq_len]
                                weighted_rep = (all_token_reps * token_attn.unsqueeze(-1)).sum(dim=0)
                                weighted_reps.append(weighted_rep)
                            
                            # Stack and average across tokens in the chunk
                            if weighted_reps:
                                chunk_rep = torch.stack(weighted_reps).mean(dim=0)
                            else:
                                chunk_rep = torch.zeros_like(layer_activations[0, 0])
                        elif pooling_method == "stats_pooling":
                            # Calculate multiple statistics and concatenate them
                            mean_rep = chunk_act.mean(dim=0)
                            min_rep = torch.min(chunk_act, dim=0)[0]
                            max_rep = torch.max(chunk_act, dim=0)[0]
                            
                            # For median and std, handle the case where there's only one token
                            if chunk_act.size(0) > 1:
                                median_rep = torch.median(chunk_act, dim=0)[0]
                                std_rep = torch.std(chunk_act, dim=0)
                            else:
                                median_rep = chunk_act[0]
                                std_rep = torch.zeros_like(chunk_act[0])
                            
                            # Concatenate all statistics
                            chunk_rep = torch.cat([mean_rep, min_rep, max_rep, median_rep, std_rep])
                        else:  # Default to mean pooling
                            chunk_rep = chunk_act.mean(dim=0)
                        
                        chunk_reps.append(weight * chunk_rep)
                    
                    # Sum the weighted representations
                    final_rep = sum(chunk_reps)
                    all_activations.append(final_rep)
                else:
                    # Use single layer
                    layer_activations = activations[layers_to_extract[0]]
                    chunk_act = layer_activations[0, start_idx:end_idx, :]
                    chunk_act = chunk_act.to(torch.float32)  # Convert to float32
                    
                    try:
                    
                        # Apply pooling method
                        if pooling_method == "positional_aware":
                            chunk_rep = get_positional_aware_pooling(
                                chunk_act, 
                                hidden_size=chunk_act.size(1),
                                kernel_size=3, 
                                fixed_kernel=True
                            )
                        elif pooling_method == "frequency_domain":
                            chunk_rep = get_frequency_domain_pooling(chunk_act, k=5)
                        elif pooling_method == "attention_pooling":
                            weights = chunk_act.norm(dim=-1)
                            if weights.sum() > 0:
                                chunk_rep = (chunk_act * weights.unsqueeze(-1)).sum(dim=0) / weights.sum()
                            else:
                                chunk_rep = chunk_act.mean(dim=0)
                        elif pooling_method == "cls_pooling":
                            chunk_rep = chunk_act[0] if chunk_act.size(0) > 0 else torch.zeros_like(chunk_act[0])
                        elif pooling_method == "first_last_pooling":
                            if chunk_act.size(0) > 1:
                                chunk_rep = torch.cat([chunk_act[0], chunk_act[-1]])
                            else:
                                chunk_rep = chunk_act[0] if chunk_act.size(0) > 0 else torch.zeros_like(chunk_act[0])
                        elif pooling_method == "max_pooling":
                            chunk_rep = torch.max(chunk_act, dim=0)[0] if chunk_act.size(0) > 0 else torch.zeros_like(chunk_act[0])
                        elif pooling_method == "svd_pooling":
                            if chunk_act.size(0) > 1:
                                try:
                                    U, S, V = torch.svd(chunk_act)
                                    chunk_rep = V[:, 0]
                                except:
                                    # Fallback if SVD fails
                                    chunk_rep = chunk_act.mean(dim=0)
                            else:
                                chunk_rep = chunk_act[0] if chunk_act.size(0) > 0 else torch.zeros_like(chunk_act[0])
                        elif pooling_method == "attention_weighted":
                            # Get attention matrix for this layer
                            attn_matrix = attention_matrices[layers_to_extract[0]]
                            
                            # Average across attention heads
                            avg_attn = attn_matrix.mean(dim=1)[0]  # Shape: [seq_len, seq_len]
                            
                            # Get attention weights for tokens in this chunk
                            # For each token in the chunk, look at what it attends to
                            chunk_attn = avg_attn[start_idx:end_idx, :]  # Shape: [chunk_len, seq_len]
                            
                            # Normalize attention weights within the chunk
                            chunk_attn_sum = chunk_attn.sum(dim=1, keepdim=True)
                            chunk_attn_norm = chunk_attn / chunk_attn_sum.clamp(min=1e-10)
                            
                            # Get weighted average of all token representations based on attention
                            all_token_reps = layer_activations[0]  # Shape: [seq_len, hidden_size]
                            
                            # For each token in the chunk, compute its representation as a weighted average
                            # of all tokens it attends to
                            weighted_reps = []
                            for token_idx in range(chunk_attn.size(0)):
                                token_attn = chunk_attn_norm[token_idx]  # Shape: [seq_len]
                                weighted_rep = (all_token_reps * token_attn.unsqueeze(-1)).sum(dim=0)
                                weighted_reps.append(weighted_rep)
                            
                            # Stack and average across tokens in the chunk
                            if weighted_reps:
                                chunk_rep = torch.stack(weighted_reps).mean(dim=0)
                            else:
                                chunk_rep = torch.zeros_like(layer_activations[0, 0])
                        elif pooling_method == "stats_pooling":
                            # Calculate multiple statistics and concatenate them
                            mean_rep = chunk_act.mean(dim=0)
                            min_rep = torch.min(chunk_act, dim=0)[0]
                            max_rep = torch.max(chunk_act, dim=0)[0]
                            
                            # For median and std, handle the case where there's only one token
                            if chunk_act.size(0) > 1:
                                median_rep = torch.median(chunk_act, dim=0)[0]
                                std_rep = torch.std(chunk_act, dim=0)
                            else:
                                median_rep = chunk_act[0]
                                std_rep = torch.zeros_like(chunk_act[0])
                            
                            # Concatenate all statistics
                            chunk_rep = torch.cat([mean_rep, min_rep, max_rep, median_rep, std_rep])
                        else:  # Default to mean pooling
                            chunk_rep = chunk_act.mean(dim=0)
                        
                        all_activations.append(chunk_rep)
                    except Exception as e:
                        print(f"Error processing chunk {i}: {e}")
                        continue
                
                # Store metadata about this chunk
                metadata = {
                    "problem_id": problem_dir.name,
                    "seed_id": seed_id,
                    "chunk_idx": i,
                    "chunk_text": chunks[i][:100] + "..." if len(chunks[i]) > 100 else chunks[i],
                    "category": chunks_data[i].get("category", "Unknown"),
                    "full_chunk": chunks_data[i]
                }
                all_chunk_metadata.append(metadata)
    
    return all_activations, all_chunk_metadata

def get_positional_aware_pooling(
    chunk_act: torch.Tensor,
    hidden_size: int = None,
    kernel_size: int = 3,
    fixed_kernel: bool = True
) -> torch.Tensor:
    """
    Pool chunk activations using positional-aware projection with 1D convolution.
    
    Args:
        chunk_act: Chunk activations tensor of shape [seq_len, hidden_dim]
        hidden_size: Hidden dimension size (if None, inferred from chunk_act)
        kernel_size: Size of the convolutional kernel
        fixed_kernel: Whether to use a fixed kernel (Gaussian) or random initialization
        
    Returns:
        Pooled representation
    """
    # Handle edge case of very short sequences
    if chunk_act.size(0) <= 1:
        return chunk_act.mean(dim=0)
    
    # Get hidden dimension if not provided
    if hidden_size is None:
        hidden_size = chunk_act.size(1)
    
    # Create convolutional layer
    conv = torch.nn.Conv1d(
        in_channels=hidden_size,
        out_channels=hidden_size,
        kernel_size=kernel_size,
        padding=kernel_size // 2,  # Same padding
        bias=False  # No bias for simplicity
    )
    
    # If using fixed kernel, initialize with Gaussian weights
    if fixed_kernel:
        # Create Gaussian kernel
        center = kernel_size // 2
        sigma = kernel_size / 6.0  # Standard deviation
        kernel = torch.zeros(kernel_size)
        for i in range(kernel_size):
            kernel[i] = torch.exp(torch.tensor(-0.5 * ((i - center) / sigma) ** 2))
        kernel = kernel / kernel.sum()  # Normalize
        
        # Expand to full conv weights [out_channels, in_channels, kernel_size]
        kernel = kernel.view(1, 1, kernel_size).repeat(hidden_size, 1, 1)
        
        # Set as conv weights
        with torch.no_grad():
            conv.weight.copy_(kernel)
    
    # Prepare input for convolution [batch, channels, seq_len]
    x = chunk_act.transpose(0, 1).unsqueeze(0)  # [1, hidden_dim, seq_len]
    
    # Apply convolution
    with torch.no_grad():
        pooled = conv(x).squeeze(0)  # [hidden_dim, seq_len]
    
    # Pool across sequence dimension
    # We can use different pooling strategies here
    pooled_mean = pooled.mean(dim=1)  # Mean pooling [hidden_dim]
    pooled_max, _ = pooled.max(dim=1)  # Max pooling [hidden_dim]
    
    # Concatenate different pooling results
    # return torch.cat([pooled_mean, pooled_max])
    return pooled_mean

def prepare_activation_matrix(activations: List[torch.Tensor]) -> np.ndarray:
    """
    Prepare activation matrix for clustering.
    
    Args:
        activations: List of activation tensors
        
    Returns:
        Numpy array of activations
    """
    # Convert tensors to numpy arrays
    activation_arrays = [act.numpy() for act in activations]
    
    # Ensure all activations have the same dimension
    max_dim = max(act.shape[0] for act in activation_arrays)
    padded_activations = []
    
    for act in activation_arrays:
        if act.shape[0] < max_dim:
            # Create a new array with the maximum dimension
            padded = np.zeros(max_dim)
            # Only copy as many elements as we have
            padded[:act.shape[0]] = act
            padded_activations.append(padded)
        else:
            padded_activations.append(act)
    
    # Stack into a 2D array
    return np.vstack(padded_activations)

def run_kmeans_clustering(
    activation_matrix: np.ndarray, 
    n_clusters: int = 10
) -> Tuple[np.ndarray, KMeans]:
    """
    Run K-means clustering on activation matrix.
    
    Args:
        activation_matrix: Matrix of activations
        n_clusters: Number of clusters
        
    Returns:
        Tuple of (cluster labels, kmeans model)
    """
    print(f"Running K-means clustering with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=100)
    labels = kmeans.fit_predict(activation_matrix)
    
    # Calculate silhouette score
    silhouette = silhouette_score(activation_matrix, labels)
    print(f"Silhouette score: {silhouette:.4f}")
    
    return labels, kmeans

def run_dbscan_clustering(
    activation_matrix: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 10
) -> np.ndarray:
    """
    Run DBSCAN clustering on activation matrix.
    
    Args:
        activation_matrix: Matrix of activations
        eps: Maximum distance between samples
        min_samples: Minimum number of samples in a neighborhood
        
    Returns:
        Cluster labels
    """
    print(f"Running DBSCAN clustering with eps={eps}, min_samples={min_samples}...")
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(activation_matrix)
    
    # Count number of clusters (excluding noise points labeled as -1)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Number of clusters found: {n_clusters}")
    
    # Calculate silhouette score if there are at least 2 clusters and no noise points
    if n_clusters >= 2 and -1 not in labels:
        silhouette = silhouette_score(activation_matrix, labels)
        print(f"Silhouette score: {silhouette:.4f}")
    
    return labels

def run_hierarchical_clustering(
    activation_matrix: np.ndarray,
    n_clusters: int = 10
) -> np.ndarray:
    """
    Run hierarchical clustering on activation matrix.
    
    Args:
        activation_matrix: Matrix of activations
        n_clusters: Number of clusters
        
    Returns:
        Cluster labels
    """
    print(f"Running hierarchical clustering with {n_clusters} clusters...")
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    labels = clustering.fit_predict(activation_matrix)
    
    # Calculate silhouette score
    silhouette = silhouette_score(activation_matrix, labels)
    print(f"Silhouette score: {silhouette:.4f}")
    
    return labels

def run_dimensionality_reduction(
    activation_matrix: np.ndarray,
    method: str = "tsne",
    n_components: int = 2,
    random_state: int = 42,
    perplexity: int = 30
) -> np.ndarray:
    """
    Run dimensionality reduction on activation matrix.
    
    Args:
        activation_matrix: Matrix of activations
        method: Method to use ('tsne', 'pca', or 'umap')
        n_components: Number of components
        random_state: Random state for reproducibility
        perplexity: Perplexity for t-SNE
    Returns:
        Reduced matrix
    """
    print(f"Running {method.upper()} dimensionality reduction...")
    
    if method == "tsne":
        reducer = TSNE(n_components=n_components, random_state=random_state, perplexity=perplexity)
        reduced_matrix = reducer.fit_transform(activation_matrix)
    elif method == "pca":
        reducer = PCA(n_components=n_components, random_state=random_state)
        reduced_matrix = reducer.fit_transform(activation_matrix)
    elif method == "umap":
        reducer = umap.UMAP(n_components=n_components, random_state=random_state)
        reduced_matrix = reducer.fit_transform(activation_matrix)
    else:
        raise ValueError(f"Unknown dimensionality reduction method: {method}")
    
    return reduced_matrix

def plot_clusters(
    reduced_matrix: np.ndarray,
    labels: np.ndarray,
    metadata: List[Dict],
    output_path: Path,
    title: str = "Chunk Clusters",
    color_by: str = "cluster"
):
    """
    Plot clusters in 2D space.
    
    Args:
        reduced_matrix: Reduced matrix from dimensionality reduction
        labels: Cluster labels
        metadata: Chunk metadata
        output_path: Path to save the plot
        title: Plot title
        color_by: What to color points by ('cluster' or 'category')
    """
    plt.figure(figsize=(12, 10))
    
    if color_by == "cluster":
        # Color by cluster
        scatter = plt.scatter(reduced_matrix[:, 0], reduced_matrix[:, 1], c=labels, cmap='tab10', alpha=0.7)
        
        # Add legend for clusters
        unique_labels = np.unique(labels)
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.tab10(i / 10), markersize=10) for i in range(len(unique_labels))]
        plt.legend(handles, [f'Cluster {label}' for label in unique_labels], loc='upper left')
    
    elif color_by == "category":
        # Extract categories
        categories = [meta.get("category", "Unknown") for meta in metadata]
        unique_categories = sorted(set(categories))
        
        # Use a colormap with more distinct colors
        # Define a list of distinct colors manually
        distinct_colors = [
            '#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', 
            '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4', 
            '#469990', '#dcbeff', '#9A6324', '#fffac8', '#800000', 
            '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9',
            '#000000', '#ffffff', '#1f77b4', '#ff7f0e', '#2ca02c',
            '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'
        ]
        
        # If we have more categories than colors, cycle through the colors
        if len(unique_categories) > len(distinct_colors):
            distinct_colors = distinct_colors * (len(unique_categories) // len(distinct_colors) + 1)
        
        # Create a mapping from category to color
        category_to_color = {cat: distinct_colors[i] for i, cat in enumerate(unique_categories)}
        
        # Create a list of colors for each point
        point_colors = [category_to_color[cat] for cat in categories]
        
        # Plot with distinct colors
        scatter = plt.scatter(reduced_matrix[:, 0], reduced_matrix[:, 1], c=point_colors, alpha=0.7)
        
        # Add legend for categories
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) 
                  for cat, color in category_to_color.items()]
        
        # Add count of each category to the legend
        category_counts = Counter(categories)
        legend_labels = [f"{cat} ({category_counts[cat]})" for cat in unique_categories]
        
        plt.legend(handles, legend_labels, loc='upper left', ncol=2)
    
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def analyze_clusters(
    labels: np.ndarray,
    metadata: List[Dict],
    output_path: Path
):
    """
    Analyze cluster contents and save results.
    
    Args:
        labels: Cluster labels
        metadata: Chunk metadata
        output_path: Path to save the analysis
    """
    # Create a DataFrame for analysis
    df = pd.DataFrame({
        "problem_id": [meta["problem_id"] for meta in metadata],
        "chunk_idx": [meta["chunk_idx"] for meta in metadata],
        "category": [meta.get("category", "Unknown") for meta in metadata],
        "cluster": labels
    })
    
    # Count categories per cluster
    cluster_categories = defaultdict(lambda: defaultdict(int))
    for cluster, category in zip(labels, df["category"]):
        # Convert NumPy int32 to Python int
        cluster_key = int(cluster)
        cluster_categories[cluster_key][category] += 1
    
    # Prepare analysis results
    analysis = {
        "num_clusters": len(set(labels)),
        "cluster_sizes": {int(k): v for k, v in Counter(labels).items()},
        "cluster_categories": {int(k): dict(v) for k, v in cluster_categories.items()},
        "dominant_categories": {}
    }
    
    # Find dominant category for each cluster
    for cluster, categories in cluster_categories.items():
        if categories:
            dominant_category = max(categories.items(), key=lambda x: x[1])
            # Convert NumPy int32 to Python int
            cluster_key = int(cluster)
            analysis["dominant_categories"][cluster_key] = {
                "category": dominant_category[0],
                "count": dominant_category[1],
                "percentage": dominant_category[1] / sum(categories.values()) * 100
            }
    
    # Save analysis
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2)
    
    # Print summary
    print("\nCluster Analysis Summary:")
    print(f"Number of clusters: {analysis['num_clusters']}")
    print("\nCluster sizes:")
    for cluster, size in sorted(analysis["cluster_sizes"].items()):
        print(f"  Cluster {cluster}: {size} chunks")
    
    print("\nDominant categories per cluster:")
    for cluster, info in sorted(analysis["dominant_categories"].items()):
        print(f"  Cluster {cluster}: {info['category']} ({info['percentage']:.1f}%)")

def get_frequency_domain_pooling(
    chunk_act: torch.Tensor, 
    k: int = 5
) -> torch.Tensor:
    """
    Pool chunk activations using frequency domain transform (FFT).
    
    Args:
        chunk_act: Chunk activations tensor of shape [seq_len, hidden_dim]
        k: Number of top frequency components to keep
        
    Returns:
        Pooled representation
    """
    # Handle edge case of very short sequences
    if chunk_act.size(0) <= 1:
        return chunk_act.mean(dim=0)
    
    # Apply FFT along the token dimension
    fft_rep = torch.fft.rfft(chunk_act, dim=0)  # shape: [seq_len//2+1, hidden_dim]
    
    # Get magnitude of complex FFT coefficients
    fft_mag = torch.abs(fft_rep)
    
    # If we have fewer frequency components than k, pad with zeros
    if fft_mag.size(0) < k:
        # Pad with zeros to get k components
        padding = torch.zeros((k - fft_mag.size(0), fft_mag.size(1)), device=fft_mag.device)
        fft_mag = torch.cat([fft_mag, padding], dim=0)
    
    # Get top-k frequency components for each dimension
    # This preserves the most important frequency patterns
    top_k_values, _ = torch.topk(fft_mag, k=k, dim=0)
    
    # Flatten to get final representation
    return top_k_values.flatten()


def find_optimal_k(
    activation_matrix: np.ndarray,
    max_k: int = 30,
) -> int:
    """
    Find optimal number of clusters using silhouette score.
    
    Args:
        activation_matrix: Matrix of activations
        max_k: Maximum number of clusters to try
        
    Returns:
        Optimal number of clusters
    """
    print("Finding optimal number of clusters...")
    silhouette_scores = []
    
    # Try different values of k
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(activation_matrix)
        score = silhouette_score(activation_matrix, labels)
        silhouette_scores.append(score)
        print(f"  k={k}, silhouette={score:.4f}")
    
    # Find k with highest silhouette score
    optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2  # +2 because we start at k=2
    
    # Plot silhouette scores
    plt.figure(figsize=(10, 6))
    plt.plot(range(2, max_k + 1), silhouette_scores, 'o-')
    plt.axvline(x=optimal_k, color='r', linestyle='--')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs. Number of Clusters')
    plt.grid(True)
    plt.savefig(output_dir / "silhouette_scores.png", dpi=300)
    plt.close()
    
    print(f"Optimal number of clusters: {optimal_k}")
    return optimal_k

def get_gradient_projection_pooling(
    model: AutoModelForCausalLM, 
    tokenizer: AutoTokenizer,
    text: str, 
    layer_idx: int,
    chunk_ranges: List[Tuple[int, int]]
) -> List[torch.Tensor]:
    """
    Pool chunk activations using gradient-based attribution.
    
    Args:
        model: The language model
        tokenizer: Tokenizer
        text: Full text input
        layer_idx: Layer to extract activations from
        chunk_ranges: List of (start_idx, end_idx) token ranges for each chunk
        
    Returns:
        List of pooled representations for each chunk
    """
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids
    
    # Process each chunk
    chunk_representations = []
    
    for chunk_start, chunk_end in chunk_ranges:
        # We need to run a separate forward pass for each chunk to get proper gradients
        model.zero_grad()
        
        # Set up hooks to capture and modify the activations
        activations = {}
        gradients = {}
        
        def save_activations_hook(name):
            def hook(module, input, output):
                # Store the output (activations)
                if isinstance(output, tuple):
                    activations[name] = output[0].detach().clone()
                    # Make it require gradients
                    activations[name].requires_grad_(True)
                    return (activations[name],) + output[1:]
                else:
                    activations[name] = output.detach().clone()
                    # Make it require gradients
                    activations[name].requires_grad_(True)
                    return activations[name]
            return hook
        
        def save_gradients_hook(name):
            def hook(grad):
                gradients[name] = grad.clone()
            return hook
        
        # Register hooks
        layer_name = f"layer_{layer_idx}"
        handle = model.model.layers[layer_idx].register_forward_hook(save_activations_hook(layer_name))
        
        # Forward pass
        outputs = model(**inputs, output_hidden_states=True)
        logits = outputs.logits
        
        # Get the activations
        if layer_name not in activations:
            print(f"Warning: Layer {layer_name} activations not captured. Available keys: {list(activations.keys())}")
            # Try to find the correct layer name
            available_layers = [k for k in activations.keys() if 'layer' in k.lower()]
            if available_layers:
                layer_name = available_layers[0]
                print(f"Using {layer_name} instead")
            else:
                # Fallback to mean pooling
                chunk_rep = torch.zeros(model.config.hidden_size * 2)
                chunk_representations.append(chunk_rep)
                handle.remove()
                continue
        
        # Register backward hook to capture gradients
        activations[layer_name].register_hook(save_gradients_hook(layer_name))
        
        # Create a target for this chunk (prediction logits for the last token in the chunk)
        target_pos = min(chunk_end, input_ids.size(1) - 1)
        target = logits[0, target_pos].sum()
        
        # Backward pass
        target.backward(retain_graph=True)
        
        # Check if gradients were captured
        if layer_name not in gradients:
            print(f"Warning: No gradients captured for {layer_name}")
            # Fallback to uniform weighting
            chunk_act = activations[layer_name][0, chunk_start:chunk_end].detach()
            chunk_rep = torch.cat([
                chunk_act.mean(dim=0),
                chunk_act[0] if chunk_act.size(0) > 0 else torch.zeros_like(chunk_act[0])
            ])
        else:
            # Get activations and gradients for the chunk tokens
            chunk_act = activations[layer_name][0, chunk_start:chunk_end].detach()
            chunk_grad = gradients[layer_name][0, chunk_start:chunk_end].detach()
            
            # Compute importance weights based on gradient magnitudes
            grad_norm = chunk_grad.norm(dim=1, keepdim=True)
            
            # Handle case where gradients are zero
            if grad_norm.sum() > 0:
                grad_weights = grad_norm / grad_norm.sum()
            else:
                # Fallback to uniform weights
                grad_weights = torch.ones_like(grad_norm) / grad_norm.size(0)
            
            # Weight activations by gradient importance
            weighted_act = chunk_act * grad_weights
            
            # Sum across tokens for gradient-weighted representation
            pooled_sum = weighted_act.sum(dim=0)
            
            # Also include max-weighted activation as additional signal
            if chunk_act.size(0) > 0:
                max_idx = grad_weights.squeeze().argmax()
                max_act = chunk_act[max_idx]
            else:
                max_act = torch.zeros_like(pooled_sum)
            
            # Concatenate for final representation
            chunk_rep = torch.cat([pooled_sum, max_act])
        
        chunk_representations.append(chunk_rep.cpu())
        
        # Remove hook
        handle.remove()
    
    return chunk_representations

def main():
    parser = argparse.ArgumentParser(description="Cluster chunks based on residual stream activations")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", help="Model name")
    parser.add_argument("--layer", type=int, default=31, help="Layer to extract activations from")
    parser.add_argument("--max_problems", type=int, default=100, help="Maximum number of problems to process")
    parser.add_argument("--n_clusters", type=int, default=0, help="Number of clusters (0 to find optimal)")
    parser.add_argument("--dim_reduction", type=str, default="tsne", choices=["tsne", "pca", "umap"], help="Dimensionality reduction method")
    parser.add_argument("--pooling", type=str, default="attention_pooling", choices=["mean", "attention_pooling", "cls_pooling", "first_last_pooling", "max_pooling", "svd_pooling", "attention_weighted", "stats_pooling"], help="Method to pool token representations")
    parser.add_argument("--multi_layer", action="store_true", help="Use multiple layers")
    parser.add_argument("--layer_weights", type=str, default="0.1,0.2,0.3,0.4", help="Comma-separated weights for layers when using multi_layer")
    args = parser.parse_args()
    
    # Parse layer weights if using multi-layer
    layer_weights = None
    if args.multi_layer:
        layer_weights = [float(w) for w in args.layer_weights.split(",")]
        # Normalize weights to sum to 1
        total = sum(layer_weights)
        if total > 0:
            layer_weights = [w / total for w in layer_weights]
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model)
    
    # Get problem directories
    problem_dirs = get_problem_dirs(cots_dir)
    
    # Extract chunk activations
    activations, chunk_metadata = extract_chunk_activations(
        problem_dirs, model, tokenizer, args.layer, args.max_problems,
        pooling_method=args.pooling,
        multi_layer=args.multi_layer,
        layer_weights=layer_weights
    )
    
    if not activations:
        print("No activations extracted. Exiting.")
        return
    
    print(f"Extracted {len(activations)} chunks")
    
    # Prepare activation matrix
    activation_matrix = prepare_activation_matrix(activations)
    
    # Standardize features
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(activation_matrix)
    
    # Find optimal number of clusters if not specified
    if args.n_clusters <= 0:
        n_clusters = find_optimal_k(scaled_matrix)
    else:
        n_clusters = args.n_clusters
    
    # Run K-means clustering
    labels, kmeans = run_kmeans_clustering(scaled_matrix, n_clusters=n_clusters)
    
    # Run dimensionality reduction
    reduced_matrix = run_dimensionality_reduction(scaled_matrix, method=args.dim_reduction)
    
    # Plot clusters
    pooling_str = f"{args.pooling}_{'multi' if args.multi_layer else 'single'}"
    output_path = output_dir / f"clusters_layer{args.layer}_{pooling_str}_{args.dim_reduction}.png"
    plot_clusters(reduced_matrix, labels, chunk_metadata, output_path, 
                 title=f"Chunk Clusters (Layer {args.layer}, {pooling_str}, {n_clusters} clusters)")
    
    # Plot by category
    output_path = output_dir / f"categories_layer{args.layer}_{pooling_str}_{args.dim_reduction}.png"
    plot_clusters(reduced_matrix, labels, chunk_metadata, output_path, 
                 title=f"Chunk Categories (Layer {args.layer}, {pooling_str})", color_by="category")
    
    # Analyze clusters
    analysis_path = output_dir / f"cluster_analysis_layer{args.layer}_{pooling_str}.json"
    analyze_clusters(labels, chunk_metadata, analysis_path)
    
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()