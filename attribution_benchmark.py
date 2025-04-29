import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from openai import OpenAI
from dotenv import load_dotenv
import torch
import scipy.stats as stats
import gc
from sklearn.metrics import ndcg_score
import traceback
from utils import get_chunk_ranges, get_chunk_token_ranges
import re

# Load environment variables
load_dotenv()

# Set up OpenAI API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if not client.api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")

# Global font size for plots
FONT_SIZE = 15

# Set font size for all plots
plt.rcParams.update({
    'font.size': FONT_SIZE,
    'axes.titlesize': FONT_SIZE + 2,
    'axes.labelsize': FONT_SIZE,
    'xtick.labelsize': FONT_SIZE - 2,
    'ytick.labelsize': FONT_SIZE - 2,
    'legend.fontsize': FONT_SIZE - 2,
    'figure.titlesize': FONT_SIZE + 4
})

def load_model_and_tokenizer(model_name: str) -> Tuple[Any, AutoTokenizer]:
    """
    Load model and tokenizer.
    
    Args:
        model_name: Name of the model to load
        
    Returns:
        Tuple of model and tokenizer
    """
    print(f"Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # Import bitsandbytes for quantization
        import bitsandbytes as bnb
        from transformers import BitsAndBytesConfig, AutoModelForCausalLM
        
        # Configure 4-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        print("Loading model with 4-bit quantization...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map=device,
            attn_implementation="eager",
        )
        print("Model loaded with 4-bit quantization")
        
    except ImportError:
        print("bitsandbytes not installed. Falling back to 16-bit precision.")
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device,
            attn_implementation="eager"
        )
    
    # Set model to evaluation mode
    model.eval()
    
    return model, tokenizer

def get_problem_dirs(analysis_dir: Path, correct_only: bool = True, limit: Optional[int] = None) -> List[Path]:
    """
    Get problem directories in the analysis directory that have complete chunk folders.
    
    Args:
        analysis_dir: Path to the analysis directory
        correct_only: Whether to only include correct solution directories
        limit: Optional limit on number of directories to return
        
    Returns:
        List of problem directory paths with complete chunk folders
    """
    # Determine which subdirectory to use based on correct_only
    subdir = "correct_base_solution" if correct_only else "incorrect_base_solution"
    base_dir = analysis_dir / subdir
    
    if not base_dir.exists():
        print(f"Directory {base_dir} does not exist")
        return []
    
    # Get all problem directories
    all_problem_dirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("problem_")]
    
    # Filter for directories with complete chunk folders
    complete_problem_dirs = []
    incomplete_problem_dirs = []
    no_chunks_problem_dirs = []
    
    for problem_dir in all_problem_dirs:
        # Check if chunks_labeled.json exists
        chunks_file = problem_dir / "chunks_labeled.json"
        if not chunks_file.exists():
            no_chunks_problem_dirs.append(problem_dir)
            continue
        
        # Load chunks data
        with open(chunks_file, 'r', encoding='utf-8') as f:
            try:
                chunks_data = json.load(f)
            except json.JSONDecodeError:
                no_chunks_problem_dirs.append(problem_dir)
                continue
        
        # Check if all chunks have folders
        all_chunks_have_folders = True
        for chunk in chunks_data:
            chunk_idx = chunk.get("chunk_idx", -1)
            if chunk_idx == -1:
                all_chunks_have_folders = False
                break
                
            chunk_dir = problem_dir / f"chunk_{chunk_idx}"
            if not chunk_dir.exists() or not chunk_dir.is_dir():
                all_chunks_have_folders = False
                break
        
        if all_chunks_have_folders:
            complete_problem_dirs.append(problem_dir)
        else:
            incomplete_problem_dirs.append(problem_dir)
    
    # Print statistics
    total_problems = len(all_problem_dirs)
    complete_count = len(complete_problem_dirs)
    incomplete_count = len(incomplete_problem_dirs)
    no_chunks_count = len(no_chunks_problem_dirs)
    
    print(f"Found {total_problems} problem directories in {base_dir}")
    print(f"  - {complete_count} ({complete_count/total_problems*100:.1f}%) have complete chunk folders")
    print(f"  - {incomplete_count} ({incomplete_count/total_problems*100:.1f}%) have incomplete chunk folders")
    print(f"  - {no_chunks_count} ({no_chunks_count/total_problems*100:.1f}%) have no chunks data")
    
    # Apply limit if specified
    if limit and limit < len(complete_problem_dirs):
        complete_problem_dirs = complete_problem_dirs[:limit]
        print(f"Limited to {limit} problem directories")
    
    return complete_problem_dirs

def load_labeled_chunks(problem_dir: Path) -> List[Dict]:
    """
    Load labeled chunks from a problem directory.
    
    Args:
        problem_dir: Path to the problem directory
        
    Returns:
        List of labeled chunks with importance scores
    """
    chunks_file = problem_dir / "chunks_labeled.json"
    
    if not chunks_file.exists():
        return []
    
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)
    
    return chunks_data

def get_base_solution(problem_dir: Path) -> Dict:
    """
    Get base solution from a problem directory.
    """
    base_solution_path = problem_dir / "base_solution.json"
    if not base_solution_path.exists():
        return None
    
    return json.load(open(base_solution_path, 'r', encoding='utf-8'))

def load_problem(problem_dir: Path) -> Dict:
    """
    Load problem from a problem directory.
    
    Args:
        problem_dir: Path to the problem directory
        
    Returns:
        Problem dictionary
    """
    problem_file = problem_dir / "problem.json"
    
    if not problem_file.exists():
        return {}
    
    with open(problem_file, 'r', encoding='utf-8') as f:
        problem_data = json.load(f)
    
    return problem_data

def calculate_pairwise_accuracy(predicted_ranks: List[int], true_scores: List[float]) -> float:
    """
    Calculate pairwise accuracy between predicted ranks and true scores.
    
    Args:
        predicted_ranks: List of predicted ranks (lower is more important)
        true_scores: List of true importance scores (higher is more important)
        
    Returns:
        Pairwise accuracy (0-1)
    """
    # Convert ranks to scores (lower rank = higher score)
    max_rank = max([r for r in predicted_ranks if not np.isnan(r)], default=0)
    predicted_scores = []
    for rank in predicted_ranks:
        if np.isnan(rank):
            predicted_scores.append(0.0)  # Assign 0 score for NaN ranks
        else:
            predicted_scores.append(max_rank - rank + 1)
    
    # Count correct pairs
    correct_pairs = 0
    total_pairs = 0
    
    for i in range(len(true_scores)):
        for j in range(i+1, len(true_scores)):
            # Skip pairs with NaN or equal values
            if np.isnan(predicted_scores[i]) or np.isnan(predicted_scores[j]) or \
               np.isnan(true_scores[i]) or np.isnan(true_scores[j]):
                continue
                
            total_pairs += 1
            
            # Check if the order is correct
            if (true_scores[i] > true_scores[j] and predicted_scores[i] > predicted_scores[j]) or \
               (true_scores[i] < true_scores[j] and predicted_scores[i] < predicted_scores[j]) or \
               (true_scores[i] == true_scores[j] and predicted_scores[i] == predicted_scores[j]):
                correct_pairs += 1
    
    if total_pairs == 0:
        return 0.0
    
    return correct_pairs / total_pairs

def calculate_kendall_tau(predicted_ranks: List[int], true_scores: List[float]) -> float:
    """
    Calculate Kendall's Tau correlation between predicted ranks and true scores.
    
    Args:
        predicted_ranks: List of predicted ranks (lower is more important)
        true_scores: List of true importance scores (higher is more important)
        
    Returns:
        Kendall's Tau correlation (-1 to 1)
    """
    # Convert ranks to scores (lower rank = higher score)
    max_rank = max([r for r in predicted_ranks if not np.isnan(r)], default=0)
    predicted_scores = []
    for rank in predicted_ranks:
        if np.isnan(rank):
            predicted_scores.append(0.0)  # Assign 0 score for NaN ranks
        else:
            predicted_scores.append(max_rank - rank + 1)
    
    # Filter out NaN values
    valid_pairs = [(p, t) for p, t in zip(predicted_scores, true_scores) 
                  if not np.isnan(p) and not np.isnan(t)]
    
    if not valid_pairs:
        return 0.0
    
    valid_predicted, valid_true = zip(*valid_pairs)
    
    # Calculate Kendall's Tau
    tau, _ = stats.kendalltau(valid_predicted, valid_true)
    
    # Handle NaN (can happen with ties)
    if np.isnan(tau):
        return 0.0
    
    return tau

def calculate_ndcg(predicted_ranks: List[int], true_scores: List[float]) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG).
    
    Args:
        predicted_ranks: List of predicted ranks (lower is more important)
        true_scores: List of true importance scores (higher is more important)
        
    Returns:
        NDCG score (0-1)
    """
    # Convert ranks to scores (lower rank = higher score)
    max_rank = max([r for r in predicted_ranks if not np.isnan(r)], default=0)
    predicted_scores = []
    for rank in predicted_ranks:
        if np.isnan(rank):
            predicted_scores.append(0.0)  # Assign 0 score for NaN ranks
        else:
            predicted_scores.append(max_rank - rank + 1)
    
    # Filter out NaN values
    valid_indices = [i for i, (p, t) in enumerate(zip(predicted_scores, true_scores)) 
                    if not np.isnan(p) and not np.isnan(t)]
    
    if not valid_indices:
        return 0.0
    
    valid_predicted = [predicted_scores[i] for i in valid_indices]
    valid_true = [true_scores[i] for i in valid_indices]
    
    # Reshape for sklearn's ndcg_score
    y_true = np.array([valid_true])
    y_score = np.array([valid_predicted])
    
    # Calculate NDCG
    try:
        score = ndcg_score(y_true, y_score)
        return score
    except:
        return 0.0

def llm_attribution(problem: Dict, chunks: List[Dict], base_solution: Dict, **kwargs) -> List[float]:
    """
    Use GPT-4o to rate chunks by importance in a single call.
    
    Args:
        problem: Problem dictionary
        chunks: List of chunk dictionaries
        base_solution: Base solution dictionary
        **kwargs: Additional arguments
        
    Returns:
        List of normalized importance scores (0-1 range)
    """
    # Extract problem text and chunks
    problem_text = problem.get("problem", "")
    chunk_texts = [chunk.get("chunk", "") for chunk in chunks]
    
    # Create the prompt with all chunks
    prompt = f"""
    You are analyzing a mathematical problem and its solution broken into chunks. 
    Your task is to give a rating to each chunk based on their importance to the solution.

    Problem:
    {problem_text}

    Rate each solution chunk on a scale from 0 to 10, where:
    - 0 means completely irrelevant or harmful to the solution
    - 10 means absolutely critical to the solution

    For each chunk, provide ONLY a single number between 0 and 10.
    Format your response as a comma-separated list of numbers, one for each chunk.
    Remember, you need a rating for each chunk, do not skip any chunks!
    You will be given chunks in the order of their importance to the solution, it'll look like this:
    
    Solution chunks to rate:
    1. Chunk 1
    2. Chunk 2
    3. Chunk 3
    4. Chunk 4
    
    """
    
    # Add each chunk with its index
    for i, chunk_text in enumerate(chunk_texts):
        prompt += f"\n{i+1}. {chunk_text}\n"
    
    prompt += "\n\nYour ratings (comma-separated list of numbers only):"
    
    try:
        # Make a single call to the LLM
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=2048
        )
        
        # Extract scores from response
        response_text = response.choices[0].message.content.strip()
        
        # Parse comma-separated numbers
        score_matches = re.findall(r'(\d+(\.\d+)?)', response_text)
        scores = [float(match[0]) for match in score_matches]
        
        # Ensure we have the right number of scores
        if len(scores) < len(chunk_texts):
            print(f"Warning: Got {len(scores)} scores but expected {len(chunk_texts)}. Filling missing with 5.0")
            scores.extend([5.0] * (len(chunk_texts) - len(scores)))
        elif len(scores) > len(chunk_texts):
            print(f"Warning: Got {len(scores)} scores but expected {len(chunk_texts)}. Truncating.")
            scores = scores[:len(chunk_texts)]
        
        # Normalize to 0-10 range
        scores = [min(max(score, 0), 10) for score in scores]
        
    except Exception as e:
        print(f"Error getting LLM attribution: {e}")
        scores = [5.0] * len(chunk_texts)  # Default to middle scores
    
    # Normalize scores to 0-1 range
    if scores:
        min_score = min(scores)
        max_score = max(scores)
        if max_score > min_score:
            scores = [(s - min_score) / (max_score - min_score) for s in scores]
    
    return scores

def kl_attribution(problem: Dict, labeled_chunks: List[Dict], base_solution: Dict, **kwargs) -> List[float]:
    """
    Compute KL divergence attribution for chunks.
    
    Args:
        problem: Problem dictionary
        labeled_chunks: List of labeled chunks
        base_solution: Base solution dictionary
        **kwargs: Additional arguments including:
            - thinking_model: The thinking model
            - base_model: The base model
            - tokenizer: Tokenizer for both models
            - layer_idx: Layer index to analyze
        
    Returns:
        List of attribution scores (higher = more important)
    """
    thinking_model = kwargs.get("thinking_model")
    base_model = kwargs.get("base_model")
    tokenizer = kwargs.get("tokenizer")
    layer_idx = kwargs.get("layer_idx", 47)
    
    if not thinking_model or not base_model or not tokenizer:
        print("Missing required models or tokenizer for KL attribution")
        return [0.0] * len(labeled_chunks)
    
    try:
        # Get the full CoT
        full_cot = base_solution.get("full_cot", "")
        if not full_cot:
            print(f"No full CoT found in base solution")
            return [0.0] * len(labeled_chunks)
        
        # Extract chunks from labeled_chunks
        chunks = [chunk.get("chunk", "") for chunk in labeled_chunks]
        
        # Get character ranges for each chunk in the full CoT
        chunk_ranges = get_chunk_ranges(full_cot, chunks)
        
        # Convert character ranges to token ranges
        chunk_token_ranges = get_chunk_token_ranges(full_cot, chunk_ranges, tokenizer)
        
        # Tokenize the full CoT
        device = next(thinking_model.parameters()).device
        inputs = tokenizer(full_cot, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # Get base model logits (no gradients)
        with torch.no_grad():
            base_outputs = base_model(input_ids=input_ids, attention_mask=attention_mask)
            base_logits = base_outputs.logits
            
        torch.cuda.empty_cache()
        gc.collect()
        
        # Get thinking model logits
        with torch.no_grad():
            thinking_outputs = thinking_model(input_ids=input_ids, attention_mask=attention_mask)
            thinking_logits = thinking_outputs.logits
        
        torch.cuda.empty_cache()
        gc.collect()
        
        # Compute KL divergence for each token position
        kl_divergences = []
        
        for pos in range(input_ids.shape[1] - 1):  # Exclude the last token
            # Get logits for the next token
            base_next_token_logits = base_logits[:, pos, :]
            thinking_next_token_logits = thinking_logits[:, pos, :]
            
            # Convert to probabilities
            base_probs = torch.nn.functional.softmax(base_next_token_logits, dim=-1)
            base_log_probs = torch.nn.functional.log_softmax(base_next_token_logits, dim=-1)
            thinking_log_probs = torch.nn.functional.log_softmax(thinking_next_token_logits, dim=-1)
            
            kl_div = (base_probs * (base_log_probs - thinking_log_probs)).sum(dim=-1).item()
            kl_divergences.append(kl_div)
        
        # Compute attribution scores for each chunk by averaging KL divergence over chunk tokens
        attribution_scores = []
        
        for start_idx, end_idx in chunk_token_ranges:
            # Ensure indices are within bounds
            start_idx = min(start_idx, len(kl_divergences) - 1)
            end_idx = min(end_idx, len(kl_divergences))
            
            if start_idx >= end_idx:
                attribution_scores.append(0.0)
                continue
            
            # Average KL divergence over chunk tokens
            chunk_kl = np.mean(kl_divergences[start_idx:end_idx])
            attribution_scores.append(float(chunk_kl))
        
        # If we couldn't match some chunks, fill with zeros
        if len(attribution_scores) < len(labeled_chunks):
            attribution_scores.extend([0.0] * (len(labeled_chunks) - len(attribution_scores)))
        
        # Replace NaN values with zeros
        attribution_scores = [0.0 if np.isnan(score) else score for score in attribution_scores]
        
        return attribution_scores
    
    except Exception as e:
        print(f"Error in KL attribution: {e}")
        traceback.print_exc()
        return [0.0] * len(labeled_chunks)

def kl_loss_attribution(
    problem: Dict, 
    labeled_chunks: List[Dict],
    base_solution: Dict,
    thinking_model: Any = None,
    base_model: Any = None,
    tokenizer: Any = None,
    layer_idx: int = 47,
    **kwargs
) -> List[float]:
    """
    Compute KL divergence attribution using loss and backward pass.
    
    Args:
        problem: Problem directory
        labeled_chunks: List of labeled chunks
        base_solution: Base solution dictionary
        thinking_model: The thinking model
        base_model: The base model
        tokenizer: Tokenizer for both models
        layer_idx: Layer index to analyze
        
    Returns:
        List of attribution scores (higher = more important)
    """
    try:
        full_cot = base_solution.get("full_cot", "")
        if not full_cot:
            print(f"No full CoT found in base solution")
            return [0.0] * len(labeled_chunks)
        
        # Extract chunks from labeled_chunks
        chunks = [chunk.get("chunk", "") for chunk in labeled_chunks]
        
        # Get character ranges for each chunk in the full CoT
        chunk_ranges = get_chunk_ranges(full_cot, chunks)
        
        # Convert character ranges to token ranges
        chunk_token_ranges = get_chunk_token_ranges(full_cot, chunk_ranges, tokenizer)
        
        # Tokenize the full CoT
        device = next(thinking_model.parameters()).device
        inputs = tokenizer(full_cot, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # Register hooks to capture activations and gradients
        activations = {}
        gradients = {}
        hooks = []
        
        def get_activation(name):
            def hook(module, input, output):
                activations[name] = output
            return hook
        
        def get_gradient(name):
            def hook(module, grad_in, grad_out):
                gradients[name] = grad_out[0]
            return hook
        
        # Register hooks for attention module
        attn_module = thinking_model.model.layers[layer_idx].self_attn
        hooks.append(attn_module.register_forward_hook(get_activation(f"attn_{layer_idx}")))
        hooks.append(attn_module.register_full_backward_hook(get_gradient(f"attn_{layer_idx}")))
        
        # Get base model logits (no gradients)
        with torch.no_grad():
            base_outputs = base_model(input_ids=input_ids, attention_mask=attention_mask)
            base_logits = base_outputs.logits
        
        # Get thinking model logits (with gradients)
        thinking_outputs = thinking_model(input_ids=input_ids, attention_mask=attention_mask)
        thinking_logits = thinking_outputs.logits
        
        # Compute KL divergence for each token position
        kl_divergences = []
        
        for pos in range(input_ids.shape[1] - 1):  # Exclude the last token
            print(f'pos: {pos}')
            # Get logits for the next token
            base_next_token_logits = base_logits[:, pos, :]
            thinking_next_token_logits = thinking_logits[:, pos, :]
            
            # Convert to probabilities
            base_probs = torch.nn.functional.softmax(base_next_token_logits, dim=-1)
            base_log_probs = torch.nn.functional.log_softmax(base_next_token_logits, dim=-1)
            thinking_log_probs = torch.nn.functional.log_softmax(thinking_next_token_logits, dim=-1)
            
            # Detach base_probs and base_log_probs to prevent gradients
            kl_div = (base_probs.detach() * (base_log_probs.detach() - thinking_log_probs)).sum(dim=-1)
            kl_divergences.append(kl_div)
        
        # Sum KL divergences to get total loss
        total_kl = torch.stack(kl_divergences).sum()
        
        # Compute gradients
        total_kl.backward()
        
        # Compute attribution scores
        attn_key = f"attn_{layer_idx}"
        if attn_key not in activations or attn_key not in gradients:
            print(f"Attention activations or gradients not found for layer {layer_idx}")
            return [0.0] * len(labeled_chunks)
        
        # Compute attribution for attention module
        attribution = (activations[attn_key][0] * gradients[attn_key][0]).sum(dim=2).abs()
        
        # Convert to numpy for easier handling
        attribution_np = attribution.detach().cpu().numpy()
        
        # Compute attribution scores for each chunk by averaging attribution over chunk tokens
        attribution_scores = []
        
        for start_idx, end_idx in chunk_token_ranges:
            # Ensure indices are within bounds
            start_idx = min(start_idx, attribution_np.shape[1] - 1)
            end_idx = min(end_idx, attribution_np.shape[1])
            
            if start_idx >= end_idx:
                attribution_scores.append(0.0)
                continue
            
            # Average attribution over chunk tokens
            chunk_attribution = np.mean(attribution_np[:, start_idx:end_idx])
            attribution_scores.append(float(chunk_attribution))
        
        # If we couldn't match some chunks, fill with zeros
        if len(attribution_scores) < len(labeled_chunks):
            attribution_scores.extend([0.0] * (len(labeled_chunks) - len(attribution_scores)))
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return attribution_scores
    
    except Exception as e:
        print(f"Error in KL loss attribution: {e}")
        traceback.print_exc()
        return [0.0] * len(labeled_chunks)

def entropy_attribution(problem: Dict, labeled_chunks: List[Dict], base_solution: Dict, **kwargs) -> List[float]:
    """
    Use token entropy to measure chunk importance.
    
    Args:
        problem: Problem dictionary
        labeled_chunks: List of labeled chunks
        base_solution: Base solution dictionary
        **kwargs: Additional arguments including:
            - model: The model
            - tokenizer: The tokenizer
        
    Returns:
        List of attribution scores (higher = more important)
    """
    model = kwargs.get("model")
    tokenizer = kwargs.get("tokenizer")
    
    if not model or not tokenizer:
        print("Missing required model or tokenizer for entropy attribution")
        return [0.0] * len(labeled_chunks)
    
    try:
        # Get the full CoT
        full_cot = base_solution.get("full_cot", "")
        if not full_cot:
            print(f"No full CoT found in base solution")
            return [0.0] * len(labeled_chunks)
        
        # Extract chunks from labeled_chunks
        chunks = [chunk.get("chunk", "") for chunk in labeled_chunks]
        
        # Get character ranges for each chunk in the full CoT
        chunk_ranges = get_chunk_ranges(full_cot, chunks)
        
        # Convert character ranges to token ranges
        chunk_token_ranges = get_chunk_token_ranges(full_cot, chunk_ranges, tokenizer)
        
        # Tokenize the full CoT
        device = next(model.parameters()).device
        inputs = tokenizer(full_cot, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # Get model outputs
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
        
        # Calculate entropy for each token position
        token_entropies = []
        
        for pos in range(input_ids.shape[1] - 1):  # Exclude the last token
            # Get logits for the next token
            next_token_logits = logits[:, pos, :]
            
            # Convert to probabilities
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            log_probs = torch.nn.functional.log_softmax(next_token_logits, dim=-1)
            
            # Calculate entropy: -sum(p * log(p))
            entropy = -torch.sum(probs * log_probs, dim=-1).item()
            token_entropies.append(entropy)
        
        # Compute attribution scores for each chunk by averaging entropy over chunk tokens
        attribution_scores = []
        
        for start_idx, end_idx in chunk_token_ranges:
            # Ensure indices are within bounds
            start_idx = min(start_idx, len(token_entropies) - 1)
            end_idx = min(end_idx, len(token_entropies))
            
            if start_idx >= end_idx:
                attribution_scores.append(0.0)
                continue
            
            # Average entropy over chunk tokens
            chunk_entropy = np.mean(token_entropies[start_idx:end_idx])
            attribution_scores.append(float(chunk_entropy))
        
        # If we couldn't match some chunks, fill with zeros
        if len(attribution_scores) < len(labeled_chunks):
            attribution_scores.extend([0.0] * (len(labeled_chunks) - len(attribution_scores)))
        
        # Replace NaN values with zeros
        attribution_scores = [0.0 if np.isnan(score) else score for score in attribution_scores]
        
        return attribution_scores
    
    except Exception as e:
        print(f"Error in entropy attribution: {e}")
        traceback.print_exc()
        return [0.0] * len(labeled_chunks)

def perplexity_attribution(problem: Dict, labeled_chunks: List[Dict], base_solution: Dict, **kwargs) -> List[float]:
    """
    Use token perplexity to measure chunk importance.
    
    Args:
        problem: Problem dictionary
        labeled_chunks: List of labeled chunks
        base_solution: Base solution dictionary
        **kwargs: Additional arguments including:
            - model: The model
            - tokenizer: The tokenizer
        
    Returns:
        List of attribution scores (higher = more important)
    """
    model = kwargs.get("model")
    tokenizer = kwargs.get("tokenizer")
    
    if not model or not tokenizer:
        print("Missing required model or tokenizer for perplexity attribution")
        return [0.0] * len(labeled_chunks)
    
    try:
        # Get the full CoT
        full_cot = base_solution.get("full_cot", "")
        if not full_cot:
            print(f"No full CoT found in base solution")
            return [0.0] * len(labeled_chunks)
        
        # Extract chunks from labeled_chunks
        chunks = [chunk.get("chunk", "") for chunk in labeled_chunks]
        
        # Get character ranges for each chunk in the full CoT
        chunk_ranges = get_chunk_ranges(full_cot, chunks)
        
        # Convert character ranges to token ranges
        chunk_token_ranges = get_chunk_token_ranges(full_cot, chunk_ranges, tokenizer)
        
        # Tokenize the full CoT
        device = next(model.parameters()).device
        inputs = tokenizer(full_cot, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # Get model outputs
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
        
        # Calculate perplexity for each token position
        token_perplexities = []
        
        for pos in range(input_ids.shape[1] - 1):  # Exclude the last token
            # Get logits for the next token
            next_token_logits = logits[:, pos, :]
            
            # Get the target token
            target_id = input_ids[:, pos + 1]
            
            # Calculate loss for this token
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(next_token_logits, target_id)
            
            # Calculate perplexity: exp(loss)
            perplexity = torch.exp(loss).item()
            # Cap perplexity to avoid extreme values
            perplexity = min(perplexity, 1000.0)
            token_perplexities.append(perplexity)
        
        # Compute attribution scores for each chunk by averaging perplexity over chunk tokens
        attribution_scores = []
        
        for start_idx, end_idx in chunk_token_ranges:
            # Ensure indices are within bounds
            start_idx = min(start_idx, len(token_perplexities) - 1)
            end_idx = min(end_idx, len(token_perplexities))
            
            if start_idx >= end_idx:
                attribution_scores.append(0.0)
                continue
            
            # Average perplexity over chunk tokens
            chunk_perplexity = np.mean(token_perplexities[start_idx:end_idx])
            attribution_scores.append(float(chunk_perplexity))
        
        # If we couldn't match some chunks, fill with zeros
        if len(attribution_scores) < len(labeled_chunks):
            attribution_scores.extend([0.0] * (len(labeled_chunks) - len(attribution_scores)))
        
        # Replace NaN and inf values with zeros or max value
        attribution_scores = [0.0 if np.isnan(score) else (1000.0 if np.isinf(score) else score) for score in attribution_scores]
        
        return attribution_scores
    
    except Exception as e:
        print(f"Error in perplexity attribution: {e}")
        traceback.print_exc()
        return [0.0] * len(labeled_chunks)

def token_length_attribution(problem: Dict, chunks: List[Dict], base_solution: Dict, **kwargs) -> List[int]:
    """
    Use token length to rank chunks by importance.
    
    Args:
        problem: Problem dictionary
        chunks: List of chunk dictionaries
        base_solution: Base solution dictionary
        **kwargs: Additional arguments
        
    Returns:
        List of ranks for each chunk (lower is more important)
    """
    # Extract chunk texts
    chunk_texts = [chunk.get("chunk", "") for chunk in chunks]
    
    # Calculate token length for each chunk
    tokenizer = kwargs.get("tokenizer")
    
    if tokenizer:
        # Use the provided tokenizer
        token_lengths = [len(tokenizer.encode(chunk)) for chunk in chunk_texts]
    else:
        # Approximate token length
        token_lengths = [len(chunk.split()) for chunk in chunk_texts]
    
    # Convert lengths to ranks (longer = more important = lower rank)
    ranks = [0] * len(chunks)
    sorted_indices = np.argsort(token_lengths)[::-1]  # Sort in descending order
    
    for rank, idx in enumerate(sorted_indices):
        ranks[idx] = rank
    
    return ranks

def evaluate_attribution_method(
    problem_dirs: List[Path],
    attribution_method: Callable,
    method_name: str,
    **kwargs
) -> Dict:
    """
    Evaluate an attribution method on multiple problems.
    
    Args:
        problem_dirs: List of problem directories
        attribution_method: Function that implements the attribution method
        method_name: Name of the attribution method
        **kwargs: Additional arguments to pass to the attribution method
        
    Returns:
        Dictionary with evaluation results
    """
    print(f"Evaluating {method_name} attribution method...")
    
    correlations = []
    
    for problem_dir in tqdm(problem_dirs, desc=f"Evaluating {method_name}"):
        # Load labeled chunks
        labeled_chunks = load_labeled_chunks(problem_dir)
        base_solution = get_base_solution(problem_dir)
        
        if not labeled_chunks:
            continue
        
        # Load problem
        problem = load_problem(problem_dir)
        
        if not problem:
            continue
        
        # Extract true importance scores
        true_scores = [np.abs(chunk.get("importance", 0.0)) if kwargs.get("use_abs_importance") else chunk.get("importance", 0.0) for chunk in labeled_chunks]
        
        # Skip if all scores are the same
        if len(set(true_scores)) <= 1:
            continue
        
        # Normalize true scores to 0-1 range
        min_true = min(true_scores)
        max_true = max(true_scores)
        if max_true > min_true:
            true_scores = [(s - min_true) / (max_true - min_true) for s in true_scores]
        
        # Get predicted scores
        predicted_scores = attribution_method(problem, labeled_chunks, base_solution, **kwargs)
        
        # Normalize predicted scores to 0-1 range
        if predicted_scores:
            min_pred = min(predicted_scores)
            max_pred = max(predicted_scores)
            if max_pred > min_pred:
                predicted_scores = [(s - min_pred) / (max_pred - min_pred) for s in predicted_scores]
        
        # Skip if lengths don't match
        if len(predicted_scores) != len(true_scores):
            continue
        
        # Calculate correlation
        correlation = np.corrcoef(true_scores, predicted_scores)[0, 1]
        if not np.isnan(correlation):
            correlations.append(correlation)
    
    # Calculate average correlation
    avg_correlation = np.mean(correlations) if correlations else 0.0
    
    return {
        "method": method_name,
        "correlation": avg_correlation,
        "num_problems": len(correlations)
    }

def plot_results(results: List[Dict], output_dir: Path) -> None:
    """
    Plot evaluation results.
    
    Args:
        results: List of evaluation results
        output_dir: Directory to save plots
    """
    # Create plots directory
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Sort by correlation
    df = df.sort_values("correlation", ascending=False)
    
    # Create correlation plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x="method", y="correlation", data=df)
    plt.title("Correlation with Ground Truth by Attribution Method")
    plt.xlabel("Method")
    plt.ylabel("Correlation")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(plots_dir / "correlation.png")
    plt.close()
    
    # Save results to CSV
    df.to_csv(output_dir / "attribution_results.csv", index=False)
    
    print(f"Results saved to {output_dir}")

def plot_attribution_metrics(
    problem_dirs: List[Path],
    attribution_methods: Dict[str, Callable],
    output_dir: Path,
    **kwargs
) -> None:
    """
    Plot attribution metrics alongside ground truth importance for each problem.
    
    Args:
        problem_dirs: List of problem directories
        attribution_methods: Dictionary mapping method names to attribution functions
        output_dir: Directory to save plots
        **kwargs: Additional arguments to pass to attribution methods
    """
    print("Plotting attribution metrics for each problem...")

    # Create plots directory
    plots_dir = output_dir / "attribution_plots"
    plots_dir.mkdir(exist_ok=True, parents=True)
    use_abs_importance = kwargs.get("use_abs_importance", True)
    
    for problem_dir in tqdm(problem_dirs, desc="Plotting attribution metrics"):
        # Load labeled chunks
        labeled_chunks = load_labeled_chunks(problem_dir)
        base_solution = get_base_solution(problem_dir)
        
        if not labeled_chunks:
            continue
        
        # Load problem
        problem = load_problem(problem_dir)
        
        if not problem:
            continue
        
        # Extract problem index from directory name
        problem_idx = problem_dir.name.split('_')[-1]
        
        # Extract true importance scores
        true_scores = [np.abs(chunk.get("importance", 0.0)) if use_abs_importance else chunk.get("importance", 0.0) for chunk in labeled_chunks]
        
        # Skip if all scores are the same
        if len(set(true_scores)) <= 1:
            continue
        
        # Get attribution scores for each method
        method_scores = {}
        for method_name, attribution_method in attribution_methods.items():
            try:
                scores = attribution_method(problem, labeled_chunks, base_solution, **kwargs)
                method_scores[method_name] = scores
            except Exception as e:
                print(f"Error computing {method_name} for problem {problem_idx}: {e}")
                traceback.print_exc()
        
        # Skip if no methods produced scores
        if not method_scores:
            continue
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Plot ground truth importance
        plt.plot(range(len(true_scores)), true_scores, 'k-', linewidth=2, label="Ground Truth Importance")
        
        # Plot each attribution method
        colors = ['b', 'r', 'g', 'm', 'c', 'y']
        styles = ['-', '--', '-.', ':']
        
        for i, (method_name, scores) in enumerate(method_scores.items()):
            if len(scores) != len(true_scores):
                print(f"Warning: {method_name} scores length ({len(scores)}) doesn't match true scores length ({len(true_scores)}) for problem {problem_idx}")
                continue
                
            color = colors[i % len(colors)]
            style = styles[(i // len(colors)) % len(styles)]
            
            # Normalize scores to same scale as ground truth for better comparison
            max_true = max(abs(s) for s in true_scores)
            max_score = max(abs(s) for s in scores) if any(scores) else 1.0
            normalized_scores = [s * (max_true / max_score) if max_score > 0 else 0 for s in scores]
            
            plt.plot(range(len(scores)), normalized_scores, f'{color}{style}', linewidth=1.5, label=f"{method_name} (normalized)")
        
        # Add labels and title
        plt.xlabel("Chunk Index")
        plt.ylabel("Attribution Score")
        plt.title(f"Problem {problem_idx}: Attribution Metrics vs Ground Truth")
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        # Add legend
        plt.legend(loc='best')
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(plots_dir / f"problem_{problem_idx}_attribution.png")
        plt.close()
    
    print(f"Attribution metric plots saved to {plots_dir}")

def main():
    parser = argparse.ArgumentParser(description="Benchmark different attribution methods")
    parser.add_argument("--analysis_dir", type=str, default="math_rollouts/deepseek-r1-distill-qwen-14b/temperature_0.6_top_p_0.92", help="Directory containing analysis results")
    parser.add_argument("--output_dir", type=str, default="analysis/attribution_benchmark", help="Directory to save benchmark results")
    parser.add_argument("--thinking_model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", help="Thinking model name")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-14B", help="Base model name")
    parser.add_argument("--max_problems", type=int, default=None, help="Maximum number of problems to analyze")
    parser.add_argument("--layer", type=int, default=47, help="Layer to analyze for KL attribution")
    parser.add_argument("--correct_only", action="store_true", default=True, help="Only analyze correct solutions")
    parser.add_argument("--use_abs_importance", action="store_true", default=True, help="Use absolute importance values")
    args = parser.parse_args()
    
    # Set up directories
    analysis_dir = Path(args.analysis_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Get problem directories
    problem_dirs = get_problem_dirs(analysis_dir, correct_only=args.correct_only, limit=args.max_problems)
    
    if not problem_dirs:
        print(f"No problem directories found in {analysis_dir}")
        return
    
    print(f"Found {len(problem_dirs)} problem directories")
    
    # Load models
    print("Loading models...")
    thinking_model, tokenizer = load_model_and_tokenizer(args.thinking_model)
    base_model, _ = load_model_and_tokenizer(args.base_model)
    
    # Evaluate attribution methods
    results = []
    
    # Method 1: LLM-based attribution (using GPT-4o)
    llm_results = evaluate_attribution_method(
        problem_dirs, 
        llm_attribution, 
        "LLM-based (GPT-4o)",
        use_abs_importance=args.use_abs_importance
    )
    results.append(llm_results)
    
    # Method 2: KL-divergence attribution
    kl_results = evaluate_attribution_method(
        problem_dirs,
        kl_attribution,
        "KL-divergence",
        thinking_model=thinking_model,
        base_model=base_model,
        tokenizer=tokenizer,
        layer_idx=args.layer,
        use_abs_importance=args.use_abs_importance
    )
    results.append(kl_results)
    
    # Method 3: KL loss attribution (TODO: Too memory intensive)
    """
    kl_loss_results = evaluate_attribution_method(
        problem_dirs,
        kl_loss_attribution,
        "KL-loss",
        thinking_model=thinking_model,
        base_model=base_model,
        tokenizer=tokenizer,
        layer_idx=args.layer,
        use_abs_importance=args.use_abs_importance
    )
    results.append(kl_loss_results)
    """
    
    # Method 4: Entropy-based attribution
    entropy_results = evaluate_attribution_method(
        problem_dirs,
        entropy_attribution,
        "Entropy-based",
        model=thinking_model,
        tokenizer=tokenizer,
        use_abs_importance=args.use_abs_importance
    )
    results.append(entropy_results)
    
    # Method 5: Perplexity-based attribution
    perplexity_results = evaluate_attribution_method(
        problem_dirs,
        perplexity_attribution,
        "Perplexity-based",
        model=thinking_model,
        tokenizer=tokenizer,
        use_abs_importance=args.use_abs_importance
    )
    results.append(perplexity_results)
    
    # Method 6: Token length attribution
    length_results = evaluate_attribution_method(
        problem_dirs,
        token_length_attribution,
        "Token-length",
        tokenizer=tokenizer,
        use_abs_importance=args.use_abs_importance
    )
    results.append(length_results)
    
    # Plot results
    plot_results(results, output_dir)
    
    # Plot attribution metrics for each problem
    plot_attribution_metrics(
        problem_dirs=problem_dirs,
        attribution_methods={
            "Entropy": entropy_attribution,
            "KL Divergence": kl_attribution,
            # "KL Loss": kl_loss_attribution,
            # "Perplexity": perplexity_attribution
        },
        output_dir=output_dir,
        model=thinking_model,
        tokenizer=tokenizer,
        thinking_model=thinking_model,
        base_model=base_model,
        use_abs_importance=args.use_abs_importance
    )
    
    # Clean up
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()