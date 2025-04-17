import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Tuple, Union, Optional, Set
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import argparse
import pandas as pd
from collections import Counter, defaultdict
import scipy.stats as stats

# Set up paths
cots_dir = Path("cots")
analysis_dir = Path("analysis")
output_dir = Path("uncertainty_analysis")
output_dir.mkdir(exist_ok=True)

# Categories of interest
TARGET_CATEGORIES = {"Uncertainty Estimation", "Backtracking"}

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
    Load the model and tokenizer for token probability analysis.
    
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
    
    return model, tokenizer

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

def has_target_categories(chunks_data: List[Dict]) -> bool:
    """
    Check if chunks data contains any of the target categories.
    
    Args:
        chunks_data: List of chunk dictionaries
        
    Returns:
        True if any chunk has a target category, False otherwise
    """
    for chunk in chunks_data:
        if chunk.get("category", "Unknown") in TARGET_CATEGORIES:
            return True
    return False

def calculate_token_entropy(logits: torch.Tensor) -> float:
    """
    Calculate entropy of token distribution from logits.
    
    Args:
        logits: Logits tensor from model output
        
    Returns:
        Entropy value
    """
    # Convert logits to probabilities using softmax
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    
    # Calculate entropy: -sum(p * log(p))
    entropy = -torch.sum(torch.exp(log_probs) * log_probs, dim=-1) / torch.log(torch.tensor(2.0))
    
    return entropy.item()

def calculate_token_uncertainty(
    model: AutoModelForCausalLM, 
    tokenizer: AutoTokenizer,
    text: str,
    chunk_ranges: List[Tuple[int, int]],
    method: str = "entropy"
) -> List[float]:
    """
    Calculate token-level uncertainty for each chunk.
    
    Args:
        model: Hugging Face model
        tokenizer: Tokenizer
        text: Full text
        chunk_ranges: List of (start, end) character positions for each chunk
        method: Method to calculate uncertainty ('entropy' or 'perplexity')
        
    Returns:
        List of uncertainty values for each chunk
    """
    # Tokenize the full text
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids
    
    # Get token ranges for each chunk
    token_ranges = []
    for start_char, end_char in chunk_ranges:
        # Convert character positions to token indices
        start_tokens = tokenizer(text[:start_char], return_tensors="pt")
        start_idx = len(start_tokens.input_ids[0])  # Adjust for special tokens
        
        end_tokens = tokenizer(text[:end_char], return_tensors="pt")
        end_idx = len(end_tokens.input_ids[0])  # Adjust for special tokens
        
        token_ranges.append((max(0, start_idx), end_idx))
    
    # Calculate uncertainty for each chunk
    uncertainties = []
    
    with torch.no_grad():
        # Get model outputs with logits
        outputs = model(**inputs, return_dict=True)
        logits = outputs.logits
        # print('Logits shape:', logits.shape)
        
        # Calculate uncertainty for each chunk
        for start_idx, end_idx in token_ranges:
            if method == "entropy":
                # Calculate average entropy across tokens in the chunk
                chunk_entropies = []
                for i in range(start_idx, min(end_idx, logits.shape[1]-1)):
                    token_entropy = calculate_token_entropy(logits[0, i])
                    chunk_entropies.append(token_entropy)
                
                # Average entropy across tokens
                if chunk_entropies:
                    avg_entropy = sum(chunk_entropies) / len(chunk_entropies)
                    uncertainties.append(avg_entropy)
                else:
                    uncertainties.append(0.0)
                    
            elif method == "perplexity":
                # Calculate perplexity for the chunk
                chunk_logits = logits[0, start_idx:min(end_idx, logits.shape[1]-1)]
                chunk_ids = input_ids[0, start_idx+1:min(end_idx+1, input_ids.shape[1])]
                
                if len(chunk_ids) > 0:
                    # Get log probabilities of the actual next tokens
                    log_probs = torch.nn.functional.log_softmax(chunk_logits, dim=-1)
                    token_log_probs = log_probs[torch.arange(len(chunk_ids)), chunk_ids]
                    
                    # Calculate negative log likelihood (lower is better)
                    nll = -torch.mean(token_log_probs).item()
                    
                    # Perplexity = exp(nll)
                    perplexity = np.exp(nll)
                    uncertainties.append(perplexity)
                else:
                    uncertainties.append(0.0)
            
            elif method == "max_prob":
                # Calculate 1 - max probability for each token (higher means more uncertainty)
                chunk_probs = []
                for i in range(start_idx, min(end_idx, logits.shape[1]-1)):
                    probs = torch.nn.functional.softmax(logits[0, i], dim=-1)
                    max_prob = torch.max(probs).item()
                    chunk_probs.append(1 - max_prob)  # Uncertainty = 1 - max_prob
                
                # Average across tokens
                if chunk_probs:
                    avg_uncertainty = sum(chunk_probs) / len(chunk_probs)
                    uncertainties.append(avg_uncertainty)
                else:
                    uncertainties.append(0.0)
    
    return uncertainties

def analyze_cot_uncertainty(
    problem_dirs: List[Path],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    uncertainty_method: str = "entropy",
    max_problems: int = 100
) -> List[Dict]:
    """
    Analyze uncertainty in CoTs with target categories.
    
    Args:
        problem_dirs: List of problem directory paths
        model: Hugging Face model
        tokenizer: Tokenizer
        uncertainty_method: Method to calculate uncertainty
        max_problems: Maximum number of problems to process
        
    Returns:
        List of CoT data with uncertainty metrics
    """
    all_cot_data = []
    
    # Process a limited number of problems
    for problem_dir in tqdm(problem_dirs[:max_problems], desc="Processing problems"):
        # Load problem and solutions
        problem, solutions = load_problem_and_solutions(problem_dir)
        if not solutions:
            continue
        
        # Process all seeds
        analysis_dir_path = analysis_dir / problem_dir.name
        seed_dirs = [d for d in analysis_dir_path.iterdir() if d.is_dir() and d.name.startswith("seed_")]
        
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
                
            # Find the first chunk containing <think>
            think_start_idx = -1
            for i, chunk in enumerate(chunks_data):
                if "<think>" in chunk["text"]:
                    think_start_idx = i
                    break
            
            # Skip if we couldn't find the <think> token
            if think_start_idx == -1:
                continue
                
            # Only keep chunks from the <think> chunk onwards
            chunks_data = chunks_data[think_start_idx:]
            
            # Check if this CoT has any of the target categories
            if not has_target_categories(chunks_data):
                continue
                
            # Extract chunks
            chunks = [item["text"] for item in chunks_data]
            
            # Get character ranges for each chunk in the full text
            chunk_ranges = []
            current_pos = 0
            
            for chunk in chunks:
                # Find the chunk in the full text starting from current position
                chunk_start = full_text.find(chunk, current_pos)
                if chunk_start == -1:
                    # If exact match not found, try with some flexibility
                    # This handles cases where whitespace might differ
                    chunk_words = chunk.split()
                    for i in range(current_pos, len(full_text) - len(chunk)):
                        if i + len(chunk_words[0]) < len(full_text) and full_text[i:i+len(chunk_words[0])] == chunk_words[0]:
                            potential_match = full_text[i:i+len(chunk)]
                            if potential_match.split() == chunk_words:
                                chunk_start = i
                                break
                
                if chunk_start == -1:
                    print(f"Warning: Chunk not found in full text: {chunk[:50]}...")
                    continue
                    
                chunk_end = chunk_start + len(chunk)
                current_pos = chunk_end
                
                chunk_ranges.append((chunk_start, chunk_end))
            
            # Calculate uncertainty for each chunk
            uncertainties = calculate_token_uncertainty(model, tokenizer, full_text, chunk_ranges, method=uncertainty_method)
            
            # Create CoT data with uncertainty metrics
            cot_data = {"problem_id": problem_dir.name, "seed_id": seed_id, "chunks": [], "has_target_category": True}
            
            # Add chunk data with uncertainty metrics
            for i, chunk_data in enumerate(chunks_data):
                if i < len(uncertainties):
                    chunk_info = {
                        "index": chunk_data.get("index", i),
                        "text": chunk_data.get("text", ""),
                        "category": chunk_data.get("category", "Unknown"),
                        "uncertainty": uncertainties[i],
                        "is_target": chunk_data.get("category", "Unknown") in TARGET_CATEGORIES
                    }
                    cot_data["chunks"].append(chunk_info)
            
            all_cot_data.append(cot_data)
    
    return all_cot_data

def plot_uncertainty_over_cot(
    cot_data: Dict,
    output_path: Path,
    uncertainty_method: str,
    normalize_x: bool = True
):
    """
    Plot uncertainty over the CoT length for a single CoT.
    
    Args:
        cot_data: CoT data with uncertainty metrics
        output_path: Path to save the plot
        uncertainty_method: Method used to calculate uncertainty
        normalize_x: Whether to normalize x-axis to [0, 1]
    """
    chunks = cot_data["chunks"]
    indices = [chunk["index"] for chunk in chunks]
    uncertainties = [chunk["uncertainty"] for chunk in chunks]
    categories = [chunk["category"] for chunk in chunks]
    is_target = [chunk["is_target"] for chunk in chunks]
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot uncertainty
    x = np.array(indices)
    if normalize_x and len(indices) > 1:
        x = x / max(indices)
    
    # Plot line
    plt.plot(x, uncertainties, 'o-', color='blue', alpha=0.7)
    
    # Highlight target categories
    for i, (idx, uncertainty, target) in enumerate(zip(x, uncertainties, is_target)):
        if target:
            plt.plot(idx, uncertainty, 'o', color='red', markersize=10)
    
    # Add category labels
    for i, (idx, uncertainty, category) in enumerate(zip(x, uncertainties, categories)):
        if category in TARGET_CATEGORIES:
            # Use abbreviated labels: "UN" for Uncertainty Estimation, "B" for Backtracking
            label = "UN" if category == "Uncertainty Estimation" else "B"
            plt.annotate(
                label,
                (idx, uncertainty),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
                fontweight='bold',
                color='red'
            )
    
    # Set labels and title
    plt.xlabel("Normalized Position in CoT" if normalize_x else "Chunk Index")
    
    if uncertainty_method == "entropy":
        plt.ylabel("Token Entropy (higher = more uncertain)")
        method_name = "Entropy"
    elif uncertainty_method == "perplexity":
        plt.ylabel("Perplexity (higher = more uncertain)")
        method_name = "Perplexity"
    else:
        plt.ylabel("Uncertainty")
        method_name = uncertainty_method.replace("_", " ").title()
    
    plt.title(f"{method_name} Uncertainty Over CoT\nProblem: {cot_data['problem_id']}, Seed: {cot_data['seed_id']}")
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Add legend
    target_patches = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Backtracking (B)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Uncertainty Estimation (UN)')
    ]
    normal_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Other Category')
    plt.legend(handles=[*target_patches, normal_patch], loc='upper right')
    
    # Save plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_aggregated_uncertainty(
    all_cot_data: List[Dict],
    output_path: Path,
    uncertainty_method: str
):
    """
    Plot aggregated uncertainty patterns across all CoTs.
    
    Args:
        all_cot_data: List of CoT data with uncertainty metrics
        output_path: Path to save the plot
        uncertainty_method: Method used to calculate uncertainty
    """
    # Prepare data for aggregation
    all_positions = []
    all_uncertainties = []
    all_is_target = []
    all_categories = []
    
    for cot_data in all_cot_data:
        chunks = cot_data["chunks"]
        max_idx = max(chunk["index"] for chunk in chunks)
        
        for chunk in chunks:
            # Normalize position to [0, 1]
            norm_pos = chunk["index"] / max_idx if max_idx > 0 else 0
            all_positions.append(norm_pos)
            all_uncertainties.append(chunk["uncertainty"])
            all_is_target.append(chunk["is_target"])
            all_categories.append(chunk["category"])
    
    # Create DataFrame
    df = pd.DataFrame({
        "position": all_positions,
        "uncertainty": all_uncertainties,
        "is_target": all_is_target,
        "category": all_categories
    })
    
    # Create bins for position
    df["position_bin"] = pd.cut(df["position"], bins=10, labels=False) / 9  # Normalize to [0, 1]
    
    # Calculate statistics per bin
    bin_stats = df.groupby(["position_bin", "is_target"]).agg({"uncertainty": ["mean", "std", "count"]}).reset_index()
    
    bin_stats.columns = ["position_bin", "is_target", "mean", "std", "count"]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot for target and non-target categories
    for is_target, color, label in [(True, "red", "Target Categories"), (False, "blue", "Other Categories")]:
        data = bin_stats[bin_stats["is_target"] == is_target]
        
        if len(data) > 0:
            plt.plot(data["position_bin"], data["mean"], 'o-', color=color, label=label)
            
            # Add error bands
            plt.fill_between(
                data["position_bin"],
                data["mean"] - data["std"],
                data["mean"] + data["std"],
                color=color,
                alpha=0.2
            )
    
    # Set labels and title
    plt.xlabel("Normalized Position in CoT")
    
    if uncertainty_method == "entropy":
        plt.ylabel("Token Entropy (higher = more uncertain)")
        method_name = "Entropy"
    elif uncertainty_method == "perplexity":
        plt.ylabel("Perplexity (higher = more uncertain)")
        method_name = "Perplexity"
    else:
        plt.ylabel("Uncertainty")
        method_name = uncertainty_method.replace("_", " ").title()
    
    plt.title(f"Aggregated {method_name} Uncertainty Over CoT Position")
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Add legend
    plt.legend()
    
    # Save plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    # Create category-specific plot
    plt.figure(figsize=(14, 8))
    
    # Get counts for each category
    category_counts = Counter(all_categories)
    
    # Only include categories with sufficient data
    min_count = 10
    common_categories = [cat for cat, count in category_counts.items() if count >= min_count]
    
    # Calculate statistics per category and position bin
    cat_stats = df[df["category"].isin(common_categories)].groupby(["position_bin", "category"]).agg({"uncertainty": ["mean", "std", "count"]}).reset_index()
    
    cat_stats.columns = ["position_bin", "category", "mean", "std", "count"]
    
    # Define a color palette with highly distinct colors for each category
    # Use a combination of distinct color palettes for maximum differentiation
    category_colors = {}
    
    # Assign specific colors to target categories
    category_colors["Uncertainty Estimation"] = "#FF0000"  # Bright red
    category_colors["Backtracking"] = "#8B0000"  # Dark red
    
    # Use a set of maximally distinct colors for other categories
    distinct_colors = [
        "#1f77b4",  # Blue
        "#ff7f0e",  # Orange
        "#2ca02c",  # Green
        "#9467bd",  # Purple
        "#8c564b",  # Brown
        "#e377c2",  # Pink
        "#7f7f7f",  # Gray
        "#bcbd22",  # Olive
        "#17becf",  # Cyan
        "#d62728",  # Brick red
        "#9edae5",  # Light blue
        "#ffbb78",  # Light orange
        "#98df8a",  # Light green
        "#f7b6d2",  # Light pink
        "#c5b0d5",  # Light purple
        "#c49c94",  # Light brown
        "#dbdb8d",  # Light olive
        "#aec7e8",  # Pale blue
        "#ff9896",  # Pale red
        "#00FF00",  # Lime green
        "#0000FF",  # Blue
        "#FFFF00",  # Yellow
        "#00FFFF",  # Cyan
        "#FF00FF",  # Magenta
        "#800080",  # Purple
        "#008000",  # Green
        "#000080",  # Navy
        "#808000",  # Olive
        "#800000",  # Maroon
        "#008080",  # Teal
    ]
    
    # Assign distinct colors to other categories
    other_categories = [cat for cat in common_categories if cat not in TARGET_CATEGORIES]
    for i, cat in enumerate(other_categories):
        category_colors[cat] = distinct_colors[i % len(distinct_colors)]
    
    # Plot for each category with distinct markers as well
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd', '|', '_']
    
    for i, category in enumerate(sorted(common_categories)):
        data = cat_stats[cat_stats["category"] == category]
        
        if len(data) > 0:
            color = category_colors[category]
            linestyle = "-" if category in TARGET_CATEGORIES else "--"
            marker = markers[i % len(markers)]
            
            plt.plot(
                data["position_bin"], 
                data["mean"], 
                marker=marker,
                linestyle=linestyle,
                color=color,
                linewidth=2,
                markersize=8,
                label=f"{category} (n={category_counts[category]})"
            )
    
    # Set labels and title
    plt.xlabel("Normalized Position in CoT", fontsize=12)
    
    if uncertainty_method == "entropy":
        plt.ylabel("Token Entropy (higher = more uncertain)", fontsize=12)
    elif uncertainty_method == "perplexity":
        plt.ylabel("Perplexity (higher = more uncertain)", fontsize=12)
    else:
        plt.ylabel("Uncertainty", fontsize=12)
    
    method_name = uncertainty_method.replace("_", " ").title()
    plt.title(f"Category-Specific {method_name} Uncertainty Over CoT Position", fontsize=14)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Add legend with smaller font in two columns
    plt.legend(fontsize='small', loc='upper left', bbox_to_anchor=(1, 1), ncol=2)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(output_path.with_name(f"{output_path.stem}_by_category{output_path.suffix}"), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Analyze uncertainty in CoTs with target categories")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", help="Model name")
    parser.add_argument("--max_problems", type=int, default=100, help="Maximum number of problems to process")
    parser.add_argument("--uncertainty_method", type=str, default="entropy", choices=["entropy", "perplexity", "max_prob"], help="Method to calculate uncertainty")
    parser.add_argument("--plot_individual", action="store_true", help="Plot individual CoTs")
    args = parser.parse_args()
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model)
    
    # Get problem directories
    problem_dirs = get_problem_dirs(cots_dir)
    
    # Analyze CoT uncertainty
    all_cot_data = analyze_cot_uncertainty(
        problem_dirs, model, tokenizer, 
        uncertainty_method=args.uncertainty_method,
        max_problems=args.max_problems
    )
    
    if not all_cot_data:
        print("No CoTs with target categories found. Exiting.")
        return
    
    print(f"Analyzed {len(all_cot_data)} CoTs with target categories")
    
    # Save CoT data
    with open(output_dir / f"uncertainty_data_{args.uncertainty_method}.json", 'w', encoding='utf-8') as f:
        json.dump(all_cot_data, f, indent=2)
    
    # Plot individual CoTs
    if args.plot_individual:
        individual_plots_dir = output_dir / "individual_plots"
        individual_plots_dir.mkdir(exist_ok=True)
        
        for i, cot_data in enumerate(all_cot_data):
            output_path = individual_plots_dir / f"uncertainty_{cot_data['problem_id']}_{cot_data['seed_id']}_{args.uncertainty_method}.png"
            plot_uncertainty_over_cot(cot_data, output_path, args.uncertainty_method)
    
    # Plot aggregated uncertainty
    output_path = output_dir / f"aggregated_uncertainty_{args.uncertainty_method}.png"
    plot_aggregated_uncertainty(all_cot_data, output_path, args.uncertainty_method)
    
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()