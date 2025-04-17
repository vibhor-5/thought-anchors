import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from collections import defaultdict

# Set up paths
cots_dir = Path("cots")
analysis_dir = Path("analysis")
output_dir = Path("deduction_circuits")
output_dir.mkdir(exist_ok=True)

# Target category for deduction analysis
DEDUCTION_CATEGORIES = {"Deduction"}

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
    Load the model and tokenizer for analysis.
    
    Args:
        model_name: Name of the model to load
        
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model with attention outputs enabled
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
        output_attentions=True,
        output_hidden_states=True
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

def has_deduction_categories(chunks_data: List[Dict]) -> bool:
    """
    Check if chunks data contains any of the deduction categories.
    
    Args:
        chunks_data: List of chunk dictionaries
        
    Returns:
        True if any chunk has a deduction category, False otherwise
    """
    for chunk in chunks_data:
        if chunk.get("category", "Unknown") in DEDUCTION_CATEGORIES:
            return True
    return False

def get_token_ranges_for_chunks(full_text: str, chunks: List[str], tokenizer) -> List[Tuple[int, int]]:
    """
    Get token ranges for each chunk in the full text.
    
    Args:
        full_text: Full text of the solution
        chunks: List of chunk texts
        tokenizer: Tokenizer to use
        
    Returns:
        List of (start_idx, end_idx) tuples for each chunk
    """
    # Get character ranges for each chunk in the full text
    chunk_char_ranges = []
    current_pos = 0
    
    for chunk in chunks:
        # Find the chunk in the full text starting from current position
        chunk_start = full_text.find(chunk, current_pos)
        if chunk_start == -1:
            # If exact match not found, try with some flexibility
            chunk_words = chunk.split()
            for i in range(current_pos, len(full_text) - len(chunk)):
                if i + len(chunk_words[0]) < len(full_text) and full_text[i:i+len(chunk_words[0])] == chunk_words[0]:
                    potential_match = full_text[i:i+len(chunk)]
                    if potential_match.split() == chunk_words:
                        chunk_start = i
                        break
        
        if chunk_start == -1:
            print(f"Warning: Chunk not found in full text: {chunk[:50]}...")
            chunk_char_ranges.append((-1, -1))  # Mark as not found
            continue
            
        chunk_end = chunk_start + len(chunk)
        current_pos = chunk_end
        
        chunk_char_ranges.append((chunk_start, chunk_end))
    
    # Convert character ranges to token ranges
    token_ranges = []
    for start_char, end_char in chunk_char_ranges:
        if start_char == -1:
            token_ranges.append((-1, -1))
            continue
            
        # Convert character positions to token indices
        start_tokens = tokenizer(full_text[:start_char], return_tensors="pt")
        start_idx = len(start_tokens.input_ids[0])
        
        end_tokens = tokenizer(full_text[:end_char], return_tensors="pt")
        end_idx = len(end_tokens.input_ids[0])
        
        token_ranges.append((start_idx, end_idx))
    
    return token_ranges

def get_model_outputs_with_attention(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, text: str) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
    """
    Get model outputs with attention weights and hidden states.
    
    Args:
        model: Hugging Face model
        tokenizer: Tokenizer
        text: Input text
        
    Returns:
        Tuple of (logits, attention_weights, hidden_states)
    """
    try:
        # Tokenize input
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        # Check if input is too long
        if inputs.input_ids.shape[1] > 2048:
            print(f"Warning: Input sequence length {inputs.input_ids.shape[1]} exceeds 2048 tokens. Truncating.")
            inputs = {k: v[:, :2048] for k, v in inputs.items()}
        
        # Run model with attention outputs
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True, output_hidden_states=True)
        
        # Extract logits, attention weights, and hidden states
        logits = outputs.logits
        attentions = outputs.attentions
        hidden_states = outputs.hidden_states
        
        return logits, attentions, hidden_states
    
    except torch.cuda.OutOfMemoryError as e:
        print(f"CUDA out of memory error: {e}")
        # Clear CUDA cache
        torch.cuda.empty_cache()
        # Return empty tensors as a fallback
        return None, None, None
    except Exception as e:
        print(f"Error in model inference: {e}")
        return None, None, None

def analyze_attention_patterns(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    full_text: str,
    chunks_data: List[Dict],
    token_ranges: List[Tuple[int, int]]
) -> Dict:
    """
    Analyze attention patterns for deduction.
    
    Args:
        model: Hugging Face model
        tokenizer: Tokenizer
        full_text: Full text of the solution
        chunks_data: List of chunk dictionaries
        token_ranges: List of token ranges for each chunk
        
    Returns:
        Dictionary with attention analysis results
    """
    # Initialize results
    attention_analysis = {
        "has_deduction": False,
        "deductions": []
    }
    
    # Get model outputs with attention
    logits, attentions, hidden_states = get_model_outputs_with_attention(model, tokenizer, full_text)
    
    # Check if model outputs are valid
    if logits is None or attentions is None:
        print("Skipping attention analysis due to model inference error")
        return attention_analysis
    
    # Find deduction chunks and their premises
    deduction_chunks = []
    for i, chunk in enumerate(chunks_data):
        if chunk.get("category", "Unknown") in DEDUCTION_CATEGORIES:
            # Find premises for this deduction
            premises = []
            for j, premise_chunk in enumerate(chunks_data[:i]):  # Only consider chunks before the deduction
                if premise_chunk.get("category", "Unknown") not in DEDUCTION_CATEGORIES:
                    premises.append({
                        "index": premise_chunk.get("index", j),
                        "text": premise_chunk.get("text", ""),
                        "token_range": token_ranges[j]
                    })
            
            deduction_chunks.append({
                "index": chunk.get("index", i),
                "text": chunk.get("text", ""),
                "token_range": token_ranges[i],
                "premises": premises
            })
    
    if not deduction_chunks:
        return attention_analysis
    
    attention_analysis["has_deduction"] = True
    
    # Get model dimensions
    num_layers = len(attentions)
    num_heads = attentions[0].shape[1]
    
    # Analyze attention for each deduction
    for deduction in deduction_chunks:
        deduction_start, deduction_end = deduction["token_range"]
        premises = deduction["premises"]
        
        # Skip if token range is invalid
        if deduction_start == -1 or deduction_end == -1:
            continue
        
        # Initialize attention scores
        deduction_to_premise_attention = np.zeros((num_layers, num_heads))
        
        # For each token in the deduction
        for token_idx in range(deduction_start, deduction_end):
            if token_idx >= attentions[0].shape[2]:  # Skip if token index is out of bounds
                continue
                
            # Get attention weights for this token
            for layer in range(num_layers):
                layer_head_attention = attentions[layer][0]  # Batch size is 1
                
                # Skip if attention shape is unexpected
                if token_idx >= layer_head_attention.shape[1]:
                    continue
                    
                # For each premise, calculate attention
                for premise in premises:
                    premise_start, premise_end = premise["token_range"]
                    
                    # Skip if token range is invalid
                    if premise_start == -1 or premise_end == -1:
                        continue
                    
                    # Sum attention from this token to all tokens in the premise
                    for premise_token in range(premise_start, premise_end):
                        if premise_token < layer_head_attention.shape[1]:
                            for head in range(num_heads):
                                deduction_to_premise_attention[layer, head] += layer_head_attention[head, token_idx, premise_token].item()
        
        # Normalize by number of tokens
        num_deduction_tokens = deduction_end - deduction_start
        if num_deduction_tokens > 0:
            deduction_to_premise_attention /= num_deduction_tokens
        
        # Find top attention heads
        top_heads = []
        for layer in range(num_layers):
            for head in range(num_heads):
                attention_score = deduction_to_premise_attention[layer, head]
                if attention_score > 0.1:  # Threshold for significant attention
                    top_heads.append({"layer": layer, "head": head, "attention_score": float(attention_score)})
        
        # Sort by attention score
        top_heads.sort(key=lambda x: x["attention_score"], reverse=True)
        
        # Add to analysis
        deduction_analysis = {
            "deduction_text": deduction["text"][:100] + "..." if len(deduction["text"]) > 100 else deduction["text"],
            "deduction_index": deduction["index"],
            "premise_indices": [p["index"] for p in premises],
            "top_attention_heads": top_heads[:10]  # Top 10 heads
        }
        
        attention_analysis["deductions"].append(deduction_analysis)
    
    return attention_analysis

def ablate_attention_heads(
    model: AutoModelForCausalLM,
    text: str,
    tokenizer: AutoTokenizer,
    target_heads: List[Dict]
) -> torch.Tensor:
    """
    Run the model with specific attention heads ablated (set to zero).
    
    Args:
        model: Hugging Face model
        text: Input text
        tokenizer: Tokenizer
        target_heads: List of {layer, head} dictionaries to ablate
        
    Returns:
        Logits from the ablated model
    """
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # Create a hook function to zero out specific attention heads
    def attention_hook(module, input_tensors, output_tensors):
        # output_tensors is a tuple with the attention output and attention weights
        output, attention_weights = output_tensors
        
        # Create a modified copy of the attention weights
        modified_weights = attention_weights.clone()
        
        # Zero out the specified heads
        for head_info in target_heads:
            if head_info["layer"] == current_layer and "head" in head_info:
                modified_weights[0, head_info["head"]] = 0
        
        # Return modified output
        return (output, modified_weights)
    
    # Register hooks for each layer
    hooks = []
    for layer_idx in range(model.config.num_hidden_layers):
        current_layer = layer_idx
        # The exact path to attention layers might vary by model architecture
        try:
            attention_layer = model.model.layers[layer_idx].self_attn
            hook = attention_layer.register_forward_hook(attention_hook)
            hooks.append(hook)
        except (AttributeError, IndexError) as e:
            print(f"Could not access attention layer {layer_idx}: {e}")
    
    # Run the model with hooks
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return outputs.logits

def run_head_ablation_experiment(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    full_text: str,
    target_heads: List[Dict]
) -> Dict:
    """
    Run an experiment to measure the effect of ablating specific attention heads.
    
    Args:
        model: Hugging Face model
        tokenizer: Tokenizer
        full_text: Full text to process
        target_heads: List of {layer, head} dictionaries to ablate
        
    Returns:
        Dictionary with experiment results
    """
    # Get baseline logits
    baseline_logits, _, _ = get_model_outputs_with_attention(model, tokenizer, full_text)
    
    # Get ablated logits
    ablated_logits = ablate_attention_heads(model, full_text, tokenizer, target_heads)
    
    # Calculate logit difference
    logit_diff = (baseline_logits - ablated_logits).abs().mean().item()
    
    # Get top token predictions for comparison
    baseline_top_tokens = torch.topk(baseline_logits[0, -1], k=5)
    ablated_top_tokens = torch.topk(ablated_logits[0, -1], k=5)
    
    # Convert token IDs to strings
    baseline_tokens = [tokenizer.decode(token_id.item()) for token_id in baseline_top_tokens.indices]
    ablated_tokens = [tokenizer.decode(token_id.item()) for token_id in ablated_top_tokens.indices]
    
    # Calculate prediction change
    prediction_changed = baseline_tokens[0] != ablated_tokens[0]
    
    return {
        "logit_difference": logit_diff,
        "prediction_changed": prediction_changed,
        "baseline_top_tokens": baseline_tokens,
        "ablated_top_tokens": ablated_tokens
    }

def analyze_deduction_circuits(
    problem_dirs: List[Path],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_problems: int = 50,
    run_ablation: bool = False
) -> Dict:
    """
    Analyze deduction circuits across multiple problems.
    
    Args:
        problem_dirs: List of problem directory paths
        model: Hugging Face model
        tokenizer: Tokenizer
        max_problems: Maximum number of problems to process
        run_ablation: Whether to run head ablation experiments
        
    Returns:
        Dictionary with analysis results
    """
    all_results = []
    head_importance = defaultdict(float)
    
    # Process a limited number of problems
    for problem_dir in tqdm(problem_dirs[:max_problems], desc="Analyzing deduction circuits"):
        try:
            # Load problem and solutions
            problem, solutions = load_problem_and_solutions(problem_dir)
            if not solutions:
                continue
            
            # Process all seeds
            analysis_dir_path = analysis_dir / problem_dir.name
            seed_dirs = [d for d in analysis_dir_path.iterdir() if d.is_dir() and d.name.startswith("seed_")]
            
            for seed_dir in seed_dirs:
                try:
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
                        
                    # Check if this CoT has any deduction categories
                    if not has_deduction_categories(chunks_data):
                        continue
                        
                    # Extract chunks
                    chunks = [item["text"] for item in chunks_data]
                    
                    # Get token ranges for each chunk
                    token_ranges = get_token_ranges_for_chunks(full_text, chunks, tokenizer)
                    
                    # Analyze attention patterns
                    attention_analysis = analyze_attention_patterns(model, tokenizer, full_text, chunks_data, token_ranges)
                    
                    if not attention_analysis["has_deduction"]:
                        continue
                    
                    # Add problem and seed info
                    attention_analysis["problem_id"] = problem_dir.name
                    attention_analysis["seed_id"] = seed_id
                    
                    # Update head importance based on attention scores
                    for deduction in attention_analysis["deductions"]:
                        for head_info in deduction["top_attention_heads"]:
                            head_key = f"L{head_info['layer']}_H{head_info['head']}"
                            head_importance[head_key] += head_info["attention_score"]
                    
                    # Run ablation experiments if requested
                    if run_ablation and len(attention_analysis["deductions"]) > 0:
                        try:
                            # Get top heads across all deductions
                            all_top_heads = []
                            for deduction in attention_analysis["deductions"]:
                                all_top_heads.extend(deduction["top_attention_heads"][:3])  # Top 3 heads per deduction
                            
                            # Sort by attention score and take top 5
                            all_top_heads.sort(key=lambda x: x["attention_score"], reverse=True)
                            top_heads_to_ablate = all_top_heads[:5]
                            
                            # Run ablation experiment
                            ablation_results = run_head_ablation_experiment(model, tokenizer, full_text, top_heads_to_ablate)
                            
                            attention_analysis["ablation_results"] = ablation_results
                            
                            # Update head importance based on logit difference
                            for head_info in top_heads_to_ablate:
                                head_key = f"L{head_info['layer']}_H{head_info['head']}"
                                head_importance[head_key] += ablation_results["logit_difference"]
                        except Exception as e:
                            print(f"Error in ablation experiment: {e}")
                            # Clear CUDA cache
                            torch.cuda.empty_cache()
                    
                    all_results.append(attention_analysis)
                    
                except Exception as e:
                    print(f"Error processing seed {seed_dir}: {e}")
                    # Clear CUDA cache
                    torch.cuda.empty_cache()
                    continue
        
        except Exception as e:
            print(f"Error processing problem {problem_dir}: {e}")
            # Clear CUDA cache
            torch.cuda.empty_cache()
            continue
    
    # Aggregate results
    aggregated_results = {
        "total_cots_analyzed": len(all_results),
        "total_deductions_found": sum(len(result["deductions"]) for result in all_results),
        "head_importance": dict(sorted(head_importance.items(), key=lambda x: x[1], reverse=True)),
        "detailed_results": all_results
    }
    
    return aggregated_results

def plot_head_attention_heatmap(attention_data: Dict, output_path: Path):
    """
    Plot a heatmap of attention head importance for deduction.
    
    Args:
        attention_data: Dictionary with attention analysis results
        output_path: Path to save the plot
    """
    # Extract head importance data
    head_importance = attention_data["head_importance"]
    
    if not head_importance:
        print("No head importance data to plot")
        return
    
    # Create a matrix of head importance
    max_layer = max([int(k.split('_')[0][1:]) for k in head_importance.keys()])
    max_head = max([int(k.split('_')[1][1:]) for k in head_importance.keys()])
    
    importance_matrix = np.zeros((max_layer + 1, max_head + 1))
    
    for key, value in head_importance.items():
        layer = int(key.split('_')[0][1:])
        head = int(key.split('_')[1][1:])
        importance_matrix[layer, head] = value
    
    # Normalize
    if importance_matrix.max() > 0:
        importance_matrix = importance_matrix / importance_matrix.max()
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Plot heatmap
    sns.heatmap(
        importance_matrix,
        cmap="viridis",
        annot=False,
        fmt=".2f",
        cbar_kws={"label": "Normalized Importance for Deduction"}
    )
    
    # Set labels and title
    plt.xlabel("Attention Head")
    plt.ylabel("Layer")
    plt.title("Attention Head Importance for Deduction")
    
    # Save plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_top_attention_heads(attention_data: Dict, output_path: Path, top_n: int = 10):
    """
    Plot the top N attention heads that attend to deduction across all examples.
    
    Args:
        attention_data: Dictionary with attention analysis results
        output_path: Path to save the plot
        top_n: Number of top heads to display
    """
    # Extract head importance data
    head_importance = attention_data["head_importance"]
    
    if not head_importance:
        print("No head importance data to plot")
        return
    
    # Sort heads by importance and get top N
    sorted_heads = sorted(head_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    # Extract head names and scores
    head_names = [head[0] for head in sorted_heads]
    head_scores = [head[1] for head in sorted_heads]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create bar chart
    bars = plt.bar(range(len(head_names)), head_scores, color='skyblue')
    
    # Add layer information with color coding
    for i, (head_name, _) in enumerate(sorted_heads):
        layer = int(head_name.split('_')[0][1:])
        # Color gradient based on layer (deeper layers = darker color)
        layer_color = plt.cm.viridis(layer / 32)  # Assuming max 32 layers
        bars[i].set_color(layer_color)
    
    # Add value labels on top of bars
    for i, v in enumerate(head_scores):
        plt.text(i, v + 0.01 * max(head_scores), f"{v:.3f}", ha='center', fontsize=9)
    
    # Set x-axis labels with head names
    plt.xticks(range(len(head_names)), head_names, rotation=45)
    
    # Add a color bar to show layer mapping
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=32))
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('Layer')
    
    # Set labels and title
    plt.xlabel("Attention Head")
    plt.ylabel("Average Attention Score")
    plt.title(f"Top {top_n} Attention Heads for Deduction (Averaged Across All Examples)")
    
    # Add grid
    plt.grid(True, alpha=0.3, axis='y')
    
    # Save plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_attention_patterns(
    attention_data: Dict,
    output_dir: Path,
    max_examples: int = 5
):
    """
    Plot attention patterns for specific deduction examples.
    
    Args:
        attention_data: Dictionary with attention analysis results
        output_dir: Directory to save plots
        max_examples: Maximum number of examples to plot
    """
    # Create directory for attention pattern plots
    attention_plots_dir = output_dir / "attention_patterns"
    attention_plots_dir.mkdir(exist_ok=True)
    
    # Get detailed results
    detailed_results = attention_data["detailed_results"]
    
    # Plot attention patterns for a few examples
    for i, result in enumerate(detailed_results[:max_examples]):
        for j, deduction in enumerate(result["deductions"][:3]):  # Up to 3 deductions per example
            plt.figure(figsize=(10, 6))
            top_heads = deduction["top_attention_heads"]
            
            if not top_heads:
                continue
                
            # Create bar chart of top attention heads
            head_labels = [f"L{h['layer']}_H{h['head']}" for h in top_heads[:10]]
            attention_scores = [h["attention_score"] for h in top_heads[:10]]
            
            plt.bar(head_labels, attention_scores, color="skyblue")
            
            # Add labels and title
            plt.xlabel("Attention Head")
            plt.ylabel("Attention Score")
            plt.title(f"Top Attention Heads for Deduction\nProblem: {result['problem_id']}, Seed: {result['seed_id']}")
            
            # Add deduction text
            plt.figtext(0.5, 0.01, f"Deduction: {deduction['deduction_text'][:100]}...", wrap=True, horizontalalignment='center', fontsize=8)
            
            # Rotate x-axis labels
            plt.xticks(rotation=45)
            
            # Save plot
            output_path = attention_plots_dir / f"attention_pattern_{result['problem_id']}_{result['seed_id']}_{j}.png"
            plt.tight_layout()
            plt.savefig(output_path, dpi=300)
            plt.close()

def main():
    parser = argparse.ArgumentParser(description="Analyze deduction circuits in language models")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", help="Model name")
    parser.add_argument("--max_problems", type=int, default=50, help="Maximum number of problems to process")
    parser.add_argument("--run_ablation", action="store_true", help="Run head ablation experiments")
    parser.add_argument("--top_n", type=int, default=10, help="Number of top heads to display in plots")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing")
    args = parser.parse_args()
    
    try:
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(args.model)
        
        # Get problem directories
        problem_dirs = get_problem_dirs(cots_dir)
        
        # Analyze deduction circuits
        results = analyze_deduction_circuits(
            problem_dirs, model, tokenizer, 
            max_problems=args.max_problems,
            run_ablation=args.run_ablation
        )
        
        # Save results
        with open(output_dir / "deduction_circuit_analysis.json", 'w', encoding='utf-8') as f:
            # Convert any non-serializable objects
            import json
            class NpEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    if isinstance(obj, np.floating):
                        return float(obj)
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return super(NpEncoder, self).default(obj)
            
            json.dump(results, f, indent=2, cls=NpEncoder)
        
        # Plot head importance heatmap
        if results["total_deductions_found"] > 0:
            plot_head_attention_heatmap(results, output_dir / "head_importance_heatmap.png")
            
            # Plot top attention heads
            plot_top_attention_heads(results, output_dir / "top_attention_heads.png", top_n=args.top_n)
            
            # Plot attention patterns for specific examples
            plot_attention_patterns(results, output_dir)
        
        print(f"Analysis complete. Found {results['total_deductions_found']} deductions in {results['total_cots_analyzed']} CoTs.")
        print(f"Results saved to {output_dir}")
    
    except Exception as e:
        print(f"Error in main execution: {e}")
        # Clear CUDA cache
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()