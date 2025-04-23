import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import pandas as pd
from collections import Counter, defaultdict
import re
from copy import deepcopy

# Set up paths
cots_dir = Path("cots")
analysis_dir = Path("analysis")
output_dir = Path("knowledge_retrieval_analysis")
output_dir.mkdir(exist_ok=True)

# Target categories for knowledge retrieval
KNOWLEDGE_CATEGORIES = {"Knowledge Recall", "Fact Retrieval", "Formula Application", "External Knowledge"}

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
    
    # Load model with attention outputs and hidden states enabled
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

def has_knowledge_categories(chunks_data: List[Dict]) -> bool:
    """
    Check if chunks data contains any of the knowledge categories.
    
    Args:
        chunks_data: List of chunk dictionaries
        
    Returns:
        True if any chunk has a knowledge category, False otherwise
    """
    for chunk in chunks_data:
        if chunk.get("category", "Unknown") in KNOWLEDGE_CATEGORIES:
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

def get_model_outputs_with_hidden_states(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text: str
) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
    """
    Get model outputs with hidden states and attention weights.
    
    Args:
        model: Hugging Face model
        tokenizer: Tokenizer
        text: Input text
        
    Returns:
        Tuple of (logits, hidden_states, attention_weights)
    """
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, output_attentions=True)
    
    return outputs.logits, outputs.hidden_states, outputs.attentions

def analyze_knowledge_retrieval(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    full_text: str,
    chunks_data: List[Dict],
    token_ranges: List[Tuple[int, int]]
) -> Dict:
    """
    Analyze knowledge retrieval patterns in the model.
    
    Args:
        model: Hugging Face model
        tokenizer: Tokenizer
        full_text: Full text of the solution
        chunks_data: List of chunk dictionaries
        token_ranges: List of token ranges for each chunk
        
    Returns:
        Dictionary with analysis results
    """
    # Get model outputs with hidden states and attention weights
    logits, hidden_states, attention_weights = get_model_outputs_with_hidden_states(model, tokenizer, full_text)
    
    # Get number of layers
    num_layers = len(hidden_states) - 1  # Subtract 1 for embedding layer
    
    # Find knowledge chunks
    knowledge_chunks = []
    for i, chunk in enumerate(chunks_data):
        if chunk.get("category", "Unknown") in KNOWLEDGE_CATEGORIES:
            if i < len(token_ranges) and token_ranges[i][0] != -1:  # Skip chunks with invalid token ranges
                knowledge_chunks.append({"index": i, "text": chunk["text"], "token_range": token_ranges[i]})
    
    if not knowledge_chunks:
        return {"has_knowledge": False}
    
    # Analyze hidden states and attention for knowledge chunks
    knowledge_analysis = {"has_knowledge": True, "knowledge_chunks": []}
    
    for knowledge in knowledge_chunks:
        knowledge_start, knowledge_end = knowledge["token_range"]
        
        # Calculate activation statistics for each layer
        layer_activations = []
        for layer_idx in range(num_layers + 1):  # +1 to include embedding layer
            layer_hidden = hidden_states[layer_idx][0]  # Batch size is 1
            
            # Get activations for knowledge tokens
            knowledge_activations = layer_hidden[knowledge_start:knowledge_end]
            
            # Calculate statistics
            mean_activation = knowledge_activations.mean(dim=0)
            max_activation = knowledge_activations.max(dim=0).values
            
            # Find top activated neurons
            top_neurons = torch.topk(max_activation, k=10)
            
            layer_activations.append({
                "layer": layer_idx,
                "mean_activation": mean_activation.detach().cpu().numpy().tolist(),
                "max_activation": max_activation.detach().cpu().numpy().tolist(),
                "top_neurons": {
                    "indices": top_neurons.indices.detach().cpu().numpy().tolist(),
                    "values": top_neurons.values.detach().cpu().numpy().tolist()
                }
            })
        
        # Analyze attention patterns
        attention_analysis = []
        
        # Skip embedding layer (no attention)
        for layer_idx in range(num_layers):
            layer_attention = attention_weights[layer_idx][0]  # Batch size is 1
            
            # Calculate attention statistics for each head
            head_attention = []
            for head_idx in range(layer_attention.shape[1]):
                head_weights = layer_attention[head_idx]
                
                # Calculate attention from knowledge tokens to all other tokens
                knowledge_attention = head_weights[knowledge_start:knowledge_end, :]
                
                # Calculate statistics
                mean_attention = knowledge_attention.mean(dim=0)
                max_attention = knowledge_attention.max(dim=0).values
                
                # Find top attended tokens
                top_attended = torch.topk(max_attention, k=5)
                top_indices = top_attended.indices.detach().cpu().numpy().tolist()
                top_values = top_attended.values.detach().cpu().numpy().tolist()
                
                # Get token strings for top attended tokens
                top_tokens = []
                for idx in top_indices:
                    if idx < len(tokenizer.input_ids[0]):
                        token = tokenizer.decode([tokenizer.input_ids[0][idx]])
                        top_tokens.append(token)
                    else:
                        top_tokens.append("<unknown>")
                
                head_attention.append({
                    "head": head_idx,
                    "mean_attention": mean_attention.detach().cpu().numpy().tolist(),
                    "max_attention": max_attention.detach().cpu().numpy().tolist(),
                    "top_attended": {
                        "indices": top_indices,
                        "values": top_values,
                        "tokens": top_tokens
                    }
                })
            
            attention_analysis.append({
                "layer": layer_idx,
                "head_attention": head_attention
            })
        
        # Calculate direct logit attribution
        knowledge_logits = []
        
        # For each token in the knowledge chunk
        for token_idx in range(knowledge_start, knowledge_end):
            if token_idx >= logits.shape[1]:
                continue
                
            # Get the actual token
            token_id = tokenizer.input_ids[0][token_idx]
            token_str = tokenizer.decode([token_id])
            
            # Get logit for this token
            token_logit = logits[0, token_idx - 1, token_id].item()
            
            # Calculate layer contributions using residual stream
            layer_contributions = []
            
            # Start with embedding layer
            prev_hidden = hidden_states[0][0, token_idx]
            
            for layer_idx in range(1, num_layers + 1):
                current_hidden = hidden_states[layer_idx][0, token_idx]
                
                # Calculate contribution as the change in hidden state
                contribution = current_hidden - prev_hidden
                
                # Update previous hidden state
                prev_hidden = current_hidden
                
                # Calculate contribution to logit
                contribution_norm = torch.norm(contribution).item()
                
                layer_contributions.append({
                    "layer": layer_idx,
                    "contribution": contribution_norm
                })
            
            knowledge_logits.append({
                "token_idx": token_idx,
                "token": token_str,
                "logit": token_logit,
                "layer_contributions": layer_contributions
            })
        
        # Add to analysis
        knowledge_analysis["knowledge_chunks"].append({
            "text": knowledge["text"][:100] + "..." if len(knowledge["text"]) > 100 else knowledge["text"],
            "index": knowledge["index"],
            "token_range": knowledge["token_range"],
            "layer_activations": layer_activations,
            "attention_analysis": attention_analysis,
            "logit_attribution": knowledge_logits
        })
    
    return knowledge_analysis

def analyze_knowledge_circuits(
    problem_dirs: List[Path],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_problems: int = 50
) -> Dict:
    """
    Analyze knowledge retrieval circuits across multiple problems.
    
    Args:
        problem_dirs: List of problem directory paths
        model: Hugging Face model
        tokenizer: Tokenizer
        max_problems: Maximum number of problems to process
        
    Returns:
        Dictionary with analysis results
    """
    all_results = []
    neuron_importance = defaultdict(float)
    head_importance = defaultdict(float)
    layer_importance = defaultdict(float)
    
    # Process a limited number of problems
    for problem_dir in tqdm(problem_dirs[:max_problems], desc="Analyzing knowledge retrieval"):
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
                
            # Check if this CoT has any knowledge categories
            if not has_knowledge_categories(chunks_data):
                continue
                
            # Extract chunks
            chunks = [item["text"] for item in chunks_data]
            
            # Get token ranges for each chunk
            token_ranges = get_token_ranges_for_chunks(full_text, chunks, tokenizer)
            
            # Analyze knowledge retrieval patterns
            knowledge_analysis = analyze_knowledge_retrieval(model, tokenizer, full_text, chunks_data, token_ranges)
            
            if not knowledge_analysis["has_knowledge"]:
                continue
            
            # Add problem and seed info
            knowledge_analysis["problem_id"] = problem_dir.name
            knowledge_analysis["seed_id"] = seed_id
            
            # Update importance metrics
            for knowledge_chunk in knowledge_analysis["knowledge_chunks"]:
                # Update neuron importance based on top neurons
                for layer_activation in knowledge_chunk["layer_activations"]:
                    layer = layer_activation["layer"]
                    
                    # Update layer importance
                    layer_importance[f"Layer_{layer}"] += 1.0
                    
                    # Update neuron importance
                    for neuron_idx, neuron_value in zip(
                        layer_activation["top_neurons"]["indices"],
                        layer_activation["top_neurons"]["values"]
                    ):
                        neuron_key = f"L{layer}_N{neuron_idx}"
                        neuron_importance[neuron_key] += float(neuron_value)
                
                # Update head importance based on attention analysis
                for layer_attention in knowledge_chunk["attention_analysis"]:
                    layer = layer_attention["layer"]
                    
                    for head_data in layer_attention["head_attention"]:
                        head = head_data["head"]
                        head_key = f"L{layer}_H{head}"
                        
                        # Use max attention as importance score
                        max_attention = max(head_data["max_attention"]) if head_data["max_attention"] else 0
                        head_importance[head_key] += float(max_attention)
            
            all_results.append(knowledge_analysis)
    
    # Aggregate results
    aggregated_results = {
        "total_cots_analyzed": len(all_results),
        "total_knowledge_chunks_found": sum(len(result["knowledge_chunks"]) for result in all_results),
        "neuron_importance": dict(sorted(neuron_importance.items(), key=lambda x: x[1], reverse=True)),
        "head_importance": dict(sorted(head_importance.items(), key=lambda x: x[1], reverse=True)),
        "layer_importance": dict(sorted(layer_importance.items(), key=lambda x: x[1], reverse=True)),
        "detailed_results": all_results
    }
    
    return aggregated_results

def plot_top_neurons(neuron_data: Dict, output_path: Path, top_n: int = 20):
    """
    Plot the top N neurons by importance for knowledge retrieval.
    
    Args:
        neuron_data: Dictionary with neuron importance data
        output_path: Path to save the plot
        top_n: Number of top neurons to display
    """
    # Sort neurons by importance and get top N
    sorted_neurons = sorted(neuron_data.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    # Extract neuron names and scores
    neuron_names = [neuron[0] for neuron in sorted_neurons]
    neuron_scores = [neuron[1] for neuron in sorted_neurons]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create bar chart
    bars = plt.bar(range(len(neuron_names)), neuron_scores, color='skyblue')
    
    # Add layer information with color coding
    for i, (neuron_name, _) in enumerate(sorted_neurons):
        layer = int(neuron_name.split('_')[0][1:])
        # Color gradient based on layer (deeper layers = darker color)
        layer_color = plt.cm.viridis(layer / 32)  # Assuming max 32 layers
        bars[i].set_color(layer_color)
    
    # Add value labels on top of bars
    for i, v in enumerate(neuron_scores):
        plt.text(i, v + 0.01 * max(neuron_scores), f"{v:.2f}", ha='center', fontsize=9)
    
    # Set x-axis labels with neuron names
    plt.xticks(range(len(neuron_names)), neuron_names, rotation=45)
    
    # Add a color bar to show layer mapping
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=32))
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('Layer')
    
    # Set labels and title
    plt.xlabel("Neuron")
    plt.ylabel("Activation Score")
    plt.title(f"Top {top_n} Neurons for Knowledge Retrieval")
    
    # Add grid
    plt.grid(True, alpha=0.3, axis='y')
    
    # Save plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_top_attention_heads(head_data: Dict, output_path: Path, top_n: int = 20):
    """
    Plot the top N attention heads by importance for knowledge retrieval.
    
    Args:
        head_data: Dictionary with head importance data
        output_path: Path to save the plot
        top_n: Number of top heads to display
    """
    # Sort heads by importance and get top N
    sorted_heads = sorted(head_data.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
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
        plt.text(i, v + 0.01 * max(head_scores), f"{v:.2f}", ha='center', fontsize=9)
    
    # Set x-axis labels with head names
    plt.xticks(range(len(head_names)), head_names, rotation=45)
    
    # Add a color bar to show layer mapping
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=32))
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('Layer')
    
    # Set labels and title
    plt.xlabel("Attention Head")
    plt.ylabel("Attention Score")
    plt.title(f"Top {top_n} Attention Heads for Knowledge Retrieval")
    
    # Add grid
    plt.grid(True, alpha=0.3, axis='y')
    
    # Save plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_layer_importance(layer_data: Dict, output_path: Path):
    """
    Plot the importance of each layer for knowledge retrieval.
    
    Args:
        layer_data: Dictionary with layer importance data
        output_path: Path to save the plot
    """
    # Sort layers by number
    sorted_layers = sorted(layer_data.items(), key=lambda x: int(x[0].split('_')[1]))
    
    # Extract layer names and scores
    layer_names = [layer[0].replace('Layer_', '') for layer in sorted_layers]
    layer_scores = [layer[1] for layer in sorted_layers]
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Create bar chart with color gradient
    colors = plt.cm.viridis(np.linspace(0, 1, len(layer_names)))
    bars = plt.bar(range(len(layer_names)), layer_scores, color=colors)
    
    # Add value labels on top of bars
    for i, v in enumerate(layer_scores):
        plt.text(i, v + 0.01 * max(layer_scores), f"{v:.1f}", ha='center', fontsize=9)
    
    # Set x-axis labels with layer names
    plt.xticks(range(len(layer_names)), layer_names, rotation=0)
    
    # Set labels and title
    plt.xlabel("Layer")
    plt.ylabel("Frequency")
    plt.title("Layer Importance for Knowledge Retrieval")
    
    # Add grid
    plt.grid(True, alpha=0.3, axis='y')
    
    # Save plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_logit_attribution(results: Dict, output_path: Path, max_examples: int = 5):
    """
    Plot logit attribution for knowledge tokens.
    
    Args:
        results: Dictionary with analysis results
        output_path: Path to save the plot
        max_examples: Maximum number of examples to plot
    """
    # Create directory for logit attribution plots
    logit_plots_dir = output_path.parent / "logit_attribution"
    logit_plots_dir.mkdir(exist_ok=True)
    
    # Get detailed results
    detailed_results = results["detailed_results"]
    
    # Plot logit attribution for a few examples
    for i, result in enumerate(detailed_results[:max_examples]):
        for j, knowledge_chunk in enumerate(result["knowledge_chunks"][:3]):  # Up to 3 knowledge chunks per example
            # Get logit attribution data
            logit_data = knowledge_chunk["logit_attribution"]
            
            if not logit_data:
                continue
            
            # Create figure
            plt.figure(figsize=(12, 8))
            
            # For each token, plot layer contributions
            for k, token_data in enumerate(logit_data[:5]):  # Up to 5 tokens per chunk
                token = token_data["token"]
                layers = [contrib["layer"] for contrib in token_data["layer_contributions"]]
                contributions = [contrib["contribution"] for contrib in token_data["layer_contributions"]]
                
                plt.plot(layers, contributions, 'o-', label=f"Token: {token}")
            
            # Set labels and title
            plt.xlabel("Layer")
            plt.ylabel("Contribution Magnitude")
            plt.title(f"Layer Contributions to Knowledge Tokens\nProblem: {result['problem_id']}, Seed: {result['seed_id']}")
            
            # Add knowledge text
            plt.figtext(0.5, 0.01, f"Knowledge: {knowledge_chunk['text'][:100]}...", 
                       wrap=True, horizontalalignment='center', fontsize=8)
            
            # Add legend
            plt.legend()
            
            # Add grid
            plt.grid(True, alpha=0.3)
            
            # Save plot
            output_file = logit_plots_dir / f"logit_attribution_{result['problem_id']}_{result['seed_id']}_{j}.png"
            plt.tight_layout()
            plt.savefig(output_file, dpi=300)
            plt.close()

def main():
    parser = argparse.ArgumentParser(description="Analyze knowledge retrieval in language models")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", help="Model name")
    parser.add_argument("--max_problems", type=int, default=50, help="Maximum number of problems to process")
    parser.add_argument("--top_n", type=int, default=20, help="Number of top neurons/heads to display in plots")
    args = parser.parse_args()
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model)
    
    # Get problem directories
    problem_dirs = get_problem_dirs(cots_dir)
    
    # Analyze knowledge retrieval circuits
    results = analyze_knowledge_circuits(
        problem_dirs, model, tokenizer, 
        max_problems=args.max_problems
    )
    
    # Save results
    with open(output_dir / "knowledge_retrieval_analysis.json", 'w', encoding='utf-8') as f:
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
    
    # Plot top neurons
    plot_top_neurons(results["neuron_importance"], output_dir / "top_neurons.png", top_n=args.top_n)
    
    # Plot top attention heads
    plot_top_attention_heads(results["head_importance"], output_dir / "top_attention_heads.png", top_n=args.top_n)
    
    # Plot layer importance
    plot_layer_importance(results["layer_importance"], output_dir / "layer_importance.png")
    
    # Plot logit attribution
    plot_logit_attribution(results, output_dir / "logit_attribution.png")
    
    print(f"Analysis complete. Found {results['total_knowledge_chunks_found']} knowledge chunks in {results['total_cots_analyzed']} CoTs.")
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()