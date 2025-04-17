import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Tuple, Union, Optional
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
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score

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
    max_problems: int = 100
) -> Tuple[List[torch.Tensor], List[Dict]]:
    """
    Extract residual stream activations for chunks across multiple problems.
    
    Args:
        problem_dirs: List of problem directory paths
        model: Hugging Face model
        tokenizer: Tokenizer
        layer: Layer to extract activations from
        max_problems: Maximum number of problems to process
        
    Returns:
        Tuple of (activations, chunk_metadata)
    """
    all_activations = []
    all_chunk_metadata = []
    
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
            
            # Get token indices for each chunk in the full text
            chunk_token_ranges = []
            full_tokens = tokenizer(full_text, return_tensors="pt").to(model.device)
            
            # Find token ranges for each chunk
            current_pos = 0
            for chunk in chunks:
                # Find the chunk in the full text starting from current position
                chunk_start = full_text.find(chunk, current_pos)
                if chunk_start == -1:
                    # If exact match not found, try with some flexibility
                    # This handles cases where whitespace might differ
                    chunk_words = chunk.split()
                    for i in range(current_pos, len(full_text) - len(chunk)):
                        if full_text[i:i+len(chunk_words[0])] == chunk_words[0]:
                            potential_match = full_text[i:i+len(chunk)]
                            if potential_match.split() == chunk_words:
                                chunk_start = i
                                break
                
                if chunk_start == -1:
                    print(f"Warning: Chunk not found in full text: {chunk[:50]}...")
                    continue
                    
                chunk_end = chunk_start + len(chunk)
                current_pos = chunk_end
                
                # Convert character positions to token indices
                chunk_tokens = tokenizer(full_text[:chunk_start], return_tensors="pt")
                start_idx = len(chunk_tokens.input_ids[0]) - 2
                
                chunk_tokens = tokenizer(full_text[:chunk_end], return_tensors="pt")
                end_idx = len(chunk_tokens.input_ids[0]) + 3
                
                chunk_token_ranges.append((max(0, start_idx), end_idx))
            
            # Get activations for the full text
            with torch.no_grad():
                activations = get_residual_stream_activations(model, tokenizer, full_text, [layer])
            
            if layer not in activations:
                continue
                
            # Extract activations for each chunk
            layer_activations = activations[layer]
            
            for i, (start_idx, end_idx) in enumerate(chunk_token_ranges):
                if i >= len(chunks_data):
                    continue
                    
                # Extract activations for this chunk (average across tokens)
                # chunk_act = layer_activations[0, start_idx:end_idx, :].mean(dim=0)
                # Extract activations for this chunk using weighted average
                chunk_act = layer_activations[0, start_idx:end_idx, :]
                # Convert to float32 to avoid overflow issues with float16
                chunk_act = chunk_act.to(torch.float32)
                weights = chunk_act.norm(dim=-1)
                # Handle case where weights sum to zero to avoid division by zero
                if weights.sum() > 0:
                    chunk_act = (chunk_act * weights.unsqueeze(-1)).sum(dim=0) / weights.sum()
                else:
                    # Fallback to simple mean if weights are all zero
                    chunk_act = chunk_act.mean(dim=0)
                all_activations.append(chunk_act)
                
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

def find_optimal_k(
    activation_matrix: np.ndarray,
    max_k: int = 30
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

def main():
    parser = argparse.ArgumentParser(description="Cluster chunks based on residual stream activations")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", help="Model name")
    parser.add_argument("--layer", type=int, default=31, help="Layer to extract activations from")
    parser.add_argument("--max_problems", type=int, default=100, help="Maximum number of problems to process")
    parser.add_argument("--n_clusters", type=int, default=0, help="Number of clusters (0 to find optimal)")
    parser.add_argument("--dim_reduction", type=str, default="tsne", choices=["tsne", "pca", "umap"], help="Dimensionality reduction method")
    args = parser.parse_args()
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model)
    
    # Get problem directories
    problem_dirs = get_problem_dirs(cots_dir)
    
    # Extract chunk activations
    activations, chunk_metadata = extract_chunk_activations(
        problem_dirs, model, tokenizer, args.layer, args.max_problems
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
    output_path = output_dir / f"clusters_layer{args.layer}_{args.dim_reduction}.png"
    plot_clusters(reduced_matrix, labels, chunk_metadata, output_path, 
                 title=f"Chunk Clusters (Layer {args.layer}, {n_clusters} clusters)")
    
    # Plot by category
    output_path = output_dir / f"categories_layer{args.layer}_{args.dim_reduction}.png"
    plot_clusters(reduced_matrix, labels, chunk_metadata, output_path, 
                 title=f"Chunk Categories (Layer {args.layer})", color_by="category")
    
    # Analyze clusters
    analysis_path = output_dir / f"cluster_analysis_layer{args.layer}.json"
    analyze_clusters(labels, chunk_metadata, analysis_path)
    
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()