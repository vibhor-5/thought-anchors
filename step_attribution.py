import multiprocessing
# Set the start method to 'spawn' for CUDA compatibility
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from utils import split_solution_into_chunks
import gc
import argparse
import multiprocessing
from functools import partial

def load_problem(problem_dir: Path) -> Dict:
    """Load problem from problem directory."""
    problem_file = problem_dir / "problem.json"
    if problem_file.exists():
        with open(problem_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def load_base_solution(problem_dir: Path) -> Dict:
    """Load base solution from problem directory."""
    base_solution_file = problem_dir / "base_solution.json"
    if base_solution_file.exists():
        with open(base_solution_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def load_chunks(problem_dir: Path) -> List[str]:
    """Load chunks from problem directory."""
    chunks_file = problem_dir / "chunks_labeled.json"
    if chunks_file.exists():
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
            return chunks_data
    return []

def load_chunk_rollouts(problem_dir: Path, chunk_idx: int) -> List[Dict]:
    """Load rollouts for a specific chunk."""
    chunk_dir = problem_dir / f"chunk_{chunk_idx}"
    solutions_file = chunk_dir / "solutions.json"
    
    if solutions_file.exists():
        with open(solutions_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def extract_steps_from_rollout(rollout_text: str) -> List[str]:
    """Extract steps from a rollout text."""
    return split_solution_into_chunks(rollout_text)

def compute_embedding_similarity(text1: str, text2: str, model) -> float:
    """Compute cosine similarity between embeddings of two texts."""
    if not text1 or not text2:
        return 0.0
    
    # Get embeddings
    embedding1 = model.encode([text1])[0]
    embedding2 = model.encode([text2])[0]
    
    # Compute cosine similarity
    similarity = cosine_similarity([embedding1], [embedding2])[0][0]
    
    return float(similarity)

def find_best_matches_fully_batched(target_step: str, all_rollout_steps: List[List[str]], model, threshold: float = 0.7) -> List[Tuple[int, float]]:
    """
    Find the best matching step in each rollout for the target_step using full batching.
    
    Args:
        target_step: The step to find a match for
        all_rollout_steps: List of lists, where each inner list contains steps from one rollout
        model: Sentence embedding model
        threshold: Minimum similarity threshold to consider a match
        
    Returns:
        List of tuples (best_match_index, similarity_score) for each rollout
    """
    if not all_rollout_steps:
        return []
    
    # Flatten all steps from all rollouts and keep track of which rollout they belong to
    all_steps = []
    rollout_indices = []
    step_indices = []
    
    for rollout_idx, rollout_steps in enumerate(all_rollout_steps):
        for step_idx, step in enumerate(rollout_steps):
            all_steps.append(step)
            rollout_indices.append(rollout_idx)
            step_indices.append(step_idx)
    
    if not all_steps:
        return [(-1, 0.0)] * len(all_rollout_steps)
    
    # Get embedding for target step and all steps in a single batch
    all_embeddings = model.encode([target_step] + all_steps)
    target_embedding = all_embeddings[0]
    step_embeddings = all_embeddings[1:]
    
    # Compute similarities with target
    all_similarities = cosine_similarity([target_embedding], step_embeddings)[0]
    
    # Initialize results with default values
    best_matches = [(-1, 0.0)] * len(all_rollout_steps)
    
    # Group by rollout and find best match for each
    for i, (rollout_idx, step_idx, similarity) in enumerate(zip(rollout_indices, step_indices, all_similarities)):
        if similarity >= threshold and (best_matches[rollout_idx][0] == -1 or similarity > best_matches[rollout_idx][1]):
            best_matches[rollout_idx] = (step_idx, float(similarity))
    
    return best_matches

def process_chunk_pair(args):
    """
    Process a pair of chunks to compute importance.
    This function is designed to be used with multiprocessing.
    
    Args:
        args: Tuple containing (i, j, target_step, problem_dir, embedding_model, similarity_threshold)
        
    Returns:
        Tuple of (i, j, importance)
    """
    i, j, target_step, problem_dir, embedding_model, similarity_threshold = args
    
    # Get rollouts where step i was kept (resampling started at i+1)
    include_i_rollouts = load_chunk_rollouts(problem_dir, i+1)
    
    # Get rollouts where step i was removed (resampling started at i)
    exclude_i_rollouts = load_chunk_rollouts(problem_dir, i)
    
    # Filter for valid rollouts (has rollout text)
    include_i_rollouts = [r for r in include_i_rollouts if 'rollout' in r and r['rollout']]
    exclude_i_rollouts = [r for r in exclude_i_rollouts if 'rollout' in r and r['rollout']]
    
    if not include_i_rollouts or not exclude_i_rollouts:
        return i, j, 0.0
    
    # Extract steps from all rollouts
    include_i_steps = [extract_steps_from_rollout(rollout['rollout']) for rollout in include_i_rollouts]
    exclude_i_steps = [extract_steps_from_rollout(rollout['rollout']) for rollout in exclude_i_rollouts]
    
    # Process all rollouts where step i was kept
    include_i_matches = find_best_matches_fully_batched(target_step, include_i_steps, embedding_model, similarity_threshold)
    
    # Process all rollouts where step i was removed
    exclude_i_matches = find_best_matches_fully_batched(target_step, exclude_i_steps, embedding_model, similarity_threshold)
    
    # Calculate average similarity for both cases
    include_positives = [sim for _, sim in include_i_matches if sim > 0]
    exclude_positives = [sim for _, sim in exclude_i_matches if sim > 0]

    include_i_avg_similarity = np.mean(include_positives) if include_positives else 0.0
    exclude_i_avg_similarity = np.mean(exclude_positives) if exclude_positives else 0.0

    # Calculate match rate for both cases
    include_i_match_rate = len(include_positives) / len(include_i_matches) if include_i_matches else 0
    exclude_i_match_rate = len(exclude_positives) / len(exclude_i_matches) if exclude_i_matches else 0

    # Calculate importance as difference in match rate and similarity
    importance = 0.5 * (include_i_match_rate - exclude_i_match_rate)
    if include_positives and exclude_positives:  # Only add similarity difference if both have matches
        importance += 0.5 * (include_i_avg_similarity - exclude_i_avg_similarity)
    elif include_positives:  # Only include_i has matches
        importance += 0.5 * include_i_avg_similarity
    elif exclude_positives:  # Only exclude_i has matches
        importance -= 0.5 * exclude_i_avg_similarity
    
    return i, j, importance

def compute_step_importance_matrix(
    problem_dir: Path,
    embedding_model,
    similarity_threshold: float = 0.7,
    max_chunks: int = 50,
    use_cache: bool = True,
    n_processes: int = None,
    output_dir: Path = None
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute the step importance matrix for a problem.
    
    Args:
        problem_dir: Path to problem directory
        embedding_model: Sentence embedding model
        similarity_threshold: Threshold for considering steps similar
        max_chunks: Maximum number of chunks to analyze
        use_cache: Whether to use cached results
        n_processes: Number of processes to use for parallel computation
        output_dir: Directory to save cache files (if None, uses problem_dir)
        
    Returns:
        Tuple of (importance_matrix, chunk_texts)
        importance_matrix[i, j] represents the causal importance of step i on step j
    """
    # Determine cache directory location - use output_dir if provided
    if output_dir is not None:
        cache_dir = output_dir / "attribution_cache"
    else:
        raise ValueError("output_dir must be provided")
    
    cache_dir.mkdir(exist_ok=True, parents=True)
    cache_file = cache_dir / f"step_importance_matrix_t{similarity_threshold}.npz"
    
    if use_cache and cache_file.exists():
        print(f"Loading cached step importance matrix from {cache_file}")
        data = np.load(cache_file, allow_pickle=True)
        return data['matrix'], data['chunk_texts']
    
    # Load base solution and chunks
    base_solution = load_base_solution(problem_dir)
    original_chunks = load_chunks(problem_dir)
    
    if not original_chunks:
        print(f"No chunks found for problem {problem_dir.name}")
        return np.array([]), []
    
    # Limit number of chunks to analyze
    original_chunks = original_chunks[:max_chunks]
    num_chunks = len(original_chunks)
    
    print(f"Analyzing {num_chunks} chunks for problem {problem_dir.name}")
    
    # Initialize importance matrix
    # importance_matrix[i, j] represents the causal importance of step i on step j
    importance_matrix = np.zeros((num_chunks, num_chunks))
    
    # Pre-compute embeddings for all original chunks
    print("Computing embeddings for all original chunks...")
    original_chunk_texts = [chunk['chunk'] for chunk in original_chunks]
    original_embeddings = embedding_model.encode(original_chunk_texts)
    
    # Process all chunk pairs
    print("Processing all chunk pairs...")
    for i in tqdm(range(num_chunks - 1), desc="Processing source steps"):
        # Get target steps (all steps after i)
        target_indices = list(range(i+1, num_chunks))
        target_embeddings = original_embeddings[target_indices]
        
        # Get rollouts where step i was kept (resampling started at i+1)
        include_i_rollouts = load_chunk_rollouts(problem_dir, i+1)
        
        # Get rollouts where step i was removed (resampling started at i)
        exclude_i_rollouts = load_chunk_rollouts(problem_dir, i)
        
        # Filter for valid rollouts (has rollout text)
        include_i_rollouts = [r for r in include_i_rollouts if 'rollout' in r and r['rollout']]
        exclude_i_rollouts = [r for r in exclude_i_rollouts if 'rollout' in r and r['rollout']]
        
        if not include_i_rollouts or not exclude_i_rollouts:
            continue
        
        # Extract steps from all rollouts
        include_i_steps = [extract_steps_from_rollout(rollout['rollout']) for rollout in include_i_rollouts]
        exclude_i_steps = [extract_steps_from_rollout(rollout['rollout']) for rollout in exclude_i_rollouts]
        
        # Flatten all steps from all include_i rollouts
        include_all_steps = []
        include_rollout_indices = []
        include_step_indices = []
        
        for rollout_idx, rollout_steps in enumerate(include_i_steps):
            for step_idx, step in enumerate(rollout_steps):
                include_all_steps.append(step)
                include_rollout_indices.append(rollout_idx)
                include_step_indices.append(step_idx)
        
        # Flatten all steps from all exclude_i rollouts
        exclude_all_steps = []
        exclude_rollout_indices = []
        exclude_step_indices = []
        
        for rollout_idx, rollout_steps in enumerate(exclude_i_steps):
            for step_idx, step in enumerate(rollout_steps):
                exclude_all_steps.append(step)
                exclude_rollout_indices.append(rollout_idx)
                exclude_step_indices.append(step_idx)
        
        # Compute embeddings for all steps at once
        if include_all_steps:
            include_embeddings = embedding_model.encode(include_all_steps)
        else:
            include_embeddings = np.array([])
            
        if exclude_all_steps:
            exclude_embeddings = embedding_model.encode(exclude_all_steps)
        else:
            exclude_embeddings = np.array([])
        
        # Process each target step
        for j_idx, j in enumerate(target_indices):
            target_embedding = target_embeddings[j_idx].reshape(1, -1)
            
            # Process include_i rollouts
            include_i_matches = [(-1, 0.0)] * len(include_i_steps)
            if len(include_embeddings) > 0:
                include_similarities = cosine_similarity(target_embedding, include_embeddings)[0]
                
                # Find best match for each rollout
                for step_idx, (rollout_idx, similarity) in enumerate(zip(include_rollout_indices, include_similarities)):
                    if similarity >= similarity_threshold and (include_i_matches[rollout_idx][0] == -1 or similarity > include_i_matches[rollout_idx][1]):
                        include_i_matches[rollout_idx] = (include_step_indices[step_idx], float(similarity))
            
            # Process exclude_i rollouts
            exclude_i_matches = [(-1, 0.0)] * len(exclude_i_steps)
            if len(exclude_embeddings) > 0:
                exclude_similarities = cosine_similarity(target_embedding, exclude_embeddings)[0]
                
                # Find best match for each rollout
                for step_idx, (rollout_idx, similarity) in enumerate(zip(exclude_rollout_indices, exclude_similarities)):
                    if similarity >= similarity_threshold and (exclude_i_matches[rollout_idx][0] == -1 or similarity > exclude_i_matches[rollout_idx][1]):
                        exclude_i_matches[rollout_idx] = (exclude_step_indices[step_idx], float(similarity))
            
            # Calculate average similarity for both cases
            include_positives = [sim for _, sim in include_i_matches if sim > 0]
            exclude_positives = [sim for _, sim in exclude_i_matches if sim > 0]

            include_i_avg_similarity = np.mean(include_positives) if include_positives else 0.0
            exclude_i_avg_similarity = np.mean(exclude_positives) if exclude_positives else 0.0

            # Calculate match rate for both cases
            include_i_match_rate = len(include_positives) / len(include_i_matches) if include_i_matches else 0
            exclude_i_match_rate = len(exclude_positives) / len(exclude_i_matches) if exclude_i_matches else 0

            # Calculate importance as difference in match rate and similarity
            importance = 0.5 * (include_i_match_rate - exclude_i_match_rate)
            if include_positives and exclude_positives:  # Only add similarity difference if both have matches
                importance += 0.5 * (include_i_avg_similarity - exclude_i_avg_similarity)
            elif include_positives:  # Only include_i has matches
                importance += 0.5 * include_i_avg_similarity
            elif exclude_positives:  # Only exclude_i has matches
                importance -= 0.5 * exclude_i_avg_similarity
            
            importance_matrix[i, j] = importance
    
    # Save to cache
    np.savez(cache_file, matrix=importance_matrix, chunk_texts=original_chunks)
    
    return importance_matrix, original_chunks

def plot_step_importance_matrix(
    importance_matrix: np.ndarray,
    chunk_texts: List[str],
    output_file: Path,
    problem_idx: str,
    max_text_length: int = 30
) -> None:
    """
    Plot the step importance matrix as a heatmap.
    
    Args:
        importance_matrix: Step importance matrix
        chunk_texts: List of chunk texts
        output_file: Path to save the plot
        problem_idx: Problem index for the title
        max_text_length: Maximum length of chunk text to display
    """
    # Create shortened labels
    labels = []
    for i, text in enumerate(chunk_texts):
        if len(text) > max_text_length:
            text = text[:max_text_length] + "..."
        labels.append(f"{i}: {text}")
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Create heatmap
    sns.heatmap(
        importance_matrix,
        cmap="viridis",
        annot=False,
        fmt=".2f",
        linewidths=0.5,
        xticklabels=range(len(chunk_texts)),
        yticklabels=range(len(chunk_texts))
    )
    
    # Add labels and title
    plt.xlabel("Target Step (j)")
    plt.ylabel("Source Step (i)")
    plt.title(f"Problem {problem_idx}: Step-to-Step Causal Importance Matrix")
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    # Create a second figure with text labels for steps
    fig, ax = plt.subplots(figsize=(12, len(chunk_texts) * 0.4))
    ax.axis('off')
    
    # Add step texts
    for i, label in enumerate(labels):
        ax.text(0, 1 - (i / len(labels)), f"{label}", fontsize=10)
    
    # Save step text figure
    plt.tight_layout()
    plt.savefig(output_file.parent / f"{output_file.stem}_steps.png")
    plt.close()

def get_function_tag_prefix(chunk):
    """Extract the first two uppercase letters from the first function tag."""
    if isinstance(chunk, dict) and 'function_tags' in chunk and chunk['function_tags']:
        # Get the first function tag
        first_tag = chunk['function_tags'][0]
        # Extract uppercase letters
        uppercase_letters = ''.join([word[0].upper() for word in first_tag.split('_')])
        # Return the first two (or fewer if only one exists)
        return uppercase_letters
    return ""

def analyze_step_attribution(
    problem_dirs: List[Path],
    output_dir: Path,
    similarity_threshold: float = 0.7,
    max_chunks: int = 50,
    use_cache: bool = True,
    n_processes: int = None
) -> None:
    """
    Analyze step attribution for a list of problems.
    
    Args:
        problem_dirs: List of problem directories
        output_dir: Directory to save analysis results
        similarity_threshold: Threshold for considering steps similar
        max_chunks: Maximum number of chunks to analyze
        use_cache: Whether to use cached results
        n_processes: Number of processes to use for parallel computation
    """
    # Create output directory
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create directories for correct and incorrect solutions
    correct_dir = output_dir / "correct_base_solution"
    incorrect_dir = output_dir / "incorrect_base_solution"
    correct_dir.mkdir(exist_ok=True, parents=True)
    incorrect_dir.mkdir(exist_ok=True, parents=True)
    
    # Load sentence embedding model
    print("Loading sentence embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2').to('cuda:0')
    model.eval()
    
    # Process each problem
    for problem_dir in tqdm(problem_dirs, desc="Analyzing problems"):
        problem_idx = int(problem_dir.name.split('_')[-1])
        
        # Determine if this is a correct or incorrect solution
        is_correct = "correct_base_solution" in str(problem_dir) and "incorrect_base_solution" not in str(problem_dir)
        
        # Create problem-specific output directory in the appropriate location
        if is_correct:
            problem_output_dir = correct_dir / f"problem_{problem_idx}"
        else:
            problem_output_dir = incorrect_dir / f"problem_{problem_idx}"
        problem_output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create plots directory
        plots_dir = problem_output_dir / "plots"
        plots_dir.mkdir(exist_ok=True, parents=True)
        
        # Create cache directory
        cache_dir = problem_output_dir / "cache"
        cache_dir.mkdir(exist_ok=True, parents=True)
        
        # Save base solution
        base_solution = load_base_solution(problem_dir)
        if base_solution:
            with open(problem_output_dir / "base_solution.json", 'w', encoding='utf-8') as f:
                json.dump(base_solution, f, indent=2)
        
        # Check if cached result exists
        cache_file = cache_dir / f"step_importance_matrix_t{similarity_threshold}.npz"
        
        if use_cache and cache_file.exists():
            print(f"Loading cached step importance matrix for problem {problem_idx}")
            data = np.load(cache_file, allow_pickle=True)
            importance_matrix = data['matrix']
            chunk_texts = data['chunk_texts'].tolist()
        else:
            # Compute step importance matrix
            importance_matrix, chunk_texts = compute_step_importance_matrix(
                problem_dir=problem_dir,
                embedding_model=model,
                similarity_threshold=similarity_threshold,
                max_chunks=max_chunks,
                use_cache=use_cache,
                n_processes=n_processes,
                output_dir=problem_output_dir  # Pass the output directory
            )
            
            # Save to cache
            np.savez(
                cache_file,
                matrix=importance_matrix,
                chunk_texts=np.array(chunk_texts, dtype=object)
            )
        
        # Create axis labels with function tag prefixes
        x_labels = []
        y_labels = []
        for i, chunk in enumerate(chunk_texts):
            tag_prefix = get_function_tag_prefix(chunk)
            label = f"{i}-{tag_prefix}" if tag_prefix else f"{i}"
            x_labels.append(label)
            y_labels.append(label)

        # Plot importance matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            importance_matrix.T,  # Transpose the matrix
            cmap="viridis",
            xticklabels=x_labels,
            yticklabels=y_labels,
            cbar_kws={"label": "Importance Score"}
        )
        plt.title(f"Step Importance Matrix for Problem {problem_idx}")
        plt.xlabel("Source Step (i)")
        plt.ylabel("Target Step (j)")
        plt.tight_layout()
        plt.savefig(plots_dir / f"step_importance_matrix.png", dpi=300)
        plt.close()
        
        # Plot outgoing importance for each step
        outgoing_importance = np.zeros(len(chunk_texts))
        for i in range(len(chunk_texts)):
            # Average importance of step i on all future steps
            future_steps = importance_matrix[i, i+1:]
            outgoing_importance[i] = np.mean(future_steps) if len(future_steps) > 0 else 0
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(chunk_texts)), outgoing_importance)
        plt.title(f"Average Outgoing Importance for Problem {problem_idx}")
        plt.xlabel("Step Index")
        plt.ylabel("Average Importance on Future Steps")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / f"outgoing_importance.png", dpi=300)
        plt.close()
        
        # Plot incoming importance for each step
        incoming_importance = np.zeros(len(chunk_texts))
        for j in range(len(chunk_texts)):
            # Average importance of all previous steps on step j
            previous_steps = importance_matrix[:j, j]
            incoming_importance[j] = np.mean(previous_steps) if len(previous_steps) > 0 else 0
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(chunk_texts)), incoming_importance)
        plt.title(f"Average Incoming Importance for Problem {problem_idx}")
        plt.xlabel("Step Index")
        plt.ylabel("Average Importance from Previous Steps")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / f"incoming_importance.png", dpi=300)
        plt.close()
        
        # Save chunk texts
        with open(problem_output_dir / "chunk_texts.json", 'w', encoding='utf-8') as f:
            json.dump(chunk_texts, f, indent=2)
        
        # Save importance scores in JSON format for easier access
        importance_data = []
        for i in range(len(chunk_texts)):
            # For each source step, create a list of its impacts on target steps
            target_impacts = []
            for j in range(len(chunk_texts)):
                if j > i:  # We only have importance scores for j > i
                    target_impacts.append({
                        "target_chunk_idx": j,
                        "importance_score": float(importance_matrix[i, j])
                    })
            
            # Create entry for this source step
            step_data = {
                "source_chunk_idx": i,
                "source_chunk_text": chunk_texts[i].get('chunk', str(chunk_texts[i])) if isinstance(chunk_texts[i], dict) else str(chunk_texts[i]),
                "target_impacts": target_impacts
            }
            importance_data.append(step_data)

        # Save to JSON file
        with open(problem_output_dir / "step_importance.json", 'w', encoding='utf-8') as f:
            json.dump(importance_data, f, indent=2)
        
        # Generate summary statistics
        summary = {
            "problem_idx": problem_idx,
            "num_chunks": len(chunk_texts),
            "avg_importance": float(np.mean(importance_matrix)),
            "max_importance": float(np.max(importance_matrix)),
            "min_importance": float(np.min(importance_matrix)),
        }
        
        # Find most influential steps (highest average outgoing importance)
        top_influential_indices = np.argsort(outgoing_importance)[::-1][:5]
        summary["top_influential_steps"] = [
            {
                "step_idx": int(idx),
                "step_text": chunk_texts[idx],
                "avg_outgoing_importance": float(outgoing_importance[idx])
            }
            for idx in top_influential_indices if outgoing_importance[idx] > 0
        ]
        
        # Find most dependent steps (highest average incoming importance)
        top_dependent_indices = np.argsort(incoming_importance)[::-1][:5]
        summary["top_dependent_steps"] = [
            {
                "step_idx": int(idx),
                "step_text": chunk_texts[idx],
                "avg_incoming_importance": float(incoming_importance[idx])
            }
            for idx in top_dependent_indices if incoming_importance[idx] > 0
        ]
        
        # Save summary
        with open(problem_output_dir / "summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        # Free memory
        del importance_matrix
        gc.collect()
    
    print(f"Step attribution analysis complete. Results saved to {output_dir}")
    
def get_problem_dirs(analysis_dir: Path, correct_only: bool = True, limit: Optional[int] = None, include_problems: Optional[str] = None) -> List[Path]:
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
    print(f"Correct only: {correct_only}")
    subdir = "correct_base_solution" if correct_only else "incorrect_base_solution"
    print(f"Using {subdir} subdirectory")
    base_dir = analysis_dir / subdir
    
    if not base_dir.exists():
        print(f"Directory {base_dir} does not exist")
        return []
    
    # Get all problem directories
    all_problem_dirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("problem_")]
    
    if include_problems:
        all_problem_dirs = [d for d in all_problem_dirs if int(d.name.split('_')[-1]) in [int(p) for p in include_problems.split(',')]]
    
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

def main():
    parser = argparse.ArgumentParser(description="Analyze step-to-step attribution in chain-of-thought reasoning")
    parser.add_argument("-ad", "--analysis_dir", type=str, default="math_rollouts/deepseek-r1-distill-qwen-14b/temperature_0.6_top_p_0.95", help="Directory containing analysis results")
    parser.add_argument("-od", "--output_dir", type=str, default="analysis/step_attribution", help="Directory to save analysis results")
    parser.add_argument("-st", "--similarity_threshold", type=float, default=0.7, help="Threshold for considering steps similar")
    parser.add_argument("-mc", "--max_chunks", type=int, default=None, help="Maximum number of chunks to analyze")
    parser.add_argument("-mp", "--max_problems", type=int, default=None, help="Maximum number of problems to analyze")
    parser.add_argument("-ip", "--include_problems", type=str, default=None, help="Comma-separated list of problem indices to include")
    parser.add_argument("-co", "--correct_only", action="store_true", help="Only analyze correct solutions, if not specified, only incorrect solutions will be analyzed")
    parser.add_argument("-nc", "--no_cache", action="store_true", default=True, help="Don't use cached results")
    parser.add_argument("-np", "--n_processes", type=int, default=1, help="Number of processes to use for parallel computation")
    args = parser.parse_args()
    
    # Set up directories
    analysis_dir = Path(args.analysis_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Get problem directories
    problem_dirs = get_problem_dirs(analysis_dir, correct_only=args.correct_only, limit=args.max_problems, include_problems=args.include_problems)
    
    if not problem_dirs:
        print(f"No problem directories found in {analysis_dir}")
        return
    
    print(f"Found {len(problem_dirs)} problem directories")
    
    # Analyze step attribution
    analyze_step_attribution(
        problem_dirs=problem_dirs,
        output_dir=output_dir,
        similarity_threshold=args.similarity_threshold,
        max_chunks=args.max_chunks,
        use_cache=not args.no_cache,
        n_processes=args.n_processes
    )

if __name__ == "__main__":
    main()