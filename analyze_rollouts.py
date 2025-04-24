import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from openai import OpenAI
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from prompts import ss_categories, DAG_PROMPT

# Load environment variables
load_dotenv()

# Set up OpenAI API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if not client.api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")

# Initialize the r1-distill-qwen-14b tokenizer
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")

def count_tokens(text: str) -> int:
    """Count the number of tokens in a text string using the r1-distill-qwen-14b tokenizer."""
    return len(tokenizer.encode(text))

def label_chunk(problem_text: str, chunks: List[str], chunk_idx: int) -> Dict:
    """
    Label a chunk using OpenAI API with the DAG prompt.
    
    Args:
        problem_text: The problem text
        chunks: All chunks for context
        chunk_idx: The index of the chunk to label
        
    Returns:
        Dictionary with the label information
    """
    # Create the full chunked text with indices
    full_chunked_text = ""
    for i, chunk in enumerate(chunks):
        full_chunked_text += f"Chunk {i}:\n{chunk}\n\n"
    
    # Format the DAG prompt with the problem and chunks
    formatted_prompt = DAG_PROMPT.format(problem_text=problem_text, full_chunked_text=full_chunked_text)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": formatted_prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Add the chunk and its index to the result
        result["chunk"] = chunks[chunk_idx]
        result["chunk_idx"] = chunk_idx
        
        return result
    except Exception as e:
        print(f"Error labeling chunk {chunk_idx}: {e}")
        return {
            "categories": ["Unknown"],
            "explanation": f"Error: {str(e)}",
            "chunk": chunks[chunk_idx],
            "chunk_idx": chunk_idx
        }

def calculate_chunk_importance(chunk_dir: Path, next_chunk_dir: Path, use_absolute: bool = False) -> float:
    """
    Calculate the importance of a chunk by comparing accuracies.
    
    Args:
        chunk_dir: Directory for the current chunk
        next_chunk_dir: Directory for the next chunk
        use_absolute: Whether to use absolute value of accuracy difference
        
    Returns:
        Importance score (difference in accuracy)
    """
    # Load solutions for current chunk
    current_solutions_file = chunk_dir / "solutions.json"
    if not current_solutions_file.exists():
        return 0.0
    
    with open(current_solutions_file, 'r', encoding='utf-8') as f:
        current_solutions = json.load(f)
    
    # Load solutions for next chunk
    next_solutions_file = next_chunk_dir / "solutions.json"
    if not next_solutions_file.exists():
        return 0.0
    
    with open(next_solutions_file, 'r', encoding='utf-8') as f:
        next_solutions = json.load(f)
    
    # Calculate accuracy for current chunk
    current_correct = sum(1 for sol in current_solutions if sol.get("is_correct", False))
    current_accuracy = current_correct / len(current_solutions) if current_solutions else 0
    
    # Calculate accuracy for next chunk
    next_correct = sum(1 for sol in next_solutions if sol.get("is_correct", False))
    next_accuracy = next_correct / len(next_solutions) if next_solutions else 0
    
    # The importance is the difference in accuracy
    # NOTE: If removing the current chunk decreases accuracy more than removing the next chunk, then the current chunk is more important
    diff = next_accuracy - current_accuracy
    
    # Use absolute value if requested
    if use_absolute:
        return abs(diff)
    else:
        return diff

def analyze_problem(problem_dir: Path, problem_idx: int, use_absolute: bool = False, force_relabel: bool = False) -> Dict:
    """
    Analyze a single problem.
    
    Args:
        problem_dir: Directory containing the problem data
        problem_idx: Index of the problem
        use_absolute: Whether to use absolute value for importance calculation
        
    Returns:
        Dictionary with analysis results
    """
    # Check if required files exist
    base_solution_file = problem_dir / "base_solution.json"
    chunks_file = problem_dir / "chunks.json"
    problem_file = problem_dir / "problem.json"
    
    if not (base_solution_file.exists() and chunks_file.exists() and problem_file.exists()):
        print(f"Problem {problem_idx}: Missing required files")
        return None
    
    # Load problem
    with open(problem_file, 'r', encoding='utf-8') as f:
        problem = json.load(f)
    
    # Load base solution
    with open(base_solution_file, 'r', encoding='utf-8') as f:
        base_solution = json.load(f)
    
    # Load chunks
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)
    
    chunks = chunks_data["chunks"]
    
    # Check if at least 50% of chunks have corresponding chunk folders
    chunk_folders = [problem_dir / f"chunk_{i}" for i in range(len(chunks))]
    existing_chunk_folders = [folder for folder in chunk_folders if folder.exists()]
    
    if len(existing_chunk_folders) < len(chunks) * 0.5:
        print(f"Problem {problem_idx}: Only {len(existing_chunk_folders)}/{len(chunks)} chunk folders exist (less than 50%)")
        return None
    
    # Calculate token counts for each chunk's full_cot
    token_counts = []
    for chunk_idx in range(len(chunks)):
        chunk_dir = problem_dir / f"chunk_{chunk_idx}"
        solutions_file = chunk_dir / "solutions.json"
        
        if solutions_file.exists():
            with open(solutions_file, 'r', encoding='utf-8') as f:
                solutions = json.load(f)
                
            # Calculate average token count
            if solutions:
                avg_tokens = np.mean([count_tokens(sol.get("full_cot", "")) for sol in solutions])
                token_counts.append((chunk_idx, avg_tokens))
    
    labeled_chunks_file = problem_dir / "chunks_labeled.json"
    if labeled_chunks_file.exists() and not force_relabel:
        with open(labeled_chunks_file, 'r', encoding='utf-8') as f:
            labeled_chunks = json.load(f)
        
        return {
            "problem_idx": problem_idx,
            "problem_type": problem.get("type"),
            "problem_level": problem.get("level"),
            "base_accuracy": base_solution.get("is_correct", False),
            "num_chunks": len(chunks),
            "labeled_chunks": labeled_chunks,
            "token_counts": token_counts
        }
    
    # Label each chunk
    print(f"Problem {problem_idx}: Labeling {len(chunks)} chunks")
    
    # Use the DAG prompt to label all chunks at once
    try:
        labeled_chunks_result = label_chunk(problem["problem"], chunks, 0)
        
        # Process the result into the expected format
        labeled_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_data = { "chunk": chunk, "chunk_idx": i }
            
            # Extract function tags and dependencies for this chunk
            chunk_key = str(i)
            if chunk_key in labeled_chunks_result:
                chunk_mapping = labeled_chunks_result[chunk_key]
                chunk_data["function_tags"] = chunk_mapping.get("function_tags", ["unknown"])
                chunk_data["depends_on"] = chunk_mapping.get("depends_on", [])
            else:
                chunk_data["function_tags"] = ["unknown"]
                chunk_data["depends_on"] = []
            
            labeled_chunks.append(chunk_data)
    except Exception as e:
        print(f"Error using DAG prompt for problem {problem_idx}: {e}")
        return None
    
    # Calculate importance for each chunk
    print(f"Problem {problem_idx}: Calculating chunk importance")
    for i in range(len(chunks) - 1):
        chunk_dir = problem_dir / f"chunk_{i}"
        next_chunk_dir = problem_dir / f"chunk_{i+1}"
        
        if chunk_dir.exists() and next_chunk_dir.exists():
            importance = calculate_chunk_importance(chunk_dir, next_chunk_dir, use_absolute)
            labeled_chunks[i]["importance"] = importance
    
    # The last chunk doesn't have a next chunk for comparison
    if labeled_chunks:
        labeled_chunks[-1]["importance"] = 0.0
    
    # Save labeled chunks
    with open(labeled_chunks_file, 'w', encoding='utf-8') as f:
        json.dump(labeled_chunks, f, indent=2)
    
    print(f"Problem {problem_idx}: Saved labeled chunks to {labeled_chunks_file}")
    
    # Return analysis results
    return {
        "problem_idx": problem_idx,
        "problem_type": problem.get("type"),
        "problem_level": problem.get("level"),
        "base_accuracy": base_solution.get("is_correct", False),
        "num_chunks": len(chunks),
        "labeled_chunks": labeled_chunks,
        "token_counts": token_counts
    }

def generate_plots(results: List[Dict], output_dir: Path) -> pd.DataFrame:
    """
    Generate plots from the analysis results.
    
    Args:
        results: List of problem analysis results
        output_dir: Directory to save plots
        
    Returns:
        DataFrame with category importance rankings
    """
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    # Prepare data for plots
    all_chunks = []
    for result in results:
        if not result:
            continue
            
        problem_idx = result["problem_idx"]
        problem_type = result.get("problem_type", "Unknown")
        problem_level = result.get("problem_level", "Unknown")
        
        for chunk in result.get("labeled_chunks", []):
            chunk_data = {
                "problem_idx": problem_idx,
                "problem_type": problem_type,
                "problem_level": problem_level,
                "chunk_idx": chunk.get("chunk_idx"),
                "function_tags": chunk.get("function_tags", ["unknown"]),
                "importance": chunk.get("importance", 0.0),
                "chunk_length": len(chunk.get("chunk", ""))
            }
            all_chunks.append(chunk_data)
    
    # Convert to DataFrame
    df_chunks = pd.DataFrame(all_chunks)
    
    # Explode function_tags to have one row per tag
    df_exploded = df_chunks.explode("function_tags")
    
    # 1. Plot importance by function tag (category)
    plt.figure(figsize=(12, 8))
    sns.boxplot(x="function_tags", y="importance", data=df_exploded)
    plt.title("Chunk Importance by Category")
    plt.xlabel("Category")
    plt.ylabel("Importance (Accuracy Difference %)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(plots_dir / "importance_by_category.png")
    plt.close()
    
    # 2. Plot average importance by problem level
    plt.figure(figsize=(10, 6))
    level_importance = df_chunks.groupby("problem_level")["importance"].mean().reset_index()
    sns.barplot(x="problem_level", y="importance", data=level_importance)
    plt.title("Average Chunk Importance by Problem Level")
    plt.xlabel("Problem Level")
    plt.ylabel("Average Importance (Accuracy Difference %)")
    plt.tight_layout()
    plt.savefig(plots_dir / "importance_by_level.png")
    plt.close()
    
    # 3. Plot average importance by problem type
    plt.figure(figsize=(12, 8))
    type_importance = df_chunks.groupby("problem_type")["importance"].mean().reset_index()
    type_importance = type_importance.sort_values("importance", ascending=False)
    sns.barplot(x="problem_type", y="importance", data=type_importance)
    plt.title("Average Chunk Importance by Problem Type")
    plt.xlabel("Problem Type")
    plt.ylabel("Average Importance (Accuracy Difference %)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(plots_dir / "importance_by_type.png")
    plt.close()
    
    # 4. Plot token counts by chunk index
    token_data = []
    for result in results:
        if not result:
            continue
            
        problem_idx = result["problem_idx"]
        for chunk_idx, token_count in result.get("token_counts", []):
            token_data.append({
                "problem_idx": problem_idx,
                "chunk_idx": chunk_idx,
                "token_count": token_count
            })
    
    df_tokens = pd.DataFrame(token_data)
    
    # Get function tag for each chunk
    chunk_tags = {}
    for result in results:
        if not result:
            continue
            
        problem_idx = result["problem_idx"]
        for chunk in result.get("labeled_chunks", []):
            chunk_idx = chunk.get("chunk_idx")
            tags = chunk.get("function_tags", ["unknown"])
            if tags:
                chunk_tags[(problem_idx, chunk_idx)] = tags[0]
            else:
                chunk_tags[(problem_idx, chunk_idx)] = "unknown"
    
    # Add function tag to token data
    df_tokens["function_tag"] = df_tokens.apply(
        lambda row: chunk_tags.get((row["problem_idx"], row["chunk_idx"]), "unknown"), 
        axis=1
    )
    
    # Plot token counts by function tag (category)
    plt.figure(figsize=(12, 8))
    sns.boxplot(x="function_tag", y="token_count", data=df_tokens)
    plt.title("Token Count by Category")
    plt.xlabel("Category")
    plt.ylabel("Token Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(plots_dir / "token_count_by_category.png")
    plt.close()
    
    # 5. Plot distribution of function tags (categories)
    plt.figure(figsize=(12, 8))
    tag_counts = df_exploded["function_tags"].value_counts()
    sns.barplot(x=tag_counts.index, y=tag_counts.values)
    plt.title("Distribution of Categories")
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(plots_dir / "category_distribution.png")
    plt.close()
    
    # 6. Plot importance vs. chunk position
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="chunk_idx", y="importance", data=df_chunks)
    plt.title("Chunk Importance by Position")
    plt.xlabel("Chunk Index")
    plt.ylabel("Importance (Accuracy Difference %)")
    plt.tight_layout()
    plt.savefig(plots_dir / "importance_by_position.png")
    plt.close()
    
    # 7. Calculate and plot average importance by function tag (category) with error bars
    tag_importance = df_exploded.groupby("function_tags").agg({
        "importance": ["mean", "std", "count"]
    }).reset_index()
    tag_importance.columns = ["categories", "mean", "std", "count"]
    tag_importance = tag_importance.sort_values("mean", ascending=False)
    
    # Convert to percentages for display
    tag_importance["mean_pct"] = tag_importance["mean"] * 100
    tag_importance["std_pct"] = tag_importance["std"] * 100
    
    plt.figure(figsize=(12, 8))
    plt.errorbar(
        x=range(len(tag_importance)), 
        y=tag_importance["mean_pct"], 
        yerr=tag_importance["std_pct"], 
        fmt="o", 
        capsize=5
    )
    plt.xticks(range(len(tag_importance)), tag_importance["categories"], rotation=45, ha="right")
    plt.title("Average Importance by Category")
    plt.xlabel("Category")
    plt.ylabel("Average Importance (Accuracy Difference %)")
    plt.tight_layout()
    plt.savefig(plots_dir / "avg_importance_by_category.png")
    plt.close()
    
    print(f"Generated plots in {plots_dir}")
    
    # Return the category importance ranking
    return tag_importance

def main():
    parser = argparse.ArgumentParser(description='Analyze rollout data and label chunks')
    parser.add_argument('-i', '--rollouts_dir', type=str, default="math_rollouts/deepseek-r1-distill-qwen-14b/temperature_0.6", help='Directory containing rollout data')
    parser.add_argument('-o', '--output_dir', type=str, default="math_rollouts_analysis", help='Directory to save analysis results (defaults to rollouts_dir)')
    parser.add_argument('-p', '--problems', type=str, default=None, help='Comma-separated list of problem indices to analyze (default: all)')
    parser.add_argument('-m', '--max_problems', type=int, default=None, help='Maximum number of problems to analyze')
    parser.add_argument('-a', '--absolute', default=False, action='store_true', help='Use absolute value for importance calculation')
    parser.add_argument('-f', '--force_relabel', default=False, action='store_true', help='Force relabeling of chunks')
    args = parser.parse_args()
    
    # Set up directories
    rollouts_dir = Path(args.rollouts_dir)
    output_dir = Path(args.output_dir) if args.output_dir else rollouts_dir
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Get problem directories
    problem_dirs = sorted([d for d in rollouts_dir.iterdir() if d.is_dir() and d.name.startswith("problem_")])
    
    # Filter problems if specified
    if args.problems:
        problem_indices = [int(idx) for idx in args.problems.split(",")]
        problem_dirs = [d for d in problem_dirs if int(d.name.split("_")[1]) in problem_indices]
    
    # Limit number of problems if specified
    if args.max_problems:
        problem_dirs = problem_dirs[:args.max_problems]
    
    print(f"Found {len(problem_dirs)} problems to analyze")
    
    # Analyze each problem
    results = []
    for problem_dir in tqdm(problem_dirs, desc="Analyzing problems"):
        problem_idx = int(problem_dir.name.split("_")[1])
        result = analyze_problem(problem_dir, problem_idx, args.absolute, args.force_relabel)
        if result:
            results.append(result)
    
    # Generate plots
    category_importance = generate_plots(results, output_dir)
    
    # Print category importance ranking with percentages
    print("\nCategory Importance Ranking:")
    for idx, row in category_importance.iterrows():
        print(f"{idx+1}. {row['categories']}: {row['mean_pct']:.2f}% ± {row['std_pct']:.2f}% (n={int(row['count'])})")
    
    # Save overall results
    results_file = output_dir / "analysis_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        # Convert numpy values to Python native types for JSON serialization
        serializable_results = []
        for result in results:
            if not result:
                continue
                
            serializable_result = {}
            for key, value in result.items():
                if isinstance(value, np.ndarray):
                    serializable_result[key] = value.tolist()
                elif isinstance(value, np.integer):
                    serializable_result[key] = int(value)
                elif isinstance(value, np.floating):
                    serializable_result[key] = float(value)
                else:
                    serializable_result[key] = value
            serializable_results.append(serializable_result)
            
        json.dump(serializable_results, f, indent=2)
    
    print(f"Analysis complete. Results saved to {results_file}")

if __name__ == "__main__":
    main()