import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from openai import OpenAI
from dotenv import load_dotenv
from prompts import DAG_PROMPT
import re
from collections import Counter

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

# Load environment variables
load_dotenv()

# Set up OpenAI API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if not client.api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")

# Initialize the r1-distill-qwen-14b tokenizer
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-r1-distill-qwen-14b")

# Define stopwords to filter out
stopwords = {
    "the", "a", "an", "and", "or", "at", "from", "for", "with", "about", "into", "through", "above", "ve",
    "below", "under", "again", "further", "here", "there", "all", "most", "other", "some", "such", "to", "on",
    "only", "own", "too", "very", "will", "wasn", "weren", "wouldn", "this", "that", "these", "those", "of",
    "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "got",
    "does", "did", "doing", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "get", "in",
    "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "so",
    "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "whose", "as", "us",  
}

def count_tokens(text: str, approximate: bool = True) -> int:
    """Count the number of tokens in a text string using the r1-distill-qwen-14b tokenizer."""
    if approximate:
        return len(text) // 4
    else:
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

def analyze_problem(
    problem_dir: Path, 
    output_dir: Path, 
    use_absolute: bool = False,
    force_relabel: bool = False,
    rollout_type: str = "correct",
    forced_answer_dir: Optional[Path] = None
) -> Dict:
    """
    Analyze rollout data for a single problem.
    
    Args:
        problem_dir: Directory containing rollout data for a problem
        output_dir: Directory to save analysis results
        use_absolute: Whether to use absolute value for importance calculation
        force_relabel: Whether to force relabeling of chunks
        rollout_type: Type of rollouts ("correct" or "incorrect")
        forced_answer_dir: Directory containing forced answer rollouts
        
    Returns:
        Dictionary with analysis results
    """
    # Check if required files exist
    base_solution_file = problem_dir / "base_solution.json"
    chunks_file = problem_dir / "chunks.json"
    problem_file = problem_dir / "problem.json"
    
    if not (base_solution_file.exists() and chunks_file.exists() and problem_file.exists()):
        print(f"Problem {problem_dir.name}: Missing required files")
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
    
    # Filter out chunks shorter than 4 characters
    valid_chunks = []
    valid_chunk_indices = []
    for i, chunk in enumerate(chunks):
        if len(chunk) >= 4:
            valid_chunks.append(chunk)
            valid_chunk_indices.append(i)
    
    if len(valid_chunks) < len(chunks):
        print(f"Problem {problem_dir.name}: Filtered out {len(chunks) - len(valid_chunks)} chunks shorter than 3 characters")
        chunks = valid_chunks
    
    # Check if at least 25% of chunks have corresponding chunk folders
    chunk_folders = [problem_dir / f"chunk_{i}" for i in valid_chunk_indices]
    existing_chunk_folders = [folder for folder in chunk_folders if folder.exists()]
    
    if len(existing_chunk_folders) < 0.1 * len(chunks):
        print(f"Problem {problem_dir.name}: Only {len(existing_chunk_folders)}/{len(chunks)} chunk folders exist")
        return None
    
    # Calculate token counts for each chunk's full_cot
    token_counts = []
    
    # Pre-calculate accuracies for all chunks - do this once instead of repeatedly
    print(f"Problem {problem_dir.name}: Pre-calculating chunk accuracies")
    chunk_accuracies = {}
    
    for chunk_idx in valid_chunk_indices:
        chunk_dir = problem_dir / f"chunk_{chunk_idx}"
        solutions_file = chunk_dir / "solutions.json"
        
        if solutions_file.exists():
            with open(solutions_file, 'r', encoding='utf-8') as f:
                solutions = json.load(f)
                
            # Calculate accuracy
            correct = sum(1 for sol in solutions if sol.get("is_correct", False) is True)
            total = sum(1 for sol in solutions if sol.get("is_correct", None) is not None)
            
            if total > 0:
                chunk_accuracies[chunk_idx] = correct / total
            else:
                chunk_accuracies[chunk_idx] = 0.0
                
            # Calculate average token count
            if solutions:
                avg_tokens = np.mean([count_tokens(sol.get("full_cot", "")) for sol in solutions])
                token_counts.append((chunk_idx, avg_tokens))
    
    # Function to calculate importance using pre-calculated accuracies
    def calculate_importance(chunk_idx, use_absolute=False):
        if chunk_idx not in chunk_accuracies:
            return 0.0
        
        # Get accuracies of all other chunks
        current_accuracy = chunk_accuracies[chunk_idx]
        prev_accuracies = [acc for idx, acc in chunk_accuracies.items() if idx <= chunk_idx]
        next_accuracies = [acc for idx, acc in chunk_accuracies.items() if idx == chunk_idx + 1]
        
        if not prev_accuracies or not next_accuracies:
            return 0.0
        
        prev_avg_accuracy = sum(prev_accuracies) / len(prev_accuracies)
        next_avg_accuracy = sum(next_accuracies) / len(next_accuracies)
        
        # The importance is how much this chunk's accuracy differs from others
        # NOTE: We can also take next_avg_accuracy as idx > chunk_idx
        diff = next_avg_accuracy - current_accuracy
        # diff = diff if diff >= 0.05 else 0.0
        
        return abs(diff) if use_absolute else diff
    
    labeled_chunks_file = problem_dir / "chunks_labeled.json"
    
    # If labeled chunks exist and we're not forcing relabeling, load them
    if labeled_chunks_file.exists() and not force_relabel:
        with open(labeled_chunks_file, 'r', encoding='utf-8') as f:
            labeled_chunks = json.load(f)
        
        # Filter out chunks shorter than 3 characters
        labeled_chunks = [chunk for chunk in labeled_chunks if chunk.get("chunk_idx") in valid_chunk_indices]
        
        # Recalculate importance for each chunk using pre-calculated accuracies
        print(f"Problem {problem_dir.name}: Recalculating chunk importance")
        for chunk in labeled_chunks:
            chunk_idx = chunk.get("chunk_idx")
            chunk["importance"] = calculate_importance(chunk_idx, use_absolute)
            chunk["accuracy"] = chunk_accuracies[chunk_idx] if chunk_idx in chunk_accuracies else 0.0
        
        # Save updated labeled chunks
        with open(labeled_chunks_file, 'w', encoding='utf-8') as f:
            json.dump(labeled_chunks, f, indent=2)
            
        print(f"Problem {problem_dir.name}: Updated importance scores in {labeled_chunks_file}")
    else:
        # Label each chunk
        print(f"Problem {problem_dir.name}: Labeling {len(chunks)} chunks")
        
        # Use the DAG prompt to label all chunks at once
        try:
            labeled_chunks_result = label_chunk(problem["problem"], chunks, 0)
            
            # Process the result into the expected format
            labeled_chunks = []
            for i, chunk_idx in enumerate(valid_chunk_indices):
                chunk = chunks[i]  # Use the filtered chunks list
                chunk_data = {
                    "chunk": chunk,
                    "chunk_idx": chunk_idx
                }
                
                # Extract function tags and dependencies for this chunk
                chunk_key = str(i)
                if chunk_key in labeled_chunks_result:
                    chunk_mapping = labeled_chunks_result[chunk_key]
                    chunk_data["function_tags"] = chunk_mapping.get("function_tags", ["unknown"])
                    chunk_data["depends_on"] = chunk_mapping.get("depends_on", [])
                else:
                    chunk_data["function_tags"] = ["unknown"]
                    chunk_data["depends_on"] = []
                
                # Calculate importance using pre-calculated accuracies
                chunk_data["importance"] = calculate_importance(chunk_idx, use_absolute)
                
                labeled_chunks.append(chunk_data)
        except Exception as e:
            print(f"Error using DAG prompt for problem {problem_dir.name}: {e}")
            return None
        
        # Save labeled chunks
        with open(labeled_chunks_file, 'w', encoding='utf-8') as f:
            json.dump(labeled_chunks, f, indent=2)
        
        print(f"Problem {problem_dir.name}: Saved labeled chunks to {labeled_chunks_file}")
    
    # Load forced answer data if available
    forced_answer_accuracies = None
    if forced_answer_dir:
        forced_problem_dir = forced_answer_dir / problem_dir.name
        if forced_problem_dir.exists():
            forced_answer_accuracies = []
            
            # Iterate through chunk directories
            for chunk_idx in valid_chunk_indices:
                chunk_dir = forced_problem_dir / f"chunk_{chunk_idx}"
                
                if chunk_dir.exists():
                    # Load solutions.json file to calculate accuracy
                    solutions_file = chunk_dir / "solutions.json"
                    if solutions_file.exists():
                        try:
                            with open(solutions_file, 'r', encoding='utf-8') as f:
                                solutions = json.load(f)
                            
                            # Calculate accuracy from solutions
                            correct_count = sum(1 for sol in solutions if sol.get("is_correct", False) is True)
                            total_count = sum(1 for sol in solutions if sol.get("is_correct", None) is not None and sol.get("answer", "") != "")
                            
                            if total_count > 0:
                                accuracy = correct_count / total_count
                            else:
                                accuracy = 0.0
                                
                            forced_answer_accuracies.append(accuracy)
                            
                        except Exception as e:
                            print(f"Error loading solutions from {solutions_file}: {e}")
                            forced_answer_accuracies.append(0.0)
                    else:
                        forced_answer_accuracies.append(0.0)
                else:
                    forced_answer_accuracies.append(0.0)
    
    # Return analysis results
    return {
        "problem_idx": problem_dir.name.split("_")[1],
        "problem_type": problem.get("type"),
        "problem_level": problem.get("level"),
        "base_accuracy": base_solution.get("is_correct", False),
        "num_chunks": len(chunks),
        "labeled_chunks": labeled_chunks,
        "token_counts": token_counts,
        "forced_answer_accuracies": forced_answer_accuracies
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
            # Format function tags for better display
            raw_tags = chunk.get("function_tags", [])
            formatted_tags = []
            
            for tag in raw_tags:
                if tag.lower() == "unknown":
                    continue  # Skip unknown tags
                
                # Format tag for better display (e.g., "planning_step" -> "Planning Step")
                formatted_tag = " ".join(word.capitalize() for word in tag.split("_"))
                formatted_tags.append(formatted_tag)
            
            # If no valid tags after filtering, skip this chunk
            if not formatted_tags:
                continue
                
            chunk_data = {
                "problem_idx": problem_idx,
                "problem_type": problem_type,
                "problem_level": problem_level,
                "chunk_idx": chunk.get("chunk_idx"),
                "function_tags": formatted_tags,
                "importance": chunk.get("importance", 0.0),
                "chunk_length": len(chunk.get("chunk", ""))
            }
            all_chunks.append(chunk_data)
    
    # Convert to DataFrame
    df_chunks = pd.DataFrame(all_chunks)
    
    # Explode function_tags to have one row per tag
    df_exploded = df_chunks.explode("function_tags")
    
    # 1. Plot importance by function tag (category) using violin plot with means
    plt.figure(figsize=(12, 8))
    # Calculate mean importance for each category to sort by
    # Convert to percentage for display
    df_exploded['importance_pct'] = df_exploded['importance'] * 100
    category_means = df_exploded.groupby("function_tags", observed=True)["importance_pct"].mean().sort_values(ascending=False)
    # Reorder the data based on sorted categories
    df_exploded_sorted = df_exploded.copy()
    df_exploded_sorted["function_tags"] = pd.Categorical(
        df_exploded_sorted["function_tags"], 
        categories=category_means.index, 
        ordered=True
    )
    # Create the sorted violin plot
    ax = sns.violinplot(x="function_tags", y="importance_pct", data=df_exploded_sorted, inner="quartile", cut=0)

    # Add mean markers
    means = df_exploded_sorted.groupby("function_tags", observed=True)["importance_pct"].mean()
    for i, mean_val in enumerate(means[means.index]):
        ax.plot([i], [mean_val], 'o', color='red', markersize=8)

    # Add a legend for the mean marker
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Mean')]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.title("Chunk Importance by Category")
    plt.ylabel("Importance (Accuracy Difference %)")
    plt.xlabel(None)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(plots_dir / "importance_by_category.png")
    plt.close()
    
    # 2. Plot average importance by problem level
    plt.figure(figsize=(10, 6))
    level_importance = df_chunks.groupby("problem_level")["importance"].mean().reset_index()
    sns.barplot(x="problem_level", y="importance", data=level_importance)
    plt.title("Average Chunk Importance by Problem Level")
    plt.xlabel(None)
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
    plt.xlabel(None)
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
            raw_tags = chunk.get("function_tags", [])
            
            # Format and filter tags
            formatted_tags = []
            for tag in raw_tags:
                if tag.lower() == "unknown":
                    continue
                formatted_tag = " ".join(word.capitalize() for word in tag.split("_"))
                formatted_tags.append(formatted_tag)
            
            if formatted_tags:
                chunk_tags[(problem_idx, chunk_idx)] = formatted_tags[0]
    
    # Add function tag to token data
    df_tokens["function_tag"] = df_tokens.apply(
        lambda row: chunk_tags.get((row["problem_idx"], row["chunk_idx"]), "Other"), 
        axis=1
    )
    
    # Remove rows with no valid function tag
    df_tokens = df_tokens[df_tokens["function_tag"] != "Other"]
    
    if not df_tokens.empty:
        # Plot token counts by function tag (category)
        plt.figure(figsize=(12, 8))
        sns.boxplot(x="function_tag", y="token_count", data=df_tokens)
        plt.title("Token Count by Category")
        plt.ylabel("Token Count")
        plt.xlabel(None)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(plots_dir / "token_count_by_category.png")
        plt.close()
    
    # 5. Plot distribution of function tags (categories)
    plt.figure(figsize=(12, 8))
    tag_counts = df_exploded["function_tags"].value_counts()
    sns.barplot(x=tag_counts.index, y=tag_counts.values)
    plt.title("Distribution of Categories")
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
    tag_importance = df_exploded.groupby("function_tags").agg({"importance": ["mean", "std", "count"]}).reset_index()
    tag_importance.columns = ["categories", "mean", "std", "count"]
    tag_importance = tag_importance.sort_values("mean", ascending=False)
    
    # Convert to percentages for display
    tag_importance["mean_pct"] = tag_importance["mean"] * 100
    tag_importance["std_pct"] = tag_importance["std"] * 100
    
    # Calculate standard error (std/sqrt(n)) instead of using raw standard deviation
    tag_importance["se_pct"] = tag_importance["std_pct"] / np.sqrt(tag_importance["count"])
    
    plt.figure(figsize=(12, 8))
    plt.errorbar(
        x=range(len(tag_importance)), 
        y=tag_importance["mean_pct"], 
        yerr=tag_importance["se_pct"],
        fmt="o", 
        capsize=5
    )
    plt.xticks(range(len(tag_importance)), tag_importance["categories"], rotation=45, ha="right")
    plt.title("Average Importance by Category")
    plt.xlabel(None)
    plt.ylabel("Average Importance (Accuracy Difference %)")
    plt.tight_layout()
    plt.savefig(plots_dir / "avg_importance_by_category.png")
    plt.close()
    
    print(f"Generated plots in {plots_dir}")
        
    # Add the new analysis of top steps by category
    for top_n in [1, 3, 5, 10, 20, 30]:
        analyze_top_steps_by_category(results, output_dir, top_n=top_n, use_abs=False)
        
    # Add the new analysis of steps with high z-score by category
    for z_threshold in [1.5, 2, 2.5, 3]:
        analyze_high_zscore_steps_by_category(results, output_dir, z_threshold=z_threshold, use_abs=False)
    
    # Return the category importance ranking
    return tag_importance

def analyze_chunk_variance(results: List[Dict], output_dir: Path) -> None:
    """
    Analyze variance in chunk importance within individual problems to identify
    potential "fork reasoning steps" where the model's behavior diverges significantly.
    
    Args:
        results: List of problem analysis results
        output_dir: Directory to save analysis results
    """
    print("Analyzing chunk variance within problems to identify potential reasoning forks...")
    
    variance_dir = output_dir / "variance_analysis"
    variance_dir.mkdir(exist_ok=True, parents=True)
    
    # Collect all chunks across problems
    all_chunks = []
    problem_chunks = {}
    
    for result in results:
        if not result:
            continue
            
        problem_idx = result["problem_idx"]
        problem_chunks[problem_idx] = []
        
        for chunk in result.get("labeled_chunks", []):
            chunk_data = {
                "problem_idx": problem_idx,
                "chunk_idx": chunk.get("chunk_idx"),
                "chunk_text": chunk.get("chunk", ""),
                "function_tags": chunk.get("function_tags", []),
                "importance": chunk.get("importance", 0.0)
            }
            all_chunks.append(chunk_data)
            problem_chunks[problem_idx].append(chunk_data)
    
    # Calculate variance in importance within each problem
    problem_variances = {}
    high_variance_problems = []
    
    for problem_idx, chunks in problem_chunks.items():
        if len(chunks) < 3:  # Need at least 3 chunks for meaningful variance
            continue
            
        # Calculate importance variance
        importance_values = [chunk["importance"] for chunk in chunks]
        variance = np.var(importance_values)
        
        problem_variances[problem_idx] = {
            "variance": variance,
            "chunks": chunks,
            "importance_values": importance_values
        }
        
        # Track problems with high variance
        high_variance_problems.append((problem_idx, variance))
    
    # Sort problems by variance
    high_variance_problems.sort(key=lambda x: x[1], reverse=True)
    
    # Save results
    with open(variance_dir / "chunk_variance.txt", 'w', encoding='utf-8') as f:
        f.write("Problems with highest variance in chunk importance (potential reasoning forks):\n\n")
        
        for problem_idx, variance in high_variance_problems[:20]:  # Top 20 problems
            f.write(f"Problem {problem_idx}: Variance = {variance:.6f}\n")
            
            # Get chunks for this problem
            chunks = problem_chunks[problem_idx]
            
            # Sort chunks by importance
            sorted_chunks = sorted(chunks, key=lambda x: x["importance"], reverse=True)
            
            # Write chunk information
            f.write("  Chunks by importance:\n")
            for i, chunk in enumerate(sorted_chunks):
                chunk_idx = chunk["chunk_idx"]
                importance = chunk["importance"]
                tags = ", ".join(chunk["function_tags"])
                
                # Truncate chunk text for display
                chunk_text = chunk["chunk_text"]
                if len(chunk_text) > 50:
                    chunk_text = chunk_text[:47] + "..."
                
                f.write(f"    {i+1}. Chunk {chunk_idx}: {importance:.4f} - {tags} - '{chunk_text}'\n")
            
            # Identify potential reasoning forks (clusters of important chunks)
            f.write("  Potential reasoning forks:\n")
            
            # Sort chunks by index to maintain sequence
            sequence_chunks = sorted(chunks, key=lambda x: x["chunk_idx"])
            
            # Find clusters of important chunks
            clusters = []
            current_cluster = []
            avg_importance = np.mean([c["importance"] for c in chunks])
            
            for chunk in sequence_chunks:
                if chunk["importance"] > avg_importance:
                    if not current_cluster or chunk["chunk_idx"] - current_cluster[-1]["chunk_idx"] <= 2:
                        current_cluster.append(chunk)
                    else:
                        if len(current_cluster) >= 2:  # At least 2 chunks in a cluster
                            clusters.append(current_cluster)
                        current_cluster = [chunk]
            
            if len(current_cluster) >= 2:
                clusters.append(current_cluster)
            
            # Write clusters
            for i, cluster in enumerate(clusters):
                start_idx = cluster[0]["chunk_idx"]
                end_idx = cluster[-1]["chunk_idx"]
                f.write(f"    Fork {i+1}: Chunks {start_idx}-{end_idx}\n")
                
                # Combine chunk text
                combined_text = " ".join([c["chunk_text"] for c in cluster])
                if len(combined_text) > 100:
                    combined_text = combined_text[:97] + "..."
                
                f.write(f"      Text: '{combined_text}'\n")
                
                # List tags
                all_tags = set()
                for chunk in cluster:
                    all_tags.update(chunk["function_tags"])
                f.write(f"      Tags: {', '.join(all_tags)}\n")
            
            f.write("\n")
    
    # Create visualization of variance distribution
    plt.figure(figsize=(12, 8))
    variances = [v for _, v in high_variance_problems]
    plt.hist(variances, bins=20)
    plt.xlabel("Variance in Chunk Importance")
    plt.ylabel("Number of Problems")
    plt.title("Distribution of Variance in Chunk Importance Across Problems")
    plt.tight_layout()
    plt.savefig(variance_dir / "chunk_variance_distribution.png")
    plt.close()
    
    # Create visualization of top high-variance problems
    plt.figure(figsize=(15, 10))
    top_problems = high_variance_problems[:20]
    problem_ids = [str(p[0]) for p in top_problems]
    problem_variances = [p[1] for p in top_problems]
    
    plt.bar(range(len(problem_ids)), problem_variances)
    plt.xticks(range(len(problem_ids)), problem_ids, rotation=45)
    plt.xlabel("Problem ID")
    plt.ylabel("Variance in Chunk Importance")
    plt.title("Top 20 Problems with Highest Variance in Chunk Importance")
    plt.tight_layout()
    plt.savefig(variance_dir / "top_variance_problems.png")
    plt.close()
    
    print(f"Chunk variance analysis saved to {variance_dir}")

def analyze_function_tag_variance(results: List[Dict], output_dir: Path) -> None:
    """
    Analyze variance in importance across different function tags to identify
    which types of reasoning steps show the most variability.
    
    Args:
        results: List of problem analysis results
        output_dir: Directory to save analysis results
    """
    print("Analyzing variance in importance across function tags...")
    
    variance_dir = output_dir / "variance_analysis"
    variance_dir.mkdir(exist_ok=True, parents=True)
    
    # Collect chunks by function tag
    tag_chunks = {}
    
    for result in results:
        if not result:
            continue
            
        for chunk in result.get("labeled_chunks", []):
            for tag in chunk.get("function_tags", []):
                if tag.lower() == "unknown":
                    continue
                    
                # Format tag for better display
                formatted_tag = " ".join(word.capitalize() for word in tag.split("_"))
                
                if formatted_tag not in tag_chunks:
                    tag_chunks[formatted_tag] = []
                
                tag_chunks[formatted_tag].append({
                    "problem_idx": result["problem_idx"],
                    "chunk_idx": chunk.get("chunk_idx"),
                    "importance": chunk.get("importance", 0.0)
                })
    
    # Calculate variance for each tag
    tag_variances = {}
    
    for tag, chunks in tag_chunks.items():
        if len(chunks) < 5:  # Need at least 5 chunks for meaningful variance
            continue
            
        importance_values = [chunk["importance"] for chunk in chunks]
        variance = np.var(importance_values)
        mean = np.mean(importance_values)
        count = len(chunks)
        
        tag_variances[tag] = {
            "variance": variance,
            "mean": mean,
            "count": count,
            "coefficient_of_variation": variance / mean if mean != 0 else 0
        }
    
    # Sort tags by variance
    sorted_tags = sorted(tag_variances.items(), key=lambda x: x[1]["variance"], reverse=True)
    
    # Save results
    with open(variance_dir / "function_tag_variance.txt", 'w', encoding='utf-8') as f:
        f.write("Function tags with highest variance in importance:\n\n")
        
        for tag, stats in sorted_tags:
            variance = stats["variance"]
            mean = stats["mean"]
            count = stats["count"]
            cv = stats["coefficient_of_variation"]
            
            f.write(f"{tag}:\n")
            f.write(f"  Variance: {variance:.6f}\n")
            f.write(f"  Mean: {mean:.6f}\n")
            f.write(f"  Count: {count}\n")
            f.write(f"  Coefficient of Variation: {cv:.6f}\n")
            f.write("\n")
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Plot top 15 tags by variance
    top_tags = sorted_tags[:15]
    tags = [t[0] for t in top_tags]
    variances = [t[1]["variance"] for t in top_tags]
    
    plt.bar(range(len(tags)), variances)
    plt.xticks(range(len(tags)), tags, rotation=45, ha="right")
    plt.xlabel("Function Tag")
    plt.ylabel("Variance in Importance")
    plt.title("Top 15 Function Tags by Variance in Importance")
    plt.tight_layout()
    plt.savefig(variance_dir / "function_tag_variance.png")
    plt.close()
    
    # Plot coefficient of variation (normalized variance)
    plt.figure(figsize=(15, 10))
    
    # Sort by coefficient of variation
    sorted_by_cv = sorted(tag_variances.items(), key=lambda x: x[1]["coefficient_of_variation"], reverse=True)
    top_cv_tags = sorted_by_cv[:15]
    
    cv_tags = [t[0] for t in top_cv_tags]
    cvs = [t[1]["coefficient_of_variation"] for t in top_cv_tags]
    
    plt.bar(range(len(cv_tags)), cvs)
    plt.xticks(range(len(cv_tags)), cv_tags, rotation=45, ha="right")
    plt.xlabel("Function Tag")
    plt.ylabel("Coefficient of Variation (σ/μ)")
    plt.title("Top 15 Function Tags by Coefficient of Variation in Importance")
    plt.tight_layout()
    plt.savefig(variance_dir / "function_tag_cv.png")
    plt.close()
    
    print(f"Function tag variance analysis saved to {variance_dir}")

def analyze_within_problem_variance(results: List[Dict], output_dir: Path) -> None:
    """
    Analyze variance in chunk importance within individual problems to identify
    potential "fork reasoning steps" where the model's behavior diverges significantly.
    
    Args:
        results: List of problem analysis results
        output_dir: Directory to save analysis results
    """
    print("Analyzing within-problem variance to identify potential reasoning forks...")
    
    variance_dir = output_dir / "variance_analysis"
    variance_dir.mkdir(exist_ok=True, parents=True)
    
    # Analyze each problem for high-variance chunks (potential reasoning forks)
    high_variance_problems = []
    
    for result in results:
        if not result:
            continue
            
        problem_idx = result["problem_idx"]
        chunks = result.get("labeled_chunks", [])
        
        if len(chunks) < 3:  # Need at least 3 chunks for meaningful analysis
            continue
        
        # Calculate importance values and their variance
        importance_values = [chunk.get("importance", 0.0) for chunk in chunks]
        mean_importance = np.mean(importance_values)
        variance = np.var(importance_values)
        
        # Identify chunks with significantly higher or lower importance than average
        # These could represent "fork reasoning steps"
        potential_forks = []
        
        for i, chunk in enumerate(chunks):
            importance = chunk.get("importance", 0.0)
            z_score = (importance - mean_importance) / (np.std(importance_values) if np.std(importance_values) > 0 else 1)
            
            # Consider chunks with importance significantly different from mean as potential forks
            if abs(z_score) > 1.5:  # Threshold can be adjusted
                potential_forks.append({
                    "chunk_idx": chunk.get("chunk_idx"),
                    "chunk_text": chunk.get("chunk", ""),
                    "importance": importance,
                    "z_score": z_score,
                    "function_tags": chunk.get("function_tags", [])
                })
        
        if potential_forks:
            high_variance_problems.append({
                "problem_idx": problem_idx,
                "variance": variance,
                "mean_importance": mean_importance,
                "potential_forks": potential_forks
            })
    
    # Sort problems by variance
    high_variance_problems.sort(key=lambda x: x["variance"], reverse=True)
    
    # Save results
    with open(variance_dir / "within_problem_variance.txt", 'w', encoding='utf-8') as f:
        f.write("Problems with high variance in chunk importance (potential reasoning forks):\n\n")
        
        for problem in high_variance_problems:
            problem_idx = problem["problem_idx"]
            variance = problem["variance"]
            mean_importance = problem["mean_importance"]
            
            f.write(f"Problem {problem_idx}:\n")
            f.write(f"  Overall variance: {variance:.6f}\n")
            f.write(f"  Mean importance: {mean_importance:.6f}\n")
            f.write("  Potential fork reasoning steps:\n")
            
            # Sort potential forks by absolute z-score
            sorted_forks = sorted(problem["potential_forks"], key=lambda x: abs(x["z_score"]), reverse=True)
            
            for fork in sorted_forks:
                chunk_idx = fork["chunk_idx"]
                importance = fork["importance"]
                z_score = fork["z_score"]
                tags = ", ".join(fork["function_tags"]) if fork["function_tags"] else "No tags"
                
                f.write(f"    Chunk {chunk_idx}:\n")
                f.write(f"      Importance: {importance:.6f} (z-score: {z_score:.2f})\n")
                f.write(f"      Function tags: {tags}\n")
                f.write(f"      Text: {fork['chunk_text'][:100]}{'...' if len(fork['chunk_text']) > 100 else ''}\n\n")
    
    # Create visualization of fork distribution
    if high_variance_problems:
        plt.figure(figsize=(12, 8))
        
        # Collect data for visualization
        problem_indices = [p["problem_idx"] for p in high_variance_problems[:15]]  # Top 15 problems
        variances = [p["variance"] for p in high_variance_problems[:15]]
        fork_counts = [len(p["potential_forks"]) for p in high_variance_problems[:15]]
        
        # Create bar chart
        plt.bar(range(len(problem_indices)), variances)
        
        # Add fork count as text on bars
        for i, count in enumerate(fork_counts):
            plt.text(i, variances[i] * 0.5, f"{count} forks", ha='center', color='white', fontweight='bold')
        
        plt.xticks(range(len(problem_indices)), [f"Problem {idx}" for idx in problem_indices], rotation=45, ha="right")
        plt.xlabel("Problem")
        plt.ylabel("Variance in Chunk Importance")
        plt.title("Problems with Highest Variance in Chunk Importance (Potential Reasoning Forks)")
        plt.tight_layout()
        plt.savefig(variance_dir / "within_problem_variance.png")
        plt.close()
    
    print(f"Within-problem variance analysis saved to {variance_dir}")

def plot_chunk_accuracy_by_position(results: List[Dict], output_dir: Path, rollout_type: str = "correct", max_chunks_to_show: Optional[int] = None) -> None:
    """
    Plot chunk accuracy by position for all processed problems with focus on early chunks.
    
    Args:
        results: List of problem analysis results
        output_dir: Directory to save the plot
        rollout_type: Type of rollouts ("correct" or "incorrect")
        max_chunks_to_show: Maximum number of chunks to show in plots
    """
    print("Plotting chunk accuracy by position...")
    
    # Create explore directory
    explore_dir = output_dir / "explore"
    explore_dir.mkdir(exist_ok=True, parents=True)
    
    # Create problems directory for individual plots
    problems_dir = explore_dir / "problems"
    problems_dir.mkdir(exist_ok=True, parents=True)
    
    # Collect data for all chunks across problems
    chunk_data = []
    forced_chunk_data = []  # New list for forced answer data
    
    for result in results:
        if not result:
            continue
            
        problem_idx = result["problem_idx"]
        
        # Get the solutions for each chunk
        for chunk in result.get("labeled_chunks", []):
            chunk_idx = chunk.get("chunk_idx")
            
            # Only include the first 100 chunks
            if max_chunks_to_show is not None and chunk_idx > max_chunks_to_show:
                continue
                
            # Get the solutions for this chunk
            accuracy = chunk.get("accuracy", 0.0)
            
            # Get the first function tag if available
            function_tags = chunk.get("function_tags", [])
            first_tag = ""
            if function_tags and isinstance(function_tags, list) and len(function_tags) > 0:
                # Get first tag and convert to initials
                tag = function_tags[0]
                if isinstance(tag, str):
                    # Convert tag like "planning_step" to "PS"
                    words = tag.split('_')
                    first_tag = ''.join(word[0].upper() for word in words if word)
            
            chunk_data.append({
                "problem_idx": problem_idx,
                "chunk_idx": chunk_idx,
                "accuracy": accuracy,
                "tag": first_tag
            })
            
            # Add forced answer accuracy if available
            if "forced_answer_accuracies" in result and result["forced_answer_accuracies"] is not None and len(result["forced_answer_accuracies"]) > chunk_idx:
                forced_accuracy = result["forced_answer_accuracies"][chunk_idx]
                forced_chunk_data.append({
                    "problem_idx": problem_idx,
                    "chunk_idx": chunk_idx,
                    "accuracy": forced_accuracy,
                    "tag": first_tag
                })
    
    if not chunk_data:
        print("No chunk data available for plotting.")
        return
    
    # Convert to DataFrame
    df_chunks = pd.DataFrame(chunk_data)
    df_forced = pd.DataFrame(forced_chunk_data) if forced_chunk_data else None
    
    # Get unique problem indices
    problem_indices = df_chunks["problem_idx"].unique()
    
    # Create a colormap for the problems (other options: plasma, inferno, magma, cividis)
    import matplotlib.cm as cm
    colors = cm.viridis(np.linspace(0, 0.75, len(problem_indices)))
    color_map = dict(zip(sorted(problem_indices), colors))
    
    # Create a single plot focusing on first 100 chunks
    plt.figure(figsize=(15, 10))
    
    # Plot each problem with a unique color
    for problem_idx in problem_indices:
        problem_data = df_chunks[df_chunks["problem_idx"] == problem_idx]
        
        # Sort by chunk index
        problem_data = problem_data.sort_values("chunk_idx")
        
        # Convert to numpy arrays for plotting to avoid pandas indexing issues
        chunk_indices = problem_data["chunk_idx"].to_numpy()
        accuracies = problem_data["accuracy"].to_numpy()
        tags = problem_data["tag"].to_numpy()
        
        # Plot with clear label
        line = plt.plot(
            chunk_indices,
            accuracies,
            marker='o',
            linestyle='-',
            color=color_map[problem_idx],
            alpha=0.7,
            label=f"Problem {problem_idx}"
        )[0]
        
        # Identify accuracy extrema (both minima and maxima)
        # Convert to numpy arrays for easier manipulation
        for i in range(1, len(chunk_indices) - 1):
            # For correct rollouts, annotate minima (lower than both neighbors)
            is_minimum = accuracies[i] < accuracies[i-1] and accuracies[i] < accuracies[i+1]
            # For all rollouts, annotate maxima (higher than both neighbors)
            is_maximum = accuracies[i] > accuracies[i-1] and accuracies[i] > accuracies[i+1]
                
            if tags[i]:  # Only add if there's a tag
                if (rollout_type == "correct" and is_minimum) or is_maximum:
                    # Position below for minima, above for maxima
                    y_offset = -15 if (rollout_type == "correct" and is_minimum) else 7.5
                    
                    plt.annotate(
                        tags[i],
                        (chunk_indices[i], accuracies[i]),
                        textcoords="offset points",
                        xytext=(0, y_offset),
                        ha='center',
                        fontsize=8,
                        color=line.get_color(),
                        alpha=0.9,
                        weight='bold'
                    )
        
        # Plot forced answer accuracy if available
        if df_forced is not None and False: # NOTE: We're disabling this for now because it looks too packed
            forced_problem_data = df_forced[df_forced["problem_idx"] == problem_idx]
            if not forced_problem_data.empty:
                forced_problem_data = forced_problem_data.sort_values("chunk_idx")
                plt.plot(
                    forced_problem_data["chunk_idx"],
                    forced_problem_data["accuracy"],
                    marker='x',
                    linestyle='--',
                    color=color_map[problem_idx],
                    alpha=0.5,
                    label=f"Problem {problem_idx} Forced Answer"
                )
    
    # Calculate and plot the average across all problems
    avg_by_chunk = df_chunks.groupby("chunk_idx")["accuracy"].agg(['mean']).reset_index()
    
    avg_by_chunk_idx = avg_by_chunk["chunk_idx"].to_numpy()
    avg_by_chunk_mean = avg_by_chunk["mean"].to_numpy()
    
    # Plot average without error bars, in gray and thinner
    plt.plot(
        avg_by_chunk_idx,
        avg_by_chunk_mean,
        marker='.',
        markersize=4,
        linestyle='-',
        linewidth=1,
        color='gray',
        alpha=0.5,
        label="Average"
    )
    
    # Plot average for forced answer if available
    if df_forced is not None and False: # NOTE: We're disabling this for now because it looks too packed
        forced_avg_by_chunk = df_forced.groupby("chunk_idx")["accuracy"].agg(['mean']).reset_index()
        plt.plot(
            forced_avg_by_chunk["chunk_idx"],
            forced_avg_by_chunk["mean"],
            marker='.',
            markersize=4,
            linestyle='--',
            linewidth=1,
            color='black',
            alpha=0.5,
            label="Average Forced Answer"
        )
    
    # Add labels and title
    plt.xlabel("Sentence Index")
    plt.ylabel("Accuracy")
    plt.title("Sentence Accuracy by Position (First 100 Sentences)")
    
    # Set x-axis limits to focus on first 100 chunks
    plt.xlim(-3, 300 if max_chunks_to_show is None else max_chunks_to_show)
    
    # Set y-axis limits
    plt.ylim(-0.1, 1.1)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # If not too many problems, include all in the main legend
    plt.legend(loc='lower right' if rollout_type == "correct" else 'upper right', ncol=2)
    
    # Save the main plot
    plt.tight_layout()
    plt.savefig(explore_dir / "chunk_accuracy_by_position.png")
    plt.close()
    
    # Create individual plots for each problem
    print("Creating individual problem plots...")
    for problem_idx in problem_indices:
        problem_data = df_chunks[df_chunks["problem_idx"] == problem_idx]
        
        # Sort by chunk index
        problem_data = problem_data.sort_values("chunk_idx")
        
        if len(problem_data) == 0:
            continue
        
        # Create a new figure for this problem
        plt.figure(figsize=(10, 6))
        
        # Get the color for this problem
        color = color_map[problem_idx]
        
        problem_data_idx = problem_data["chunk_idx"].to_numpy()
        problem_data_accuracy = problem_data["accuracy"].to_numpy()
        
        # Plot the problem data
        line = plt.plot(
            problem_data_idx,
            problem_data_accuracy,
            marker='o',
            linestyle='-',
            color=color,
            label=f"Problem {problem_idx}"
        )[0]
        
        # Identify accuracy extrema (minima for correct, maxima for incorrect)
        # Convert to numpy arrays for easier manipulation
        chunk_indices = problem_data["chunk_idx"].values
        accuracies = problem_data["accuracy"].values
        tags = problem_data["tag"].values
        
        # Add function tag labels for both minima and maxima
        for i in range(1, len(chunk_indices) - 1):
            # For correct rollouts, annotate minima (lower than both neighbors)
            is_minimum = accuracies[i] < accuracies[i-1] and accuracies[i] < accuracies[i+1]
            # For all rollouts, annotate maxima (higher than both neighbors)
            is_maximum = accuracies[i] > accuracies[i-1] and accuracies[i] > accuracies[i+1]
                
            if tags[i]:  # Only add if there's a tag
                if (rollout_type == "correct" and is_minimum) or is_maximum:
                    # Position below for minima, above for maxima
                    y_offset = -15 if (rollout_type == "correct" and is_minimum) else 7.5
                    
                    plt.annotate(
                        tags[i],
                        (chunk_indices[i], accuracies[i]),
                        textcoords="offset points",
                        xytext=(0, y_offset),
                        ha='center',
                        fontsize=8,
                        color=color,
                        alpha=0.9,
                        weight='bold'
                    )
        
        # Plot forced answer accuracy if available
        if df_forced is not None:
            forced_problem_data = df_forced[df_forced["problem_idx"] == problem_idx]
            if not forced_problem_data.empty:
                forced_problem_data = forced_problem_data.sort_values("chunk_idx")
                forced_problem_data_idx = forced_problem_data["chunk_idx"].to_numpy()
                forced_problem_data_accuracy = forced_problem_data["accuracy"].to_numpy()
                
                plt.plot(
                    forced_problem_data_idx,
                    forced_problem_data_accuracy,
                    marker='.',
                    linestyle='--',
                    color=color,
                    alpha=0.7,
                    label=f"Forced Answer"
                )
        
        # Add labels and title
        plt.xlabel("Sentence Index")
        plt.ylabel("Accuracy")
        plt.title(f"Problem {problem_idx}: Sentence Accuracy by Position")
        
        # Set x-axis limits to focus on first 100 chunks
        plt.xlim(-3, 300 if max_chunks_to_show is None else max_chunks_to_show)
        
        # Set y-axis limits
        plt.ylim(-0.1, 1.1)
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        # Add legend
        plt.legend(loc='lower right')
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(problems_dir / f"problem_{problem_idx}_accuracy.png")
        plt.close()
    
    print(f"Chunk accuracy plots saved to {explore_dir}")

def process_rollouts(
    rollouts_dir: Path, 
    output_dir: Path, 
    problems: str = None, 
    max_problems: int = None, 
    absolute: bool = False, 
    force_relabel: bool = False,
    rollout_type: str = "correct",
    dag_dir: Optional[str] = None,
    forced_answer_dir: Optional[str] = None,
    get_token_frequencies: bool = False,
    max_chunks_to_show: int = 100
) -> None:
    """
    Process rollouts from a specific directory and save analysis results.
    
    Args:
        rollouts_dir: Directory containing rollout data
        output_dir: Directory to save analysis results
        problems: Comma-separated list of problem indices to analyze
        max_problems: Maximum number of problems to analyze
        absolute: Use absolute value for importance calculation
        force_relabel: Force relabeling of chunks
        rollout_type: Type of rollouts ("correct" or "incorrect")
        dag_dir: Directory containing DAG-improved chunks for token frequency analysis
        forced_answer_dir: Directory containing correct rollout data with forced answers
        get_token_frequencies: Whether to get token frequencies
        max_chunks_to_show: Maximum number of chunks to show in plots
    """
    # Get problem directories
    problem_dirs = sorted([d for d in rollouts_dir.iterdir() if d.is_dir() and d.name.startswith("problem_")])
    
    # Filter problems if specified
    if problems:
        problem_indices = [int(idx) for idx in problems.split(",")]
        problem_dirs = [d for d in problem_dirs if int(d.name.split("_")[1]) in problem_indices]
    
    # Limit number of problems if specified
    if max_problems:
        problem_dirs = problem_dirs[:max_problems]
    
    # Count problems with complete chunk folders
    total_problems = len(problem_dirs)
    problems_with_complete_chunks = 0
    problems_with_partial_chunks = 0
    problems_with_no_chunks = 0
    
    for problem_dir in problem_dirs:
        chunks_file = problem_dir / "chunks.json"
        if not chunks_file.exists():
            problems_with_no_chunks += 1
            continue
            
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
            
        chunks = chunks_data.get("chunks", [])
        if not chunks:
            problems_with_no_chunks += 1
            continue
            
        # Check if all chunk folders exist
        chunk_folders = [problem_dir / f"chunk_{i}" for i in range(len(chunks))]
        existing_chunk_folders = [folder for folder in chunk_folders if folder.exists()]
        
        if len(existing_chunk_folders) == len(chunks):
            problems_with_complete_chunks += 1
        elif len(existing_chunk_folders) > 0:
            problems_with_partial_chunks += 1
        else:
            problems_with_no_chunks += 1
    
    print(f"\n=== {rollout_type.capitalize()} Rollouts Summary ===")
    print(f"Total problems found: {total_problems}")
    print(f"Problems with complete chunk folders: {problems_with_complete_chunks} ({problems_with_complete_chunks/total_problems*100:.1f}%)")
    print(f"Problems with partial chunk folders: {problems_with_partial_chunks} ({problems_with_partial_chunks/total_problems*100:.1f}%)")
    print(f"Problems with no chunk folders: {problems_with_no_chunks} ({problems_with_no_chunks/total_problems*100:.1f}%)")
    
    # Only analyze problems with at least some chunk folders
    analyzable_problem_dirs = []
    for problem_dir in problem_dirs:
        chunks_file = problem_dir / "chunks.json"
        if not chunks_file.exists():
            continue
            
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
            
        chunks = chunks_data.get("chunks", [])
        if not chunks:
            continue
            
        # Check if at least one chunk folder exists
        chunk_folders = [problem_dir / f"chunk_{i}" for i in range(len(chunks))]
        existing_chunk_folders = [folder for folder in chunk_folders if folder.exists()]
        
        if existing_chunk_folders:
            analyzable_problem_dirs.append(problem_dir)
    
    print(f"Analyzing {len(analyzable_problem_dirs)} problems with at least some chunk folders")
    
    # Analyze each problem
    results = []
    for problem_dir in tqdm(analyzable_problem_dirs, desc=f"Analyzing {rollout_type} problems"):
        problem_idx = int(problem_dir.name.split("_")[1])
        result = analyze_problem(problem_dir, output_dir, absolute, force_relabel, rollout_type, forced_answer_dir)
        if result:
            results.append(result)
    
    # Generate plots
    category_importance = generate_plots(results, output_dir)
    
    # Plot chunk accuracy by position
    plot_chunk_accuracy_by_position(results, output_dir, rollout_type, max_chunks_to_show)
    
    # Print category importance ranking with percentages
    print(f"\n{rollout_type.capitalize()} Category Importance Ranking:")
    for idx, row in category_importance.iterrows():
        print(f"{idx+1}. {row['categories']}: {row['mean_pct']:.2f}% ± {row['se_pct']:.2f}% (n={int(row['count'])})")
    
    # Analyze token frequencies
    if get_token_frequencies:
        if dag_dir:
            print(f"\nAnalyzing token frequencies from DAG-improved chunks in {dag_dir}")
            analyze_dag_token_frequencies(Path(dag_dir), output_dir)
        else:
            print("\nAnalyzing token frequencies from rollout results")
            analyze_token_frequencies(results, output_dir)
    
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
    
    # Analyze variance to identify potential reasoning forks
    analyze_chunk_variance(results, output_dir)
    analyze_function_tag_variance(results, output_dir)
    analyze_within_problem_variance(results, output_dir)
    
    print(f"{rollout_type.capitalize()} analysis complete. Results saved to {output_dir}")

def analyze_dag_token_frequencies(dag_dir: Path, output_dir: Path) -> None:
    """
    Analyze token frequencies from DAG-improved chunks.
    
    Args:
        dag_dir: Directory containing DAG-improved chunks
        output_dir: Directory to save analysis results
    """
    print("Analyzing token frequencies from DAG-improved chunks...")
    
    # Create plots directory
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    # Collect all chunks by category
    category_chunks = {}
    
    # Find all problem directories
    problem_dirs = sorted([d for d in dag_dir.iterdir() if d.is_dir() and d.name.startswith("problem_")])
    
    for problem_dir in problem_dirs:
        # Find seed directories
        seed_dirs = sorted([d for d in problem_dir.iterdir() if d.is_dir() and d.name.startswith("seed_")])
        
        for seed_dir in seed_dirs:
            # Look for chunks_dag_improved.json
            chunks_file = seed_dir / "chunks_dag_improved.json"
            
            if not chunks_file.exists():
                continue
                
            # Load chunks
            with open(chunks_file, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
                
            # Process each chunk
            for chunk in chunks_data:
                # Get function tags (categories)
                function_tags = chunk.get("function_tags", [])
                
                # Skip chunks with no tags
                if not function_tags:
                    function_tags = chunk.get("categories", [])
                    if not function_tags:
                        continue
                
                # Get chunk text
                chunk_text = chunk.get("chunk", "")
                
                # Skip empty chunks
                if not chunk_text:
                    chunk_text = chunk.get("text", "")
                    if not chunk_text:
                        continue
                    
                # Add chunk to each of its categories
                for tag in function_tags:
                    # Format tag for better display
                    if isinstance(tag, str):
                        formatted_tag = " ".join(word.capitalize() for word in tag.split("_"))
                    else:
                        continue  # Skip non-string tags
                        
                    # Skip unknown category
                    if formatted_tag.lower() == "unknown":
                        continue
                        
                    if formatted_tag not in category_chunks:
                        category_chunks[formatted_tag] = []
                        
                    category_chunks[formatted_tag].append(chunk_text)
    
    # Skip if no categories found
    if not category_chunks:
        print("No categories found for token frequency analysis")
        return
    
    print(f"Found {len(category_chunks)} categories with {sum(len(chunks) for chunks in category_chunks.values())} total chunks")
    
    # Generate plots for unigrams, bigrams, and trigrams
    for n in [1, 2, 3]:
        print(f"Analyzing {n}-gram frequencies...")
        
        # Tokenize chunks and count frequencies
        category_ngram_frequencies = {}
        
        for category, chunks in category_chunks.items():
            # Tokenize all chunks
            all_tokens = []
            for chunk in chunks:
                # Simple tokenization by splitting on whitespace and punctuation
                tokens = re.findall(r'\b\w+\b', chunk.lower())
                
                # Filter out stopwords and numbers
                filtered_tokens = [token for token in tokens if token not in stopwords and not token.isdigit() and len(token) > 1]
                
                # Generate n-grams
                ngrams = []
                for i in range(len(filtered_tokens) - (n-1)):
                    ngram = " ".join(filtered_tokens[i:i+n])
                    ngrams.append(ngram)
                
                all_tokens.extend(ngrams)
                
            # Count token frequencies
            token_counts = Counter(all_tokens)
            
            # Calculate percentages
            total_chunks = len(chunks)
            token_percentages = {}
            
            for token, count in token_counts.items():
                # Count in how many chunks this n-gram appears
                chunks_with_token = sum(1 for chunk in chunks if re.search(r'\b' + re.escape(token) + r'\b', chunk.lower()))
                percentage = (chunks_with_token / total_chunks) * 100
                token_percentages[token] = percentage
                
            # Store results
            category_ngram_frequencies[category] = token_percentages
        
        # Create a master plot with subplots for each category
        num_categories = len(category_ngram_frequencies)
        
        # Calculate grid dimensions
        cols = min(3, num_categories)
        rows = (num_categories + cols - 1) // cols  # Ceiling division
        
        # Create figure
        fig = plt.figure(figsize=(20, 5 * rows))
        
        # Create subplots
        for i, (category, token_percentages) in enumerate(category_ngram_frequencies.items()):
            # Sort tokens by percentage (descending)
            sorted_tokens = sorted(token_percentages.items(), key=lambda x: x[1], reverse=True)
            
            # Take top 10 tokens
            top_tokens = sorted_tokens[:10]
            
            # Create subplot
            ax = fig.add_subplot(rows, cols, i + 1)
            
            # Extract token names and percentages
            token_names = [token for token, _ in top_tokens]
            percentages = [percentage for _, percentage in top_tokens]
            
            # Create horizontal bar plot
            y_pos = range(len(token_names))
            ax.barh(y_pos, percentages, align='center')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(token_names)
            ax.invert_yaxis()  # Labels read top-to-bottom
            ax.set_xlabel('Percentage of Chunks (%)')
            ax.set_title(f'Top 10 {n}-grams in {category} Chunks')
            
            # Add percentage labels
            for j, percentage in enumerate(percentages):
                ax.text(percentage + 1, j, f'{percentage:.1f}%', va='center')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        ngram_name = "unigrams" if n == 1 else f"{n}-grams"
        plt.savefig(plots_dir / f"dag_token_{ngram_name}_by_category.png", dpi=300)
        plt.close()
        
        print(f"{n}-gram frequency analysis complete. Plot saved to {plots_dir / f'dag_token_{ngram_name}_by_category.png'}")
        
        # Save token frequencies to JSON
        token_frequencies_file = output_dir / f"dag_token_{ngram_name}_by_category.json"
        with open(token_frequencies_file, 'w', encoding='utf-8') as f:
            json.dump(category_ngram_frequencies, f, indent=2)

def analyze_token_frequencies(results: List[Dict], output_dir: Path) -> None:
    """
    Analyze token frequencies by category and generate plots.
    
    Args:
        results: List of problem analysis results
        output_dir: Directory to save plots
    """
    print("Analyzing token frequencies by category...")
    
    # Create plots directory
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    # Collect all chunks by category
    category_chunks = {}
    
    for result in results:
        if not result or "labeled_chunks" not in result:
            continue
            
        for chunk in result.get("labeled_chunks", []):
            # Get function tags (categories)
            function_tags = chunk.get("function_tags", [])
            
            # Skip chunks with no tags
            if not function_tags:
                continue
                
            # Get chunk text
            chunk_text = chunk.get("chunk", "")
            
            # Skip empty chunks
            if not chunk_text:
                continue
                
            # Add chunk to each of its categories
            for tag in function_tags:
                # Format tag for better display
                if isinstance(tag, str):
                    formatted_tag = " ".join(word.capitalize() for word in tag.split("_"))
                else:
                    continue  # Skip non-string tags
                    
                # Skip unknown category
                if formatted_tag.lower() == "unknown":
                    continue
                    
                if formatted_tag not in category_chunks:
                    category_chunks[formatted_tag] = []
                    
                category_chunks[formatted_tag].append(chunk_text)
    
    # Skip if no categories found
    if not category_chunks:
        print("No categories found for token frequency analysis")
        return
    
    # Generate plots for unigrams, bigrams, and trigrams
    for n in [1, 2, 3]:
        print(f"Analyzing {n}-gram frequencies...")
        
        # Tokenize chunks and count frequencies
        category_ngram_frequencies = {}
        
        for category, chunks in category_chunks.items():
            # Tokenize all chunks
            all_tokens = []
            for chunk in chunks:
                # Simple tokenization by splitting on whitespace and punctuation
                tokens = re.findall(r'\b\w+\b', chunk.lower())
                
                # Filter out stopwords and numbers
                filtered_tokens = [token for token in tokens if token not in stopwords and not token.isdigit() and len(token) > 1]
                
                # Generate n-grams
                ngrams = []
                for i in range(len(filtered_tokens) - (n-1)):
                    ngram = " ".join(filtered_tokens[i:i+n])
                    ngrams.append(ngram)
                
                all_tokens.extend(ngrams)
                
            # Count token frequencies
            token_counts = Counter(all_tokens)
            
            # Calculate percentages
            total_chunks = len(chunks)
            token_percentages = {}
            
            for token, count in token_counts.items():
                # Count in how many chunks this n-gram appears
                chunks_with_token = sum(1 for chunk in chunks if re.search(r'\b' + re.escape(token) + r'\b', chunk.lower()))
                percentage = (chunks_with_token / total_chunks) * 100
                token_percentages[token] = percentage
                
            # Store results
            category_ngram_frequencies[category] = token_percentages
        
        # Create a master plot with subplots for each category
        num_categories = len(category_ngram_frequencies)
        
        # Calculate grid dimensions
        cols = min(3, num_categories)
        rows = (num_categories + cols - 1) // cols  # Ceiling division
        
        # Create figure
        fig = plt.figure(figsize=(20, 5 * rows))
        
        # Create subplots
        for i, (category, token_percentages) in enumerate(category_ngram_frequencies.items()):
            # Sort tokens by percentage (descending)
            sorted_tokens = sorted(token_percentages.items(), key=lambda x: x[1], reverse=True)
            
            # Take top 10 tokens
            top_tokens = sorted_tokens[:10]
            
            # Create subplot
            ax = fig.add_subplot(rows, cols, i + 1)
            
            # Extract token names and percentages
            token_names = [token for token, _ in top_tokens]
            percentages = [percentage for _, percentage in top_tokens]
            
            # Create horizontal bar plot
            y_pos = range(len(token_names))
            ax.barh(y_pos, percentages, align='center')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(token_names)
            ax.invert_yaxis()  # Labels read top-to-bottom
            ax.set_xlabel('Percentage of Chunks (%)')
            ax.set_title(f'Top 10 {n}-grams in {category} Chunks')
            
            # Add percentage labels
            for j, percentage in enumerate(percentages):
                ax.text(percentage + 1, j, f'{percentage:.1f}%', va='center')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        ngram_name = "unigrams" if n == 1 else f"{n}-grams"
        plt.savefig(plots_dir / f"token_{ngram_name}_by_category.png", dpi=300)
        plt.close()
        
        print(f"{n}-gram frequency analysis complete. Plot saved to {plots_dir / f'token_{ngram_name}_by_category.png'}")
        
        # Save token frequencies to JSON
        token_frequencies_file = output_dir / f"token_{ngram_name}_by_category.json"
        with open(token_frequencies_file, 'w', encoding='utf-8') as f:
            json.dump(category_ngram_frequencies, f, indent=2)

def analyze_top_steps_by_category(results: List[Dict], output_dir: Path, top_n: int = 20, use_abs: bool = True) -> None:
    """
    Analyze and plot average z-scores of top N steps by function tag category.
    
    Args:
        results: List of problem analysis results
        output_dir: Directory to save the plot
        top_n: Number of top steps to consider for each problem
        use_abs: Whether to use absolute values of z-scores for ranking and averaging
    """
    print(f"Analyzing top {top_n} steps by category")
    
    # Create plots directory
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    # Create a dictionary to store z-scores by category
    category_zscores = {}
    
    # Process each problem
    for result in results:
        if not result:
            continue
            
        labeled_chunks = result.get("labeled_chunks", [])
        if not labeled_chunks:
            continue
            
        # Extract importance scores and convert to z-scores
        importance_scores = [chunk.get("importance", 0.0) if not use_abs else abs(chunk.get("importance", 0.0)) for chunk in labeled_chunks]
        
        # Skip if all scores are the same or if there are too few chunks
        if len(set(importance_scores)) <= 1 or len(importance_scores) < 3:
            continue
            
        # Calculate z-scores
        mean_importance = np.mean(importance_scores)
        std_importance = np.std(importance_scores)
        
        if std_importance == 0:
            continue
            
        z_scores = [(score - mean_importance) / std_importance for score in importance_scores]
        
        # Create a list of (chunk_idx, z_score, function_tags) tuples
        chunk_data = []
        for i, (chunk, z_score) in enumerate(zip(labeled_chunks, z_scores)):
            function_tags = chunk.get("function_tags", ["unknown"])
            if not function_tags:
                function_tags = ["unknown"]
            # Use absolute or raw z-score based on parameter
            score_for_ranking = z_score
            chunk_data.append((i, z_score, score_for_ranking, function_tags))
        
        # Sort by z-score (absolute or raw) and get top N
        top_chunks = sorted(chunk_data, key=lambda x: x[2], reverse=True)[:top_n]
        
        # Add to category dictionary - each chunk can have multiple tags
        for _, z_score, _, function_tags in top_chunks:
            # Use the actual z-score (not the ranking score)
            score_to_store = z_score
            
            for tag in function_tags:
                # Format tag for better display
                formatted_tag = " ".join(word.capitalize() for word in tag.split("_"))
                if formatted_tag.lower() == "unknown":
                    continue
                    
                if formatted_tag not in category_zscores:
                    category_zscores[formatted_tag] = []
                category_zscores[formatted_tag].append(score_to_store)
    
    # Calculate average z-score for each category
    category_avg_zscores = {}
    category_std_zscores = {}
    category_counts = {}
    
    for category, zscores in category_zscores.items():
        if zscores:
            category_avg_zscores[category] = np.mean(zscores)
            category_std_zscores[category] = np.std(zscores)
            category_counts[category] = len(zscores)
    
    # Sort categories by average z-score
    sorted_categories = sorted(category_avg_zscores.keys(), key=lambda x: category_avg_zscores[x], reverse=True)
    
    # Create the plot
    plt.figure(figsize=(15, 10))
    
    # Plot average z-scores with standard error bars
    avg_zscores = [category_avg_zscores[cat] for cat in sorted_categories]
    std_zscores = [category_std_zscores[cat] for cat in sorted_categories]
    counts = [category_counts[cat] for cat in sorted_categories]
    
    # Calculate standard error (SE = standard deviation / sqrt(sample size))
    standard_errors = [std / np.sqrt(count) for std, count in zip(std_zscores, counts)]
    
    # Create bar plot with standard error bars
    bars = plt.bar(range(len(sorted_categories)), avg_zscores, yerr=standard_errors, capsize=5, alpha=0.7, color='skyblue', edgecolor='black')
    
    # Add count labels on top of bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'n={count}', ha='center', va='bottom', fontsize=9)
    
    # Set labels and title
    plt.xlabel('Function Tag Category', fontsize=12)
    plt.ylabel(f'Average Z-Score (Top {top_n} Steps) ± SE', fontsize=12)
    plt.title(f'Average Z-Score of Top {top_n} Steps by Category', fontsize=14)
    
    # Set x-tick labels
    plt.xticks(range(len(sorted_categories)), sorted_categories, rotation=45, ha='right')
    
    # Add grid
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add a legend explaining the error bars
    plt.figtext(0.91, 0.01, "Error bars: Standard Error (SE)", ha="right", fontsize=10, style='italic')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plot_path = plots_dir / f"top_{top_n}_steps_by_category.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"Plot saved to {plot_path}")
    
    # Also save the data as CSV
    csv_data = []
    for i, category in enumerate(sorted_categories):
        csv_data.append({
            'category': category,
            'avg_zscore': category_avg_zscores[category],
            'std_zscore': category_std_zscores[category],
            'standard_error': category_std_zscores[category] / np.sqrt(category_counts[category]),
            'count': category_counts[category]
        })
    
    csv_path = plots_dir / f"top_{top_n}_steps_by_category.csv"
    pd.DataFrame(csv_data).to_csv(csv_path, index=False)
    print(f"Data saved to {csv_path}")

def analyze_high_zscore_steps_by_category(results: List[Dict], output_dir: Path, z_threshold: float = 1.5, use_abs: bool = True) -> None:
    """
    Analyze and plot average z-scores of steps with z-scores above threshold by function tag category.
    
    Args:
        results: List of problem analysis results
        output_dir: Directory to save the plot
        z_threshold: Minimum z-score threshold for including steps
        use_abs: Whether to use absolute values of z-scores for thresholding and averaging
    """
    print(f"Analyzing steps with z-score > {z_threshold} by category...")
    
    # Create plots directory
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    # Create a dictionary to store z-scores by category
    category_zscores = {}
    total_high_zscore_steps = 0
    total_steps_analyzed = 0
    
    # Process each problem
    for result in results:
        if not result:
            continue
            
        labeled_chunks = result.get("labeled_chunks", [])
        if not labeled_chunks:
            continue
            
        # Extract importance scores and convert to z-scores
        importance_scores = [chunk.get("importance", 0.0) if not use_abs else abs(chunk.get("importance", 0.0)) for chunk in labeled_chunks]
        
        # Skip if all scores are the same or if there are too few chunks
        if len(set(importance_scores)) <= 1 or len(importance_scores) < 3:
            continue
            
        # Calculate z-scores
        mean_importance = np.mean(importance_scores)
        std_importance = np.std(importance_scores)
        
        if std_importance == 0:
            continue
            
        z_scores = [(score - mean_importance) / std_importance for score in importance_scores]
        total_steps_analyzed += len(z_scores)
        
        # Create a list of (chunk_idx, z_score, function_tags) tuples
        chunk_data = []
        for i, (chunk, z_score) in enumerate(zip(labeled_chunks, z_scores)):
            function_tags = chunk.get("function_tags", ["unknown"])
            if not function_tags:
                function_tags = ["unknown"]
            chunk_data.append((i, z_score, function_tags))
        
        # Filter chunks by z-score threshold
        high_zscore_chunks = [chunk for chunk in chunk_data if abs(chunk[1]) > z_threshold]
        total_high_zscore_steps += len(high_zscore_chunks)
        
        # Add to category dictionary - each chunk can have multiple tags
        for _, z_score, function_tags in high_zscore_chunks:
            # Use the actual z-score (not the ranking score)
            score_to_store = z_score
            
            for tag in function_tags:
                # Format tag for better display
                formatted_tag = " ".join(word.capitalize() for word in tag.split("_"))
                if formatted_tag.lower() == "unknown":
                    continue
                    
                if formatted_tag not in category_zscores:
                    category_zscores[formatted_tag] = []
                category_zscores[formatted_tag].append(score_to_store)
    
    print(f"Found {total_high_zscore_steps} steps with z-score > {z_threshold} out of {total_steps_analyzed} total steps ({total_high_zscore_steps/total_steps_analyzed:.1%})")
    
    # Skip if no categories found
    if not category_zscores:
        print(f"No steps with z-score > {z_threshold} found")
        return
    
    # Calculate average z-score for each category
    category_avg_zscores = {}
    category_std_zscores = {}
    category_counts = {}
    
    for category, zscores in category_zscores.items():
        if zscores:
            category_avg_zscores[category] = np.mean(zscores)
            category_std_zscores[category] = np.std(zscores)
            category_counts[category] = len(zscores)
    
    # Sort categories by average z-score
    sorted_categories = sorted(category_avg_zscores.keys(), key=lambda x: category_avg_zscores[x], reverse=True)
    
    # Create the plot
    plt.figure(figsize=(15, 10))
    
    # Plot average z-scores with standard error bars
    avg_zscores = [category_avg_zscores[cat] for cat in sorted_categories]
    std_zscores = [category_std_zscores[cat] for cat in sorted_categories]
    counts = [category_counts[cat] for cat in sorted_categories]
    
    # Calculate standard error (SE = standard deviation / sqrt(sample size))
    standard_errors = [std / np.sqrt(count) for std, count in zip(std_zscores, counts)]
    
    # Create bar plot with standard error bars
    bars = plt.bar(range(len(sorted_categories)), avg_zscores, yerr=standard_errors, capsize=5, alpha=0.7, color='skyblue', edgecolor='black')
    
    # Add count labels on top of bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'n={count}', ha='center', va='bottom', fontsize=9)
    
    # Set labels and title
    plt.xlabel('Function Tag Category', fontsize=12)
    plt.ylabel(f'Average Z-Score (Steps with |Z| > {z_threshold}) ± SE', fontsize=12)
    plt.title(f'Average Z-Score of Steps with |Z| > {z_threshold} by Category', fontsize=14)
    
    # Set x-tick labels
    plt.xticks(range(len(sorted_categories)), sorted_categories, rotation=45, ha='right')
    
    # Add grid
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add a legend explaining the error bars
    plt.figtext(0.91, 0.01, "Error bars: Standard Error (SE)", ha="right", fontsize=10, style='italic')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plot_path = plots_dir / f"high_zscore_{z_threshold}_steps_by_category.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"Plot saved to {plot_path}")
    
    # Also save the data as CSV
    csv_data = []
    for i, category in enumerate(sorted_categories):
        csv_data.append({
            'category': category,
            'avg_zscore': category_avg_zscores[category],
            'std_zscore': category_std_zscores[category],
            'standard_error': category_std_zscores[category] / np.sqrt(category_counts[category]),
            'count': category_counts[category]
        })
    
    csv_path = plots_dir / f"high_zscore_{z_threshold}_steps_by_category.csv"
    pd.DataFrame(csv_data).to_csv(csv_path, index=False)
    print(f"Data saved to {csv_path}")

def main():
    parser = argparse.ArgumentParser(description='Analyze rollout data and label chunks')
    parser.add_argument('-ic', '--correct_rollouts_dir', type=str, default="math_rollouts/deepseek-r1-distill-qwen-14b/temperature_0.6_top_p_0.95/correct_base_solution", help='Directory containing correct rollout data')
    parser.add_argument('-ii', '--incorrect_rollouts_dir', type=str, default="math_rollouts/deepseek-r1-distill-qwen-14b/temperature_0.6_top_p_0.95/incorrect_base_solution", help='Directory containing incorrect rollout data')
    parser.add_argument('-icf', '--correct_forced_answer_rollouts_dir', type=str, default="math_rollouts/deepseek-r1-distill-qwen-14b/temperature_0.6_top_p_0.95/correct_base_solution_forced_answer", help='Directory containing correct rollout data with forced answers')
    parser.add_argument('-o', '--output_dir', type=str, default="analysis/basic", help='Directory to save analysis results (defaults to rollouts_dir)')
    parser.add_argument('-p', '--problems', type=str, default=None, help='Comma-separated list of problem indices to analyze (default: all)')
    parser.add_argument('-m', '--max_problems', type=int, default=None, help='Maximum number of problems to analyze')
    parser.add_argument('-a', '--absolute', default=False, action='store_true', help='Use absolute value for importance calculation')
    parser.add_argument('-f', '--force_relabel', default=False, action='store_true', help='Force relabeling of chunks')
    parser.add_argument('-d', '--dag_dir', type=str, default="archive/analysis/math", help='Directory containing DAG-improved chunks for token frequency analysis')
    parser.add_argument('-t', '--token_analysis_source', type=str, default="dag", choices=["dag", "rollouts"], help='Source for token frequency analysis: "dag" for DAG-improved chunks or "rollouts" for rollout data')
    parser.add_argument('-tf', '--get_token_frequencies', default=False, action='store_true', help='Get token frequencies')
    parser.add_argument('-mc', '--max_chunks_to_show', type=int, default=100, help='Maximum number of chunks to show in plots')
    args = parser.parse_args()
    
    # Set up directories
    correct_rollouts_dir = Path(args.correct_rollouts_dir) if args.correct_rollouts_dir and len(args.correct_rollouts_dir) > 0 else None
    incorrect_rollouts_dir = Path(args.incorrect_rollouts_dir) if args.incorrect_rollouts_dir and len(args.incorrect_rollouts_dir) > 0 else None
    correct_forced_answer_dir = Path(args.correct_forced_answer_rollouts_dir) if args.correct_forced_answer_rollouts_dir and len(args.correct_forced_answer_rollouts_dir) > 0 else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Check if at least one rollouts directory is provided
    if not correct_rollouts_dir and not incorrect_rollouts_dir:
        print("Error: At least one of --correct_rollouts_dir or --incorrect_rollouts_dir must be provided")
        return
    
    # Process each rollout type if provided
    if correct_rollouts_dir:
        print(f"\n=== Processing CORRECT rollouts from {correct_rollouts_dir} ===\n")
        correct_output_dir = output_dir / "correct_base_solution"
        correct_output_dir.mkdir(exist_ok=True, parents=True)
        process_rollouts(
            rollouts_dir=correct_rollouts_dir,
            output_dir=correct_output_dir,
            problems=args.problems,
            max_problems=args.max_problems,
            absolute=args.absolute,
            force_relabel=args.force_relabel,
            rollout_type="correct",
            dag_dir=args.dag_dir if args.token_analysis_source == "dag" else None,
            forced_answer_dir=correct_forced_answer_dir,
            get_token_frequencies=args.get_token_frequencies,
            max_chunks_to_show=args.max_chunks_to_show
        )
    
    if incorrect_rollouts_dir:
        print(f"\n=== Processing INCORRECT rollouts from {incorrect_rollouts_dir} ===\n")
        incorrect_output_dir = output_dir / "incorrect_base_solution"
        incorrect_output_dir.mkdir(exist_ok=True, parents=True)
        process_rollouts(
            rollouts_dir=incorrect_rollouts_dir,
            output_dir=incorrect_output_dir,
            problems=args.problems,
            max_problems=args.max_problems,
            absolute=args.absolute,
            force_relabel=args.force_relabel,
            rollout_type="incorrect",
            dag_dir=args.dag_dir if args.token_analysis_source == "dag" else None,
            get_token_frequencies=args.get_token_frequencies,
            max_chunks_to_show=args.max_chunks_to_show
        )

if __name__ == "__main__":
    main()