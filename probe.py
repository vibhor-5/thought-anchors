import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Tuple, Union, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import pandas as pd
from collections import Counter
import pickle
import seaborn as sns

# Set up paths
cots_dir = Path("cots")
analysis_dir = Path("analysis")
output_dir = Path("probe_analysis")
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
    # Tokenize the input
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # Store activations
    activations = {}
    
    # Define hook function
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output[0].detach()
        return hook
    
    # Register hooks
    hooks = []
    for layer_idx in layers:
        hook = model.model.layers[layer_idx].register_forward_hook(get_activation(layer_idx))
        hooks.append(hook)
    
    # Forward pass
    with torch.no_grad():
        model(**inputs)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return activations

def extract_chunk_pairs_with_activations(
    problem_dirs: List[Path],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    layer: int,
    max_problems: int = 100
) -> Tuple[List[torch.Tensor], List[str], List[bool], List[Dict]]:
    """
    Extract pairs of consecutive chunks with activations and next chunk categories.
    
    Args:
        problem_dirs: List of problem directory paths
        model: Hugging Face model
        tokenizer: Tokenizer
        layer: Layer to extract activations from
        max_problems: Maximum number of problems to process
        
    Returns:
        Tuple of (activations, next_categories, is_backtracking, metadata)
    """
    all_activations = []
    all_next_categories = []
    all_is_backtracking = []
    all_metadata = []
    
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
            
            # Find the corresponding solution
            solution_dict = None
            for sol in solutions:
                if sol.get("seed") == int(seed_id.replace("seed_", "")):
                    solution_dict = sol
                    break
            
            if not solution_dict:
                continue
                
            full_text = solution_dict["solution"]
            
            # Load chunks data
            chunks_file = seed_dir / "chunks.json"
            
            if not chunks_file.exists():
                continue
                
            with open(chunks_file, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
            
            if not chunks_data:
                continue
            
            # Filter out chunks with "Unknown" category
            chunks_data = [chunk for chunk in chunks_data if chunk.get("category", "Unknown") != "Unknown"]
            
            if len(chunks_data) < 2:  # Need at least 2 chunks to form a pair
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
                start_idx = len(chunk_tokens.input_ids[0]) - 2  # Adjust for special tokens
                
                chunk_tokens = tokenizer(full_text[:chunk_end], return_tensors="pt")
                end_idx = len(chunk_tokens.input_ids[0]) - 2  # Adjust for special tokens
                
                chunk_token_ranges.append((max(0, start_idx), end_idx))
            
            # Get activations for the full text
            with torch.no_grad():
                activations = get_residual_stream_activations(model, tokenizer, full_text, [layer])
            
            if layer not in activations:
                continue
                
            # Extract activations for each chunk
            layer_activations = activations[layer]
            
            # For each chunk (except the last one), get its activation and the next chunk's category
            for i in range(len(chunks_data) - 1):
                if i >= len(chunk_token_ranges) or i+1 >= len(chunks_data):
                    continue
                
                start_idx, end_idx = chunk_token_ranges[i]
                
                # Extract activations for this chunk (average across tokens)
                chunk_act = layer_activations[0, start_idx:end_idx, :].mean(dim=0)
                
                # Get the category of the next chunk
                next_category = chunks_data[i+1]["category"]
                
                # Check if the next chunk is backtracking
                is_backtracking = chunks_data[i+1]["category"] == "Backtracking"
                
                all_activations.append(chunk_act)
                all_next_categories.append(next_category)
                all_is_backtracking.append(is_backtracking)
                
                # Add metadata for each chunk
                metadata = {
                    "problem_id": problem_dir.name,
                    "seed_id": seed_id,
                    "chunk_idx": i,
                    "next_category": next_category
                }
                all_metadata.append(metadata)
    
    return all_activations, all_next_categories, all_is_backtracking, all_metadata

def train_category_probe(
    activations: List[torch.Tensor],
    categories: List[str],
    problem_ids: List[str],
    test_size: float = 0.2,
    model_type: str = "logistic"
) -> Tuple[object, LabelEncoder, float, Dict]:
    """
    Train a probe to predict the category of the next chunk.
    
    Args:
        activations: List of activation tensors
        categories: List of category labels
        problem_ids: List of problem IDs for stratified splitting
        test_size: Proportion of data to use for testing
        model_type: Type of model to use ('logistic' or 'mlp')
        
    Returns:
        Tuple of (trained model, label encoder, accuracy, metrics)
    """
    print(f"Training category probe with {len(activations)} examples...")
    
    # Convert activations to numpy array
    X = torch.stack(activations).cpu().numpy()
    
    # Check for NaN values and handle them
    if np.isnan(X).any():
        print(f"Warning: Found {np.isnan(X).sum()} NaN values in the data.")
        print(f"NaN distribution: {np.isnan(X).sum(axis=0).max()} max in a single feature")
        
        # Option 1: Remove samples with NaN values
        nan_rows = np.isnan(X).any(axis=1)
        print(f"Removing {nan_rows.sum()} samples with NaN values")
        X = X[~nan_rows]
        categories = [cat for i, cat in enumerate(categories) if not nan_rows[i]]
        problem_ids = [pid for i, pid in enumerate(problem_ids) if not nan_rows[i]]
        
        # Option 2 (alternative): Replace NaN values with zeros
        # X = np.nan_to_num(X, nan=0.0)
    
    # Encode categories
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(categories)
    
    # Get unique problem IDs for splitting
    unique_problems = list(set(problem_ids))
    
    # Split problems into train and test sets
    train_problems, test_problems = train_test_split(
        unique_problems, test_size=test_size, random_state=42
    )
    
    # Create train and test masks
    train_mask = np.array([pid in train_problems for pid in problem_ids])
    test_mask = np.array([pid in test_problems for pid in problem_ids])
    
    # Split data based on problem IDs
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    
    print(f"Train set: {X_train.shape[0]} examples, Test set: {X_test.shape[0]} examples")
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train model
    if model_type == "logistic":
        model = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced')
    else:  # MLP
        model = MLPClassifier(hidden_layer_sizes=(512, 128), max_iter=2000, early_stopping=True)
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Get detailed metrics
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
    
    print(f"Category probe accuracy: {accuracy:.4f}")
    print(f"Categories: {label_encoder.classes_}")
    
    # Print per-class metrics
    for category in label_encoder.classes_:
        if category in report:
            print(f"  {category}: F1={report[category]['f1-score']:.4f}, Precision={report[category]['precision']:.4f}, Recall={report[category]['recall']:.4f}")
    
    # Save model and scaler
    with open(output_dir / f"category_probe_layer{layer}.pkl", 'wb') as f:
        pickle.dump((model, scaler, label_encoder), f)
    
    return model, label_encoder, accuracy, report

def train_backtracking_probe(
    activations: List[torch.Tensor],
    is_backtracking: List[bool],
    metadata: List[Dict],
    test_size: float = 0.2,
    model_type: str = "logistic",
    balance_method: str = "undersample"  # Options: "undersample", "oversample", "class_weight"
) -> Tuple[object, object, float, Dict]:
    """
    Train a binary probe to predict if the next chunk is backtracking.
    
    Args:
        activations: List of activation tensors
        is_backtracking: List of boolean flags indicating if next chunk is backtracking
        metadata: Metadata for each chunk
        test_size: Proportion of data to use for testing
        model_type: Type of model to use ('logistic' or 'mlp')
        balance_method: Method to balance the dataset ('undersample', 'oversample', 'class_weight')
        
    Returns:
        Tuple of (trained model, scaler, accuracy, metrics)
    """
    print(f"Training backtracking probe with {len(activations)} examples...")
    
    # Convert activations to numpy array
    X = torch.stack(activations).cpu().numpy()
    y = np.array(is_backtracking, dtype=int)
    
    # Print original class distribution
    backtracking_count = sum(is_backtracking)
    non_backtracking_count = len(is_backtracking) - backtracking_count
    print(f"Original class distribution: Backtracking={backtracking_count}, Non-backtracking={non_backtracking_count}")
    
    # Get problem IDs for splitting
    problem_ids = [meta["problem_id"] for meta in metadata]
    
    # Get unique problem IDs for splitting
    unique_problems = list(set(problem_ids))
    
    # Split problems into train and test sets
    train_problems, test_problems = train_test_split(
        unique_problems, test_size=test_size, random_state=42
    )
    
    # Create train and test masks
    train_mask = np.array([pid in train_problems for pid in problem_ids])
    test_mask = np.array([pid in test_problems for pid in problem_ids])
    
    # Split data based on problem IDs
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    
    print(f"Train set: {X_train.shape[0]} examples, Test set: {X_test.shape[0]} examples")
    
    # Balance the training data
    if balance_method == "undersample":
        # Undersample the majority class
        from sklearn.utils import resample
        
        # Separate majority and minority classes
        X_train_majority = X_train[y_train == 0]
        X_train_minority = X_train[y_train == 1]
        y_train_majority = y_train[y_train == 0]
        y_train_minority = y_train[y_train == 1]
        
        # If backtracking is the minority class
        if len(X_train_minority) < len(X_train_majority):
            # Undersample majority class
            X_train_majority_downsampled = resample(
                X_train_majority,
                replace=False,
                n_samples=len(X_train_minority),
                random_state=42
            )
            y_train_majority_downsampled = np.zeros(len(X_train_majority_downsampled))
            
            # Combine minority class with downsampled majority class
            X_train = np.vstack((X_train_majority_downsampled, X_train_minority))
            y_train = np.hstack((y_train_majority_downsampled, y_train_minority))
        else:
            # Undersample minority class
            X_train_minority_downsampled = resample(
                X_train_minority,
                replace=False,
                n_samples=len(X_train_majority),
                random_state=42
            )
            y_train_minority_downsampled = np.ones(len(X_train_minority_downsampled))
            
            # Combine majority class with downsampled minority class
            X_train = np.vstack((X_train_majority, X_train_minority_downsampled))
            y_train = np.hstack((y_train_majority, y_train_minority_downsampled))
            
    elif balance_method == "oversample":
        # Oversample the minority class
        from sklearn.utils import resample
        
        # Separate majority and minority classes
        X_train_majority = X_train[y_train == 0]
        X_train_minority = X_train[y_train == 1]
        y_train_majority = y_train[y_train == 0]
        y_train_minority = y_train[y_train == 1]
        
        # If backtracking is the minority class
        if len(X_train_minority) < len(X_train_majority):
            # Oversample minority class
            X_train_minority_upsampled = resample(
                X_train_minority,
                replace=True,
                n_samples=len(X_train_majority),
                random_state=42
            )
            y_train_minority_upsampled = np.ones(len(X_train_minority_upsampled))
            
            # Combine majority class with upsampled minority class
            X_train = np.vstack((X_train_majority, X_train_minority_upsampled))
            y_train = np.hstack((y_train_majority, y_train_minority_upsampled))
        else:
            # Oversample majority class
            X_train_majority_upsampled = resample(
                X_train_majority,
                replace=True,
                n_samples=len(X_train_minority),
                random_state=42
            )
            y_train_majority_upsampled = np.zeros(len(X_train_majority_upsampled))
            
            # Combine upsampled majority class with minority class
            X_train = np.vstack((X_train_majority_upsampled, X_train_minority))
            y_train = np.hstack((y_train_majority_upsampled, y_train_minority))
    
    # Print balanced class distribution
    balanced_backtracking_count = sum(y_train == 1)
    balanced_non_backtracking_count = sum(y_train == 0)
    print(f"Balanced training class distribution: Backtracking={balanced_backtracking_count}, Non-backtracking={balanced_non_backtracking_count}")
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train model
    if model_type == "logistic":
        if balance_method == "class_weight":
            model = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced')
        else:
            model = LogisticRegression(max_iter=1000, C=1.0)
    else:  # MLP
        if balance_method == "class_weight":
            model = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=1000, early_stopping=True, class_weight='balanced')
        else:
            model = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=1000, early_stopping=True)
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Get detailed metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    
    print(f"Backtracking probe accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    # Save model and scaler
    with open(output_dir / f"backtracking_probe_layer{layer}.pkl", 'wb') as f:
        pickle.dump((model, scaler), f)
    
    return model, scaler, accuracy, {"precision": precision, "recall": recall, "f1": f1}

def plot_category_importance(model, label_encoder, layer, top_n=10):
    """
    Plot feature importance for category prediction.
    
    Args:
        model: Trained logistic regression model
        label_encoder: Label encoder for categories
        layer: Layer number for the plot title
        top_n: Number of top features to show
    """
    if not hasattr(model, 'coef_'):
        print("Model doesn't have feature coefficients, skipping importance plot")
        return
    
    plt.figure(figsize=(15, 10))
    
    # Get feature importance for each class
    coefs = model.coef_
    classes = label_encoder.classes_
    
    # For each class, plot top features
    for i, (coef, cls) in enumerate(zip(coefs, classes)):
        # Get absolute importance
        importance = np.abs(coef)
        # Get indices of top features
        top_indices = importance.argsort()[-top_n:][::-1]
        
        plt.subplot(len(classes), 1, i+1)
        plt.barh(range(top_n), importance[top_indices])
        plt.yticks(range(top_n), [f"Feature {idx}" for idx in top_indices])
        plt.title(f"Top {top_n} features for predicting '{cls}'")
        plt.tight_layout()
    
    plt.savefig(output_dir / f"category_importance_layer{layer}.png", dpi=300)
    plt.close()

def plot_backtracking_probability_by_length(
    model,
    scaler,
    activations: List[torch.Tensor],
    is_backtracking: List[bool],
    metadata: List[Dict],
    layer: int,
    test_size: float = 0.2
):
    """
    Plot the probability of backtracking as a function of sequence length.
    
    Args:
        model: Trained backtracking probe model
        scaler: Feature scaler used for the model
        activations: List of activation tensors
        is_backtracking: List of boolean flags indicating if next chunk is backtracking
        metadata: Metadata for each chunk
        layer: Layer number for the plot title
        test_size: Proportion of data used for testing
    """
    # Convert activations to numpy array
    X = torch.stack(activations).cpu().numpy()
    y = np.array(is_backtracking, dtype=int)
    
    # Split data to use only test set (to avoid data leakage)
    _, X_test, _, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    # Get corresponding metadata for test set
    _, meta_test = train_test_split(metadata, test_size=test_size, random_state=42, stratify=y)
    
    # Standardize features
    X_test = scaler.transform(X_test)
    
    # Get probabilities from model
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_test)[:, 1]  # Probability of class 1 (backtracking)
    else:
        # For models without predict_proba, use decision function and sigmoid
        decision_values = model.decision_function(X_test)
        probs = 1 / (1 + np.exp(-decision_values))
    
    # Extract sequence lengths (use chunk_idx as proxy for position in sequence)
    chunk_indices = [meta.get("chunk_idx", 0) for meta in meta_test]
    
    # Get problem IDs to group by problem
    problem_ids = [meta.get("problem_id", "") for meta in meta_test]
    
    # Create DataFrame for analysis
    df = pd.DataFrame({
        "problem_id": problem_ids,
        "chunk_idx": chunk_indices,
        "backtracking_prob": probs,
        "is_backtracking": y_test
    })
    
    # Calculate average probability by chunk index
    avg_probs = df.groupby("chunk_idx")["backtracking_prob"].mean()
    std_probs = df.groupby("chunk_idx")["backtracking_prob"].std()
    counts = df.groupby("chunk_idx").size()
    
    # Filter to positions with enough samples (at least 5)
    valid_positions = counts[counts >= 5].index
    avg_probs = avg_probs[valid_positions]
    std_probs = std_probs[valid_positions]
    counts = counts[valid_positions]
    
    # Plot average probability by position
    plt.figure(figsize=(12, 6))
    
    # Plot line with error bands
    plt.plot(avg_probs.index, avg_probs.values, 'b-', label='P(backtracking)')
    plt.fill_between(
        avg_probs.index, 
        avg_probs.values - std_probs.values / np.sqrt(counts), 
        avg_probs.values + std_probs.values / np.sqrt(counts),
        alpha=0.3, color='b'
    )
    
    # Add sample size as text
    for idx, count in zip(counts.index, counts.values):
        plt.text(idx, avg_probs[idx] + 0.02, f"n={count}", fontsize=8, ha='center')
    
    # Add actual backtracking rate for comparison
    actual_rate = df.groupby("chunk_idx")["is_backtracking"].mean()
    actual_rate = actual_rate[valid_positions]
    plt.plot(actual_rate.index, actual_rate.values, 'r--', label='Actual backtracking rate')
    
    plt.xlabel('Chunk Position in Sequence')
    plt.ylabel('Probability of Backtracking')
    plt.title(f'Backtracking Probability by Sequence Position (Layer {layer})')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Add horizontal line at 0.5 probability
    plt.axhline(y=0.5, color='gray', linestyle='-', alpha=0.5)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(output_dir / f"backtracking_probability_by_position_layer{layer}.png", dpi=300)
    plt.close()
    
    # Also plot by problem length
    # Calculate max chunk index per problem
    problem_lengths = df.groupby("problem_id")["chunk_idx"].max()
    
    # Add problem length to DataFrame
    df["problem_length"] = df["problem_id"].map(lambda x: problem_lengths.get(x, 0))
    
    # Group problems by length ranges
    df["length_bin"] = pd.cut(df["problem_length"], bins=[0, 5, 10, 15, 20, 30, 50, 100], 
                             labels=["1-5", "6-10", "11-15", "16-20", "21-30", "31-50", "51+"])
    
    # Calculate average probability by relative position and length bin
    df["relative_position"] = df["chunk_idx"] / df["problem_length"]
    df["position_bin"] = pd.cut(df["relative_position"], bins=10)
    
    # Group by length bin and position bin
    grouped = df.groupby(["length_bin", "position_bin"])["backtracking_prob"].mean().reset_index()
    
    # Plot heatmap
    plt.figure(figsize=(12, 8))
    pivot = grouped.pivot(index="length_bin", columns="position_bin", values="backtracking_prob")
    sns.heatmap(pivot, cmap="YlOrRd", annot=True, fmt=".2f", cbar_kws={'label': 'P(backtracking)'})
    plt.title(f'Backtracking Probability by Relative Position and Problem Length (Layer {layer})')
    plt.xlabel('Relative Position in Solution (deciles)')
    plt.ylabel('Problem Length (chunks)')
    plt.tight_layout()
    plt.savefig(output_dir / f"backtracking_probability_heatmap_layer{layer}.png", dpi=300)
    plt.close()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train probes to predict chunk categories")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", help="Model name")
    parser.add_argument("--layer", type=int, default=31, help="Layer to extract activations from")
    parser.add_argument("--max_problems", type=int, default=100, help="Maximum number of problems to process")
    parser.add_argument("--model_type", type=str, default="mlp", choices=["logistic", "mlp"], help="Type of model to use")
    parser.add_argument("--balance_method", type=str, default="undersample", choices=["undersample", "oversample", "class_weight"], help="Method to balance the dataset for backtracking probe")
    args = parser.parse_args()
    
    # Set global layer variable for saving files
    global layer
    layer = args.layer
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model)
    
    # Get problem directories
    problem_dirs = get_problem_dirs(cots_dir)
    
    # Extract chunk pairs with activations
    activations, next_categories, is_backtracking, chunk_metadata = extract_chunk_pairs_with_activations(
        problem_dirs, model, tokenizer, args.layer, args.max_problems
    )
    
    if not activations:
        print("No activations extracted. Exiting.")
        return
    
    print(f"Extracted {len(activations)} chunk pairs")
    
    # Print category distribution
    category_counts = Counter(next_categories)
    print("Category distribution:")
    for category, count in category_counts.most_common():
        print(f"  {category}: {count} ({count/len(next_categories)*100:.1f}%)")
    
    # Extract problem IDs for train/test splitting
    problem_ids = [meta["problem_id"] for meta in chunk_metadata]
    
    # Train category probe
    category_model, label_encoder, category_accuracy, category_metrics = train_category_probe(
        activations, next_categories, problem_ids, model_type=args.model_type
    )
    
    # Train backtracking probe with balanced dataset
    backtracking_model, backtracking_scaler, backtracking_accuracy, backtracking_metrics = train_backtracking_probe(
        activations, is_backtracking, chunk_metadata, model_type=args.model_type, balance_method=args.balance_method
    )
    
    # Plot feature importance for category prediction (if using logistic regression)
    if args.model_type == "logistic":
        plot_category_importance(category_model, label_encoder, args.layer)
    
    # Plot backtracking probability by sequence length
    plot_backtracking_probability_by_length(
        backtracking_model, 
        backtracking_scaler, 
        activations, 
        is_backtracking, 
        chunk_metadata, 
        args.layer
    )
    
    # Save results
    results = {
        "layer": args.layer,
        "num_examples": len(activations),
        "category_distribution": {k: v for k, v in category_counts.items()},
        "category_probe": {
            "accuracy": category_accuracy,
            "metrics": category_metrics
        },
        "backtracking_probe": {
            "accuracy": backtracking_accuracy,
            "metrics": backtracking_metrics,
            "balance_method": args.balance_method
        }
    }
    
    with open(output_dir / f"probe_results_layer{args.layer}.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()