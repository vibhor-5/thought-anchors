import torch
import random
import numpy as np
import os
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from typing import List, Dict, Tuple

model_choices = ['deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B']

# Set up argument parser
parser = argparse.ArgumentParser(description='Generate chain-of-thought solutions for reasoning problems')
parser.add_argument('-m', '--model', type=str, default='deepseek-ai/DeepSeek-R1-Distill-Llama-8B', choices=model_choices, help='Model to use for generation')
parser.add_argument('-ns', '--num_seeds', type=int, default=10, help='Number of random seeds to use')
parser.add_argument('-t', '--temperature', type=float, default=0.6, help='Temperature for generation')
parser.add_argument('-mt', '--max_new_tokens', type=int, default=4096, help='Maximum number of tokens to generate')
parser.add_argument('-tp', '--top_p', type=float, default=0.95, help='Top-p value for nucleus sampling')
parser.add_argument('-np', '--num_problems', type=int, default=1000, help='Number of problems to sample')
parser.add_argument('-od', '--output_dir', type=str, default='results', help='Directory to save results')
parser.add_argument('-s', '--seed', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument('-i', '--input_file', type=str, default='reasoning_problems.json', help='Input JSON file with reasoning problems')
parser.add_argument('-bs', '--batch_size', type=int, default=1, help='Batch size for generation')
parser.add_argument('-f', '--force', action='store_true', help='Force regeneration even if solutions exist')
parser.add_argument('-d', '--difficulty', type=str, default='Easy', choices=['Easy', 'Medium', 'Hard'], help='Difficulty level of problems to select')
args = parser.parse_args()

# Create output directory
output_dir = os.path.join(args.output_dir)
os.makedirs(output_dir, exist_ok=True)

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f'Got device: {device}')

# Disable gradient computation for inference
torch.set_grad_enabled(False)

# Set global random seed for reproducibility
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

def load_model(model_name: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load the model and tokenizer and move them to the appropriate device.
    
    Args:
        model_name: Name of the model to load
        
    Returns:
        The loaded model and tokenizer
    """
    print(f"Loading model: {model_name}")
    print("Loading HuggingFace model config...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Ensure padding token is set
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = tokenizer.eos_token = "</s>"
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device
    )
    
    # Get model config
    config = model.config
    
    print(f"📏 Model context length: {config.max_position_embeddings if hasattr(config, 'max_position_embeddings') else 'N/A'}")
    print(f"🧠 Model layers: {config.num_hidden_layers}")
    print(f"🔤 Vocabulary size: {config.vocab_size}")
    print(f"📊 Hidden dimension: {config.hidden_size}")
    print(f"🧩 Attention heads: {config.num_attention_heads}")
    print(f"🏷️ Model name: {model_name}")
    
    return model, tokenizer

def load_reasoning_problems(file_path: str, difficulty: str = None, num_problems: int = None) -> List[Dict]:
    """
    Load reasoning problems from a JSON file.
    
    Args:
        file_path: Path to the JSON file containing reasoning problems
        difficulty: Difficulty level to filter by (if None, use all difficulties)
        num_problems: Number of problems to sample (if None, use all problems)
        
    Returns:
        List of reasoning problems with their original indices
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            all_problems = json.load(f)
        
        # Add original indices to problems
        indexed_problems = [(i, problem) for i, problem in enumerate(all_problems)]
        
        # Filter by difficulty if specified
        if difficulty is not None:
            indexed_problems = [(i, problem) for i, problem in indexed_problems if problem.get('difficulty') == difficulty]
            
        # Sample if needed
        if num_problems is not None and num_problems < len(indexed_problems):
            indexed_problems = random.sample(indexed_problems, num_problems)
            
        return indexed_problems
    except Exception as e:
        print(f"Error loading reasoning problems: {e}")
        return []

def get_existing_solutions(problem_dir: str) -> List[Dict]:
    """
    Load existing solutions for a problem if they exist.
    
    Args:
        problem_dir: Directory containing problem solutions
        
    Returns:
        List of existing solutions or empty list if none exist
    """
    solutions_file = os.path.join(problem_dir, "solutions.json")
    if os.path.exists(solutions_file):
        try:
            with open(solutions_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading existing solutions: {e}")
    return []

def get_seeds_to_generate(existing_solutions: List[Dict], num_seeds: int) -> List[int]:
    """
    Determine which seeds need to be generated based on existing solutions.
    
    Args:
        existing_solutions: List of existing solutions
        num_seeds: Total number of seeds to generate
        
    Returns:
        List of seeds that need to be generated
    """
    existing_seeds = {solution["seed"] for solution in existing_solutions}
    return [seed for seed in range(num_seeds) if seed not in existing_seeds]

def generate_cot_for_problem_batch(
    model, 
    tokenizer,
    problems: List[Dict], 
    seeds: List[int],
    temperature: float = 0.6, 
    max_new_tokens: int = 2000, 
    top_p: float = 0.92
) -> List[Tuple[int, int, str, str]]:
    """
    Generate chain-of-thought solutions for multiple problems in batch.
    
    Args:
        model: The language model to use
        tokenizer: The tokenizer to use
        problems: List of reasoning problems
        seeds: List of seeds to use for generation
        temperature: Temperature for generation
        max_new_tokens: Maximum number of tokens to generate
        top_p: Top-p value for nucleus sampling
        
    Returns:
        List of tuples (problem_idx, seed, solution, prompt)
    """
    if not problems or not seeds:
        return []
    
    batch_size = min(len(problems) * len(seeds), args.batch_size)
    
    # Create all problem-seed pairs
    pairs = [(p_idx, seed) for p_idx, p in enumerate(problems) for seed in seeds]
    
    # Process in batches
    all_results = []
    for i in range(0, len(pairs), batch_size):
        batch_pairs = pairs[i:i+batch_size]
        
        # Set seeds for reproducibility
        for _, seed in batch_pairs:
            torch.manual_seed(seed)
            random.seed(seed)
        
        # Create prompts
        prompts = [
            f"Solve this problem step by step. End with 'DONE.' when you have a solution. \n\n Question: {problems[p_idx]['question']} \n\n Solution: \n<think>\n"
            for p_idx, _ in batch_pairs
        ]
        
        # Tokenize inputs
        inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
        
        # Generate solutions
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Decode outputs and extract generated parts
        for j, (p_idx, seed) in enumerate(batch_pairs):
            full_output = tokenizer.decode(outputs[j], skip_special_tokens=True)
            solution = full_output[len(prompts[j]):]
            all_results.append((p_idx, seed, solution, prompts[j]))
    
    return all_results

def generate_cot_for_problem(
    model, 
    tokenizer,
    problem: Dict, 
    seed: int,
    temperature: float = 0.6, 
    max_new_tokens: int = 2000, 
    top_p: float = 0.92
) -> Tuple[str, str]:
    """
    Generate a chain-of-thought solution for a reasoning problem.
    
    Args:
        model: The language model to use
        tokenizer: The tokenizer to use
        problem: The reasoning problem
        seed: Random seed for generation
        temperature: Temperature for generation
        max_new_tokens: Maximum number of tokens to generate
        top_p: Top-p value for nucleus sampling
        
    Returns:
        Tuple of (generated solution text, prompt)
    """
    # Set seed for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    
    # Create prompt
    prompt = f"Solve this problem step by step. End with 'DONE.' when you have a solution. \n\n Question: {problem['question']} \n\n Solution: \n<think>\n"
    
    # Generate solution
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Decode the output
    solution = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Extract only the generated part (remove the prompt)
    solution = solution[len(prompt):]
    
    return solution, prompt

def save_solutions_to_file(solutions: List[Dict], output_dir: str) -> None:
    """
    Save solutions to a JSON file.
    
    Args:
        problem_id: ID of the problem
        solutions: List of solutions
        output_dir: Directory to save the solutions
    """
    output_file = os.path.join(output_dir, f"solutions.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(solutions, f, indent=2)

def main():
    """Main function to run the script."""
    # Load reasoning problems
    problems = load_reasoning_problems(args.input_file, args.difficulty, args.num_problems)
    if not problems:
        print(f"No problems loaded from {args.input_file}. Exiting.")
        exit(1)

    print(f"Loaded {len(problems)} {args.difficulty} problems.")

    # Load model and tokenizer
    model, tokenizer = load_model(args.model)

    # Generate solutions for each problem
    for problem_idx, problem in tqdm(problems, desc="Iterating over problems"):
        problem_dir = os.path.join(output_dir, f"problem_{problem_idx}")
        os.makedirs(problem_dir, exist_ok=True)
        
        # Save problem
        with open(os.path.join(problem_dir, "problem.json"), 'w', encoding='utf-8') as f:
            json.dump(problem, f, indent=2)
        
        # Check if solutions already exist
        existing_solutions = get_existing_solutions(problem_dir)
        if existing_solutions and not args.force:
            print(f"Solutions already exist for problem {problem_idx}. Skipping. Use --force to regenerate.")
            continue
        
        # Get seeds that need to be generated
        seeds_to_generate = get_seeds_to_generate(existing_solutions, args.num_seeds)
        if not seeds_to_generate:
            print(f"All seeds already generated for problem {problem_idx}. Skipping.")
            continue
        
        # Generate solutions
        solutions = existing_solutions.copy()
        for seed_idx in tqdm(seeds_to_generate, desc="Generating solutions with seeds"):
            seed = args.seed + seed_idx
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            
            solution, prompt = generate_cot_for_problem(model, tokenizer, problem, seed, args.temperature, args.max_new_tokens, args.top_p)
            
            # Add solution to list
            solutions.append({"seed": seed_idx, "solution": prompt + solution})
            
            # Save solutions after each generation
            save_solutions_to_file(solutions, problem_dir)
            
        print(f"Generated {len(seeds_to_generate)} solutions for problem {problem_idx}")

    print("Done!")

if __name__ == "__main__":
    main()