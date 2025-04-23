import os
import json
import random
import numpy as np
import torch
import asyncio
import httpx
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv
from datasets import load_dataset
from utils import extract_boxed_answers, check_answer

# Load environment variables
load_dotenv()

# Get OpenRouter API key
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in .env file")

# Set up argument parser
import argparse
parser = argparse.ArgumentParser(description='Generate chain-of-thought solutions with rollouts')
parser.add_argument('-m', '--model', type=str, default="deepseek/deepseek-r1-distill-qwen-14b", help='Model to use')
parser.add_argument('-o', '--output_dir', type=str, default='math_rollouts', help='Directory to save results')
parser.add_argument('-np', '--num_problems', type=int, default=100, help='Number of problems to sample')
parser.add_argument('-nr', '--num_rollouts', type=int, default=100, help='Number of rollouts per chunk')
parser.add_argument('-t', '--temperature', type=float, default=0.6, help='Temperature for rollout generation')
parser.add_argument('-tp', '--top_p', type=float, default=0.92, help='Top-p sampling parameter')
parser.add_argument('-mt', '--max_tokens', type=int, default=16384, help='Maximum number of tokens for generation')
parser.add_argument('-mc', '--max_chunks', type=int, default=150, help='Maximum number of chunks to process')
parser.add_argument('-s', '--seed', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument('-f', '--force', action='store_true', help='Force regeneration even if solutions exist')
parser.add_argument('-ep', '--exclude_problems', type=str, default=None, help='Comma-separated list of problem IDs to exclude')
parser.add_argument('-ip', '--include_problems', type=str, default=None, help='Comma-separated list of problem IDs to include')
parser.add_argument('-ty', '--type', type=str, default=None, help='Problem type filter')
parser.add_argument('-l', '--level', type=str, default="Level 5", help='Problem level filter')
parser.add_argument('-sp', '--split', type=str, default='train', choices=['train', 'test'], help='Dataset split to use')
parser.add_argument('-p', '--provider', type=str, default="Novita", choices=['Novita', 'Together'], help='Provider to use')
args = parser.parse_args()

# Create output directory
output_dir = Path(args.output_dir) / args.model.split("/")[-1] / f"temperature_{str(args.temperature)}"
output_dir.mkdir(exist_ok=True, parents=True)

# Set random seed for reproducibility
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

def load_math_problems(problem_type: Optional[str] = None, level: Optional[str] = None, num_problems: Optional[int] = None, split: str = 'train') -> List[Tuple[int, Dict]]:
    """
    Load problems from the MATH dataset with optional filtering.
    
    Args:
        problem_type: Type of problems to filter by (if None, use all types)
        level: Level of problems to filter by (if None, use all levels)
        num_problems: Number of problems to sample (if None, use all problems)
        split: Dataset split to use ('train' or 'test')
        
    Returns:
        List of problems with their original indices
    """
    try:
        # Load from Hugging Face dataset
        math_dataset = load_dataset("fdyrd/math")
        dataset_split = math_dataset[split]
        
        # Add original indices to problems
        indexed_problems = [(i, {
            'problem': item['problem'],
            'level': item['level'],
            'type': item['type'],
            'gt_solution': item['solution']
        }) for i, item in enumerate(dataset_split)]
        
        # Extract ground truth answers
        for i, problem in indexed_problems:
            gt_boxed_answers = extract_boxed_answers(problem['gt_solution'])
            gt_answer = gt_boxed_answers[0] if gt_boxed_answers else ""
            problem['gt_answer'] = gt_answer
        
        # Filter by type if specified
        if problem_type is not None:
            indexed_problems = [(i, problem) for i, problem in indexed_problems if problem.get('type') == problem_type]
        
        # Filter by level if specified
        if level is not None:
            indexed_problems = [(i, problem) for i, problem in indexed_problems if problem.get('level') == level]
            
        # Sample if needed
        if num_problems is not None and num_problems < len(indexed_problems):
            indexed_problems = random.sample(indexed_problems, num_problems)
            
        if level:
            print(f"Filtered to level: {level}")
        if problem_type:
            print(f"Filtered to type: {problem_type}")
            
        return indexed_problems
    except Exception as e:
        print(f"Error loading problems: {e}")
        return []

def split_solution_into_chunks(solution_text: str) -> List[str]:
    """
    Split a solution into chunks for rollout generation.
    
    Args:
        solution_text: The full solution text
        
    Returns:
        List of chunks
    """
    # First, remove the prompt part if present
    if "<think>" in solution_text:
        solution_text = solution_text.split("<think>")[1].strip()
    
    # Remove the closing tag if present
    if "</think>" in solution_text:
        solution_text = solution_text.split("</think>")[0].strip()
    
    # Define patterns for chunk boundaries
    sentence_ending_tokens = [".", "?", "!"]
    paragraph_ending_patterns = ["\n\n", "\r\n\r\n"]
    
    # Split the text into chunks
    chunks = []
    current_chunk = ""
    
    # Process the text character by character
    i = 0
    while i < len(solution_text):
        current_chunk += solution_text[i]
        
        # Check for paragraph endings
        is_paragraph_end = False
        for pattern in paragraph_ending_patterns:
            if i + len(pattern) <= len(solution_text) and solution_text[i:i+len(pattern)] == pattern:
                is_paragraph_end = True
                break
        
        # Check for sentence endings followed by space or newline
        is_sentence_end = False
        if i < len(solution_text) - 1 and solution_text[i] in sentence_ending_tokens:
            next_char = solution_text[i+1]
            if next_char == " " or next_char == "\n":
                is_sentence_end = True
        
        # If we found a boundary, add the chunk and reset
        if is_paragraph_end or is_sentence_end:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
                current_chunk = ""
        
        i += 1
    
    # Add the last chunk if not empty
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

async def make_openrouter_request(prompt: str, temperature: float, top_p: float, max_tokens: int) -> Dict:
    """Make a direct HTTP request to OpenRouter API."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost:3000"
    }
    
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "include_reasoning": True,
        "provider": {
            "order": [args.provider],
            "ignore": ["Together" if args.provider == "Novita" else "Novita"],
            "allow_fallbacks": False
        }
    }
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(OPENROUTER_API_URL, headers=headers, json=payload)
        return response.json()

async def generate_base_solution(problem: Dict, temperature: float = 0.0) -> Dict:
    """
    Generate a base solution for a problem using OpenRouter API.
    
    Args:
        problem: Problem dictionary
        temperature: Temperature for generation
        
    Returns:
        Dictionary with the generated solution
    """
    # Create prompt similar to generate_cots_math.py
    prompt = f"Solve this math problem step by step. You MUST put your final answer in \\boxed{{}}. Problem: {problem['problem']} Solution: \n<think>\n"
    
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            response = await make_openrouter_request(prompt, temperature, args.top_p, args.max_tokens)
            
            solution_text = response['choices'][0]['message']['content']
            
            # Try to get reasoning tokens if available
            reasoning = None
            try:
                reasoning = response['choices'][0]['message']['reasoning']
            except (KeyError, TypeError):
                print("Reasoning tokens not available in response")
            
            # Create full CoT with prompt, reasoning, and solution
            full_cot = f"{prompt}{solution_text}"
            
            if reasoning:
                full_cot = f"{prompt}{reasoning}\n</think>\n{solution_text}"
            
            # Extract answer and check correctness
            extracted_answers = extract_boxed_answers(solution_text)
            answer = extracted_answers[0] if extracted_answers else ""
            is_correct = False
            
            if problem.get('gt_answer') and answer:
                is_correct = check_answer(answer, problem['gt_answer'])
            
            return {
                "prompt": prompt,
                "solution": solution_text,
                "reasoning": reasoning,
                "full_cot": full_cot,
                "temperature": temperature,
                "top_p": args.top_p,
                "answer": answer,
                "is_correct": is_correct
            }
        except Exception as e:
            print(f"API error: {e}")
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                print(f"Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            else:
                return {
                    "prompt": prompt,
                    "solution": f"Error: {str(e)}",
                    "temperature": temperature,
                    "top_p": args.top_p,
                    "error": str(e)
                }

async def generate_rollout(problem: Dict, chunk_text: str, full_cot_prefix: str, temperature: float = 0.7) -> Dict:
    """
    Generate a rollout by removing a specific chunk and regenerating from that point.
    
    Args:
        problem: Problem dictionary
        chunk_text: Text of the current chunk to remove
        full_cot_prefix: Full CoT text up to and including the current chunk
        temperature: Temperature for generation
        
    Returns:
        Dictionary with the rollout result
    """
    # Remove the current chunk from the prefix to see how it gets regenerated
    prefix_without_chunk = full_cot_prefix.replace(chunk_text, "").strip()
    
    # Create prompt with the prefix without the current chunk
    prompt = f"Solve this math problem step by step. You MUST put your final answer in \\boxed{{}}. Problem: {problem['problem']} Solution: \n<think>\n{prefix_without_chunk}"
    
    separator = ""
    if len(prefix_without_chunk) > 0:
        separator = " "
    
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            response = await make_openrouter_request(prompt, temperature, args.top_p, args.max_tokens)
            rollout_text = response['choices'][0]['message']['content']
            
            # Try to get reasoning tokens if available
            reasoning = None
            try:
                reasoning = response['choices'][0]['message']['reasoning']
            except (KeyError, TypeError):
                pass  # Silently continue if reasoning not available
            
            # Create full CoT with prompt and rollout
            full_cot = f"{prompt}{separator}{rollout_text}\n</think>"
            
            if reasoning:
                full_cot = f"{prompt}{separator}{reasoning}\n</think>\n{rollout_text}"
            
            # Extract answer and check correctness
            source_text = prefix_without_chunk + rollout_text
            extracted_answers = extract_boxed_answers(source_text)
            answer = extracted_answers[0] if extracted_answers else ""
            is_correct = False
            
            if problem.get('gt_answer') and answer:
                is_correct = check_answer(answer, problem['gt_answer'])
            
            return {
                "chunk_removed": chunk_text,
                "prefix_without_chunk": prefix_without_chunk,
                "rollout": rollout_text,
                "reasoning": reasoning,
                "full_solution": prefix_without_chunk + separator + rollout_text,
                "full_cot": full_cot,
                "temperature": temperature,
                "top_p": args.top_p,
                "answer": answer,
                "is_correct": is_correct
            }
        except Exception as e:
            print(f"API error: {e}")
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                print(f"Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            else:
                return {
                    "chunk_removed": chunk_text,
                    "prefix_without_chunk": prefix_without_chunk,
                    "rollout": f"Error: {str(e)}",
                    "temperature": temperature,
                    "top_p": args.top_p,
                    "error": str(e),
                    "is_correct": False
                }

async def process_problem(problem_idx: int, problem: Dict) -> None:
    """
    Process a single problem: generate base solution and rollouts.
    
    Args:
        problem_idx: Index of the problem
        problem: Problem dictionary
    """
    problem_dir = output_dir / f"problem_{problem_idx}"
    problem_dir.mkdir(exist_ok=True, parents=True)
    
    # Save problem
    problem_file = problem_dir / "problem.json"
    if not problem_file.exists() or args.force:
        with open(problem_file, 'w', encoding='utf-8') as f:
            json.dump(problem, f, indent=2)
    
    # Check if base solution already exists
    base_solution_file = problem_dir / "base_solution.json"
    base_solution = None
    if base_solution_file.exists() and not args.force:
        with open(base_solution_file, 'r', encoding='utf-8') as f:
            base_solution = json.load(f)
            print(f"Problem {problem_idx}: Loaded existing base solution")
    
    # Generate base solution if needed
    if base_solution is None:
        print(f"Problem {problem_idx}: Generating base solution")
        base_solution = await generate_base_solution(problem, args.temperature)
            
        if "is_correct" not in base_solution or not base_solution["is_correct"]:
            print(base_solution["solution"])
            print(f"Problem {problem_idx}: Base solution is incorrect or has error. Will not generate rollouts.")
            return
        
        # Save base solution
        with open(base_solution_file, 'w', encoding='utf-8') as f:
            json.dump(base_solution, f, indent=2)
    
    # Get the source text for chunking
    source_text = base_solution["full_cot"]
    print(f"Problem {problem_idx}: Using solution text for chunking")
    
    # Extract the solution part for chunking
    if "<think>" in source_text:
        solution_text = source_text.split("<think>")[1].strip()
        if "</think>" in solution_text:
            solution_text = solution_text.split("</think>")[0].strip()
    else:
        solution_text = source_text
    
    # Split into chunks
    chunks = split_solution_into_chunks(solution_text)
    print(f"Problem {problem_idx}: Split into {len(chunks)} chunks")
    
    if len(chunks) > args.max_chunks:
        print(f"Problem {problem_idx}: Too many chunks. Will not generate rollouts.")
        return
    
    # Save chunks to a separate file
    chunks_file = problem_dir / "chunks.json"
    with open(chunks_file, 'w', encoding='utf-8') as f:
        json.dump({
            "source_text": source_text,
            "solution_text": solution_text,
            "chunks": chunks
        }, f, indent=2)
    print(f"Problem {problem_idx}: Saved chunks to {chunks_file}")
    
    # Build cumulative chunks for proper continuation
    cumulative_chunks = []
    current_cumulative = ""
    for chunk in chunks:
        current_cumulative += chunk + " "
        cumulative_chunks.append(current_cumulative.strip())
    
    # Process each chunk
    for chunk_idx, (chunk, full_prefix) in enumerate(zip(chunks, cumulative_chunks)):
        chunk_dir = problem_dir / f"chunk_{chunk_idx}"
        chunk_dir.mkdir(exist_ok=True, parents=True)
        
        # Check if solutions already exist
        solutions_file = chunk_dir / "solutions.json"
        existing_solutions = []
        valid_existing_solutions = []
        
        if solutions_file.exists() and not args.force:
            with open(solutions_file, 'r', encoding='utf-8') as f:
                existing_solutions = json.load(f)
                # Filter for valid solutions (has answer and no error)
                valid_existing_solutions = [
                    sol for sol in existing_solutions 
                    if sol.get("answer") and len(sol.get("answer", "")) > 0 and "error" not in sol
                ]
                
                num_existing = len(existing_solutions)
                num_valid = len(valid_existing_solutions)
                
                if num_valid >= args.num_rollouts:
                    print(f"Problem {problem_idx}, Chunk {chunk_idx}: All {num_valid} valid rollouts already exist")
                    continue
                    
                rollouts_to_generate = args.num_rollouts - num_valid
                print(f"Problem {problem_idx}, Chunk {chunk_idx}: Found {num_existing} existing rollouts, but only {num_valid} are valid")
                print(f"Problem {problem_idx}, Chunk {chunk_idx}: Generating {rollouts_to_generate} additional rollouts")
        else:
            rollouts_to_generate = args.num_rollouts
            print(f"Problem {problem_idx}, Chunk {chunk_idx}: Generating {rollouts_to_generate} rollouts")
        
        # Create all rollout tasks at once
        tasks = [generate_rollout(problem, chunk, full_prefix, args.temperature) for _ in range(rollouts_to_generate)]
        
        # Execute all tasks concurrently
        print(f"Problem {problem_idx}, Chunk {chunk_idx}: Executing {len(tasks)} rollout tasks concurrently")
        new_rollouts = await asyncio.gather(*tasks)
        
        # Filter for valid new rollouts
        valid_new_rollouts = [
            rollout for rollout in new_rollouts 
            if rollout.get("answer") and len(rollout.get("answer", "")) > 0 and "error" not in rollout
        ]
        
        # Combine valid existing solutions with valid new rollouts
        all_valid_solutions = valid_existing_solutions + valid_new_rollouts
        
        # Save only the valid solutions
        with open(solutions_file, 'w', encoding='utf-8') as f:
            json.dump(all_valid_solutions, f, indent=2)
        
        print(f"Problem {problem_idx}, Chunk {chunk_idx}: Saved {len(valid_new_rollouts)} new valid rollouts")
        print(f"Problem {problem_idx}, Chunk {chunk_idx}: Total valid rollouts: {len(all_valid_solutions)}")

async def main():
    """Main function to run the script."""
    # Load problems
    problems = load_math_problems(problem_type=args.type, level=args.level, num_problems=args.num_problems, split=args.split)
    
    if args.exclude_problems:
        exclude_problems = [int(id) for id in args.exclude_problems.split(",")]
        problems = [problem for problem in problems if problem[0] not in exclude_problems]
        
    if args.include_problems:
        include_problems = [int(id) for id in args.include_problems.split(",")]
        problems = [problem for problem in problems if problem[0] in include_problems]
    
    if not problems:
        print(f"No problems loaded. Exiting.")
        exit(1)

    print(f"Loaded {len(problems)} problems.")
    
    # Process problems one by one
    for problem_idx, problem in tqdm(problems, desc="Processing problems"):
        await process_problem(problem_idx, problem)

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
