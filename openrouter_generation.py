import os
import json
import random
import numpy as np
import torch
import asyncio
import re
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI
from datasets import load_dataset
from utils import extract_boxed_answers, check_answer

# Load environment variables
load_dotenv()

# Get OpenRouter API key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in .env file")

# Constants
MODEL_NAME = "deepseek/deepseek-r1-distill-qwen-14b"

# Set up argument parser
import argparse
parser = argparse.ArgumentParser(description='Generate chain-of-thought solutions with rollouts')
parser.add_argument('-i', '--input_file', type=str, default=None, help='Input JSON file with reasoning problems (optional)')
parser.add_argument('-o', '--output_dir', type=str, default='math_rollouts', help='Directory to save results')
parser.add_argument('-np', '--num_problems', type=int, default=10, help='Number of problems to sample')
parser.add_argument('-nr', '--num_rollouts', type=int, default=100, help='Number of rollouts per chunk')
parser.add_argument('-t', '--temperature', type=float, default=0.7, help='Temperature for rollout generation')
parser.add_argument('-tp', '--top_p', type=float, default=0.92, help='Top-p sampling parameter')
parser.add_argument('-mt', '--max_tokens', type=int, default=8192, help='Maximum number of tokens for generation')
parser.add_argument('-s', '--seed', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument('-f', '--force', action='store_true', help='Force regeneration even if solutions exist')
parser.add_argument('-c', '--concurrency', type=int, default=5, help='Number of concurrent API requests')
parser.add_argument('-ty', '--type', type=str, default=None, help='Problem type filter')
parser.add_argument('-l', '--level', type=str, default=None, help='Problem level filter')
parser.add_argument('-sp', '--split', type=str, default='train', choices=['train', 'test'], help='Dataset split to use')
args = parser.parse_args()

# Create output directory
output_dir = Path(args.output_dir)
output_dir.mkdir(exist_ok=True, parents=True)

# Set random seed for reproducibility
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

def load_math_problems(file_path: Optional[str] = None, problem_type: Optional[str] = None, level: Optional[str] = None, num_problems: Optional[int] = None, split: str = 'train') -> List[Tuple[int, Dict]]:
    """
    Load problems from the MATH dataset with optional filtering.
    
    Args:
        file_path: Path to the JSON file containing problems (if None, load from HF dataset)
        problem_type: Type of problems to filter by (if None, use all types)
        level: Level of problems to filter by (if None, use all levels)
        num_problems: Number of problems to sample (if None, use all problems)
        split: Dataset split to use ('train' or 'test')
        
    Returns:
        List of problems with their original indices
    """
    try:
        if file_path:
            # Load from JSON file
            with open(file_path, 'r', encoding='utf-8') as f:
                all_problems = json.load(f)
            
            # Add original indices to problems
            indexed_problems = [(i, problem) for i, problem in enumerate(all_problems)]
        else:
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
    
    # Split by sentences or logical breaks
    chunks = []
    current_chunk = ""
    
    # Split by sentences, keeping some context
    sentences = re.split(r'(?<=[.!?])\s+', solution_text)
    
    for i, sentence in enumerate(sentences):
        if not sentence.strip():
            continue
            
        # Start a new chunk every few sentences
        if i % 3 == 0 and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += " " + sentence
    
    # Add the last chunk if not empty
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

async def generate_base_solution(client: OpenAI, problem: Dict, temperature: float = 0.0) -> Dict:
    """
    Generate a base solution for a problem using OpenRouter API.
    
    Args:
        client: OpenAI client
        problem: Problem dictionary
        temperature: Temperature for generation
        
    Returns:
        Dictionary with the generated solution
    """
    # Create prompt similar to generate_cots_math.py
    prompt = f"Solve this math problem step by step. Put your final answer in \\boxed{{}}. Problem: {problem['problem']} Solution: \n<think>\n"
    
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                extra_body={"include_reasoning": True}
            )
            
            solution_text = completion.choices[0].message.content
            
            # Try to get reasoning tokens if available
            reasoning = None
            try:
                reasoning = completion.choices[0].message.reasoning
            except AttributeError:
                print("Reasoning tokens not available in response")
            
            # Create full CoT with prompt, reasoning, and solution
            full_cot = f"{prompt}{solution_text}\n</think>"
            
            # Create a version with reasoning if available
            full_cot_with_reasoning = None
            if reasoning:
                full_cot_with_reasoning = f"Problem: {problem['problem']}\nSolution: \n<think>\n{reasoning}\n</think>\n{solution_text}"
            
            # Extract answer and check correctness from the full CoT with reasoning if available
            # Otherwise, use the regular solution text
            source_text = full_cot_with_reasoning if full_cot_with_reasoning else solution_text
            extracted_answers = extract_boxed_answers(source_text)
            answer = extracted_answers[0] if extracted_answers else ""
            is_correct = False
            
            if problem.get('gt_answer') and answer:
                is_correct = check_answer(answer, problem['gt_answer'])
            
            return {
                "prompt": prompt,
                "solution": solution_text,
                "reasoning": reasoning,
                "full_cot": full_cot,
                "full_cot_with_reasoning": full_cot_with_reasoning,
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

async def generate_rollout(client: OpenAI, problem: Dict, chunk_text: str, full_cot_prefix: str, temperature: float = 0.7) -> Dict:
    """
    Generate a rollout from a specific chunk.
    
    Args:
        client: OpenAI client
        problem: Problem dictionary
        chunk_text: Text of the current chunk
        full_cot_prefix: Full CoT text up to and including the current chunk
        temperature: Temperature for generation
        
    Returns:
        Dictionary with the rollout result
    """
    # Create prompt with the full CoT prefix
    prompt = f"Solve this math problem step by step. Put your final answer in \\boxed{{}}. Problem: {problem['problem']} Solution: \n<think>\n{full_cot_prefix}"
    
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                extra_body={"include_reasoning": True}
            )
            
            rollout_text = completion.choices[0].message.content
            
            # Try to get reasoning tokens if available
            reasoning = None
            try:
                reasoning = completion.choices[0].message.reasoning
            except AttributeError:
                print("Reasoning tokens not available in response")
            
            # Create full CoT with prompt and rollout
            full_cot = f"{prompt}{rollout_text}\n</think>"
            
            # Create a version with reasoning if available
            full_cot_with_reasoning = None
            if reasoning:
                full_cot_with_reasoning = f"Problem: {problem['problem']}\nSolution: \n<think>\n{full_cot_prefix}{reasoning}\n</think>\n{rollout_text}"
            
            # Extract answer and check correctness from the full CoT with reasoning if available
            # Otherwise, use the combined prefix + rollout text
            source_text = full_cot_with_reasoning if full_cot_with_reasoning else (full_cot_prefix + rollout_text)
            extracted_answers = extract_boxed_answers(source_text)
            answer = extracted_answers[0] if extracted_answers else ""
            is_correct = False
            
            if problem.get('gt_answer') and answer:
                is_correct = check_answer(answer, problem['gt_answer'])
            
            return {
                "chunk": chunk_text,
                "full_cot_prefix": full_cot_prefix,
                "rollout": rollout_text,
                "reasoning": reasoning,
                "full_solution": full_cot_prefix + rollout_text,
                "full_cot": full_cot,
                "full_cot_with_reasoning": full_cot_with_reasoning,
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
                    "chunk": chunk_text,
                    "full_cot_prefix": full_cot_prefix,
                    "rollout": f"Error: {str(e)}",
                    "temperature": temperature,
                    "top_p": args.top_p,
                    "error": str(e),
                    "is_correct": False
                }

async def generate_rollout_with_semaphore(client: OpenAI, problem: Dict, chunk: str, full_cot_prefix: str, temperature: float, semaphore: asyncio.Semaphore) -> Dict:
    """Helper function to apply semaphore to rollout generation"""
    async with semaphore:
        return await generate_rollout(client, problem, chunk, full_cot_prefix, temperature)

async def process_problem(problem_idx: int, problem: Dict, client: OpenAI, semaphore: asyncio.Semaphore) -> None:
    """
    Process a single problem: generate base solution and rollouts.
    
    Args:
        problem_idx: Index of the problem
        problem: Problem dictionary
        client: OpenAI client
        semaphore: Semaphore for limiting concurrent requests
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
    if base_solution_file.exists() and not args.force:
        with open(base_solution_file, 'r', encoding='utf-8') as f:
            base_solution = json.load(f)
    else:
        # Generate base solution
        async with semaphore:
            base_solution = await generate_base_solution(client, problem, args.temperature)
            
            # Save base solution
            with open(base_solution_file, 'w', encoding='utf-8') as f:
                json.dump(base_solution, f, indent=2)
    
    # Get the source text for chunking
    if base_solution.get("full_cot_with_reasoning"):
        source_text = base_solution["full_cot_with_reasoning"]
        print(f"Problem {problem_idx}: Using full CoT with reasoning for chunking")
    else:
        source_text = base_solution["solution"]
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
    
    # Build cumulative chunks for proper continuation
    cumulative_chunks = []
    current_cumulative = ""
    for chunk in chunks:
        current_cumulative += chunk + " "
        cumulative_chunks.append(current_cumulative.strip())
    
    # Process each chunk with its own progress bar
    chunk_tasks = []
    for chunk_idx, (chunk, full_prefix) in enumerate(zip(chunks, cumulative_chunks)):
        chunk_dir = problem_dir / f"chunk_{chunk_idx}"
        chunk_dir.mkdir(exist_ok=True, parents=True)
        
        # Check if solutions already exist
        solutions_file = chunk_dir / "solutions.json"
        if solutions_file.exists() and not args.force:
            with open(solutions_file, 'r', encoding='utf-8') as f:
                existing_solutions = json.load(f)
                num_existing = len(existing_solutions)
                if num_existing >= args.num_rollouts:
                    print(f"Problem {problem_idx}, Chunk {chunk_idx}: All {num_existing} rollouts already exist")
                    continue
                rollouts_to_generate = args.num_rollouts - num_existing
                print(f"Problem {problem_idx}, Chunk {chunk_idx}: Generating {rollouts_to_generate} additional rollouts")
        else:
            existing_solutions = []
            rollouts_to_generate = args.num_rollouts
            print(f"Problem {problem_idx}, Chunk {chunk_idx}: Generating {rollouts_to_generate} rollouts")
        
        # Create a task for processing this chunk
        chunk_task = asyncio.create_task(
            process_chunk(problem_idx, chunk_idx, problem, chunk, full_prefix, client, semaphore, 
                         existing_solutions, rollouts_to_generate)
        )
        chunk_tasks.append(chunk_task)
    
    # Wait for all chunk tasks to complete
    for f in tqdm(asyncio.as_completed(chunk_tasks), total=len(chunk_tasks), 
                 desc=f"Problem {problem_idx} chunks"):
        await f

async def process_chunk(problem_idx: int, chunk_idx: int, problem: Dict, chunk: str, full_prefix: str, 
                       client: OpenAI, semaphore: asyncio.Semaphore, 
                       existing_solutions: List[Dict], rollouts_to_generate: int) -> None:
    """
    Process a single chunk: generate rollouts.
    
    Args:
        problem_idx: Index of the problem
        chunk_idx: Index of the chunk
        problem: Problem dictionary
        chunk: Current chunk text
        full_prefix: Full text up to and including current chunk
        client: OpenAI client
        semaphore: Semaphore for limiting concurrent requests
        existing_solutions: Existing solutions for this chunk
        rollouts_to_generate: Number of rollouts to generate
    """
    chunk_dir = output_dir / f"problem_{problem_idx}" / f"chunk_{chunk_idx}"
    solutions_file = chunk_dir / "solutions.json"
    
    # Generate rollouts with parallelization at the rollout level
    rollout_tasks = []
    for i in range(rollouts_to_generate):
        rollout_task = asyncio.create_task(
            generate_rollout_with_semaphore(client, problem, chunk, full_prefix, args.temperature, semaphore)
        )
        rollout_tasks.append(rollout_task)
    
    # Wait for all rollouts to complete with progress bar
    rollouts = []
    for f in tqdm(asyncio.as_completed(rollout_tasks), 
                 total=len(rollout_tasks), 
                 desc=f"Problem {problem_idx}, Chunk {chunk_idx} rollouts"):
        result = await f
        rollouts.append(result)
    
    # Combine with existing solutions
    all_solutions = existing_solutions + rollouts
    
    # Save solutions
    with open(solutions_file, 'w', encoding='utf-8') as f:
        json.dump(all_solutions, f, indent=2)

async def main():
    """Main function to run the script."""
    # Load problems
    problems = load_math_problems(
        file_path=args.input_file, 
        problem_type=args.type, 
        level=args.level, 
        num_problems=args.num_problems,
        split=args.split
    )
    
    if not problems:
        print(f"No problems loaded. Exiting.")
        exit(1)

    print(f"Loaded {len(problems)} problems.")
    
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(args.concurrency)
    
    # Create OpenAI client for OpenRouter
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
    
    # Process problems
    tasks = []
    for problem_idx, problem in problems:
        task = process_problem(problem_idx, problem, client, semaphore)
        tasks.append(task)
    
    # Use tqdm to show progress
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing problems"):
        await f

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
