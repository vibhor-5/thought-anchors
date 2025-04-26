import os
import json
import random
import numpy as np
import torch
import asyncio
import httpx
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from utils import extract_boxed_answers, check_answer, split_solution_into_chunks, load_math_problems

# Load environment variables
load_dotenv()

# Get API keys
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
NOVITA_API_KEY = os.getenv("NOVITA_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")

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
parser.add_argument('-mc', '--max_chunks', type=int, default=250, help='Maximum number of chunks to process')
parser.add_argument('-s', '--seed', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument('-f', '--force', action='store_true', help='Force regeneration even if solutions exist')
parser.add_argument('-ep', '--exclude_problems', type=str, default=None, help='Comma-separated list of problem IDs to exclude')
parser.add_argument('-ip', '--include_problems', type=str, default=None, help='Comma-separated list of problem IDs to include')
parser.add_argument('-ty', '--type', type=str, default=None, help='Problem type filter')
parser.add_argument('-l', '--level', type=str, default="Level 5", help='Problem level filter')
parser.add_argument('-sp', '--split', type=str, default='train', choices=['train', 'test'], help='Dataset split to use')
parser.add_argument('-p', '--provider', type=str, default="Novita", choices=['Novita', 'Together', 'Fireworks'], help='Provider to use')
parser.add_argument('-or', '--use_openrouter', default=False, action='store_true', help='Use OpenRouter API')
parser.add_argument('-fp', '--frequency_penalty', type=float, default=None, help='Frequency penalty parameter')
parser.add_argument('-pp', '--presence_penalty', type=float, default=None, help='Presence penalty parameter')
parser.add_argument('-rp', '--repetition_penalty', type=float, default=None, help='Repetition penalty parameter')
parser.add_argument('-tk', '--top_k', type=int, default=None, help='Top-k parameter')
parser.add_argument('-mp', '--min_p', type=float, default=None, help='Min-p parameter')
parser.add_argument('-sr', '--skip_recalculate', default=False, action='store_true', help='Skip recalculating accuracy for existing rollouts')
args = parser.parse_args()

# Create output directory
output_dir = Path(args.output_dir) / args.model.split("/")[-1] / f"temperature_{str(args.temperature)}_top_p_{str(args.top_p)}"
output_dir.mkdir(exist_ok=True, parents=True)

# Set random seed for reproducibility
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

async def make_api_request(prompt: str, temperature: float, top_p: float, max_tokens: int) -> Dict:
    """Make an API request to either Novita, Together, or Fireworks based on provider setting."""
    if args.provider == "Novita":
        # Novita API request
        headers = {
            "Authorization": f"Bearer {NOVITA_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": args.model,
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "n": 1,
            "stream": False
        }
        
        api_url = "https://api.novita.ai/v3/openai/completions"
        
    elif args.provider == "Together":
        # Together API request
        headers = {
            "Authorization": f"Bearer {TOGETHER_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "deepseek-ai/deepseek-r1-distill-qwen-14b",
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "n": 1,
            "stream": False
        }
        
        api_url = "https://api.together.xyz/v1/completions"
        
    elif args.provider == "Fireworks":
        # Fireworks API request
        headers = {
            "Authorization": f"Bearer {FIREWORKS_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "accounts/fireworks/models/deepseek-r1-distill-qwen-14b",
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "n": 1,
            "stream": False
        }
        
        api_url = "https://api.fireworks.ai/inference/v1/completions"
    
    # Add optional parameters for all APIs
    if args.frequency_penalty is not None:
        payload["frequency_penalty"] = args.frequency_penalty
    if args.presence_penalty is not None:
        payload["presence_penalty"] = args.presence_penalty
    if args.repetition_penalty is not None:
        payload["repetition_penalty"] = args.repetition_penalty
    if args.top_k is not None:
        payload["top_k"] = args.top_k
    if args.min_p is not None and args.provider != "Fireworks":  # Fireworks doesn't support min_p
        payload["min_p"] = args.min_p
    if args.seed is not None:
        payload["seed"] = args.seed
    
    # Implement exponential backoff for retries
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(api_url, headers=headers, json=payload, timeout=360)
                
                # Handle different error codes
                if response.status_code == 500:
                    print(f"Server error (500) on attempt {attempt+1}/{max_retries}. Retrying...")
                    await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                    continue
                    
                elif response.status_code == 429:
                    print(f"Rate limit (429) on attempt {attempt+1}/{max_retries}. Retrying...")
                    await asyncio.sleep(retry_delay * (2 ** attempt) + random.uniform(1, 3))  # Add jitter
                    continue
                    
                elif response.status_code != 200:
                    print(f"Error from API: {response.status_code} - {response.text}")
                    
                    # If it's the last attempt, return the error
                    if attempt == max_retries - 1:
                        return {"error": f"API error: {response.status_code}", "details": response.text}
                    
                    # Otherwise retry
                    await asyncio.sleep(retry_delay * (2 ** attempt))
                    continue
                
                # Success case
                result = response.json()
                
                if args.provider == "Novita" or args.provider == "Together":
                    return {
                        "text": result["choices"][0]["text"],
                        "finish_reason": result["choices"][0].get("finish_reason", ""),
                        "usage": result.get("usage", {})
                    }
                elif args.provider == "Fireworks":
                    return {
                        "text": result["choices"][0]["text"],
                        "finish_reason": result["choices"][0].get("finish_reason", ""),
                        "usage": result.get("usage", {})
                    }
                
        except Exception as e:
            print(f"Exception during API request (attempt {attempt+1}/{max_retries}): {e}")
            
            # If it's the last attempt, return the error
            if attempt == max_retries - 1:
                return {"error": f"Request exception: {str(e)}"}
            
            # Otherwise retry
            await asyncio.sleep(retry_delay * (2 ** attempt))
    
    # If we get here, all retries failed
    return {"error": "All API request attempts failed"}

async def generate_base_solution(problem: Dict, temperature: float = 0.0) -> Dict:
    """
    Generate a base solution for a problem using Novita API.
    
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
            response = await make_api_request(prompt, temperature, args.top_p, args.max_tokens)
            solution_text = response['text']
            
            # Extract answer and check correctness
            extracted_answers = extract_boxed_answers(solution_text)
            answer = extracted_answers[0] if extracted_answers else ""
            is_correct = False
            
            if problem.get('gt_answer') and answer:
                is_correct = check_answer(answer, problem['gt_answer'])
            
            return {
                "prompt": prompt,
                "solution": solution_text,
                "full_cot": prompt + solution_text,
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
    
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            response = await make_api_request(prompt, temperature, args.top_p, args.max_tokens)
            rollout_text = response['text']
            chunk_resampled = split_solution_into_chunks(rollout_text)[0]
            
            # Extract answer and check correctness
            extracted_answers = extract_boxed_answers(rollout_text)
            answer = extracted_answers[0] if extracted_answers else ""
            is_correct = False
            
            if problem.get('gt_answer') and answer:
                is_correct = check_answer(answer, problem['gt_answer'])
            
            return {
                "chunk_removed": chunk_text,
                "prefix_without_chunk": prefix_without_chunk,
                "chunk_resampled": chunk_resampled,
                "rollout": rollout_text,
                "full_cot": f"{prompt}{rollout_text}",
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
                    "error": str(e)
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
            
            # Recalculate accuracy for base solution if needed
            if not args.skip_recalculate and 'solution' in base_solution:
                extracted_answers = extract_boxed_answers(base_solution['solution'])
                answer = extracted_answers[0] if extracted_answers else ""
                is_correct = False
                
                if problem.get('gt_answer') and answer:
                    is_correct = check_answer(answer, problem['gt_answer'])
                
                # Update if different
                if base_solution.get('answer') != answer or base_solution.get('is_correct') != is_correct:
                    print(f"Problem {problem_idx}: Updating base solution accuracy")
                    base_solution['answer'] = answer
                    base_solution['is_correct'] = is_correct
                    
                    # Save updated base solution
                    with open(base_solution_file, 'w', encoding='utf-8') as f:
                        json.dump(base_solution, f, indent=2)
    
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
    print(f"Problem {problem_idx}: Using full CoT for chunking")
    
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
                
                # Recalculate accuracy for existing rollouts if needed
                if not args.skip_recalculate:
                    updated_count = 0
                    for rollout in existing_solutions:
                        if 'rollout' in rollout and 'error' not in rollout:
                            extracted_answers = extract_boxed_answers(rollout['rollout'])
                            answer = extracted_answers[0] if extracted_answers else ""
                            is_correct = False
                            
                            if problem.get('gt_answer') and answer:
                                is_correct = check_answer(answer, problem['gt_answer'])
                            
                            # Update if different
                            if rollout.get('answer') != answer or rollout.get('is_correct') != is_correct:
                                updated_count += 1
                                rollout['answer'] = answer
                                rollout['is_correct'] = is_correct
                    
                    if updated_count > 0:
                        print(f"Problem {problem_idx}, Chunk {chunk_idx}: Updated accuracy for {updated_count} rollouts")
                        # Save updated rollouts
                        with open(solutions_file, 'w', encoding='utf-8') as f:
                            json.dump(existing_solutions, f, indent=2)
                
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
    problems = load_math_problems(problem_type=args.type, level=args.level, num_problems=args.num_problems, split=args.split, include_problems=args.include_problems)
    
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

# Add this near the top of your script where you check for API keys
if args.provider == "Novita" and not NOVITA_API_KEY:
    raise ValueError("NOVITA_API_KEY not found in environment variables")
elif args.provider == "Together" and not TOGETHER_API_KEY:
    raise ValueError("TOGETHER_API_KEY not found in environment variables")
elif args.provider == "Fireworks" and not FIREWORKS_API_KEY:
    raise ValueError("FIREWORKS_API_KEY not found in environment variables")
