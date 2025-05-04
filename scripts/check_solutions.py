import os
import json
from pathlib import Path
from collections import defaultdict
import argparse

def analyze_rollout_solutions(base_dir):
    """
    Analyze rollout solutions across all problems and chunks.
    
    Args:
        base_dir: Base directory containing problem folders
    """
    base_path = Path(base_dir)
    
    # Statistics
    total_problems = 0
    total_chunks = 0
    total_solutions = 0
    solutions_with_errors = 0
    solutions_with_correct_answers = 0
    solutions_with_incorrect_answers = 0
    solutions_without_answers = 0
    
    # Per-problem statistics
    problem_stats = defaultdict(lambda: {
        "chunks": 0,
        "solutions": 0,
        "errors": 0,
        "correct": 0,
        "incorrect": 0,
        "no_answer": 0
    })
    
    # Per-chunk statistics for empty responses
    chunk_empty_stats = []
    
    # Find all problem directories
    problem_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("problem_")]
    
    for problem_dir in sorted(problem_dirs):
        problem_idx = problem_dir.name.replace("problem_", "")
        total_problems += 1
        
        # Find all chunk directories
        chunk_dirs = [d for d in problem_dir.iterdir() if d.is_dir() and d.name.startswith("chunk_")]
        problem_stats[problem_idx]["chunks"] = len(chunk_dirs)
        total_chunks += len(chunk_dirs)
        
        for chunk_dir in sorted(chunk_dirs):
            chunk_idx = chunk_dir.name.replace("chunk_", "")
            
            # Check for solutions file
            solutions_file = chunk_dir / "solutions.json"
            if solutions_file.exists():
                try:
                    with open(solutions_file, 'r', encoding='utf-8') as f:
                        solutions = json.load(f)
                        
                    # Count solutions
                    num_solutions = len(solutions)
                    total_solutions += num_solutions
                    problem_stats[problem_idx]["solutions"] += num_solutions
                    
                    # Count empty responses for this chunk
                    empty_responses = 0
                    
                    # Analyze each solution
                    for solution in solutions:
                        if "error" in solution:
                            solutions_with_errors += 1
                            problem_stats[problem_idx]["errors"] += 1
                            empty_responses += 1
                        elif "is_correct" in solution:
                            if solution["is_correct"]:
                                solutions_with_correct_answers += 1
                                problem_stats[problem_idx]["correct"] += 1
                            else:
                                solutions_with_incorrect_answers += 1
                                problem_stats[problem_idx]["incorrect"] += 1
                        elif "answer" not in solution or not solution["answer"]:
                            solutions_without_answers += 1
                            problem_stats[problem_idx]["no_answer"] += 1
                            empty_responses += 1
                        else:
                            # Has answer but no is_correct flag
                            solutions_without_answers += 1
                            problem_stats[problem_idx]["no_answer"] += 1
                            
                    # Calculate empty response rate for this chunk
                    empty_rate = empty_responses / num_solutions if num_solutions > 0 else 0
                    
                    # Add to chunk stats
                    chunk_empty_stats.append({
                        "problem_idx": problem_idx,
                        "chunk_idx": chunk_idx,
                        "total_solutions": num_solutions,
                        "empty_responses": empty_responses,
                        "empty_rate": empty_rate
                    })
                            
                except Exception as e:
                    print(f"Error processing {solutions_file}: {e}")
    
    # Print summary
    print(f"Summary for {base_dir}:")
    print(f"Total problems: {total_problems}")
    print(f"Total chunks: {total_chunks}")
    print(f"Total solutions: {total_solutions}")
    print(f"Solutions with errors: {solutions_with_errors} ({solutions_with_errors/total_solutions*100:.2f}%)")
    print(f"Solutions with correct answers: {solutions_with_correct_answers} ({solutions_with_correct_answers/total_solutions*100:.2f}%)")
    print(f"Solutions with incorrect answers: {solutions_with_incorrect_answers} ({solutions_with_incorrect_answers/total_solutions*100:.2f}%)")
    print(f"Solutions without answers: {solutions_without_answers} ({solutions_without_answers/total_solutions*100:.2f}%)")
    
    # Print per-problem statistics
    print("\nPer-problem statistics:")
    print("Problem\tChunks\tSolutions\tErrors\tCorrect\tIncorrect\tNo Answer")
    for problem_idx, stats in sorted(problem_stats.items(), key=lambda x: int(x[0])):
        print(f"{problem_idx}\t{stats['chunks']}\t{stats['solutions']}\t{stats['errors']}\t{stats['correct']}\t{stats['incorrect']}\t{stats['no_answer']}")
    
    # Print chunks with highest empty response rates
    print("\nChunks with highest empty response rates:")
    print("Problem\tChunk\tEmpty Rate\tEmpty/Total")
    
    # Sort chunks by empty rate (highest first)
    sorted_chunks = sorted(chunk_empty_stats, key=lambda x: x["empty_rate"], reverse=True)
    
    # Print top chunks
    for chunk in sorted_chunks:  # Show top 20
        if chunk["empty_rate"] > 0.0:
            print(f"{chunk['problem_idx']}\t{chunk['chunk_idx']}\t{chunk['empty_rate']:.2f}\t{chunk['empty_responses']}/{chunk['total_solutions']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze rollout solutions')
    parser.add_argument('--base_dir', type=str, default='math_rollouts/deepseek-r1-distill-qwen-14b/temperature_0.6_top_p_0.95/correct_base_solution', help='Base directory containing problem folders')
    args = parser.parse_args()
    
    analyze_rollout_solutions(args.base_dir)