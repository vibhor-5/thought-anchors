#!/usr/bin/env python
# coding: utf-8

# In[45]:

import json
from pathlib import Path
from tqdm import tqdm
import re
from transformers import AutoTokenizer
from generate_chunk_rollouts import split_solution_into_chunks


# In[46]:

hf_model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
tokenizer = AutoTokenizer.from_pretrained(hf_model)


# In[72]:

print(tokenizer.encode("Wait", add_special_tokens=False))
print(tokenizer.encode(" wait", add_special_tokens=False))
print(tokenizer.encode(" Remember", add_special_tokens=False))
print(tokenizer.encode(" remember", add_special_tokens=False))


# In[64]:

def count_token_occurrences(file_path, token):
    # Load the JSON file
    with open(file_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # Get the by_problem section
    overall = results.get('overall', {})
    correct_count = overall.get('correct', 0)
    incorrect_count = overall.get('incorrect', 0)
    accuracy = correct_count / (correct_count + incorrect_count) if correct_count + incorrect_count > 0 else 0
    problems = results.get('by_problem', {})
    
    # Initialize counters
    total_tokens_count = 0
    total_problems = 0
    problems_with_tokens = 0
    
    # Process each problem
    problem_counts = []
    for problem_idx, problem_data in problems.items():
        total_problems += 1
        
        # Get solutions for this problem
        solutions = problem_data.get('solutions', [])
        
        # Count occurrences in each solution's full_cot
        problem_tokens_count = 0
        for solution in solutions:
            full_cot = solution.get('full_cot', '')
            
            lower_token_count = len(re.findall(r'\b' + token + r'\b', full_cot))
            upper_token_count = len(re.findall(r'\b' + token[0].upper() + token[1:].lower() + r'\b', full_cot))
            problem_tokens_count += lower_token_count + upper_token_count
        
        # Record the count for this problem
        if problem_tokens_count > 0:
            problems_with_tokens += 1
            
        total_tokens_count += problem_tokens_count
        problem_counts.append((problem_idx, problem_tokens_count))
    
    # Sort problems by wait count (highest first)
    problem_counts.sort(key=lambda x: x[1], reverse=True)
    
    print(f"Total '{token}' occurrences: {total_tokens_count}")
    print(f"Total problems analyzed: {total_problems}")
    print(f"Problems containing '{token}': {problems_with_tokens} ({problems_with_tokens/total_problems*100:.2f}%)")
    print(f"Average '{token}' per problem: {total_tokens_count / total_problems if total_problems > 0 else 0:.2f}")
    print(f"Accuracy: {accuracy:.3f}")
    
    return {
        'total_tokens_count': total_tokens_count,
        'total_problems': total_problems,
        'problems_with_tokens': problems_with_tokens,
        'avg_tokens_per_problem': total_tokens_count / total_problems if total_problems > 0 else 0,
        'top_10_problems': problem_counts[:10],
        'accuracy': accuracy
    }


# In[69]:

file_path = Path("math_cots/deepseek-r1-distill-qwen-14b/temperature_0.6_top_p_0.92_logit_bias_wait_1/results.json")
results = count_token_occurrences(file_path, "wait")


# In[70]:

file_path = Path("math_cots/deepseek-r1-distill-qwen-14b/temperature_0.6_top_p_0.92_logit_bias_wait_-100/results.json")
results = count_token_occurrences(file_path, "wait")


# In[75]:

file_path = Path("math_cots/deepseek-r1-distill-qwen-14b/temperature_0.6_top_p_0.92_logit_bias_remember_1/results.json")
results = count_token_occurrences(file_path, "remember")


# In[76]:

file_path = Path("math_cots/deepseek-r1-distill-qwen-14b/temperature_0.6_top_p_0.92_logit_bias_remember_-100/results.json")
results = count_token_occurrences(file_path, "remember")


# In[95]:

file_path = Path("math_cots/deepseek-r1-distill-qwen-14b/temperature_0.6_top_p_0.92/hardest_problems.json")
hardest_problems = json.load(open(file_path, 'r', encoding='utf-8'))
selected_problems = []

for problem_idx, metrics in tqdm(hardest_problems.items(), desc="Processing problems"):
    correct = metrics['correct'] if 'correct' in metrics and metrics['correct'] > 0 else 1
    incorrect = metrics['incorrect'] if 'incorrect' in metrics and metrics['incorrect'] > 0 else 1
    accuracy = correct / (correct + incorrect)
    
    if accuracy >= 0.25 and accuracy <= 0.75 and 'gt_answer' in metrics and metrics['gt_answer'] != '':
        solutions = metrics['solutions']
        approximate_mean_tokens = sum([len(solution['full_cot']) / 4 for solution in solutions]) / len(solutions)
        approximate_mean_chunks = sum([len(split_solution_into_chunks(solution['full_cot'])) for solution in solutions]) / len(solutions)
        
        selected_problems.append({
            'problem_idx': f"problem_{problem_idx}", 
            'level': metrics['level'],
            'type': metrics['type'],
            'problem': metrics['problem'],
            'gt_answer': metrics['gt_answer'],
            'accuracy': accuracy,
            'approximate_mean_tokens': approximate_mean_tokens,
            'approximate_mean_chunks': approximate_mean_chunks
        })

selected_problems = sorted(selected_problems, key=lambda x: x['approximate_mean_chunks'], reverse=False)
print('Number of selected problems:', len(selected_problems))

json.dump(selected_problems, open('selected_problems.json', 'w', encoding='utf-8'), indent=2)

