#!/usr/bin/env python3
"""
Test script to check available levels in the MATH dataset
"""

try:
    from datasets import load_dataset
    import collections
    
    print('Loading DAFT Math dataset...')
    math_dataset = load_dataset("metr-evals/daft-math")
    dataset_split = math_dataset['train']
    
    print('Sample items:')
    for i in range(3):
        item = dataset_split[i]
        print(f'Item {i}:')
        print(f'  Estimated Difficulty: {item.get("Estimated Difficulty", "N/A")}')
        print(f'  Topic: {item.get("topic", "N/A")}')
        print(f'  Question: {item["Integer Answer Variant Question"][:100]}...')
        print(f'  Answer: {item["Integer Variant Answer"]}')
        print()
    
    print('All unique difficulty levels:')
    levels = [item.get('Estimated Difficulty', 'N/A') for item in dataset_split]
    unique_levels = collections.Counter(levels)
    for level, count in unique_levels.most_common():
        print(f'  "{level}": {count} problems')
        
    print('\nAll unique topics:')
    types = [item.get('topic', 'N/A') for item in dataset_split]
    unique_types = collections.Counter(types)
    for ptype, count in unique_types.most_common():
        print(f'  "{ptype}": {count} problems')

except ImportError:
    print("Error: 'datasets' package not found. Please install it:")
    print("pip install datasets")
except Exception as e:
    print(f"Error: {e}")
    print("\nPlease make sure you have internet connection and the datasets package installed.")
