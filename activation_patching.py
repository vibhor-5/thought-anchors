import json
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import argparse
import glob
import re
import anthropic
from functools import lru_cache

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# Initialize Anthropic client for answer comparison
try:
    client = anthropic.Anthropic()
except:
    client = None
    print("Warning: Anthropic client not initialized. Will use string comparison for answers.")

@lru_cache(maxsize=128)
def compare_answers_with_claude(question: str, clean_answer: str, test_answer: str) -> bool:
    """
    Use Claude to determine if two answers are meaningfully different.
    
    Args:
        clean_answer: The original answer
        test_answer: The answer to compare against
        
    Returns:
        True if answers are meaningfully the same, False if different
    """
    if client is None:
        # Fall back to string comparison if no client
        return clean_answer.strip() == test_answer.strip()
    
    prompt = f"""You are an expert judge evaluating language model outputs. 
    Compare these two answers to the same question and determine if they respond with the SAME FINAL ANSWER or DIFFERENT FINAL ANSWERS.
    
    Focus on the reported final answer, not on phrasing differences or trajectory. If there is no final answer reported in the response, return "DIFFERENT". 
    If there are steps towards a final answer, but the final answer is not provided or not reached in the response, return "DIFFERENT".
    If there are steps towards a final answer, but the final answer is different, return "DIFFERENT".
    Only return "SAME" if the final answer is the same or mathematically equivalent.
    
    Question:
    {question}
    
    ANSWER RESPONSE 1 (CLEAN):
    {clean_answer}
    
    ANSWER RESPONSE 2 (TEST):
    {test_answer}
    
    Respond with EXACTLY ONE WORD: either "SAME" or "DIFFERENT".
    """
    
    try:
        response = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=10,
            system="You are a helpful assistant that responds with exactly one word: either SAME or DIFFERENT.",
            messages=[{"role": "user", "content": prompt}]
        )
        
        result = response.content[0].text.strip().upper()
        return result == "SAME"
    except Exception as e:
        print(f"API error when comparing answers: {e}")
        # Fall back to string comparison
        return clean_answer.strip() == test_answer.strip()

class ActivationPatcher:
    """Class to handle activation patching for transformer models."""
    def __init__(self, model: AutoModelForCausalLM):
        """Initialize the activation patcher with a model."""
        self.model = model
        self.hooks = []
        self.clean_activations = {}
        
    def register_hooks(self, layers: List[int], component: str = "mlp"):
        """
        Register hooks to capture activations at specified layers.
        
        Args:
            layers: List of layer indices to hook
            component: Model component to hook ('mlp', 'attn', or 'resid')
        """
        self.clean_activations = {}
        
        # Remove any existing hooks
        self.remove_hooks()
        
        # Register new hooks
        for layer_idx in layers:
            # Get the appropriate module based on component
            if component == "mlp":
                module = self.model.model.layers[layer_idx].mlp
                hook_point = "down_proj"  # After MLP, before adding to residual stream
            elif component == "attn":
                module = self.model.model.layers[layer_idx].self_attn
                hook_point = "o_proj"  # After attention, before adding to residual stream
            elif component == "resid":
                # For residual stream, we hook the output of the layer
                module = self.model.model.layers[layer_idx]
                hook_point = "forward"
            else:
                raise ValueError(f"Unknown component: {component}")
            
            # Create unique key for this hook
            hook_key = f"layer_{layer_idx}_{component}"
            
            # Register forward hook
            def get_activation_hook(key):
                def hook(module, input, output):
                    # Store the activation
                    if isinstance(output, tuple):
                        # For residual stream, the output might be a tuple
                        # Store the first element (hidden states)
                        self.clean_activations[key] = output[0].detach().clone()
                    else:
                        # For MLP or attention, output is a tensor
                        self.clean_activations[key] = output.detach().clone()
                return hook
            
            # Add the hook
            if hook_point == "forward":
                # For residual stream, hook the layer's forward method
                handle = module.register_forward_hook(get_activation_hook(hook_key))
            else:
                # For MLP or attention, hook the specific projection
                handle = getattr(module, hook_point).register_forward_hook(get_activation_hook(hook_key))
            
            self.hooks.append(handle)
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []
    
    def generate_with_patching(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        patch_layer: int,
        patch_indices_list: List[Tuple[int, int]],
        component: str = "mlp",
        max_new_tokens: int = 1024
    ) -> torch.Tensor:
        """
        Generate text with activation patching at a specific layer.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask for input
            patch_layer: Layer to patch
            patch_indices_list: List of (start_idx, end_idx) token indices to patch
            component: Component to patch ('mlp', 'attn', or 'resid')
            max_new_tokens: Maximum number of new tokens to generate
            
        Returns:
            Generated token IDs
        """
        # Create a hook that patches the activations
        def get_patch_hook(layer_idx):
            hook_key = f"layer_{layer_idx}_{component}"
            
            def hook(module, input, output):
                # Only patch if we have the clean activation
                if hook_key in self.clean_activations:                    
                    # Handle different output types
                    if isinstance(output, tuple):
                        # For residual stream, output is a tuple
                        # Get the hidden states (first element)
                        hidden_states = output[0]
                        
                        # Create a patched activation
                        patched_hidden = hidden_states.detach().clone()
                        
                        # Apply patches for each index range
                        for start_idx, end_idx in patch_indices_list:
                            # Only patch if we're within the patch indices
                            if patched_hidden.size(1) > start_idx:
                                # Calculate the actual patch range based on current sequence length
                                actual_start = start_idx
                                actual_end = min(end_idx, patched_hidden.size(1))
                                
                                # Only patch if we have a valid range
                                if actual_start < actual_end and actual_start < self.clean_activations[hook_key].size(1):
                                    # Calculate how much of the clean activation to use
                                    clean_patch_len = min(actual_end - actual_start, self.clean_activations[hook_key].size(1) - actual_start)
                                    
                                    # Apply the patch
                                    patched_hidden[:, actual_start:actual_start + clean_patch_len] = self.clean_activations[hook_key][:, actual_start:actual_start + clean_patch_len]
                        
                        # Return the patched output as a tuple with the same structure
                        return (patched_hidden,) + output[1:] if len(output) > 1 else (patched_hidden,)
                    else:
                        # For MLP or attention, output is a tensor
                        # Create a patched activation
                        patched_act = output.detach().clone()
                        
                        # Apply patches for each index range
                        for start_idx, end_idx in patch_indices_list:
                            # Only patch if we're within the patch indices
                            if patched_act.size(1) > start_idx:
                                # Calculate the actual patch range based on current sequence length
                                actual_start = start_idx
                                actual_end = min(end_idx, patched_act.size(1))
                                
                                # Only patch if we have a valid range
                                if actual_start < actual_end and actual_start < self.clean_activations[hook_key].size(1):
                                    # Calculate how much of the clean activation to use
                                    clean_patch_len = min(actual_end - actual_start, self.clean_activations[hook_key].size(1) - actual_start)
                                    
                                    # Apply the patch
                                    patched_act[:, actual_start:actual_start + clean_patch_len] = self.clean_activations[hook_key][:, actual_start:actual_start + clean_patch_len]
                        
                        return patched_act
                return output
            
            return hook
        
        # Get the appropriate module based on component
        if component == "mlp":
            module = self.model.model.layers[patch_layer].mlp
            hook_point = "down_proj"
        elif component == "attn":
            module = self.model.model.layers[patch_layer].self_attn
            hook_point = "o_proj"
        elif component == "resid":
            module = self.model.model.layers[patch_layer]
            hook_point = "forward"
        else:
            raise ValueError(f"Unknown component: {component}")
        
        # Register the patch hook
        if hook_point == "forward":
            patch_handle = module.register_forward_hook(get_patch_hook(patch_layer))
        else:
            patch_handle = getattr(module, hook_point).register_forward_hook(get_patch_hook(patch_layer))
        
        try:
            # Generate with the patch
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # Use greedy decoding for deterministic results
                    pad_token_id=self.model.config.pad_token_id if hasattr(self.model.config, 'pad_token_id') else self.model.config.eos_token_id,
                    temperature=None,  # Remove temperature when do_sample=False
                    top_p=None  # Remove top_p when do_sample=False
                )
        finally:
            # Always remove the hook
            patch_handle.remove()
        
        return outputs

def load_model(model_name: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load the model and tokenizer."""
    print(f"Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Ensure padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad_token_id to eos_token_id: {tokenizer.eos_token_id}")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map="auto"  # Use "auto" instead of device variable for better GPU utilization
    )
    
    # Ensure the model knows about the pad token
    model.config.pad_token_id = tokenizer.pad_token_id
    
    return model, tokenizer

def load_chunks_data(file_path: str) -> List[Dict]:
    """Load the chunks data from the JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def load_solution_text(file_path: str) -> str:
    """Load the solution text from the JSON file."""
    with open(file_path, 'r') as f:
        solutions = json.load(f)
        if solutions and isinstance(solutions, list) and len(solutions) > 0:
            return solutions[0].get("solution", "")
    return ""

def find_deduction_chunks(chunks_data: List[Dict]) -> List[int]:
    """
    Find indices of chunks with category 'Deduction'.
    
    Args:
        chunks_data: List of chunk dictionaries
        
    Returns:
        List of indices for deduction chunks
    """
    deduction_indices = []
    for i, chunk in enumerate(chunks_data):
        if chunk.get("category") == "Deduction":
            deduction_indices.append(i)
    return deduction_indices

def find_subsequence(source, target):
    """Find a subsequence of tokens within a larger sequence."""
    for i in range(len(source) - len(target) + 1):
        if source[i:i+len(target)] == target:
            return i, i+len(target)
    return -1, -1

def get_token_indices(tokenizer, full_text: str, target_text: str) -> Tuple[int, int]:
    """
    Get the token indices for a target text within the full text.
    
    Args:
        tokenizer: The tokenizer
        full_text: The full text
        target_text: The target text to find
        
    Returns:
        Tuple of (start_token_idx, end_token_idx)
    """
    # Tokenize the full text and target text
    full_tokens = tokenizer.encode(full_text, add_special_tokens=False)
    target_tokens = tokenizer.encode(target_text, add_special_tokens=False)
    
    # Find the target tokens in the full tokens
    start_idx, end_idx = find_subsequence(full_tokens, target_tokens)
    
    if start_idx == -1:
        target_tokens = target_tokens[:-1]
        start_idx, end_idx = find_subsequence(full_tokens, target_tokens)
        
    if start_idx == -1:
        target_tokens = target_tokens[1:]
        start_idx, end_idx = find_subsequence(full_tokens, target_tokens)
        
    if start_idx == -1:
        target_tokens = target_tokens[:-1]
        start_idx, end_idx = find_subsequence(full_tokens, target_tokens)
        
    if start_idx == -1:
        target_tokens = target_tokens[1:]
        start_idx, end_idx = find_subsequence(full_tokens, target_tokens)
    
    if start_idx == -1:
        # If not found, try a more lenient approach
        print(f"Warning: Could not find exact token match for: {target_text[:50]}...")
        
        # Try to find the character position and convert to tokens
        char_start = full_text.find(target_text)
        if char_start != -1:
            char_end = char_start + len(target_text)
            
            # Convert character positions to token indices
            prefix_tokens = tokenizer.encode(full_text[:char_start], add_special_tokens=False)
            start_idx = len(prefix_tokens)
            
            full_up_to_end_tokens = tokenizer.encode(full_text[:char_end], add_special_tokens=False)
            end_idx = len(full_up_to_end_tokens)
        else:
            # If still not found, use a default range
            print(f"Warning: Could not find target text in full text. Using default range.")
            start_idx = 0
            end_idx = 0
    
    return start_idx, end_idx

def create_corrupted_text(full_text: str, chunks_data: List[Dict], removed_chunk_indices: List[int], tokenizer: AutoTokenizer) -> str:
    """
    Create a corrupted version of the text with specified chunks removed.
    
    Args:
        full_text: The full text
        chunks_data: List of chunk dictionaries
        removed_chunk_indices: List of indices of chunks to remove
        
    Returns:
        Corrupted text with chunks removed
    """
    """
    corrupted_text = ""
    
    for i, chunk in enumerate(chunks_data):
        if i in removed_chunk_indices:
            chunk_text = chunk["text"]
            pad_token = "<｜end▁of▁sentence｜>"
            corrupted_text += pad_token * len(tokenizer.encode(chunk_text))
        else:
            current_chunk_text = chunk["text"]
            corrupted_text += current_chunk_text
            start_index = full_text.find(current_chunk_text) + len(current_chunk_text)
            end_index = start_index + 8
            eol = full_text[start_index: end_index]
            
            if "\n\n\n\n" in eol:
                corrupted_text += "\n\n\n\n"
            elif "\n\n\n" in eol:
                corrupted_text += "\n\n\n"
            elif "\n\n" in eol:
                corrupted_text += "\n\n"
            elif "\n" in eol:
                corrupted_text += "\n"
            elif "  " in eol:
                corrupted_text += "  "
            elif " " in eol:
                corrupted_text += " "
            else:
                corrupted_text += "\n"
    return corrupted_text
    """
    chunks_to_keep = [chunks_data[i] for i in range(len(chunks_data)) if i not in removed_chunk_indices]
    return "\n".join([chunk["text"] for chunk in chunks_to_keep])
    

def extract_answer_from_text(text: str) -> str:
    """
    Extract the final answer from the solution text.
    
    Args:
        text: Solution text
        
    Returns:
        Extracted answer
    """
    # Look for the answer after </think> tag
    think_end = text.find("</think>")
    if think_end != -1:
        answer_text = text[think_end + len("</think>"):].strip()
        return answer_text
    
    # If no </think> tag, look for "DONE." and take everything before it
    done_idx = text.find("DONE.")
    if done_idx != -1:
        answer_text = text[:done_idx].strip()
        return answer_text
    
    # If no clear markers, return the last paragraph
    paragraphs = text.split("\n\n")
    if paragraphs:
        return paragraphs[-1].strip()
    
    return text.strip()

def run_activation_patching_experiment(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    full_text: str,
    chunks_data: List[Dict],
    target_chunk_indices: List[int],
    layers_to_patch: List[int],
    output_dir: str = "results",
    max_new_tokens: int = 128,
    component: str = "resid"
) -> Dict:
    """
    Run the activation patching experiment for a specific problem and target chunks.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        full_text: Full solution text
        chunks_data: List of chunk dictionaries
        target_chunk_indices: List of indices of chunks to remove
        layers_to_patch: List of layers to patch
        output_dir: Directory to save results
        max_new_tokens: Maximum number of new tokens to generate
        component: Component to patch (default: resid)
    Returns:
        Dictionary of experiment results
    """
    # Get the target chunks
    target_chunks = [chunks_data[idx] for idx in target_chunk_indices]
    target_texts = [chunk.get("text", "") for chunk in target_chunks]
    
    # Create corrupted text with target chunks removed
    corrupted_text = create_corrupted_text(full_text, chunks_data, target_chunk_indices, tokenizer)
    
    # Get token indices for each target chunk
    patch_indices_list = [get_token_indices(tokenizer, full_text, text) for text in target_texts]
    
    # Reconstruct the full text from chunks
    reconstructed_text = "\n".join(chunk.get("text", "") for chunk in chunks_data)
    
    # Find the </think> position in both texts
    clean_think_end = reconstructed_text.find("</think>")
    corrupted_think_end = corrupted_text.find("</think>")
    
    # If </think> is found, append a newline to ensure generation starts after it
    if clean_think_end != -1:
        clean_prompt = reconstructed_text[:clean_think_end + len("</think>")] + "\n" + "Therefore the answer is:"
    else:
        clean_prompt = reconstructed_text + "\n" + "</think>" + "\n" + "Therefore the answer is:"
    
    if corrupted_think_end != -1:
        corrupted_prompt = corrupted_text[:corrupted_think_end + len("</think>")] + "\n" + "Therefore the answer is:"
    else:
        corrupted_prompt = corrupted_text + "\n" + "</think>" + "\n" + "Therefore the answer is:"
    
    # Tokenize the prompts
    clean_inputs = tokenizer(clean_prompt, return_tensors="pt", padding=True)
    clean_input_ids = clean_inputs.input_ids.to(device)
    clean_attention_mask = clean_inputs.attention_mask.to(device)
    
    corrupted_inputs = tokenizer(corrupted_prompt, return_tensors="pt", padding=True)
    corrupted_input_ids = corrupted_inputs.input_ids.to(device)
    corrupted_attention_mask = corrupted_inputs.attention_mask.to(device)
    
    # Determine the maximum length for padding
    max_length = max(clean_input_ids.size(1), corrupted_input_ids.size(1))
    
    if corrupted_input_ids.size(1) < max_length:
        padding = torch.full((corrupted_input_ids.size(0), max_length - corrupted_input_ids.size(1)), tokenizer.pad_token_id, dtype=corrupted_input_ids.dtype, device=corrupted_input_ids.device)
        corrupted_input_ids = torch.cat([corrupted_input_ids, padding], dim=1)
        corrupted_attention_mask = torch.cat([corrupted_attention_mask, torch.zeros_like(padding)], dim=1)
    
    if clean_input_ids.size(1) < max_length:
        padding = torch.full((clean_input_ids.size(0), max_length - clean_input_ids.size(1)), tokenizer.pad_token_id, dtype=clean_input_ids.dtype, device=clean_input_ids.device)
        clean_input_ids = torch.cat([clean_input_ids, padding], dim=1)
        clean_attention_mask = torch.cat([clean_attention_mask, torch.zeros_like(padding)], dim=1)
    
    # Initialize activation patcher
    patcher = ActivationPatcher(model)
    
    # Get the clean run outputs by generating after the prompt
    with torch.no_grad():
        clean_outputs = model.generate(
            clean_input_ids,
            attention_mask=clean_attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Use greedy decoding for deterministic results
            pad_token_id=tokenizer.pad_token_id,
            temperature=None,  # Remove temperature when do_sample=False
            top_p=None  # Remove top_p when do_sample=False
        )
        # Extract only the generated tokens (after the input)
        clean_generated = clean_outputs[0][len(clean_inputs.input_ids[0]):]
        clean_answer = tokenizer.decode(clean_generated, skip_special_tokens=True)
    
    # Get the corrupted run outputs by generating after the prompt
    with torch.no_grad():
        corrupted_outputs = model.generate(
            corrupted_input_ids,
            attention_mask=corrupted_attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Use greedy decoding for deterministic results
            pad_token_id=tokenizer.pad_token_id,
            temperature=None,  # Remove temperature when do_sample=False
            top_p=None  # Remove top_p when do_sample=False
        )
        # Extract only the generated tokens (after the input)
        corrupted_generated = corrupted_outputs[0][len(corrupted_inputs.input_ids[0]):]
        corrupted_answer = tokenizer.decode(corrupted_generated, skip_special_tokens=True)
    
    # Check if the corrupted answer is meaningfully different from the clean answer
    answers_differ = not compare_answers_with_claude(full_text.split("<think>")[0], clean_answer, corrupted_answer)
    
    # If answers don't differ meaningfully, skip patching experiments
    if not answers_differ:
        print(f"Skipping patching for chunks {target_chunk_indices} - answers are not meaningfully different")
        return {
            "target_chunk_indices": target_chunk_indices,
            "target_chunk_texts": target_texts,
            "target_chunk_categories": [chunk.get("category", "") for chunk in target_chunks],
            "clean_answer": clean_answer,
            "corrupted_answer": corrupted_answer,
            "answers_differ": False,
            "patched_answers": {},
            "answer_match": {}
        }
    
    # First, capture clean activations
    patcher.register_hooks(layers_to_patch, component=component)
    with torch.no_grad():
        _ = model(clean_input_ids, attention_mask=clean_attention_mask)
        
    # Remove hooks
    patcher.remove_hooks()
    
    # Run patching experiments for each layer
    patched_answers = {}
    
    for layer in tqdm(layers_to_patch, desc=f"Patching layers for chunks {target_chunk_indices}"):
        # Generate with patching at this layer
        patched_outputs = patcher.generate_with_patching(
            input_ids=corrupted_input_ids,
            attention_mask=corrupted_attention_mask,
            patch_layer=layer,
            patch_indices_list=patch_indices_list,
            component=component,
            max_new_tokens=max_new_tokens
        )
        
        # Extract only the generated tokens (after the input)
        patched_generated = patched_outputs[0][len(corrupted_inputs.input_ids[0]):]
        patched_answer = tokenizer.decode(patched_generated, skip_special_tokens=True)
        patched_answers[str(layer)] = patched_answer
    
    # Remove hooks
    patcher.remove_hooks()
    
    # Calculate metrics
    metrics = {
        "target_chunk_indices": target_chunk_indices,
        "target_chunk_texts": target_texts,
        "target_chunk_categories": [chunk.get("category", "") for chunk in target_chunks],
        "clean_answer": clean_answer,
        "corrupted_answer": corrupted_answer,
        "answers_differ": True,
        "patched_answers": patched_answers,
        "answer_match": { 
            str(layer): compare_answers_with_claude(full_text.split("<think>")[0], clean_answer, patched_answers[str(layer)]) 
            for layer in layers_to_patch 
        }
    }
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/patch_results_chunks{'_'.join(map(str, target_chunk_indices))}.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    
    # Plot answer match by layer
    match_by_layer = [1 if metrics["answer_match"][str(layer)] else 0 for layer in layers_to_patch]
    plt.bar(layers_to_patch, match_by_layer)
    plt.xlabel("Layer")
    plt.ylabel("Answer Match (1=Yes, 0=No)")
    plt.title(f"Effect of Patching Chunks {target_chunk_indices} by Layer")
    plt.savefig(f"{output_dir}/patch_results_chunks{'_'.join(map(str, target_chunk_indices))}.png")
    
    return metrics

def run_batch_experiments(
    model_name: str,
    problem_dirs: List[str],
    layers_to_patch: Optional[List[int]] = None,
    output_dir: str = "results",
    max_new_tokens: int = 128,
    component: str = "resid"
) -> Dict:
    """
    Run activation patching experiments for a batch of problems.
    
    Args:
        model_name: Name of the model to use
        problem_dirs: List of problem directories
        layers_to_patch: List of layers to patch (if None, use evenly spaced layers)
        output_dir: Directory to save results
        max_new_tokens: Maximum number of new tokens to generate
        component: Component to patch (default: resid)
    Returns:
        Dictionary of aggregate results
    """
    # Load model and tokenizer
    model, tokenizer = load_model(model_name)
    
    # Set default layers to patch if not provided
    if layers_to_patch is None:
        num_layers = model.config.num_hidden_layers
        layers_to_patch = list(range(0, num_layers, max(1, num_layers // 10)))
    
    # Run experiments for each problem
    all_results = []
    
    for problem_dir in tqdm(problem_dirs, desc="Processing problems"):
        # Extract problem ID and seed from directory name
        problem_match = re.search(r'problem_(\d+)', problem_dir)
        seed_match = re.search(r'seed_(\d+)', problem_dir)
        
        if not problem_match or not seed_match:
            print(f"Skipping directory with invalid format: {problem_dir}")
            continue
        
        problem_id = problem_match.group(1)
        seed = seed_match.group(1)
        
        # Create output directory for this problem
        problem_output_dir = os.path.join(output_dir, f"problem_{problem_id}_seed_{seed}")
        if os.path.exists(problem_output_dir):
            print(f"Skipping existing problem: {problem_id}, seed {seed}")
            continue
        
        os.makedirs(problem_output_dir, exist_ok=True)
        
        # Load chunks data
        chunks_file = os.path.join(problem_dir, "chunks.json")
        if not os.path.exists(chunks_file):
            print(f"Chunks file not found: {chunks_file}")
            continue
        
        chunks_data = load_chunks_data(chunks_file)
        
        # Load solution text
        solution_file = os.path.join('cots', os.path.dirname(problem_dir).split('/')[-1], "solutions.json")
        if not os.path.exists(solution_file):
            print(f"Solution file not found: {solution_file}")
            continue
        
        full_text = load_solution_text(solution_file)
        if not full_text:
            print(f"No solution text found in: {solution_file}")
            continue
        
        # Find deduction chunks
        deduction_indices = find_deduction_chunks(chunks_data)
        if not deduction_indices:
            print(f"No deduction chunks found in: {chunks_file}")
            continue
        
        # For each problem, test patching with cumulative deduction chunks
        removed_chunks = []
        for chunk_idx in deduction_indices:
            # Add this chunk to the list of chunks to remove
            removed_chunks.append(chunk_idx)

            try:
                results = run_activation_patching_experiment(
                    model=model,
                    tokenizer=tokenizer,
                    full_text=full_text,
                    chunks_data=chunks_data,
                    target_chunk_indices=removed_chunks.copy(),  # Use a copy to avoid modifying the list
                    layers_to_patch=layers_to_patch,
                    output_dir=problem_output_dir,
                    max_new_tokens=max_new_tokens,
                    component=component
                )
                
                # Add problem and seed info to results
                results["problem_id"] = problem_id
                results["seed"] = seed
                
                # Only add to results if answers differ
                if results.get("answers_differ", False):
                    all_results.append(results)
                    
                    # If we found a meaningful difference, we can stop adding more chunks
                    # Uncomment this if you want to stop after finding a meaningful difference
                    # break
            except Exception as e:
                print(f"Error processing problem {problem_id}, seed {seed}, chunks {removed_chunks}: {e}")
    
    # Aggregate results
    aggregate_results = {
        "num_problems": len(problem_dirs),
        "num_experiments": len(all_results),
        "layers_patched": layers_to_patch,
        "success_rate_by_layer": {
            str(layer): sum(1 for r in all_results if r["answer_match"].get(str(layer), False)) / max(1, len(all_results))
            for layer in layers_to_patch
        }
    }
    
    # Save aggregate results
    with open(f"{output_dir}/aggregate_results.json", "w") as f:
        json.dump(aggregate_results, f, indent=2)
    
    # Create visualization of aggregate results
    plt.figure(figsize=(12, 6))
    success_rates = [aggregate_results["success_rate_by_layer"][str(layer)] for layer in layers_to_patch]
    plt.bar(layers_to_patch, success_rates)
    plt.xlabel("Layer")
    plt.ylabel("Success Rate")
    plt.title("Aggregate Success Rate by Layer")
    plt.savefig(f"{output_dir}/aggregate_results.png")
    
    return aggregate_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run activation patching experiments for causal reasoning analysis")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", help="HuggingFace model name")
    parser.add_argument("--analysis_dir", type=str, default="analysis", help="Directory containing problem analysis")
    parser.add_argument("--num_problems", type=int, default=1000, help="Number of problems to analyze")
    parser.add_argument("--output_dir", type=str, default="activation_patching_results", help="Directory to save results")
    parser.add_argument("--layers", type=str, default=None, help="Comma-separated list of layers to patch (default: evenly spaced layers)")
    parser.add_argument("--max_new_tokens", type=int, default=150, help="Maximum number of new tokens to generate")
    parser.add_argument("--component", type=str, default="resid", help="Component to patch (default: resid)")
    args = parser.parse_args()
    
    # Parse layers
    layers_to_patch = None
    if args.layers:
        layers_to_patch = [int(layer) for layer in args.layers.split(",")]
    
    # Find problem directories
    problem_pattern = os.path.join(args.analysis_dir, "problem_*", "seed_*")
    problem_dirs = sorted(glob.glob(problem_pattern))[:args.num_problems]
    
    # TODO: Remove later
    problem_dirs = [x for x in problem_dirs if "seed_0" in x][::-1]
    
    if not problem_dirs:
        print(f"No problem directories found matching pattern: {problem_pattern}")
        exit(1)
    
    print(f"Found {len(problem_dirs)} problem directories.")
    
    # Run experiments
    aggregate_results = run_batch_experiments(
        model_name=args.model,
        problem_dirs=problem_dirs,
        layers_to_patch=layers_to_patch,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,
        component=args.component
    )
    
    print("Experiment completed. Results saved to:", args.output_dir)
    print("Aggregate success rate by layer:", aggregate_results["success_rate_by_layer"])