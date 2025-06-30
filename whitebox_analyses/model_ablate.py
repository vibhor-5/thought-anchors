import os
import random
import sys
import time
import warnings
from collections import defaultdict
from functools import cache

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Import necessary components from the original model.py
# You can also copy the get_deepseek_r1 function here if needed
from model_read import get_deepseek_r1

from pkld import pkld


class AttentionHeadAblator:
    def __init__(self, ablation_dict, verbose=False):
        """
        Initialize attention head ablator

        Args:
            ablation_dict: Dict mapping layer indices to lists of head indices to ablate
                          e.g., {0: [1, 3], 15: [0, 2, 7]}
            verbose: Whether to print debug information
        """
        self.ablation_dict = ablation_dict
        self.verbose = verbose
        self.hooks = []
        self.applied_count = defaultdict(int)
        self.num_heads = None
        self.head_dim = None
        self.logged_layers = set()  # Track which layers we've logged to avoid spam

    def get_model_config(self, model):
        """Extract attention configuration from the model"""
        config = model.config
        self.num_heads = getattr(config, "num_attention_heads", 32)
        hidden_size = getattr(config, "hidden_size", 4096)
        self.head_dim = hidden_size // self.num_heads

        if self.verbose:
            print(f"Model config: {self.num_heads} heads, {self.head_dim} dim per head")

    def ablation_hook(self, layer_idx):
        """Create hook function that ablates specified attention heads"""

        def hook_fn(module, input_tuple, output):
            # Get the heads to ablate for this layer
            heads_to_ablate = self.ablation_dict.get(layer_idx, [])
            if not heads_to_ablate:
                # No ablation needed - return output unchanged
                return output

            # Handle different output formats
            if isinstance(output, tuple):
                hidden_states = output[0]
                rest_outputs = output[1:]
            else:
                hidden_states = output
                rest_outputs = ()

            # Verify dimensions
            batch_size, seq_len, hidden_dim = hidden_states.shape
            expected_dim = self.num_heads * self.head_dim
            if hidden_dim != expected_dim:
                if self.verbose:
                    print(
                        f"Warning: Hidden dim mismatch at layer {layer_idx}: {hidden_dim} vs {expected_dim}"
                    )
                return output

            # Clone to avoid in-place modification
            modified_hidden_states = hidden_states.clone()

            # Reshape to separate heads: (batch, seq, num_heads, head_dim)
            reshaped = modified_hidden_states.view(
                batch_size, seq_len, self.num_heads, self.head_dim
            )

            # Ablate specified heads by setting them to zero
            valid_heads_ablated = []
            for head_idx in heads_to_ablate:
                if 0 <= head_idx < self.num_heads:
                    # mask_value = torch.finfo(reshaped.dtype).min
                    # print(f"{mask_value=}")
                    reshaped[:, :, head_idx, :] = 0
                    self.applied_count[f"{layer_idx}_{head_idx}"] += seq_len
                    valid_heads_ablated.append(head_idx)
                elif self.verbose:
                    print(
                        f"Warning: Invalid head index {head_idx} for layer {layer_idx} (max: {self.num_heads-1})"
                    )

            # Log only once per layer to avoid spam
            if self.verbose and valid_heads_ablated and layer_idx not in self.logged_layers:
                print(f"Ablating heads {valid_heads_ablated} at layer {layer_idx}")
                self.logged_layers.add(layer_idx)

            # Reshape back to original format
            modified_hidden_states = reshaped.view(batch_size, seq_len, hidden_dim)

            # Return in the same format as input
            if rest_outputs:
                return (modified_hidden_states,) + rest_outputs
            else:
                return modified_hidden_states

        return hook_fn

    def register_hooks(self, model):
        """Register hooks to ablate attention heads at specified layers"""
        self.remove_hooks()
        self.applied_count = defaultdict(int)
        self.logged_layers = set()
        self.get_model_config(model)

        # Only proceed if there are actually heads to ablate
        if not self.ablation_dict:
            if self.verbose:
                print("No ablation specified - skipping hook registration")
            return

        attention_modules = {}

        # Find attention modules ONLY for layers that need ablation
        for name, module in model.named_modules():
            if ".layers." in name and "self_attn" in name and name.endswith("self_attn"):
                parts = name.split(".layers.")
                if len(parts) > 1 and parts[1]:
                    layer_idx_str = parts[1].split(".")[0]
                    try:
                        layer_idx = int(layer_idx_str)
                        # Only register hooks for layers that actually need ablation
                        if layer_idx in self.ablation_dict and self.ablation_dict[layer_idx]:
                            attention_modules[layer_idx] = module
                    except ValueError:
                        pass

        if self.verbose:
            if attention_modules:
                print(f"Registering ablation hooks for layers: {sorted(attention_modules.keys())}")
                print(f"Ablation plan: {self.ablation_dict}")
            else:
                print("No attention modules found for specified layers")

        # Register hooks for attention outputs
        for layer_idx, attn_module in attention_modules.items():
            hook = attn_module.register_forward_hook(self.ablation_hook(layer_idx))
            self.hooks.append(hook)

    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

        if self.verbose and sum(self.applied_count.values()) > 0:
            print(f"Ablation summary:")
            for key, count in sorted(self.applied_count.items()):
                layer_head = key.split("_")
                if len(layer_head) == 2:
                    layer, head = layer_head
                    print(f"  Layer {layer}, Head {head}: Applied to {count} token positions")


def generate_with_ablation(
    model,
    tokenizer,
    prompt,
    ablation_dict,
    max_new_tokens=100,
    use_sampling=False,  # Simple boolean toggle
    temperature=0.8,  # More reasonable default
    verbose=False,
    seed=None,
    num_responses=1,  # NEW: Number of responses to generate
):
    """
    Generate text with specified attention heads ablated

    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input prompt string
        ablation_dict: Dict mapping layer indices to lists of head indices to ablate
                      e.g., {0: [1, 3], 15: [0, 2, 7]}
        max_new_tokens: Maximum number of new tokens to generate
        use_sampling: If True, use temperature sampling; if False, use greedy
        temperature: Temperature for sampling (only used when use_sampling=True)
        verbose: Whether to print debug information
        seed: Random seed for reproducibility
        num_responses: Number of responses to generate

    Returns:
        Dict or List[Dict] containing prompt, response, tokens, and ablation info
        If num_responses=1, returns single dict. If num_responses>1, returns list of dicts.
    """

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)

    # Set up ablation - only if there's something to ablate
    ablator = None
    if ablation_dict:
        # Filter out empty lists
        filtered_ablation = {k: v for k, v in ablation_dict.items() if v}
        if filtered_ablation:
            ablator = AttentionHeadAblator(filtered_ablation, verbose=verbose)
            ablator.register_hooks(model)

    try:
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_ids = inputs.input_ids

        if verbose:
            print(f"Prompt: '{prompt}'")
            print(f"Encoded to {input_ids.shape[1]} tokens")
            print(f"Generation mode: {'Sampling' if use_sampling else 'Greedy'}")
            print(f"Generating {num_responses} response(s)")
            if use_sampling:
                print(f"Temperature: {temperature}")

        response_start = input_ids.shape[1]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            with torch.inference_mode():
                if use_sampling:
                    # Temperature sampling with multiple sequences
                    output_ids = model.generate(
                        input_ids,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=temperature,
                        top_p=0.9,
                        pad_token_id=tokenizer.eos_token_id,
                        use_cache=True,
                        num_return_sequences=num_responses,  # Generate multiple sequences at once
                    )
                else:
                    # For greedy, all sequences will be identical, but still use num_return_sequences
                    output_ids = model.generate(
                        input_ids,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                        num_return_sequences=num_responses,  # Generate multiple sequences at once
                    )

        # Process results
        results = []
        for i in range(num_responses):
            # Get the i-th generated sequence
            sequence_ids = output_ids[i]

            # Decode response (only the new tokens)
            response = tokenizer.decode(sequence_ids[response_start:], skip_special_tokens=True)

            # Get token texts for the full sequence
            all_tokens = sequence_ids.tolist()
            token_texts = tokenizer.convert_ids_to_tokens(all_tokens)

            # Organize results
            result = {
                "prompt": prompt,
                "response": response,
                "tokens": all_tokens,
                "token_texts": token_texts,
                "input_length": input_ids.shape[1],
                "response_start": response_start,
                "ablation_dict": ablation_dict,
                "ablation_summary": dict(ablator.applied_count) if ablator else {},
                "use_sampling": use_sampling,
                "temperature": temperature if use_sampling else None,
                "response_index": i,  # Track which response this is
                "seed_used": seed,  # All sequences use the same base seed
            }

            results.append(result)

        # Return single dict if num_responses=1, otherwise return list
        return results[0] if num_responses == 1 else results

    finally:
        t_st = time.time()
        try:
            from utils import print_gpu_memory_summary

            print_gpu_memory_summary("Finished generation")
        except ImportError:
            if verbose:
                print("GPU memory summary not available")

        if verbose:
            print("Cleaning up...")

        # Always clean up hooks
        if ablator:
            ablator.remove_hooks()
        # torch.cuda.empty_cache()
        t_end = time.time()
        if verbose:
            print(f"Cleanup completed in {t_end - t_st:.2f} seconds")


def test_ablation(
    prompt,
    ablation_dict,
    max_new_tokens=100,
    use_sampling=False,  # Simple boolean toggle
    temperature=0.8,  # More reasonable default
    model_name="qwen-15b",
    verbose=True,
    seed=None,
    float32=False,
    quantize_8bit=False,
    quantize_4bit=False,
    device_map="auto",
    do_flash_attn=True,
    num_responses=1,  # NEW: Number of responses to generate
):
    """
    Test attention head ablation with a specific prompt

    Args:
        prompt: Input prompt string
        ablation_dict: Dict mapping layer indices to lists of head indices to ablate
        max_new_tokens: Maximum number of new tokens to generate
        use_sampling: If True, use temperature sampling; if False, use greedy
        temperature: Temperature for sampling
        model_name: Model name (should work with "qwen-15b")
        verbose: Whether to print debug information
        seed: Random seed for reproducibility (None for no seeding)
        num_responses: Number of responses to generate

    Returns:
        Dict or List[Dict] containing generation results and ablation info
    """

    if verbose:
        print(f"Loading model: {model_name}")
        if ablation_dict:
            print(f"Ablation plan: {ablation_dict}")
        else:
            print("No ablation specified - running normally")

    # Load model and tokenizer
    model, tokenizer = get_deepseek_r1(
        model_name=model_name,
        float32=float32,
        quantize_8bit=quantize_8bit,
        quantize_4bit=quantize_4bit,
        device_map=device_map,
        do_flash_attn=do_flash_attn,
    )
    # print("Model attention implementation:", model.config)

    # Generate with ablation
    t_start = time.time()
    result = generate_with_ablation(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        ablation_dict=ablation_dict,
        max_new_tokens=max_new_tokens,
        use_sampling=use_sampling,
        temperature=temperature,
        verbose=verbose,
        seed=seed,
        num_responses=num_responses,  # NEW: Pass through num_responses
    )
    t_end = time.time()

    # Handle both single result and multiple results
    if isinstance(result, list):
        total_tokens = sum(len(r["token_texts"]) for r in result)
        # print(f"Generation completed in {t_end - t_start:.2f} seconds ({total_tokens} total tokens, {len(result)} responses)")
        # Add generation time to each result
        for r in result:
            r["generation_time"] = t_end - t_start
    else:
        num_tokens = len(result["token_texts"])
        if verbose:
            print(f"Generation completed in {t_end - t_start:.2f} seconds ({num_tokens} tokens)")
            print(f"Response: {result['response']}")
        result["generation_time"] = t_end - t_start

    return result


@pkld
def get_ablation_response(
    prompt,
    ablation_dict,
    model_name="qwen-14b",
    seed=0,
    float32=False,
    max_new_tokens=4000,
    device_map="auto",
    verbose=False,
    quantize_8bit=False,
    quantize_4bit=False,
    do_flash_attn=True,
    num_responses=1,  # NEW: Number of responses to generate
):

    n_ablation_heads = sum(len(heads) for heads in ablation_dict.values())
    print(f"Running ablation for prompt: {prompt}")
    print(f"\tNumber of ablated heads: {n_ablation_heads}")
    print(f"\tGenerating {num_responses} response(s)")

    t_start = time.time()
    result_normal_sampling = test_ablation(
        prompt=prompt,
        ablation_dict=ablation_dict,
        max_new_tokens=max_new_tokens,
        use_sampling=True,  # Temperature sampling
        temperature=0.7,  # Reasonable temperature
        model_name=model_name,
        verbose=verbose,
        seed=seed,
        float32=float32,
        quantize_8bit=quantize_8bit,
        quantize_4bit=quantize_4bit,
        device_map=device_map,
        do_flash_attn=do_flash_attn,
        num_responses=num_responses,  # NEW: Pass through num_responses
    )

    # Handle both single result and multiple results
    if isinstance(result_normal_sampling, list):
        total_tokens = sum(len(r["token_texts"]) for r in result_normal_sampling)
        t_end = time.time()
        t_needed = t_end - t_start
        print(
            f"Total tokens across {len(result_normal_sampling)} responses: {total_tokens} ({t_needed:.2f} seconds)"
        )
    else:
        n_tokens = len(result_normal_sampling["token_texts"])
        t_end = time.time()
        t_needed = t_end - t_start
        print(f"Number of tokens (prompt + response): {n_tokens} ({t_needed:.2f} seconds)")

    return result_normal_sampling


if __name__ == "__main__":

    print("\n" + "=" * 50)
    print("Testing with temperature sampling...")

    ablation_dict = {
        5: [0, 1, 2],
        10: [3, 7],
    }
    # ablation_dict =
    ablation_dict = {}
    # for layer in range(0):
    # ablation_dict[layer] = list(range(12))

    result_normal_sampling = test_ablation(
        prompt="The capital of France is <think>",
        ablation_dict=ablation_dict,
        max_new_tokens=50,
        use_sampling=True,  # Temperature sampling
        temperature=0.7,  # Reasonable temperature
        model_name="qwen-15b",
        verbose=True,
        seed=None,
        device_map="cpu",
        float32=True,
        flash_attn=False,
        num_responses=1,
    )

    print(result_normal_sampling)

    print("\n" + "=" * 50)
    print("SAMPLING GENERATION RESULTS:")
    print("=" * 50)
    # print(f"Response: {result_normal_sampling['response']}")
    quit()

    print("\n" + "=" * 50)
    print("Testing with ablation (greedy)...")

    # Ablate heads 0, 1, 2 at layer 5 and heads 3, 7 at layer 10
    ablation_dict = {
        5: [0, 1, 2],
        10: [3, 7],
    }

    result_ablated = test_ablation(
        prompt="The capital of France is",
        ablation_dict=ablation_dict,
        max_new_tokens=50,
        use_sampling=False,  # Greedy for consistent ablation results
        model_name="qwen-15b",
        verbose=True,
        seed=None,
        num_responses=1,
    )

    print("\n" + "=" * 50)
    print("ABLATED GENERATION RESULTS:")
    print("=" * 50)
    print(f"Response: {result_ablated['response']}")
    print(f"Ablation applied: {result_ablated['ablation_dict']}")
    print(f"Ablation summary: {result_ablated['ablation_summary']}")
