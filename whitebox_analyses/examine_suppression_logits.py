import os
from pathlib import Path


import json
import pickle
import warnings
import math
from types import MethodType
from collections import defaultdict
from typing import List, Tuple, Optional, Dict, Set, Union

from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import sys
import time

from pkld import pkld

from model_read import (
    analyze_text,
    get_deepseek_r1,
)  # Use existing model loader and memory checker
from repeated_suppression_logits import (
    analyze_text_get_p_logits,
    analyze_text_get_p_logits_large,
    decompress_logits_for_position,
)
from utils import (
    # get_qwen_14b_tokens_lower,
    # get_qwen_raw_tokens,
    # get_qwen_tokenizer,
    get_raw_tokens,
    get_top_p_logits,
    hash_dict,
    print_gpu_memory_summary,
)  # Use existing utils
from run_target_problems import (
    get_full_CoT_token_ranges,
    get_problem_nums,
    load_problem_json,
)  # Use existing problem loader
from uzay_utils import (
    get_chunk_ranges,
    get_chunk_token_ranges,
)  # Use existing chunking utils

import torch


@pkld
def get_sentence_entropies(
    problem_num: int,
    p_nucleus: float = 0.9999,
    model_name: str = "qwen-14b",
    quantize_4bit: bool = False,
    quantize_8bit: bool = False,
    problem_dir=os.path.join("target_problems", "temperature_0.6_top_p_0.95"),
    temperature: float = 0.6,
    only_pre_convergence: bool = False,
    **kwargs,
):
    try:
        sentence2ranges, problem = get_full_CoT_token_ranges(
            problem_num,
            problem_dir,
            only_pre_convergence=only_pre_convergence,
            verbose=True,
            model_name=model_name,
        )
    except AssertionError as e:
        print(f"No convergence on {problem_num}: {e}")
        return

    text = (
        problem["full_cot_truncated"]
        if only_pre_convergence
        else problem["base_solution"]["full_cot"]
    )
    assert "qwen" in model_name

    text_tokens = get_raw_tokens(text, model_name)
    if len(text_tokens) > 4200:
        device_map = "cpu"
        print(f"No CPU today ({problem_num}): {len(text_tokens)=}")
        return
    else:
        device_map = "auto"

    kw = {
        "text": text,
        "model_name": (model_name, device_map),
        "seed": 0,
        "p_nucleus": p_nucleus,
        "float32": model_name == "qwen-15b",
        "quantize_8bit": quantize_8bit,
        "quantize_4bit": quantize_4bit,
        "token_range_to_mask": None,
        "layers_to_mask": None,
    }
    if device_map == "auto":
        kw["model_name"] = model_name

    try:
        baseline_data = analyze_text_get_p_logits(**kw)
    except (KeyError, torch.OutOfMemoryError, RuntimeError) as e:
        print(f"Probably CUDA failed: {e}. Sleeping for 10 seconds....")
        time.sleep(10)
        return

    entropy_l = []
    for i in range(len(text_tokens)):
        idxs, logits = decompress_logits_for_position(baseline_data, i)
        logits = logits.astype(np.float32)
        # print(f"{logits=}")
        if len(logits) == 1:
            # print("toast")
            entropy = 0
            entropy_l.append(entropy)
            continue
        # Apply temperature scaling to logits
        scaled_logits = logits / temperature
        shifted_logits = scaled_logits - np.max(scaled_logits)
        exp_logits = np.exp(shifted_logits)
        probs = exp_logits / np.sum(exp_logits)

        # Calculate entropy: -sum(p * log(p))
        # Handle the case where p is 0 (log(0) is undefined)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        entropy_l.append(entropy)
    entropy_l = np.array(entropy_l)

    if "only_first" in kwargs:
        only_first = kwargs["only_first"]
    else:
        only_first = None

    sentence_entropies = []
    for sentence_num, token_range in tqdm(
        sentence2ranges.items(), desc=f"Examining sentence2sentence ({problem_num})"
    ):
        start_idx = min(token_range[0], len(entropy_l))
        end_idx = min(token_range[1], len(entropy_l))
        if only_first is not None:
            if end_idx - start_idx > only_first:
                end_idx = start_idx + only_first

        if start_idx < end_idx:
            mean_entropy = np.nanmean(entropy_l[start_idx:end_idx])  # Use nanmean for safety
        else:
            mean_entropy = np.nan  # Assign NaN if range is empty or invalid
        sentence_entropies.append(mean_entropy)

    return sentence_entropies


@pkld
def get_sentence_sentence_KL(
    problem_num: int,
    layers_to_mask: Union[dict, defaultdict],
    p_nucleus: float = 0.9999,
    model_name: str = "qwen-14b",
    quantize_4bit: bool = False,
    quantize_8bit: bool = False,
    problem_dir=os.path.join("target_problems", "temperature_0.6_top_p_0.95"),
    output_dir: str = "suppressed_results",
    only_pre_convergence: bool = False,
    plot_sentences: bool = False,
    **kwargs,
):
    try:
        sentence2ranges, problem = get_full_CoT_token_ranges(
            problem_num,
            problem_dir,
            only_pre_convergence=only_pre_convergence,
            verbose=True,
            model_name=model_name,
        )
    except AssertionError as e:
        print(f"No convergence on {problem_num}: {e}")
        return

    text = (
        problem["full_cot_truncated"]
        if only_pre_convergence
        else problem["base_solution"]["full_cot"]
    )
    assert "qwen" in model_name

    # text = text[:int(len(text) * .75)]

    text_tokens = get_raw_tokens(text, model_name)
    # if len(text_tokens) > 4200:
    #     # device_map = "cpu"
    #     print(f"No CPU today ({problem_num}): {len(text_tokens)=}")
    #     # return
    # else:
    print(f"{len(text_tokens)=}")
    device_map = "auto"

    kw = {
        "text": text,
        "model_name": (model_name, device_map),
        "seed": 0,
        "p_nucleus": p_nucleus,
        "float32": model_name == "qwen-15b",
        "quantize_8bit": quantize_8bit,
        "quantize_4bit": quantize_4bit,
        "token_range_to_mask": None,
        "layers_to_mask": None,
        # "flash_attn": True
    }
    if device_map == "auto":
        kw["model_name"] = model_name

    try:
        if len(text_tokens) > 5100:
            # print('Too large!!')
            # return
            baseline_data = analyze_text_get_p_logits_large(**kw)
        else:
            baseline_data = analyze_text_get_p_logits(**kw)
    except (KeyError, torch.OutOfMemoryError, RuntimeError) as e:
        print(f"Probably CUDA failed: {e}. Sleeping for 10 seconds....")
        time.sleep(10)
        return

    sentence_sentence_scores = np.full((len(sentence2ranges), len(sentence2ranges)), np.nan)

    if "only_first" in kwargs:
        only_first = kwargs["only_first"]
    else:
        only_first = None

    if "take_log" in kwargs:
        take_log = kwargs["take_log"]
    else:
        take_log = True

    if "norm_entropy" in kwargs:
        norm_entropy = kwargs["norm_entropy"]
    else:
        norm_entropy = False

    if norm_entropy:
        entropy_l = get_sentence_entropies(
            problem_num,
            p_nucleus=p_nucleus,
            model_name=model_name,
            quantize_4bit=quantize_4bit,
            quantize_8bit=quantize_8bit,
            only_pre_convergence=only_pre_convergence,
        )
    for (
        sentence_num,
        token_range,
    ) in tqdm(sentence2ranges.items(), desc=f"Examining sentence2sentence ({problem_num})"):
        kw["token_range_to_mask"] = list(token_range)
        kw["layers_to_mask"] = layers_to_mask
        try:
            # baseline_data = analyze_text_get_p_logits(**kw)
            s_data = analyze_text_get_p_logits(**kw)
        except (KeyError, torch.OutOfMemoryError, RuntimeError) as e:
            print(f"Probably CUDA failed: {e}. Sleeping for 10 seconds....")
            time.sleep(10)
            return

        KL_log_l = []
        for i in range(len(text_tokens)):
            # if i == token_range[0]:
            #     print('--------')
            b_idxs, b_logits = decompress_logits_for_position(baseline_data, i)
            s_idxs, s_logits = decompress_logits_for_position(s_data, i)

            KL_sparse = calculate_kl_divergence_sparse(
                (b_idxs, b_logits), (s_idxs, s_logits), temperature=0.6, epsilon=1e-9
            )
            # if norm_entropy:

            if take_log:
                KL_log = np.log(KL_sparse + 1e-9)
            else:
                KL_log = KL_sparse
            KL_log_l.append(KL_log)
            if np.isnan(KL_log):
                print(f"{b_idxs=}")
                print(f"{b_logits=}")
                print(f"{s_idxs=}")
                print(f"{s_logits=}")
                quit()

        KL_log_l = np.array(KL_log_l)
        sentence_KL_logs = []
        # Calculate mean log KL divergence for each sentence
        for sentence_idx_loop, token_range_loop in sentence2ranges.items():
            # Ensure indices are within bounds of KL_log_l
            start_idx = min(token_range_loop[0], len(KL_log_l))
            end_idx = min(token_range_loop[1], len(KL_log_l))
            if only_first is not None:
                if end_idx - start_idx > only_first:
                    end_idx = start_idx + only_first

            if start_idx < end_idx:
                mean_log_kl = np.nanmean(KL_log_l[start_idx:end_idx])  # Use nanmean for safety
            else:
                mean_log_kl = np.nan  # Assign NaN if range is empty or invalid
            sentence_KL_logs.append(mean_log_kl)
            sentence_sentence_scores[
                sentence_idx_loop,
                sentence_num,
            ] = mean_log_kl
            # if sentence_idx_loop > 50:
            #     break
        if not plot_sentences:
            continue

        # Convert to float array explicitly *before* creating the mask
        sentence_KL_logs = np.array(sentence_KL_logs, dtype=float)  # Ensure float type

        sentence_indices = np.array(
            list(sentence2ranges.keys())
        )  # Get sentence numbers (0, 1, 2...)

        # --- Plotting Section ---
        plt.figure(figsize=(12, 7))  # Make figure a bit larger

        # Filter out potential NaNs before plotting the line to avoid gaps/errors
        valid_indices = ~np.isnan(sentence_KL_logs)  # Create boolean mask

        # 1. Plot the main line connecting all points
        if np.any(valid_indices):
            # Use the boolean mask for indexing both arrays
            plt.plot(
                sentence_indices[valid_indices],
                sentence_KL_logs[valid_indices],
                linewidth=1,
                markersize=2,
                marker=".",
                linestyle="-",
                label="Mean Log KL per Sentence",
            )
        else:
            print("Warning: All sentence KL logs are NaN, cannot plot line.")

        # 2. Add threshold highlighting and labels
        threshold = -5.0  # Define the threshold for highlighting
        highlight_count = 0
        for idx, log_kl in enumerate(sentence_KL_logs):
            if not np.isnan(log_kl) and log_kl > threshold:
                sentence_idx_plot = sentence_indices[idx]  # Get the actual sentence number
                # Add a more prominent marker
                plt.plot(
                    sentence_idx_plot,
                    log_kl,
                    marker="o",
                    markersize=6,
                    color="orange",
                    linestyle="",
                )  # No line for marker
                # Add text label
                plt.text(
                    sentence_idx_plot + 0.1,
                    log_kl,
                    f"{sentence_idx_plot}",  # Offset slightly
                    ha="left",
                    va="bottom",
                    fontsize=9,
                    color="black",
                )
                highlight_count += 1

        print(f"Highlighted {highlight_count} points above threshold {threshold}")

        # 3. Add vertical line for the suppressed sentence
        # Get the min/max for y-axis limits, ignoring NaNs
        if np.any(valid_indices):
            kl_log_low = np.nanmin(sentence_KL_logs)
            kl_log_high = np.nanmax(sentence_KL_logs)
            # Add some padding if min/max are the same or very close
            padding = (kl_log_high - kl_log_low) * 0.05
            if padding < 0.1:
                padding = 0.1  # Ensure minimum padding
            y_min = kl_log_low - padding
            y_max = kl_log_high + padding
        else:
            y_min, y_max = -6, -4  # Default range if no valid data

        plt.vlines(
            sentence_num,
            y_min,
            y_max,
            color="red",
            linestyle="--",
            linewidth=1,
            label=f"Suppressed Sent {sentence_num}",
        )
        # plt.text(sentence_num + 0.2, y_max * 0.95, f'Suppressed: {sentence_num}', # Adjusted position
        #          ha='left', va='top', fontsize=10, color='red')

        # --- Final plot adjustments ---
        plt.ylim(y_min, y_max)  # Set y-limits based on data range
        plt.ylabel("Mean Log(KL Divergence + 1e-9)")  # More descriptive label
        plt.xlabel("Sentence Number")
        plt.title(
            f"Impact of Suppressing Sentence {sentence_num} (Tokens {token_range}) on Sentence Logits\nProblem: {problem_num}, Model: {model_name}"
        )
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend()  # Show legend for the line and vline

        hash_layers_to_mask = hash_dict(layers_to_mask)
        dir_out = f"plot_suppression/{hash_layers_to_mask}"
        Path(dir_out).mkdir(parents=True, exist_ok=True)
        # --- Saving ---
        fp_out = rf"{dir_out}/{problem_num}/suppress_{sentence_num}_tok{token_range[0]}-{token_range[1]}.png"
        print(f"Plotting: {fp_out}")
        Path(fp_out).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fp_out)
        print(f"Saved plot to: {fp_out}")
        plt.close()
        # --- End Plotting Section ---
    return sentence_sentence_scores


def calculate_kl_divergence_sparse(
    baseline_data: Tuple[np.ndarray, np.ndarray],
    suppressed_data: Tuple[np.ndarray, np.ndarray],
    temperature: float = 0.6,
    epsilon: float = 1e-9,  # Epsilon currently unused in KL calc, but kept for signature
) -> float:
    """
    Calculates the KL divergence KL(P || Q) between two probability distributions
    derived from sparse top-p logits. Clips small negative results to 0.

    P is derived from baseline_data, Q from suppressed_data.

    Args:
        baseline_data (Tuple[np.ndarray, np.ndarray]): Tuple containing
            (indices, logits) for the baseline distribution (P).
            Indices should be int32, logits float16 or float32.
        suppressed_data (Tuple[np.ndarray, np.ndarray]): Tuple containing
            (indices, logits) for the suppressed distribution (Q).
        temperature (float): Temperature to use for softmax conversion. Defaults to 0.6.
        epsilon (float): Small value (currently unused here, handled internally).

    Returns:
        float: The calculated KL divergence KL(P || Q), guaranteed non-negative.
               Returns np.nan if inputs are invalid or calculation yields NaN/inf.
    """
    b_idxs, b_logits = baseline_data
    s_idxs, s_logits = suppressed_data

    if b_idxs is None or b_logits is None or s_idxs is None or s_logits is None:
        print("Warning: Invalid input data (None found). Cannot calculate KL divergence.")
        return np.nan
    if len(b_idxs) != len(b_logits) or len(s_idxs) != len(s_logits):
        print(
            "Warning: Mismatch between length of indices and logits. Cannot calculate KL divergence."
        )
        return np.nan
    # Allow empty arrays for one side? KL(P||0) is inf if P>0, KL(0||Q) is 0.
    # If both are empty, KL is 0 or NaN? Let's return 0 if both empty, NaN otherwise for now.
    if len(b_idxs) == 0 and len(s_idxs) == 0:
        return 0.0
    if len(b_idxs) == 0 or len(s_idxs) == 0:
        # If P is non-empty and Q is empty -> Div is Inf
        # If P is empty and Q is non-empty -> Div is 0
        # Let's return NaN for simplicity unless explicitly handled otherwise.
        # Returning inf might be more correct if P is non-empty.
        # For now, keep NaN to signal an edge case was hit.
        print("Warning: One distribution has no tokens. Returning NaN.")
        return np.nan

    # Ensure logits are float32 for stable softmax
    b_logits = b_logits.astype(np.float32)
    s_logits = s_logits.astype(np.float32)

    # 1. Find the union of indices
    union_indices = np.union1d(b_idxs, s_idxs)
    union_size = len(union_indices)

    # 2. Create mapping from union index to its position in the union array
    idx_to_union_pos = {idx: pos for pos, idx in enumerate(union_indices)}

    # 3. Create dense logit vectors over the union set
    min_logit_val = -1e9  # Approx -inf for softmax
    b_logits_union = np.full(union_size, min_logit_val, dtype=np.float32)
    s_logits_union = np.full(union_size, min_logit_val, dtype=np.float32)

    for idx, logit in zip(b_idxs, b_logits):
        b_logits_union[idx_to_union_pos[idx]] = logit
    for idx, logit in zip(s_idxs, s_logits):
        s_logits_union[idx_to_union_pos[idx]] = logit

    # 4. Convert to PyTorch tensors
    b_logits_tensor = torch.from_numpy(b_logits_union)
    s_logits_tensor = torch.from_numpy(s_logits_union)

    # 5. Calculate log probabilities directly (more stable)
    log_p = F.log_softmax(b_logits_tensor / temperature, dim=0)
    log_q = F.log_softmax(s_logits_tensor / temperature, dim=0)

    # Check for immediate issues after log_softmax
    if (
        torch.isinf(log_p).any()
        or torch.isnan(log_p).any()
        or torch.isinf(log_q).any()
        or torch.isnan(log_q).any()
    ):
        print("Warning: Inf or NaN detected in log-probabilities. Inputs might be too extreme.")
        return np.nan

    # 6. Calculate KL divergence KL(P || Q) = sum(exp(logP) * (logP - logQ))
    p_dist = torch.exp(log_p)  # Equivalent to softmax(logits/T)

    # Check p_dist just in case
    if torch.isnan(p_dist).any():
        print("Warning: NaN detected in P distribution.")
        return np.nan

    # Calculate terms: p * (log p - log q)
    kl_terms = p_dist * (log_p - log_q)

    # Explicitly handle the case where p_dist is 0: the term should be 0
    # This prevents 0 * log(0/q) -> 0 * (-inf) -> NaN
    kl_terms = torch.where(p_dist == 0, torch.tensor(0.0, dtype=kl_terms.dtype), kl_terms)

    # Check for NaNs that might have arisen from other issues (e.g., inf - inf if log_p and log_q were inf)
    if torch.isnan(kl_terms).any():
        print("Warning: NaN detected during KL term calculation.")
        return np.nan

    # Sum the terms
    kl_div = torch.sum(kl_terms)

    # Handle potential infinities (e.g., if p>0 and q=0 -> log_q = -inf -> term = inf)
    if torch.isinf(kl_div):
        print("Warning: KL divergence is infinite.")
        # Returning inf is mathematically correct here. Caller must handle.
        return kl_div.item()  # This will be float('inf')

    kl_div_value = kl_div.item()

    # --- CLIP SMALL NEGATIVE VALUES ---
    # Final check for small negative values due to floating point errors
    if kl_div_value < 0:
        # Check if it's significantly negative or just noise
        if kl_div_value < -1e-6:  # Adjust tolerance if needed
            print(
                f"Warning: KL divergence significantly negative ({kl_div_value:.2e}). Clipping to 0.0. This might indicate an issue."
            )
        # else:
        # print(f"Warning: KL divergence slightly negative ({kl_div_value:.2e}), clipping to 0.0.")
        return 0.0
    else:
        # Return the non-negative, non-NaN, potentially infinite value
        return kl_div_value


if __name__ == "__main__":
    problem_num = 223601
    # problem_num = 69980
    layers_to_mask = {i: list(range(40)) for i in range(48)}
    only_pre_convergence = False

    output_file = get_sentence_sentence_KL(
        # problem_num=problem_to_run,
        problem_num=problem_num,
        layers_to_mask=layers_to_mask,
        p_nucleus=0.9999,  # Example p value
        model_name="qwen-14b",  # Make sure this matches your available model
        quantize_4bit=False,  # Use 4-bit quantization for memory efficiency
        quantize_8bit=False,
        problem_dir=os.path.join(
            "target_problems", "temperature_0.6_top_p_0.95"
        ),  # Adjust if needed
        output_dir="suppressed_results_test",  # Save to a test directory
        only_pre_convergence=only_pre_convergence,
    )
