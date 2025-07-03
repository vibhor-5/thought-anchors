import json
import os
import torch

import torch


from utils import hash_dict


from examine_suppression_logits import get_sentence_sentence_KL
from repeated_suppression_logits import get_most_sensitive_heads_map
from run_target_problems import get_problem_nums
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd

from run_target_problems import (
    get_full_CoT_token_ranges,
    get_problem_nums,
    load_problem_json,
)  # Use existing problem loader

from utils import (
    # get_qwen_14b_tokens_lower,
    # get_qwen_raw_tokens,
    # get_qwen_tokenizer,
    get_raw_tokens,
    get_top_p_logits,
    hash_dict,
    print_gpu_memory_summary,
)  # Use existing utils


def plot_supp_matrix_nice(
    sentence_sentence_scores0,
    sentence_sentence_scores1,
    model_name,
    problem_num,
    only_pre_convergence,
    only_first,
    take_log,
):
    plot_dir = rf"plot_suppression/{model_name}_both/{only_pre_convergence}"
    Path(plot_dir).mkdir(parents=True, exist_ok=True)
    trils = np.tril_indices_from(sentence_sentence_scores0, k=-11)
    sentence_scores0_ = sentence_sentence_scores0[trils]
    sentence_scores1_ = sentence_sentence_scores1[trils]

    sentence_scores = np.concatenate([sentence_scores0_, sentence_scores1_], axis=0)
    vmin = np.nanquantile(sentence_scores, 0.05)
    vmax = np.nanquantile(sentence_scores, 0.95)

    fig, axs = plt.subplots(2, 2, figsize=(10, 5))
    plt.sca(axs[0, 0])
    plt.title(f"Reasoning model: {problem_num}")
    plt.imshow(sentence_sentence_scores, vmin=vmin, vmax=vmax)
    plt.colorbar()

    # Plot first histogram and get its y-axis limits
    plt.sca(axs[0, 1])
    if take_log:
        plt.hist(sentence_scores0_, bins=100, range=(-21, 0))
    else:
        plt.hist(sentence_scores0_, bins=100)
    plt.title(f"Reasoning model: {problem_num}")
    ylim0 = plt.ylim()

    plt.sca(axs[1, 0])
    plt.imshow(sentence_sentence_scores_base, vmin=vmin, vmax=vmax)
    plt.title(f"Base model: {problem_num}")
    plt.colorbar()

    # Plot second histogram
    plt.sca(axs[1, 1])
    if take_log:
        plt.hist(sentence_scores1_, bins=100, range=(-21, 0))
    else:
        plt.hist(sentence_scores1_, bins=100)
    plt.title(f"Base model: {problem_num}")
    ylim1 = plt.ylim()

    # Set both histograms to use the higher y-limit
    max_ylim = max(ylim0[1], ylim1[1])
    plt.sca(axs[0, 1])
    plt.ylim(0, max_ylim)
    plt.sca(axs[1, 1])
    plt.ylim(0, max_ylim)

    fp_out = os.path.join(plot_dir, f"mat_{problem_num}.png")
    plt.savefig(fp_out)
    plt.tight_layout()
    plt.close()
    print(f"Saved to {fp_out}")


def plot_supp_matrix_subset(
    sentence_sentence_scores, model_name, problem_num, only_pre_convergence, st, end
):
    plot_dir = rf"plot_suppression/{model_name}/{only_pre_convergence}"
    Path(plot_dir).mkdir(parents=True, exist_ok=True)
    sentence_scores_trils = np.tril(sentence_sentence_scores, k=-1)
    vmin = np.nanquantile(sentence_scores_trils, 0.05)
    vmax = np.nanquantile(sentence_scores_trils, 0.95)
    plt.clf()
    plt.imshow(sentence_sentence_scores, vmin=vmin, vmax=vmax)
    plt.colorbar()
    fp_out = os.path.join(plot_dir, f"mat_{problem_num}.png")
    plt.savefig(fp_out)
    print(f"Saved to {fp_out}")

    st = 0
    end = min(80, sentence_sentence_scores.shape[0])
    subset = sentence_sentence_scores[st:end, st:end]
    plt.clf()
    subset[np.diag_indices_from(subset)] = np.nan
    im = plt.imshow(subset, vmin=vmin, vmax=vmax)
    # plt.xticks(range(end-st), range(st, end))
    # plt.yticks(range(end-st), range(st, end))
    plt.xticks(range(0, end - st, 5), range(st, end, 5))
    plt.yticks(range(0, end - st, 5), range(st, end, 5))
    # plt.colorbar()
    cbar = plt.colorbar(im)

    # cbar.ax.set_ylabel('Suppression KL', rotation=270, labelpad=15)
    plt.title(f"Suppression matrix subset (sentences {st}-{end})\nfor problem: {problem_num}")
    cbar.ax.set_title("Suppression KL")
    fp_out = os.path.join(plot_dir, f"mat_{problem_num}_crop_{st}_{end}.png")
    plt.tight_layout()
    plt.savefig(fp_out)
    # print(f'Saved to {fp_out}')
    # quit()

    sentence_sentence_scores_nan = np.copy(sentence_sentence_scores)
    sentence_sentence_scores_nan[np.triu_indices_from(sentence_sentence_scores)] = np.nan
    subtract = np.nanmean(sentence_sentence_scores_nan, axis=1)
    sentence_sentence_scores_nan -= subtract[:, None]
    sentence_scores_trils = np.tril(sentence_sentence_scores_nan, k=-1)
    vmin = np.nanquantile(sentence_scores_trils, 0.05)
    vmax = np.nanquantile(sentence_scores_trils, 0.95)
    plt.clf()
    plt.imshow(sentence_sentence_scores_nan, vmin=vmin, vmax=vmax)
    plt.colorbar()
    fp_out = os.path.join(plot_dir, f"mat_{problem_num}_sub.png")
    plt.savefig(fp_out)
    print(f"Saved to {fp_out}")


def dict2hash(layers_to_mask):
    d_clean = {}
    for k, v in layers_to_mask.items():
        if isinstance(v, list):
            d_clean[int(k)] = [int(i) for i in v]
        elif isinstance(v, np.ndarray):
            d_clean[int(k)] = [int(i) for i in v.tolist()]
        else:
            d_clean[int(k)] = v
    dict_str = json.dumps(d_clean, sort_keys=True)
    return dict_str, d_clean


def make_dict_dir(layers_to_mask):
    dict_str, d_clean = dict2hash(layers_to_mask)
    plot_dir = f"plot_suppression/{dict_str}"
    Path(plot_dir).mkdir(parents=True, exist_ok=True)
    fp_json = os.path.join(plot_dir, f"readme.json")
    with open(fp_json, "w") as f:
        json.dump(d_clean, f)


if __name__ == "__main__":
    problem_to_run = 33001  # Example problem number from HumanEvalX
    only_pre_convergence = "semi"
    only_pre_convergence = False
    # only_pre_convergence = "semi"
    model_name = "qwen-14b"
    only_first = None
    take_log = True

    problem_dir = os.path.join("target_problems", "temperature_0.6_top_p_0.95")

    # selected_targets = targets_specific
    # selected_targets = targets_layers_only

    only = None
    problem_nums = get_problem_nums(only=only, only_pre_convergence=only_pre_convergence)
    assert len(problem_nums) > 0

    layers_to_mask = {i: list(range(40)) for i in range(48)}

    problem_nums = sorted(problem_nums)

    # text_tokens_list = []
    # num_sentences_list = []
    # for problem_num in problem_nums:
    #     sentence2ranges, problem = get_full_CoT_token_ranges(
    #         problem_num,
    #         problem_dir,
    #         only_pre_convergence=only_pre_convergence,
    #         verbose=True,
    #         model_name=model_name,
    #     )
    #     text = (
    #         problem["full_cot_truncated"]
    #         if only_pre_convergence
    #         else problem["base_solution"]["full_cot"]
    #     )
    #     assert "qwen" in model_name
    #     text_tokens = get_raw_tokens(text, model_name)
    #     print(f"{problem_num} | {len(text_tokens)=}")
    #     text_tokens_list.append(text_tokens)
    #     num_sentences_list.append(len(sentence2ranges))

    # # for problem_num, text_tokens, num_sentences in zip(problem_nums, text_tokens_list, num_sentences_list):
    # #     print(f"{problem_num} | {len(text_tokens)=} | {num_sentences=}")
    # # quit()
    problems_good = []
    problem_nums = [468201]
    for problem_num in problem_nums:
        # if int(problem_num) != 344801:
        #     continue
        model_name = "qwen-14b"
        sentence_sentence_scores = get_sentence_sentence_KL(
            # problem_num=problem_to_run,
            problem_num=problem_num,
            layers_to_mask=layers_to_mask,
            p_nucleus=0.9999,  # Example p value
            model_name=model_name,  # Make sure this matches your available model
            quantize_4bit=False,  # Use 4-bit quantization for memory efficiency
            quantize_8bit=False,
            problem_dir=problem_dir,  # Adjust if needed
            output_dir="suppressed_results_test",  # Save to a test directory
            only_pre_convergence=only_pre_convergence,
            plot_sentences=False,
            only_first=only_first,
            take_log=take_log,
        )
        if sentence_sentence_scores is None:
            print(f"Skipping {problem_num} due to None result")
            continue
        print(f"{sentence_sentence_scores.shape=}")
        continue
        # plt.imshow(sentence_sentence_scores)
        # plt.colorbar()
        # plt.show()
        # quit()

        print(f"DONE R1: {problem_num}")
        model_name = "qwen-14b-base"

        sentence_sentence_scores_base = get_sentence_sentence_KL(
            # problem_num=problem_to_run,
            problem_num=problem_num,
            layers_to_mask=layers_to_mask,
            p_nucleus=0.9999,  # Example p value
            model_name=model_name,  # Make sure this matches your available model
            quantize_4bit=False,  # Use 4-bit quantization for memory efficiency
            quantize_8bit=False,
            problem_dir=problem_dir,  # Adjust if needed
            output_dir="suppressed_results_test",  # Save to a test directory
            only_pre_convergence=only_pre_convergence,
            plot_sentences=False,
            only_first=only_first,
            take_log=take_log,
        )
        plot_supp_matrix_nice(
            sentence_sentence_scores,
            sentence_sentence_scores_base,
            model_name,
            problem_num,
            only_pre_convergence,
            only_first,
            take_log,
        )
        print(f"DONE base: {problem_num}")
        # continbase
