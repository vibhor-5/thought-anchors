from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import sys
import numpy as np

from evaluate_head_reliability import get_kurt_matrix
from examine_suppression_logits import get_sentence_entropies
from run_target_problems import get_full_CoT_token_ranges

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pkld import pkld

import matplotlib.pyplot as plt
import os

from run_target_problems import (
    get_most_sensitive_layer_heads,
    get_problem_attn_avg,
    get_problem_nums,
)


def plot_kurt_data(qwen=True):
    if qwen:
        model_name = "qwen-14b"
    else:
        model_name = "llama8"
    quantize_8bit = False
    quantize_4bit = False
    only_pre_convergence = "semi"
    qwen_llama_str = "_qwen" if qwen else "_llama"
    problem_dir = os.path.join("target_problems", f"temperature_0.6_top_p_0.95{qwen_llama_str}")
    drop_first = 4
    drop_last = 32
    proximity_ignore = 4
    vert_score_calc = None
    weighted_avg = False  # TOD: ISSUE, errors for Llama

    kurts = get_kurt_matrix(
        model_name=model_name,
        quantize_8bit=quantize_8bit,
        quantize_4bit=quantize_4bit,
        only_pre_convergence=only_pre_convergence,
        problem_dir=problem_dir,
        drop_first=drop_first,
        drop_last=drop_last,
        proximity_ignore=proximity_ignore,
        vert_score_calc=vert_score_calc,
    )
    print(f"{kurts.shape=}")
    # quit()
    kurts[:, 0, :] = np.nan  # ignore layer 0

    if weighted_avg:

        problem_nums = get_problem_nums(
            only=None,
            only_pre_convergence=only_pre_convergence,
            problem_dir=problem_dir,
        )
        problem_lens = []
        for pn in problem_nums:
            sentence2ranges, problem = pkld(get_full_CoT_token_ranges)(
                pn, problem_dir, only_pre_convergence=only_pre_convergence, model_name=model_name
            )
            n_sentences = len(sentence2ranges)
            # if n_sentences <= drop_first + drop_last:
            #     continue

            problem_lens.append(n_sentences - drop_first - drop_last)
        print(f"{len(problem_lens)=}")
        kurt = np.average(kurts, axis=0, weights=problem_lens)
    else:
        kurt = np.mean(kurts, axis=0)

    layer_l = []
    kurt_l = []
    for layer in range(kurt.shape[0]):
        for j in range(kurt.shape[1]):
            if np.isnan(kurt[layer, j]):
                continue
            kurt_l.append(kurt[layer, j])
            layer_l.append(layer)
    layer_l = np.array(layer_l)
    kurt_l = np.array(kurt_l)

    plt.rcParams["font.size"] = 11

    fig = plt.figure(figsize=(4.5, 3.5))
    # plt.axis("square")

    plt.scatter(layer_l, kurt_l, color="dodgerblue", alpha=0.25, s=20)
    if qwen:
        plt.xlim(0, 48)
    else:
        plt.xlim(0, 32)
    plt.xlabel("Layer", labelpad=7)
    plt.ylabel("Kurtosis", labelpad=7)
    plt.title("Kurtosis of each attention head's\nvertical score", fontsize=12, pad=0)
    if drop_first > 0 or drop_last > 0:
        drop_str = f"_drop_{drop_first}-{drop_last}"
    else:
        drop_str = ""

    plt.gca().spines[["top", "right"]].set_visible(False)

    pi_str = f"_pi{proximity_ignore}"
    fp_out = f"plots/kurt_plots/kurt_layer_scatter_{model_name}_{only_pre_convergence}{drop_str}{pi_str}.png"
    Path(fp_out).parent.mkdir(parents=True, exist_ok=True)
    plt.subplots_adjust(bottom=0.17, top=0.8, left=0.2, right=0.95)
    plt.savefig(fp_out, dpi=300)
    # plt.show()
    plt.close()

    fig = plt.figure(figsize=(3, 3.5))
    # Flatten the matrix
    flat = kurt.flatten()
    plt.rcParams["font.size"] = 12
    if qwen:
        plt.hist(flat, bins=80, color="dodgerblue", range=(0, 40))
    else:
        plt.hist(flat, bins=80, color="dodgerblue", range=(0, 40))
    # plt.vlines(3, 0, 100, color="k", linestyle="--")
    # print(np.nanmin(flat), np.nanmax(flat))
    # quit()
    plt.xlim(-1, None)
    plt.title(
        "Histogram of attention head\nvertical score kurtoses",
        fontsize=12,
        pad=0,
    )
    plt.ylabel("Count", labelpad=7)
    plt.xlabel("Kurtosis", labelpad=7)
    if qwen:
        plt.xticks(np.arange(0, 42, 10))
    else:
        plt.xticks(np.arange(0, 42, 10))
    plt.gca().spines[["top", "right"]].set_visible(False)

    fp_plot = (
        f"plots/kurt_plots/kurt_hist_{model_name}_{only_pre_convergence}_{drop_str}{pi_str}.png"
    )
    Path(fp_plot).parent.mkdir(parents=True, exist_ok=True)
    plt.subplots_adjust(bottom=0.17, top=0.8, left=0.25, right=0.95)
    plt.savefig(fp_plot, dpi=300)
    plt.close()


if __name__ == "__main__":
    plot_kurt_data(qwen=True)
