from pathlib import Path
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import os

import sys
import numpy as np

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examine_suppression_logits import get_sentence_sentence_KL
from run_target_problems import get_problem_nums


def plot_supp_matrix_nice(
    sentence_sentence_scores0,
    model_name,
    problem_num,
    only_pre_convergence,
    quantiles=(0.05, 0.95),
):
    plot_dir = rf"plot_suppression/{model_name}/{only_pre_convergence}"
    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    # sentence_sentence_scores0[np.diag_indices_from(sentence_sentence_scores0)] = np.nan

    trils = np.tril_indices_from(sentence_sentence_scores0, k=-11)
    sentence_scores = sentence_sentence_scores0[trils]

    vmin = np.nanquantile(sentence_scores, quantiles[0])
    vmax = np.nanquantile(sentence_scores, quantiles[1])

    plt.title(f"Reasoning model: {problem_num}")
    cmap = white_to_reds()
    plt.imshow(sentence_sentence_scores0, vmin=vmin, vmax=vmax, cmap=cmap)
    # plt.colorbar()

    fp_out = os.path.join(plot_dir, f"mat_{problem_num}.png")
    # plt.savefig(fp_out)
    # plt.tight_layout()
    # plt.close()
    # print(f"Saved to {fp_out}")


def get_white_reds():
    # Create a custom colormap from white to blue
    colors = [(1, 1, 1), (1, 0, 0)]  # White to Blue
    cmap_name = "white_to_red"
    white_red_cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors)
    return white_red_cmap


# Create a custom function to blend white with Blues
def white_to_reds(N=256):
    reds = plt.cm.Reds

    # Get Blues colormap array
    reds_colors = reds(np.linspace(0, 1, N))

    # Create an array that transitions from white to the Blues colormap
    white_color = np.array([1, 1, 1, 1])  # RGBA for white
    white_color = get_white_reds()(np.linspace(0, 1, N))

    # Create weights for blending (from 1 to 0 for white, from 0 to 1 for Blues)
    white_weights = np.linspace(1, 0, N)[:, np.newaxis]
    reds_weights = 1 - white_weights

    # Blend the colors
    colors = white_weights * white_color + reds_weights * reds_colors

    # Create a new colormap
    return mcolors.LinearSegmentedColormap.from_list("WhiteToReds", colors)


def plot_single_problem(problem_num, only_pre_convergence="semi", take_log=True, figsize=(6, 6)):
    """
    Plot a single problem's suppression matrix.

    Args:
        problem_num: The problem number to plot
        only_pre_convergence: Whether to only include pre-convergence steps
        take_log: Whether to take log of values
        figsize: Figure size as (width, height) tuple
    """
    model_name = "qwen-14b"
    problem_dir = os.path.join("target_problems", "temperature_0.6_top_p_0.95")
    layers_to_mask = {i: list(range(40)) for i in range(48)}

    plt.rcParams["font.size"] = 11
    plt.figure(figsize=figsize)

    sentence_sentence_scores = get_sentence_sentence_KL(
        problem_num=problem_num,
        layers_to_mask=layers_to_mask,
        p_nucleus=0.9999,
        model_name=model_name,
        quantize_4bit=False,
        quantize_8bit=False,
        problem_dir=problem_dir,
        output_dir="suppressed_results_test",
        only_pre_convergence=only_pre_convergence,
        plot_sentences=False,
        only_first=None,
        take_log=take_log,
    )

    sentence_sentence_scores[np.triu_indices_from(sentence_sentence_scores, k=0)] = np.nan

    # sentence_sentence_scores_ = sentence_sentence_scores.copy()
    # sentence_sentence_scores_[np.diag_indices_from(sentence_sentence_scores_)] = np.nan
    # print(np.nanmean(sentence_sentence_scores, axis=1))
    # quit()
    for i in range(sentence_sentence_scores.shape[0]):
        sentence_sentence_scores[i, :] -= np.nanmean(sentence_sentence_scores[i, :])
    sentence_sentence_scores[np.diag_indices_from(sentence_sentence_scores)] = np.nanquantile(
        sentence_sentence_scores, 0.99
    )
    # print(sentence_sentence_scores[i, :])
    # quit()

    # sentence_sentence_scores -= np.nanmean(sentence_sentence_scores, axis=1)

    plot_supp_matrix_nice(
        sentence_sentence_scores,
        model_name,
        problem_num,
        only_pre_convergence,
        quantiles=(0.5, 0.998),
    )
    # plt.colorbar()

    plt.tick_params(axis="both", labelsize=11)
    xticks = np.arange(0, sentence_sentence_scores.shape[1], 10)
    plt.xticks(xticks)
    yticks = np.arange(0, sentence_sentence_scores.shape[0], 10)
    plt.yticks(yticks)
    plt.gca().spines[["top", "right"]].set_visible(False)

    pn = str(problem_num)
    is_correct = pn[-1] == "1"
    if is_correct:
        pn_title = f"Problem: {pn[:-2]} (correct)"
    else:
        pn_title = f"Problem: {pn[:-2]} (incorrect)"
    plt.title(pn_title)

    plt.xlabel("Sentence position", fontsize=14)
    plt.ylabel("Sentence position", fontsize=14)

    plt.tight_layout()
    plt.savefig(
        f"plots/suppression_single_problem_{problem_num}_{model_name}_{only_pre_convergence}.png",
        dpi=600,
        transparent=True,
    )
    plt.show()


if __name__ == "__main__":
    plot_single_problem(468201)
    quit()

    only_pre_convergence = "semi"
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
    print(f"{problem_nums=}")

    problems_good = []
    # problem_nums = [159100, 33000, 344801, 468201]
    problem_nums = [
        468201,
        468201,
        159100,
        159100,
    ]
    plt.rcParams["font.size"] = 11

    # Change from 1 row, 4 columns to 2 rows, 2 columns
    fig, axs = plt.subplots(2, 2, figsize=(6, 6))
    # Flatten the 2D array of axes for easier indexing
    axs = axs.flatten()

    for i, problem_num in enumerate(problem_nums):
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

        plt.sca(axs[i])
        plot_supp_matrix_nice(
            sentence_sentence_scores,
            model_name,
            problem_num,
            only_pre_convergence,
            take_log,
        )

        axs[i].tick_params(axis="both", labelsize=11)
        xticks = np.arange(0, sentence_sentence_scores.shape[1], 25)
        axs[i].set_xticks(xticks)
        yticks = np.arange(0, sentence_sentence_scores.shape[0], 25)
        axs[i].set_yticks(yticks)
        axs[i].spines[["top", "right"]].set_visible(False)

        pn = str(problem_num)
        is_correct = pn[-1] == "1"
        if is_correct:
            pn_title = f"Problem: {pn[:-2]} (correct)"
        else:
            pn_title = f"Problem: {pn[:-2]} (incorrect)"
        plt.title(pn_title)

        # Set labels only for bottom row (x-axis) and left column (y-axis)
        if i >= 2:  # Bottom row
            axs[i].set_xlabel("Sentence position", fontsize=14)
        if i % 2 == 0:  # Left column
            axs[i].set_ylabel("Sentence position", fontsize=14)

    plt.tight_layout()
    plt.savefig(
        f"plots/suppression_examples_{model_name}_{only_pre_convergence}.png",
        dpi=300,
        transparent=True,
    )
    plt.show()
