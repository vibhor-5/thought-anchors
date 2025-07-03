from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import sys
import numpy as np

from evaluate_head_reliability import get_kurt_matrix
from examine_suppression_logits import get_sentence_entropies

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import matplotlib.pyplot as plt
import os

from run_target_problems import (
    get_most_sensitive_layer_heads,
    get_problem_attn_avg,
    get_problem_nums,
)


def get_white_blues():
    # Create a custom colormap from white to blue
    colors = [(1, 1, 1), (0, 0, 1)]  # White to Blue
    cmap_name = "white_to_blue"
    white_blue_cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors)
    return white_blue_cmap


# Create a custom function to blend white with Blues
def white_to_blues(N=256):
    blues = plt.cm.Blues

    # Get Blues colormap array
    blues_colors = blues(np.linspace(0, 1, N))

    # Create an array that transitions from white to the Blues colormap
    white_color = np.array([1, 1, 1, 1])  # RGBA for white
    white_color = get_white_blues()(np.linspace(0, 1, N))

    # Create weights for blending (from 1 to 0 for white, from 0 to 1 for Blues)
    white_weights = np.linspace(1, 0, N)[:, np.newaxis]
    blues_weights = 1 - white_weights

    # Blend the colors
    colors = white_weights * white_color + blues_weights * blues_colors

    # Create a new colormap
    return mcolors.LinearSegmentedColormap.from_list("WhiteToBlues", colors)


if __name__ == "__main__":
    # plot_one(pn=159101, layer=36, head=6, top_k=None)
    # # # plot_one(pn=468201, layer=36, head=6)
    # # plt.show()
    # # quit()

    quantize_8bit = False
    quantize_4bit = False

    problem_dir = os.path.join("target_problems", "temperature_0.6_top_p_0.95")
    only_pre_convergence = "semi"
    model_name = "qwen-14b"

    problem_nums = get_problem_nums(only=None, only_pre_convergence=only_pre_convergence)

    targets = [
        [(36, 6, 159100), (36, 6, 33000), (36, 6, 344801), (36, 6, 659601)],
        [(42, 33, 159100), (42, 19, 159100), (36, 25, 159100), (6, 18, 159100)],
    ]

    targets[0][0] = (36, 6, 159100)

    # targets = [
    #     [(36, 6, 159100), (36, 6, 33000), (36, 6, 344801), (36, 6, 659601)],
    #     [(42, 33, 159100), (42, 19, 159100), (36, 25, 159100), (6, 18, 159100)],
    # ]

    coords = get_most_sensitive_layer_heads(
        100,
        model_name=model_name,
        quantize_8bit=False,
        quantize_4bit=False,
        only_pre_convergence="semi",
        only=None,
        problem_dir=problem_dir,
        drop_first=4,
        drop_last=32,
        proximity_ignore=4,
        # vert_score_calc="median",
        pool_before=False,
        # pns=[pn],
    )
    # print(coords)
    # quit()

    n_row = 5
    n_col = 4

    target_layer = 36
    target_head = 6
    target_layer, target_head = coords[0]

    idxs = np.arange(20)
    idxs = np.reshape(idxs, (n_row, n_col))
    targets = [[None] * n_col for _ in range(n_row)]
    for i in range(n_row):
        for j in range(n_col):
            targets[i][j] = (target_layer, target_head, problem_nums[idxs[i, j]])

    # for i in range(n_row):
    #     for j in range(n_col):
    #         targets[i][j] = (*coords[idxs[i, j]], 159100)

    head_all_same = True
    pn_all_same = False

    fig, axs = plt.subplots(len(targets), len(targets[0]), figsize=(7, 8.5))

    for row_idx, row in enumerate(targets):
        for col_idx, target in enumerate(row):
            print(f"Cooking: {row_idx}/{col_idx}")
            # target = 36, 6, 159100
            layer, head, pn = target
            # pn = problem_nums.pop(0)
            # pn = 159100
            # layer, head = coords.pop(0)
            avg_matrix = get_problem_attn_avg(
                pn,
                layer,
                head,
                problem_dir=problem_dir,
                only_pre_convergence=only_pre_convergence,
                model_name=model_name,
                quantize_8bit=quantize_8bit,
                quantize_4bit=quantize_4bit,
            )
            avg_matrix = avg_matrix[1:-1, 1:-1]
            # avg_matrix = avg_matrix[:129, :129]
            avg_matrix_tril = np.tril(avg_matrix)
            vmin = np.nanquantile(avg_matrix_tril, 0.005)
            vmax = np.nanquantile(avg_matrix_tril, 0.995)

            # avg_matrix[:, :] = 0

            white_blue_cmap = white_to_blues()

            axs[row_idx, col_idx].imshow(avg_matrix, vmin=0, vmax=vmax, cmap=white_blue_cmap)
            pn = str(pn)
            is_correct = pn[-1] == "1"

            if pn_all_same:
                title = f"Layer: {layer}, Head: {head}"
            else:
                if is_correct:
                    pn_title = f"#{pn[:-2]} (correct)"
                else:
                    pn_title = f"#{pn[:-2]} (incorrect)"
                if not head_all_same:
                    title = f"{pn_title}\nLayer: {layer}, Head: {head}"
                else:
                    title = pn_title
            axs[row_idx, col_idx].set_title(title, fontsize=11 if head_all_same else 10)

            axs[row_idx, col_idx].tick_params(axis="both", labelsize=10)

            if avg_matrix.shape[0] > 124:
                xticks = np.arange(0, avg_matrix.shape[1], 50)
                yticks = np.arange(0, avg_matrix.shape[0], 50)
            elif avg_matrix.shape[0] > 30:
                xticks = np.arange(0, avg_matrix.shape[1], 25)
                yticks = np.arange(0, avg_matrix.shape[0], 25)
            else:
                xticks = np.arange(0, avg_matrix.shape[1], 10)
                yticks = np.arange(0, avg_matrix.shape[0], 10)

            # Format tick labels to have consistent width
            axs[row_idx, col_idx].set_xticks(xticks)
            axs[row_idx, col_idx].set_yticks(yticks)
            # axs[row_idx, col_idx].set_xticklabels([f"{x:>3}" for x in xticks])
            # axs[row_idx, col_idx].set_yticklabels([f"{y:>3}" for y in yticks])
            axs[row_idx, col_idx].set_xlim(0, avg_matrix.shape[1] - 1)
            axs[row_idx, col_idx].set_ylim(avg_matrix.shape[0] - 1, 0)

            # Remove top and right spines
            axs[row_idx, col_idx].spines["top"].set_visible(False)
            axs[row_idx, col_idx].spines["right"].set_visible(False)
            if row_idx == len(targets) - 1:
                axs[row_idx, col_idx].set_xlabel("Sentence position", fontsize=10)
            if col_idx == 0:
                axs[row_idx, col_idx].set_ylabel("Sentence position", fontsize=10, labelpad=5)

    if head_all_same:
        plt.suptitle(f"All responses for layer: {target_layer}, head: {target_head}", fontsize=13)
    elif pn_all_same:
        plt.suptitle(f"Attention weights for problem: #1591 (incorrect)", fontsize=13)

    plt.tight_layout()
    fig.align_labels()

    if head_all_same:
        plt.savefig(f"plots/receiver_head_{target_layer}_{target_head}_pn_examples.png", dpi=300)
    elif pn_all_same:
        plt.savefig(f"plots/receiver_pn_{pn}_examples.png", dpi=300)
    plt.close()
    # plt.show()
