from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import sys
import numpy as np

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import os
from matplotlib import pyplot as plt
import numpy as np
from make_final_plots.plot_Fig51_receiver_examples import white_to_blues
from run_target_problems import get_most_sensitive_layer_heads, get_problem_attn_avg


def plot_one(pn=468201, layer=36, head=6, top_k=None):
    problem_dir = os.path.join("target_problems", "temperature_0.6_top_p_0.95_qwen")
    only_pre_convergence = "semi"
    model_name = "qwen-14b"
    quantize_8bit = False
    quantize_4bit = False

    plt.figure(figsize=(3, 3))

    if top_k is not None:
        coords = get_most_sensitive_layer_heads(
            top_k,
            model_name=model_name,
            quantize_8bit=False,
            quantize_4bit=False,
            only_pre_convergence="semi",
            only=None,
            problem_dir=problem_dir,
            drop_first=4,
            drop_last=32,
        )
    else:
        coords = [(layer, head)]

    avg_matrix_l = []
    # for layer in range(48):
    #     for head in range(40):
    for layer, head in coords:

        avg_matrix = get_problem_attn_avg(
            pn,
            layer,
            head,
            problem_dir=problem_dir,
            only_pre_convergence=only_pre_convergence,
            model_name=model_name + "-base",
            quantize_8bit=quantize_8bit,
            quantize_4bit=quantize_4bit,
        )
        avg_matrix_l.append(avg_matrix)
    avg_matrix = np.nanmean(avg_matrix_l, axis=0)
    avg_matrix = avg_matrix[1:-1, 1:-1]
    # avg_matrix = avg_matrix[:129, :129]
    avg_matrix_tril = np.tril(avg_matrix)
    if top_k is not None:
        vmin = np.nanquantile(avg_matrix_tril, 0.1)
        vmax = np.nanquantile(avg_matrix_tril, 0.9)
    else:
        vmin = np.nanquantile(avg_matrix_tril, 0.01)
        vmax = np.nanquantile(avg_matrix_tril, 0.99)
        avg_matrix = avg_matrix[:129, :129]

    # avg_matrix[:, :] = 0

    white_blue_cmap = white_to_blues()

    plt.imshow(avg_matrix, vmin=0, vmax=vmax, cmap=white_blue_cmap)
    pn = str(pn)
    is_correct = pn[-1] == "1"
    if is_correct:
        pn_title = f"Problem: {pn[:-2]} (correct)"
    else:
        pn_title = f"Problem: {pn[:-2]} (incorrect)"
    title = f"{pn_title}\nLayer: {layer}, Head: {head}"
    # plt.title(title, fontsize=12)

    plt.gca().tick_params(axis="both", labelsize=11)
    xticks = np.arange(0, avg_matrix.shape[1], 25)
    plt.gca().set_xticks(xticks)
    yticks = np.arange(0, avg_matrix.shape[0], 25)
    plt.gca().set_yticks(yticks)
    plt.gca().set_xlim(0, avg_matrix.shape[1] - 1)
    plt.gca().set_ylim(avg_matrix.shape[0] - 1, 0)

    # Remove top and right spines
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.xlabel("Sentence position", fontsize=12)
    plt.ylabel("Sentence position", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"plots/receiver_head_{pn}_{layer}_{head}.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    # plot_one(pn=468201, layer=36, head=6, top_k=32)
    plot_one(pn=468200, layer=36, head=6, top_k=32)
