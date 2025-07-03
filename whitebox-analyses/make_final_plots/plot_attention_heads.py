import sys
import numpy as np
import os

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

import numpy as np
from run_target_problems import (
    get_most_sensitive_layer_heads,
    get_problem_attn_avg,
    get_vert_scores_for_heads,
)
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as ticker


def plot_dozen_layer_heads(target_layer, key_head=6, pn=159100, key_color="navy", top_k=None):
    if top_k is not None:
        coords = get_most_sensitive_layer_heads(
            top_k,
            model_name="qwen-14b",
            quantize_8bit=False,
            quantize_4bit=False,
            only_pre_convergence="semi",
            only=None,
            problem_dir=os.path.join("target_problems", "temperature_0.6_top_p_0.95_qwen"),
            drop_first=4,
            drop_last=32,
        )
    else:
        coords = [(target_layer, i) for i in range(40)]
    all_mappings = get_vert_scores_for_heads(
        coords,
        [pn],
        proximity_ignore=4,
        model_name="qwen-14b",
        quantize_8bit=False,
        quantize_4bit=False,
        only_pre_convergence="semi",  # Change to "semi" probably...
        control_depth=False,
        drop_first=0,
    )
    # quit()
    fig = plt.figure(figsize=(8, 3))

    if top_k is not None:
        plt.rcParams["font.size"] = 14
    else:
        plt.rcParams["font.size"] = 11
    if top_k is not None:
        plt.title(f"   Top {top_k} receiver heads")
    else:
        plt.title(f"Layer: {coords[0][0]}")

    # l = []
    # for layer, head in coords:
    #     key = (layer, head, pn)
    #     vert_scores = all_mappings[key]
    #     l.append(vert_scores)
    # all_mappings_ = {(0, 0, pn): np.nanmean(l, axis=0)}
    # coords = [(0, 0)]
    # all_mappings = all_mappings_
    for layer, head in coords:
        key = (layer, head, pn)
        vert_scores = all_mappings[key]
        if top_k:
            if np.nanmax(vert_scores) > 0.01:
                print(f"Too high: {layer=}, {head=}, {np.nanmax(vert_scores)=}")
                continue
        if not top_k:
            vert_scores = vert_scores[:-20]
        # vert_scores = vert_scores[:-20]
        # vert_scores = vert_scores[:105]
        # print(f"| {head=} | {np.sum(vert_scores)=} | {np.mean(vert_scores)=} | {np.std(vert_scores)=} |")
        # print(f"{vert_scores.shape=}")
        # quit()
        if head == key_head and not top_k:
            # print(f"{vert_scores=}")
            # quit()
            plt.plot(
                vert_scores,
                label=f"Head: {head}",
                color=key_color,
                zorder=100,
                linewidth=1,
            )
        else:
            plt.plot(vert_scores, label=f"Head: {head}", linewidth=1)

    fp_out = f"plots/attn_wait_case/head_plots_layer_{target_layer}.png"
    Path(fp_out).parent.mkdir(parents=True, exist_ok=True)
    if top_k is not None:
        plt.ylim(0, 0.009)
        plt.yticks([0, 0.002, 0.004, 0.006, 0.008])
    elif target_layer == 36:
        # plt.text(70, 0.0042, f"Head: {key_head}", fontsize=12, color=key_color)
        plt.ylim(0, 0.007)
    else:
        # plt.text(73, 0.135, f"Head: {key_head}", fontsize=12, color=key_color)
        plt.ylim(0, 0.149)
    if top_k is not None:
        plt.ylabel("Receiver head score", fontsize=16, labelpad=11)
    else:
        plt.ylabel("Vertical attention score", fontsize=11, labelpad=7)
    plt.xlabel("Sentence position", fontsize=14 if top_k else 11, labelpad=7)

    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-3, -3))
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.gca().spines[["top", "right"]].set_visible(False)

    print(f"{len(vert_scores)=}")
    # quit()

    if top_k is not None:
        plt.xticks(np.arange(0, len(vert_scores), 10))
        plt.xlim(0, np.sum(~np.isnan(vert_scores)) - 1)
    else:
        plt.xticks(np.arange(0, len(vert_scores), 25))
        plt.xlim(0, len(vert_scores) - 1)
    # plt.tight_layout()
    plt.subplots_adjust(bottom=0.20 if top_k is not None else 0.15, top=0.8, left=0.15, right=0.95)
    if top_k is not None:
        fp_out = f"plots/kurt_plots/heads_plot_layer_top_{top_k}.png"
    else:
        fp_out = f"plots/kurt_plots/heads_plot_layer.png"

    plt.savefig(fp_out, dpi=300)
    plt.show()
    plt.close()


def plot_attn_avg(pn, target_layer, key_head):
    avg_matrix = get_problem_attn_avg(
        pn,
        target_layer,
        key_head,
        problem_dir=os.path.join("target_problems", "temperature_0.6_top_p_0.95"),
        only_pre_convergence="semi",
        model_name="qwen-14b",
        quantize_8bit=False,
        quantize_4bit=False,
    )
    vmin = np.nanpercentile(avg_matrix, 5)
    vmax = np.nanpercentile(avg_matrix, 95)
    im = plt.imshow(avg_matrix, vmin=vmin, vmax=vmax)

    cbar = plt.colorbar(im)
    cbar.set_label("Attention weight", rotation=270, labelpad=15)

    plt.title(f"Layer: {target_layer} (Head: {key_head}), problem: #{pn}")
    fp_out = f"plots/attn_wait_case/avg_matrix_layer_{target_layer}_head_{key_head}.png"
    Path(fp_out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fp_out, dpi=300)
    plt.show()
    plt.close()


if __name__ == "__main__":
    # plot_dozen_layer_heads(pn=159100, target_layer=36, key_head=6)
    # plot_dozen_layer_heads(pn=159100, target_layer=36, key_head=6)
    plot_dozen_layer_heads(pn=468201, target_layer=36, key_head=6, top_k=32)
    # plot_dozen_layer_heads(pn=468201, target_layer=36, key_head=6, top_k=128)
    # plot_dozen_layer_heads(pn=468201, target_layer=36, key_head=6, top_k=256)
