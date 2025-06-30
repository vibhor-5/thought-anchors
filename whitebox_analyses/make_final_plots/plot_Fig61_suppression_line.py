import os
from pathlib import Path

import sys
import numpy as np

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examine_suppression_logits import get_sentence_sentence_KL
import matplotlib.pyplot as plt
import numpy as np

from run_target_problems import get_full_CoT_token_ranges
from utils import hash_dict


def plot_sentence_nice(sentence_sentence_scores, sentence_num, problem_num, fp_out):
    # fp_out = os.path.join(pn_dir, f"suppress_{sentence_num}_nice.png")
    sentence_KL_logs = sentence_sentence_scores[:, sentence_num]
    kl_log_low = np.nanmin(sentence_KL_logs)
    kl_log_high = np.nanmax(sentence_KL_logs)
    # Add some padding if min/max are the same or very close
    padding = (kl_log_high - kl_log_low) * 0.02
    if padding < 0.1:
        padding = 0.1  # Ensure minimum padding
    y_min = kl_log_low - padding
    y_max = kl_log_high + padding
    plt.rcParams["font.size"] = 11
    plt.figure(figsize=(6, 4))

    plt.vlines(
        sentence_num,
        y_min,
        y_max,
        color="k",
        linestyle="--",
        linewidth=1,
        label=f"Suppressed sentence {sentence_num}",
    )
    sentence_idxs = np.arange(len(sentence_KL_logs))
    log_kl = list(sentence_KL_logs)

    plt.plot(
        sentence_idxs, log_kl, marker=".", markersize=2, color="firebrick"
    )  # No line for marker
    # Add text label for each point
    threshold = -6
    for i in range(len(sentence_idxs)):
        x_coord = sentence_idxs[i]
        y_coord = log_kl[i]
        # Optional: Add a check to only plot labels if y_coord is not NaN
        if not np.isnan(y_coord) and y_coord > threshold:
            if i == sentence_num:
                if False:
                    plt.text(
                        # x_coord + 1.5,
                        x_coord - 1.5,
                        y_coord + 0.0,
                        f"Suppressed sentence: {x_coord}",  # Use individual index for label
                        # ha="left",
                        ha="right",
                        va="center",
                        fontsize=10,
                        color="maroon",
                    )
            else:
                plt.text(
                    x_coord + 0.8,
                    y_coord + 0.2,
                    f"{x_coord}",  # Use individual index for label
                    ha="left",
                    va="bottom",
                    fontsize=10,
                    color="k",
                )  # Smaller font, maybe grey
            plt.plot(
                x_coord,
                y_coord,
                marker="o",
                markersize=6,
                color="maroon",
                linestyle="",
            )  # No line for marker

    plt.ylim(y_min, y_max * 1.06)  # Set y-limits based on data range
    plt.ylabel("Mean Log(KL Divergence + 1e-9)", labelpad=7)  # More descriptive label
    plt.xlabel("Sentence Number", labelpad=7)

    pn = str(problem_num)
    is_correct = pn[-1] == "1"
    if is_correct:
        pn_title = f"Problem {pn[:-2]} (correct)"
    else:
        pn_title = f"Problem {pn[:-2]} (incorrect)"

    plt.title(f"Suppression effect: {pn_title}")
    # plt.grid(True, linestyle=":", alpha=0.6)
    plt.gca().spines[["top", "right"]].set_visible(False)
    plt.xlim(0, len(sentence_KL_logs) - 1)

    # plt.legend()  # Show legend for the line and vline
    plt.subplots_adjust(bottom=0.15, top=0.9, left=0.15, right=0.95)

    # plt.vlines(sentence_num, y_min, y_max, color='red',
    #    linestyle='--', linewidth=1, label=f'Suppressed Sent {sentence_num}')

    plt.savefig(fp_out, dpi=300)
    print(f"Saved to {fp_out}")
    plt.show()
    quit()


if __name__ == "__main__":
    only_pre_convergence = "semi"

    problem_dir = os.path.join("target_problems", "temperature_0.6_top_p_0.95")

    # Identify highly important sentence

    layers_to_mask = {i: list(range(40)) for i in range(48)}
    problem_num = 468201

    hash_layers_to_mask = hash_dict(layers_to_mask)

    sentence2ranges, problem = get_full_CoT_token_ranges(
        problem_num,
        problem_dir,
        only_pre_convergence=only_pre_convergence,
        verbose=True,
    )

    sentence_sentence_scores = get_sentence_sentence_KL(
        # problem_num=problem_to_run,
        problem_num=problem_num,
        layers_to_mask=layers_to_mask,
        p_nucleus=0.9999,  # Example p value
        model_name="qwen-14b",  # Make sure this matches your available model
        quantize_4bit=False,  # Use 4-bit quantization for memory efficiency
        quantize_8bit=False,
        problem_dir=problem_dir,  # Adjust if needed
        output_dir="suppressed_results_test",  # Save to a test directory
        only_pre_convergence=only_pre_convergence,
        # plot_sentences=True,
    )
    sentence_num = 32
    fp_out = f"plots/suppress_sentence_{problem_num}_{sentence_num}.png"
    plot_sentence_nice(sentence_sentence_scores, sentence_num, problem_num, fp_out)
