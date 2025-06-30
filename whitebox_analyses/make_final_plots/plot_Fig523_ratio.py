import statsmodels.formula.api as smf
from matplotlib import pyplot as plt, ticker
import pandas as pd
from scipy import stats
import numpy as np

from run_Fig62 import load_csv_good


if __name__ == "__main__":
    only_first = None
    take_log = True
    s = ""
    if not take_log:
        s += "_nolog"
    if only_first is not None:
        s += f"_of{only_first}"

    top_k = 16

    target_col = f"v_weight_ignore_8_top_{top_k}"
    # target_col = "v_supp_ignore_4"
    # target_col = "sentence_token_entropy"
    # target_col = "h_supp_ignore_20"

    # df_base = pd.read_csv(f"df_attn_base-Qwen14_May14{s}.csv")
    # df_R1 = pd.read_csv(f"df_attn_R1-Qwen14_May14{s}.csv")

    model_name = "llama8"
    df_R1 = load_csv_good(model_name, mtx=False)
    model_base = model_name + "-base"
    df_base = load_csv_good(model_base, mtx=False)

    # r, p = stats.spearmanr(df_R1[target_col], df_R1["kl_smooth"], nan_policy="omit")
    # print(f"{r=:.2f} {p=:.2f}")
    # r, p = stats.spearmanr(df_base[target_col], df_base["kl_smooth"], nan_policy="omit")
    # print(f"{r=:.2f} {p=:.2f}")
    # quit()

    # df_base = df_base[df_base["normalized_position"] < 0.8]
    # df_R1 = df_R1[df_R1["normalized_position"] < 0.8]

    df_base = df_base.dropna(subset=[target_col])
    df_R1 = df_R1.dropna(subset=[target_col])

    df_base.sort_values(by=target_col, inplace=True)
    df_R1.sort_values(by=target_col, inplace=True)

    vals_base = df_base[target_col].values
    # print(list(vals_base))
    # quit()

    vals_R1 = df_R1[target_col].values

    vals_base = vals_base[10:-10]
    vals_R1 = vals_R1[10:-10]

    plt.rcParams["font.size"] = 11

    fig, axs = plt.subplots(1, 2, figsize=(7, 2))

    plt.sca(axs[0])
    # Add whitesmoke grid to first subplot
    plt.grid(color="lightgray", linestyle="-", linewidth=0.7, alpha=0.7)

    plt.plot(vals_R1, label="R1 Qwen 14B", color="navy", zorder=100, linewidth=1.5)
    plt.plot(vals_base, label="Base Qwen 14B", color="r", linewidth=1.5)
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-3, -3))
    plt.ylim(0, 0.0021)
    plt.xlim(0, len(vals_base) * 1.01)
    plt.yticks([0, 0.001, 0.002])
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.gca().spines[["top", "right"]].set_visible(False)
    plt.ylabel(f"Top-{top_k}\nrec-head score")

    # Position the legend on the left near the y-axis
    plt.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.05))

    plt.sca(axs[1])
    # Add whitesmoke grid to second subplot
    plt.grid(color="lightgray", linestyle=":", linewidth=0.7, alpha=0.7)

    ratio = vals_R1 / vals_base
    plt.ylim(0.5, 2.05)
    plt.xlim(0, len(vals_base) * 1.01)
    plt.yticks([0.5, 1, 1.5, 2])
    plt.plot(ratio, color="g", zorder=100, linewidth=1.5)
    plt.gca().spines[["top", "right"]].set_visible(False)
    plt.axhline(1, color="k", linestyle="--")
    plt.ylabel("Ratio (R1 / base)")

    # Remove the individual xlabel and add a common one
    axs[1].set_xlabel("")

    # Add a common xlabel beneath both subplots
    fig.supxlabel("Sentence rank, sorted by receiver-head score", y=0.05, fontsize=11)

    fp_out = f"plots/receiver_ratio_top{top_k}_{model_name}.png"
    plt.tight_layout()
    plt.savefig(fp_out, dpi=300)
    print(f"Saved to {fp_out}")

    plt.show()
