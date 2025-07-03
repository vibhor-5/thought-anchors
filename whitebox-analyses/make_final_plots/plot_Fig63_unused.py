import os
import random
import sys
import seaborn as sns
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# from regress_mtx import plot_matrix_with_stats
from run_Fig62 import load_csv_good
import statsmodels.formula.api as smf
import numpy as np
import matplotlib.pyplot as plt

from plot_utils import get_type2color, get_type2label
import pandas as pd
import scipy.stats as stats


def plot_matrix_with_stats(
    mat_M,
    mat_SE,
    tags,
    title="Matrix Plot",
    cmap="viridis",
    figsize=(6, 4),
    t_mat=None,
):
    """
    Plot a matrix with mean values and standard errors in each cell.

    Args:
        mat_M: Matrix of mean values
        mat_SE: Matrix of standard errors
        tags: List of tags for row/column labels
        title: Title for the plot
        cmap: Colormap to use
        figsize: Figure size tuple
    """
    plt.figure(figsize=figsize)

    vmin = np.nanquantile(mat_M, 0.1)
    vmax = np.nanquantile(mat_M, 0.9)
    vabs = np.max(np.abs([vmin, vmax]))
    vmin = -vabs
    vmax = vabs

    # Create the main imshow plot
    im = plt.imshow(mat_M, cmap=cmap, vmin=vmin, vmax=vmax)

    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label("Mean Effect")

    # Set ticks and labels
    type2label = get_type2label()

    plt.xticks(range(len(tags)), [type2label[t] for t in tags])
    plt.yticks(range(len(tags)), [type2label[t] for t in tags])

    # Add text in each cell
    for i in range(len(tags)):
        for j in range(len(tags)):
            if not np.isnan(mat_M[i, j]):
                t = mat_M[i, j] / mat_SE[i, j]
                # if t_mat is not None:
                #     t = t_mat[i, j]
                # text = f"t = {t:.2f}"
                text = f"{mat_M[i, j]:.3f}\n({mat_SE[i, j]:.3f})"
                if np.abs(mat_M[i, j]) > 0.5 * vabs:
                    color = "white"
                else:
                    color = "black"
                plt.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    color=color,
                    # color="white" if mat_M[i, j] < np.nanmean(mat_M) else "black",
                )

    plt.title(title)
    plt.tight_layout()

    return plt.gcf()


def plot_outgoing_bars(
    key="eff",
    regress_out=False,
    truncate=False,
    random_ef=False,
    min_cnt=10,
    plot_type="box",  # New parameter: "bar", "box", or "violin"
):
    model_name = "qwen-14b"
    df = load_csv_good(model_name, mtx=True)

    df = df[df["dist"] < 5]

    df_g = df.groupby(["giving", "pn_ic"])[key].mean()

    tags = [
        "plan_generation",
        "fact_retrieval",
        "active_computation",
        "uncertainty_management",
        "result_consolidation",
        "self_checking",
    ]
    for tag_i in tags:
        for tag_j in tags:
            df_i = df_g.loc[tag_i]
            df_j = df_g.loc[tag_j]
            idxs_both = df_i.index.intersection(df_j.index)
            diff = df_j.loc[idxs_both] - df_i.loc[idxs_both]
            M = diff.mean()
            SE = diff.std() / np.sqrt(len(diff))
            t = M / SE
            print(f"{tag_i}->{tag_j}: {M:.3f} ({SE:.3f}) t={t:.3f}")
            # print(f"{tag_i}->{tag_j}: {diff:.3f}")

            # df_ij = df_g[df_g["giving"] == tag_i]
            # df_ij = df_ij[df_ij["pn_ic"] == tag_j]

    print(df_g)
    quit()

    # df = df[df["dist"] < 5]
    tags = df["giving"].unique()

    tags = [
        "plan_generation",
        "fact_retrieval",
        "active_computation",
        "uncertainty_management",
        "result_consolidation",
        "self_checking",
    ]

    df = df[df["receiving"] != "final_answer_emission"]

    type2label = get_type2label()
    type2color = get_type2color()

    df["key"] = df[key]
    if regress_out:
        formula = "key ~ dist + pn_ic + receiving + pos_receiving + pos_receiving_rel + pos_giving + pos_giving_rel"
        # formula += "+ sentence_receiving_entropy"
        # formula += " + giving * receiving"
        # formula += "+ giving"
        df["pn_ic"] = df["pn_ic"].astype(str)
        model = smf.ols(formula=formula, data=df).fit()
        # Get the residuals (these are the effects after removing dist and pn)
        df["key"] = model.resid

    plt.rcParams["font.size"] = 12  # Set default font size to 12

    plt.figure(figsize=(7, 3.5))

    if plot_type == "bar":
        Ms = []
        SE_s = []
        for i, giving_tag in enumerate(tags):
            df_g = df[df["giving"] == giving_tag]
            vals_g = []
            if random_ef:
                for pn, df_pn in df_g.groupby("pn_ic"):
                    if len(df_pn) < min_cnt:
                        continue
                    pn_g = df_pn["key"].mean()
                    vals_g.append(pn_g)
            else:
                vals_g = df_g["key"].tolist()
            vals_g = sorted(vals_g)
            if truncate:
                vals_g = vals_g[1:-1]
            M_g = np.mean(vals_g)
            # print(f"{vals_g=}")
            SE_g = np.std(vals_g) / np.sqrt(len(vals_g))
            t_g = M_g / SE_g
            N_g = len(vals_g)
            print(f"{giving_tag}: {M_g:.3f} ({SE_g:.3f}) t={t_g:.3f} N={N_g}")
            Ms.append(M_g)
            SE_s.append(SE_g)

        colors = [type2color[t] for t in tags]
        labels = [type2label[t] for t in tags]
        plt.bar(
            labels,
            Ms,
            yerr=SE_s,
            color=colors,
            label=labels,
            edgecolor="black",  # Add black outline
            linewidth=0.5,  # Set outline width
            capsize=5,
            alpha=0.8,
        )  # Add caps to error bars

        plt.xticks(fontsize=10)

        # Color the x-tick labels to match their bars
        for i, (label, color) in enumerate(zip(labels, colors)):
            plt.gca().get_xticklabels()[i].set_color(color)
    else:
        # For box and violin plots, we need to prepare the data differently
        plot_data = []
        for giving_tag in tags:
            df_g = df[df["giving"] == giving_tag]
            if random_ef:
                for pn, df_pn in df_g.groupby("pn_ic"):
                    if len(df_pn) < min_cnt:
                        continue
                    pn_g = df_pn["key"].mean()
                    plot_data.append({"tag": giving_tag, "value": pn_g})
            else:
                for val in df_g["key"]:
                    plot_data.append({"tag": giving_tag, "value": val})

        plot_df = pd.DataFrame(plot_data)

        if plot_type == "box":
            sns.boxplot(data=plot_df, x="tag", y="value")
        elif plot_type == "violin":
            sns.violinplot(data=plot_df, x="tag", y="value")

    # plt.xticks(rotation=45)
    key2ylabel = {
        "eff": "Mean outgoing\nattention-suppression effect",
        "eff_uzay": "Mean outgoing\nattention-suppression effect",
    }
    # plt.title(
    #     f"Outgoing: {key}\n({model_name}; {regress_out=}; {truncate=}; {random_ef=})"
    # )
    plt.ylabel(key2ylabel[key])
    plt.gca().spines[["top", "bottom", "right"]].set_visible(False)

    # Add horizontal line at y=0
    plt.axhline(y=0, color="black", linestyle="-", linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_interaction_imshow(
    key="eff",
    regress_out=True,
    truncate=False,
    random_ef=False,
    min_cnt=10,
    plot_type="bar",
):
    model_name = "qwen-14b"
    df = load_csv_good(model_name, mtx=True)
    # df = df[df["dist"] < 5]

    df["key"] = df[key]
    if regress_out:
        formula = "key ~ dist + pn_ic + receiving + pos_receiving + pos_receiving_rel + pos_giving + pos_giving_rel"
        # formula += " + giving * receiving"
        # formula += "+ giving"
        # formula += "+ sentence_receiving_entropy"

        df["pn_ic"] = df["pn_ic"].astype(str)
        model = smf.ols(formula=formula, data=df).fit()
        # Get the residuals (these are the effects after removing dist and pn)
        df["key"] = model.resid

    tags = df["giving"].unique()

    tags = [
        "plan_generation",
        "fact_retrieval",
        "active_computation",
        "uncertainty_management",
        "result_consolidation",
        "self_checking",
    ]

    mat_gr_M = np.full((len(tags), len(tags)), np.nan)
    mat_gr_SE = np.full((len(tags), len(tags)), np.nan)
    mat_gr_t = np.full((len(tags), len(tags)), np.nan)
    for i, giving_tag in enumerate(tags):
        for j, receiving_tag in enumerate(tags):
            df_gr = df[(df["giving"] == giving_tag) & (df["receiving"] == receiving_tag)]
            vals_gr = []
            cnts_gr = []
            if random_ef:
                for pn, df_pn in df_gr.groupby("pn_ic"):
                    if len(df_pn) < min_cnt:
                        continue
                    cnt = len(df_pn)
                    cnts_gr.append(cnt)
                    pn_gr = df_pn["key"].mean()
                    vals_gr.append(pn_gr)
            else:
                vals_gr = df_gr["key"].tolist()
            # drop lowest and highest
            vals_gr = sorted(vals_gr)
            if truncate:
                vals_gr = vals_gr[1:-1]

            w, p = stats.wilcoxon(vals_gr)
            t_wilcox = stats.t.ppf(p / 2, len(vals_gr) - 1)
            # print(f"{giving_tag}->{receiving_tag}: {w=}, {p=}")

            # vals_gr = vals_gr[2:-2]
            M_gr = np.mean(vals_gr)
            M_gr = np.average(vals_gr)  # , weights=cnts_gr)
            SE_gr = np.std(vals_gr) / np.sqrt(len(vals_gr))
            t_gr = M_gr / SE_gr
            N_gr = len(vals_gr)
            print(f"{giving_tag}->{receiving_tag}: {M_gr:.3f} ({SE_gr:.3f}) t={t_gr:.3f} N={N_gr}")

            # M_gr

            mat_gr_M[i, j] = M_gr
            mat_gr_SE[i, j] = SE_gr
            # mat_gr_t[i, j] = t_wilcox

    # After creating mat_gr_M and mat_gr_SE:
    fig = plot_matrix_with_stats(
        mat_gr_M,
        mat_gr_SE,
        tags,
        title=f"Type-type interaction effects",
        cmap="RdBu_r",  # Red-Blue diverging colormap
        t_mat=mat_gr_t,
    )

    plt.show()


if __name__ == "__main__":
    # Example usage with different plot types:
    # plot_outgoing_bars(key="eff", regress_out=True, truncate=True, random_ef=False, min_cnt=10, plot_type="bar")
    # plot_outgoing_bars(key="eff", regress_out=True, truncate=True, random_ef=False, min_cnt=10, plot_type="box")
    plot_outgoing_bars(
        key="eff_uzay",
        regress_out=False,
        truncate=False,
        random_ef=True,
        min_cnt=2,
        plot_type="box",
    )

    # plot_interaction_imshow(
    #     key="eff_uzay",
    #     regress_out=False,
    #     truncate=False,
    #     random_ef=True,
    #     min_cnt=10,
    #     plot_type="bar",
    # )
