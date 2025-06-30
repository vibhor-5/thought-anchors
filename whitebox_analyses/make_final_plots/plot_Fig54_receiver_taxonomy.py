import os
import random
import sys
from matplotlib import ticker
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


def plot_receiver_bars(
    key="v_weight_ignore_5_top_20",
    regress_out=False,
    truncate=False,
    random_ef=False,
    min_cnt=10,
    plot_type="box",  # New parameter: "bar", "box", or "violin"
):
    model_name = "qwen-14b"
    model_name = "llama8"
    df = load_csv_good(model_name, mtx=False)
    # print(df.columns)
    # df = df[df["normalized_position"] < 0.4]
    # quit()
    # print(df["pn_ci"].unique())
    # print(df[key])
    # quit()
    # quit()

    # n_s = len(df)
    # for tag, df_tag in df.groupby("tag"):
    #     print(f"{tag}: {len(df_tag)}: {len(df_tag)/n_s:.2%}")

    # # print(df["tag"].describe())
    # quit()

    # r, p = stats.spearmanr(
    #     df["h_uzay_abs_ignore_8"], df["v_uzay_abs_ignore_8"], nan_policy="omit"
    # )
    # print(f"{r=:.2f} {p=:.2f}")
    # quit()
    # print(df[key].describe())
    # quit()
    # print(df.columns)
    # quit()
    tags = df["tag"].unique()

    tags = [
        "plan_generation",
        "fact_retrieval",
        "active_computation",
        "uncertainty_management",
        "result_consolidation",
        "self_checking",
    ]

    type2label = get_type2label()
    type2color = get_type2color()

    df["key"] = df[key]
    if regress_out:
        formula = "key ~ dist + pn_ci + receiving + pos_receiving + pos_receiving_rel + pos_giving + pos_giving_rel"
        # formula += "+ sentence_receiving_entropy"
        # formula += " + giving * receiving"
        # formula += "+ giving"
        df["pn_ci"] = df["pn_ci"].astype(str)
        model = smf.ols(formula=formula, data=df).fit()
        # Get the residuals (these are the effects after removing dist and pn)
        df["key"] = model.resid

    plt.rcParams["font.size"] = 11  # Set default font size to 12

    plt.figure(figsize=(7, 2.5))

    if plot_type == "bar":
        Ms = []
        SE_s = []
        for i, giving_tag in enumerate(tags):
            df_g = df[df["tag"] == giving_tag]
            vals_g = []
            if random_ef:
                for pn, df_pn in df_g.groupby("pn_ci"):
                    if len(df_pn) < min_cnt:
                        continue
                    pn_g = df_pn["key"].mean()
                    vals_g.append(pn_g)
            else:
                vals_g = df_g["key"].tolist()
            vals_g = sorted(vals_g)
            if truncate:
                vals_g = vals_g[1:-1]
            M_g = np.nanmean(vals_g)
            # print(f"{vals_g=}")
            N = np.sum(~np.isnan(vals_g))
            SE_g = np.nanstd(vals_g) / np.sqrt(len(vals_g))
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
            yerr=np.array(SE_s) * 1.96,
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

        # df = df[df["pn_ci"].apply(lambda x: str(x)[-1] != "0")]

        # print(len(df))
        cnt_pn_tag = df.groupby(["pn_ci", "tag"]).size().reset_index(name="cnt")
        df_bad_idxs = cnt_pn_tag[cnt_pn_tag["cnt"] <= min_cnt]
        # print(df_bad_idxs)
        bad_pn_tag_pairs = set(zip(df_bad_idxs["pn_ci"], df_bad_idxs["tag"]))
        df = df[~df.apply(lambda row: (row["pn_ci"], row["tag"]) in bad_pn_tag_pairs, axis=1)]

        # print(len(df))
        # quit()

        for giving_tag in tags:
            df_g = df[df["tag"] == giving_tag]
            if random_ef:
                for pn, df_pn in df_g.groupby("pn_ci"):
                    if len(df_pn) < min_cnt:
                        continue
                    pn_g = df_pn["key"].mean()
                    plot_data.append({"tag": giving_tag, "value": pn_g, "pn_ci": pn})
            else:
                for val in df_g["key"]:
                    plot_data.append({"tag": giving_tag, "value": val, "pn_ci": pn})

        plot_df = pd.DataFrame(plot_data)

        for tag0 in tags:
            df_tag0 = plot_df[plot_df["tag"] == tag0]
            for tag1 in tags:
                df_tag1 = plot_df[plot_df["tag"] == tag1]
                df_tag1 = df_tag1[df_tag1["pn_ci"].isin(df_tag0["pn_ci"])]
                df_tag0_ = df_tag0[df_tag0["pn_ci"].isin(df_tag1["pn_ci"])]
                # print(f"{tag0} {tag1}: {len(df_tag1)}")
                t, p = stats.ttest_rel(df_tag0_["value"], df_tag1["value"], nan_policy="omit")
                print(f"{tag0} vs. {tag1}: {t=:.2f} {p=:.3f}, N = {len(df_tag0_)}")

        # print(plot_df)
        # quit()

        # Add a mapping column with readable labels
        label_map = {tag: type2label[tag] for tag in tags}
        plot_df["label"] = plot_df["tag"].map(label_map)

        # Create color mapping for boxplot
        colors = [type2color[t] for t in tags]
        labels = [type2label[t] for t in tags]

        if plot_type == "box":
            # Create a boxplot with the readable labels and colors
            ax = sns.boxplot(
                data=plot_df,
                x="label",
                y="value",
                order=labels,  # Use the same order as bar plots
                palette={label: type2color[tag] for tag, label in zip(tags, labels)},
                width=0.6,
            )

            # Color the x-tick labels to match their boxes
            for i, (label, color) in enumerate(zip(labels, colors)):
                plt.gca().get_xticklabels()[i].set_color(color)

            plt.xticks(fontsize=10)
            plt.xlabel("")

        elif plot_type == "violin":
            # Similar update for violin plots
            ax = sns.violinplot(
                data=plot_df,
                # x="label",
                y="value",
                order=labels,
                palette={label: type2color[tag] for tag, label in zip(tags, labels)},
            )

            # Color the x-tick labels to match their violins
            for i, (label, color) in enumerate(zip(labels, colors)):
                plt.gca().get_xticklabels()[i].set_color(color)

            plt.xticks(fontsize=10)

    # plt.xticks(rotation=45)
    key2ylabel = {
        "eff": "Mean outgoing\nattention-suppression effect",
        "v_weight_ignore_4_top_32": "Mean receiver-head score",
        "v_weight_ignore_8_top_16": "Mean receiver-head score",
        "v_weight_ignore_8_top_32": "Mean receiver-head score",
        "v_weight_ignore_8_top_64": "Mean receiver-head score",
        "v_weight_ignore_16_top_64": "Mean receiver-head score",
        "v_uzay_abs_ignore_8": "Resampling method score",
        "v_uzay_ignore_8": "Resampling method score",
        "v_uzay_ignore_4": "Resampling method score",
        "v_uzay_ignore_1": "Resampling method score",
        "v_uzay_ignore_0": "Resampling method score",
        "v_uzay_abs_ignore_1": "Resampling method score",
    }
    # plt.title(
    #     f"Outgoing: {key}\n({model_name}; {regress_out=}; {truncate=}; {random_ef=})"
    # )
    plt.ylabel(key2ylabel[key])
    if "uzay" not in key:
        if "llama" in model_name:
            plt.ylim(0, 0.00061)
        else:
            plt.ylim(0, 0.00085)
        plt.gca().spines[["top", "right"]].set_visible(False)
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-3, -3))
        plt.gca().yaxis.set_major_formatter(formatter)
        plt.gca().spines[["top", "right"]].set_visible(False)

    if "llama" in model_name:
        plt.title("Receiver-head scores by sentence category (R1-Llama-8B)", fontsize=11)
    else:
        plt.title("Receiver-head scores by sentence category")

    # Add horizontal line at y=0
    fp_out = f"plots/receiver_taxonomy_{model_name}.png"
    plt.subplots_adjust(bottom=0.2, top=0.85, left=0.1, right=0.95)
    plt.savefig(fp_out, dpi=300)

    plt.show()


def get_receiver_x_uzay():
    df = load_csv_good("qwen-14b", mtx=False, drop_cats=("problem_setup", "final_answer_emission"))

    # df = df[df["sentence_idx"] > 10]

    head_col = "v_weight_ignore_5_top_20"
    # head_col = "v_supp_ignore_10"
    uzay_col = "v_uzay_abs_ignore_5"
    # uzay_col = "v_supp_ignore_20"t

    formula = f"{head_col} ~ normalized_position + sentence_idx"
    model = smf.ols(formula=formula, data=df).fit()
    df[head_col] = model.resid

    formula = f"{uzay_col} ~ normalized_position + sentence_idx"
    model = smf.ols(formula=formula, data=df).fit()
    df[uzay_col] = model.resid

    r, p = stats.spearmanr(df[head_col], df[uzay_col], nan_policy="omit")
    # print(f"{r=:.2f} {p=:.2f}")
    # quit()

    # plt.scatter(df[head_col], df[uzay_col], alpha=0.01)
    # plt.show()

    rs_l = []
    ps_l = []

    for pn, df_pn in df.groupby("pn_ci"):
        r, p = stats.spearmanr(df_pn[head_col], df_pn[uzay_col], nan_policy="omit")
        if np.isnan(r):
            print(f"Has NaN for {pn}")
            continue
        rs_l.append(r)
        ps_l.append(p)
        print(f"{pn=}, {r=:.2f} {p=:.2f}")

    rs_l = sorted(rs_l)
    rs_l = rs_l[1:-1]
    M = np.mean(rs_l)
    SE = np.std(rs_l) / np.sqrt(len(rs_l))
    lower = M - 1.96 * SE
    upper = M + 1.96 * SE
    print(f"{M=:.2f} {SE=:.2f} [{lower:.2f} {upper:.2f}]")


if __name__ == "__main__":
    # Example usage with different plot types:
    # plot_outgoing_bars(key="eff", regress_out=True, truncate=True, random_ef=False, min_cnt=10, plot_type="bar")
    # plot_outgoing_bars(key="eff", regress_out=True, truncate=True, random_ef=False, min_cnt=10, plot_type="box")
    # plot_receiver_bars(
    #     # key="v_weight_ignore_4_top_32",
    #     key="v_weight_ignore_8_top_64",
    #     regress_out=False,
    #     truncate=True,
    #     random_ef=True,
    #     min_cnt=2,
    #     plot_type="box",
    # )

    plot_receiver_bars(
        key="v_weight_ignore_8_top_64",
        # key="v_weight_ignore_4_top_32", # used for qwen results
        regress_out=False,
        truncate=False,
        random_ef=True,
        min_cnt=2,
        plot_type="box",
    )

    # get_receiver_x_uzay()
