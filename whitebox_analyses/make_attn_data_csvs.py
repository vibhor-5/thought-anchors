from functools import cache
import os

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from calling_funcs import get_weights_scores
from correlate_matrices import get_importance_mtx
from repeated_suppression_logits import get_most_sensitive_heads_map

# os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3"
from examine_suppression_logits import get_sentence_entropies, get_sentence_sentence_KL
import os

from run_target_problems import (
    get_attn_direction_scores,
    get_attn_horz_scores,
    get_attn_vert_scores,
    get_pn_df_info,
    get_problem_nums,
    get_vert_scores_for_heads,
    get_full_CoT_token_ranges,
)

from scipy import stats
import statsmodels.formula.api as smf
from scipy.stats import sem  # To calculate standard error of the mean


# @pkld
def load_all_graphs(
    problem_nums,
    layers_to_mask,
    only_pre_convergence=False,
    problem_dir=os.path.join("target_problems", "temperature_0.6_top_p_0.95"),
    only_first=None,
    take_log=True,
    model_name="qwen-14b",
):
    graphs = []
    valid_pns = []
    for problem_num in problem_nums:
        # if int(problem_num) == 344801:
        # continue
        # sentence_sentence_scores = get_sentence_sentence_KL(
        #     # problem_num=problem_to_run,
        #     problem_num=problem_num,
        #     layers_to_mask=layers_to_mask,
        #     p_nucleus=0.9999,  # Example p value
        #     model_name="qwen-14b",  # Make sure this matches your available model
        #     quantize_4bit=False,  # Use 4-bit quantization for memory efficiency
        #     quantize_8bit=False,
        #     problem_dir=problem_dir,  # Adjust if needed
        #     output_dir="suppressed_results_test",  # Save to a test directory
        #     only_pre_convergence=only_pre_convergence,
        # )
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
        # if problem_num == 468201:
        #     np.save(
        #         f"{problem_num}_sentence_sentence_scores.npy", sentence_sentence_scores
        #     )
        #     quit()
        if sentence_sentence_scores is not None:
            graphs.append(sentence_sentence_scores)
            valid_pns.append(problem_num)
    return graphs, valid_pns


def plot_box_plot(df, keys=("vert_score", "horz_score", "diag_score"), plot_type="bar"):
    """
    Generates and saves box plots OR bar plots (with SE bars) for specified
    score columns, grouped by 'tag'.

    Args:
        df (pd.DataFrame): Input DataFrame containing scores and a 'tag' column.
        keys (tuple): Tuple of column names for which to generate plots.
        plot_type (str): Type of plot ('box' or 'bar'). Defaults to 'box'.
    """
    if "tag" not in df.columns:
        print("Error: 'tag' column not found in DataFrame. Cannot create plots.")
        return
    if plot_type not in ["box", "bar"]:
        print(f"Error: Invalid plot_type '{plot_type}'. Choose 'box' or 'bar'.")
        return

    output_dir = f"plots/df_{plot_type}plots"  # Different subdir for bar plots
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving {plot_type} plots to: {output_dir}")

    # Define a consistent order for tags for better visualization
    # Filter out 'unknown' tag if present, as it might dominate or be uninformative
    valid_tags = sorted(
        [tag for tag in df["tag"].unique() if tag != "unknown" and tag != "problem_setup"]
    )
    if not valid_tags:
        print("Warning: No valid tags found (excluding 'unknown'). Cannot plot.")
        return

    df_filtered_tags = df[df["tag"].isin(valid_tags)].copy()
    # Ensure 'tag' is treated as a categorical type with the defined order
    df_filtered_tags["tag"] = pd.Categorical(
        df_filtered_tags["tag"], categories=valid_tags, ordered=True
    )

    # Set default font size for all plots
    plt.rcParams.update({"font.size": 16})

    for key in keys:
        df_key = df_filtered_tags[df_filtered_tags[key].notna()]
        if key not in df_filtered_tags.columns:
            print(f"Warning: Key '{key}' not found in DataFrame. Skipping.")
            continue

        plt.figure(figsize=(12, 8))  # Adjust figure size as needed

        if plot_type == "box":
            # --- Box Plot Logic ---
            sns.boxplot(data=df_key, x="tag", y=key, order=valid_tags, palette="viridis")
            plt.title(f"Distribution of {key} by Sentence Tag (Box Plot)")

        elif plot_type == "bar":
            # --- Bar Plot Logic ---
            # Calculate mean and standard error (SE) per tag
            # Add observed=False to silence the future warning for now
            grouped_stats = (
                df_key.groupby("tag", observed=False)[key].agg(["mean", sem]).reindex(valid_tags)
            )

            # Identify tags with valid (non-NaN) mean and standard error
            valid_plot_tags_mask = grouped_stats["mean"].notna() & grouped_stats["sem"].notna()
            valid_plot_stats = grouped_stats[valid_plot_tags_mask]

            if valid_plot_stats.empty:
                print(
                    f"Warning: No tags with valid mean and SE found for key '{key}'. Skipping plot."
                )
                plt.close()
                continue  # Skip to the next key

            if valid_plot_stats.shape[0] < grouped_stats.shape[0]:
                print(
                    f"Warning: Plotting only {valid_plot_stats.shape[0]} out of {grouped_stats.shape[0]} tags for key '{key}' due to NaNs."
                )

            means = valid_plot_stats["mean"]
            errors = valid_plot_stats["sem"]  # +/- 1 SE
            plot_tags = valid_plot_stats.index  # Get the tags we are actually plotting

            bar_positions = np.arange(len(means))
            plt.bar(
                bar_positions,
                means,
                yerr=errors,
                capsize=5,
                color=sns.color_palette("viridis", len(means)),
            )

            plt.title(f"Mean {key} by Sentence Tag (Bar Plot +/- 1 SE)")
            # Use the filtered list of tags for x-ticks
            plt.xticks(bar_positions, plot_tags)

            means_errors_high = means + errors
            means_errors_low = means - errors
            gap = means_errors_high - means_errors_low
            ylim = (
                means_errors_low.min() - gap.max() * 0.1,
                means_errors_high.max() + gap.max() * 0.1,
            )
            plt.ylim(ylim)

        # --- Common Plot Adjustments ---
        # plt.xlabel("Sentence Tag")
        plt.ylabel(key.replace("_", " ").title())  # Nicer label for y-axis
        plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels for readability
        plt.grid(axis="y", linestyle="--", alpha=0.7)  # Add horizontal grid lines
        plt.tight_layout()  # Adjust layout to prevent labels overlapping

        save_path = os.path.join(output_dir, f"{key}_by_tag_{plot_type}plot.png")
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
        plt.close()  # Close the figure to free memory before the next plot


def do_regression(df):
    df = df[df["is_convergence"] == False]

    # Convert categorical variables to dummy variables if needed
    # if "tag" in df.columns:
    # df = pd.get_dummies(df, columns=["tag"], drop_first=True)

    # Create formula string with all variables as predictors of vert_score
    drop_cols = [
        "vert_score",
        "horz_score",
        "is_convergence",
        "diag_score",
        "kl_smooth",
        "js",
        "kl_prev_smooth",
        "kl_prev_no_smooth",
    ]
    # predictors = [
    # col for col in df.columns if col not in drop_cols
    # ]
    # predictors = [p for p in predictors if "has" not in p]
    # formula = "vert_score ~ " + " + ".join(predictors)
    # model = smf.ols(formula=formula, data=df).fit()
    # print(model.summary())

    # formula = "horz_score ~ " + " + ".join(predictors)
    # model = smf.ols(formula=formula, data=df).fit()
    # print(model.summary())

    # formula = "diag_score ~ " + " + ".join(predictors)
    # model = smf.ols(formula=formula, data=df).fit()
    # print(model.summary())

    # predictors_no_kl = [p for p in predictors if "kl" not in p and 'js' not in p]
    # predictors_no_kl = [p for p in predictors_no_kl if "accuracy" not in p
    #                     and "answers" not in p and "jaccard" not in p and "has" not in p
    #                     and "length" not in p]

    # df_grp = df.groupby('tag')[['vert_score', 'horz_score', 'diag_score']].mean()
    # print(df_grp)
    # quit()

    # formula = "kl_no_smooth ~ vert_score + horz_score + " + " + ".join(predictors_no_kl)
    # model = smf.ols(formula=formula, data=df).fit() answer_entropy +
    # print(model.summary())
    # formula = "js ~ pn + tag + vert_score + horz_score + correct_q + sentence_idx + normalized_position + sentence_token_len"

    # df.to_csv("test_supp.csv")
    # quit()

    formula = "horz_score ~ pn + correct_q + tag + vert_score + vert_weight + sentence_idx + normalized_position + sentence_token_len"

    model = smf.ols(formula=formula, data=df).fit()
    print(model.summary())

    r, p = stats.spearmanr(df["horz_score"], df["vert_score"], nan_policy="omit")
    print(f"vert score x horz score: {r=}, {p=}")
    r, p = stats.spearmanr(df["horz_score"], df["vert_weight"], nan_policy="omit")
    print(f"vert weight x horz score: {r=}, {p=}")

    # plt.hist(df['length'])
    # plt.savefig("length.png")
    # quit()

    # df_grp = df.groupby('tag')['delta_length'].mean()
    # print(df_grp)
    # quit()

    # formula = "delta_length ~ pn + tag + sentence_idx"# + vert_score + horz_score + correct_q + answer_entropy + sentence_idx + normalized_position + sentence_token_len"
    # model = smf.ols(formula=formula, data=df).fit()
    # print(model.summary())
    # formula = "answers_gap_jaccard ~ vert_score + horz_score + " + " + ".join(predictors_no_kl)
    # df["kl_no_smooth"] = stats.rankdata(df["kl_no_smooth"])
    # plt.hist(df["kl_no_smooth"])
    # plt.savefig("kl_no_smooth.png")
    # df = df[df["kl_no_smooth"] < 2]
    # quit()

    # print(df['js'].describe())
    # quit()

    # plt.hist(df['kl_smooth'])
    # plt.savefig("kl_smooth.png")
    # quit()

    # key = 'kl_no_smooth'
    # df.loc[df[key] < 0.01, key] = 0

    # pd.set_option('display.max_rows', None)
    # df_grp = df.groupby('tag')[key].mean() # groupby(['tag', 'pn'])['kl_smooth'].mean()
    # print(df_grp)

    # df_grp = df.groupby('tag')[key].sem() # groupby(['tag', 'pn'])['kl_smooth'].mean().
    # print(df_grp)
    # quit()

    # df = df[df['tag'].isin(['plan_generation', 'active_computation'])]

    # formula = "kl_no_smooth ~ pn + tag + sentence_idx"

    # formula = "kl_no_smooth ~ vert_score + horz_score + pn"
    # model = smf.ols(formula=formula, data=df).fit()
    # print(model.summary())


def do_vector_based_regression(
    graphs,
    valid_pns,
    problem_dir,
    only_pre_convergence,
    include_weights=False,
    model_name="qwen-14b",
):
    df = prepare_suppression_csv(
        graphs,
        valid_pns,
        problem_dir,
        only_pre_convergence,
        include_weights,
        model_name,
    )
    # if include_weights:
    #     plot_box_plot(df, keys=('vert_score', 'horz_score', 'diag_score', 'vert_weight', 'diag_weight'))
    # else:
    #     plot_box_plot(df, keys=('vert_score', 'horz_score', 'diag_score'))
    do_regression(df)


def do_itr_regression(df):
    formula = "eff ~ pn + receiving + giving + receiving:giving + dist"
    model = smf.ols(formula=formula, data=df).fit()
    print(model.summary())


def make_interaction_csv(
    graphs,
    valid_pns,
    problem_dir,
    only_pre_convergence,
    model_name,
    take_log=True,
    only_first=None,
    uzay_t=0.5,
):
    dfs_l = []
    for graph, pn in zip(graphs, valid_pns):
        df_as_l = get_pn_df_info(pn, problem_dir, only_pre_convergence, None, model_name)
        tags = df_as_l["tag"]
        uzay_mtx = get_importance_mtx(pn, old=False, t=uzay_t)
        # print(f"{uzay_mtx=}")
        # quit()
        # print(f"{uzay_mtx.shape=}")
        # uzay_mtx = get_importance_mtx(pn, old=False, t=None)
        # print(f"{uzay_mtx.shape=}")
        # quit()

        uzay_mtx = uzay_mtx[: graph.shape[0], : graph.shape[0]]
        uzay_abs_mtx = np.abs(uzay_mtx)
        n = graph.shape[0]
        assert len(tags) == n, f"{len(tags)=}, {n=}"

        entropy_l = get_sentence_entropies(
            problem_num=pn,
            p_nucleus=0.9999,
            model_name=model_name,
            only_pre_convergence=only_pre_convergence,
            problem_dir=problem_dir,
            only_first=only_first,
        )
        df_as_l["sentence_token_entropy"] = entropy_l

        for i in range(n):  # receiving sentence index
            if tags[i] == "unknown":
                continue
            for j in range(i):  # giving sentence index
                if tags[j] == "unknown":
                    continue
                # if abs(i - j) < 10:
                #     continue
                # Store giving -> receiving effect
                # Make sure you are indexing graph correctly: graph[giving_idx, receiving_idx]
                # if graph[i, j] < thresh:
                #     graph[i, j] = thresh
                row = {
                    "eff": graph[i, j],
                    "eff_uzay": uzay_mtx[j, i],
                    "eff_uzay_abs": uzay_abs_mtx[j, i],
                    "giving": tags[j],
                    "receiving": tags[i],
                    "pn_ic": str(pn),
                    "dist": abs(i - j),
                    "dist_relative": abs(j - i) / n,
                    "pos_giving": j,
                    "pos_receiving": i,
                    "pos_giving_rel": j / n,
                    "pos_receiving_rel": i / n,
                    "sentence_receiving_entropy": entropy_l[i],
                    "sentence_giving_entropy": entropy_l[j],
                }
                # print(f"{row=}")
                dfs_l.append(row)

    if not dfs_l:
        print("No valid interactions found to build DataFrame.")
        return

    df = pd.DataFrame(dfs_l)

    s = ""
    if not take_log:
        s += "_nolog"
    if only_first is not None:
        s += f"_of{only_first}"
    if model_name == "qwen-14b":
        df.to_csv(f"df_mtx_R1-Qwen14_May14{s}.csv", index=False)
    elif model_name == "qwen-14b-base":
        df.to_csv(f"df_mtx_base-Qwen14_May14{s}.csv", index=False)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def do_interaction_based_regression(
    graphs, valid_pns, problem_dir, only_pre_convergence, model_name, thresh=-5
):
    dfs_l = []
    for graph, pn in zip(graphs, valid_pns):
        df_as_l = get_pn_df_info(pn, problem_dir, only_pre_convergence, None, model_name)
        tags = df_as_l["tag"]
        n = graph.shape[0]
        assert len(tags) == n, f"{len(tags)=}, {n=}"
        for i in range(n):  # receiving sentence index
            if tags[i] == "unknown":
                continue
            for j in range(i):  # giving sentence index
                if tags[j] == "unknown":
                    continue
                # if abs(i - j) < 10:
                #     continue
                # Store giving -> receiving effect
                # Make sure you are indexing graph correctly: graph[giving_idx, receiving_idx]
                if graph[i, j] < thresh:
                    graph[i, j] = thresh
                row = {
                    "eff": graph[i, j],
                    "giving": tags[j],
                    "receiving": tags[i],
                    "pn": str(pn),
                    "dist": abs(i - j),
                }
                dfs_l.append(row)

    if not dfs_l:
        print("No valid interactions found to build DataFrame.")
        return

    df = pd.DataFrame(dfs_l)

    # Group and calculate mean effect
    df_grp = df.groupby(["giving", "receiving"])["eff"].mean().reset_index()

    # Create a list of tuples: ( (giving, receiving), mean_effect )
    pair_effects = [
        ((row["giving"], row["receiving"]), row["eff"]) for index, row in df_grp.iterrows()
    ]

    # Sort lexicographically by pair for consistent output order
    sorted_pair_effects = sorted(pair_effects, key=lambda x: x[0])

    # --- Color Mapping Setup ---
    effects = np.array(
        [eff for pair, eff in sorted_pair_effects if not np.isnan(eff)]
    )  # Filter NaNs for range calculation
    if len(effects) == 0:
        print("No valid (non-NaN) effects to process for color mapping.")
        min_eff, max_eff, eff_range = 0, 0, 0
    else:
        min_eff = np.quantile(effects, 0.05)
        max_eff = np.quantile(effects, 0.95)
        eff_range = max_eff - min_eff
        # Handle potential NaN if all effects were NaN (unlikely but possible) - already handled by filtering
        if (
            eff_range <= 1e-9 and len(effects) > 0
        ):  # Handle case where all valid effects are the same
            print("Warning: All valid effect values are nearly identical.")
            # Keep eff_range slightly positive to avoid division by zero but map everything to middle color
            eff_range = 1e-9  # Set a tiny range

    print(
        f"Debug: Effect Range (used for color): min={min_eff:.4f}, max={max_eff:.4f}, range={eff_range:.4f}"
    )

    # --- RED <-> GREEN Spectrum ---
    # Green (low values) -> Yellow (mid) -> Red (high values)
    color_spectrum_indices = [
        46,
        82,
        118,
        154,  # Greens
        190,
        226,
        220,  # Lime -> Yellows
        214,
        208,
        202,
        196,  # Oranges -> Red
    ]
    n_colors = len(color_spectrum_indices)
    COLOR_RESET = "\033[0m"

    def get_color_for_effect(eff):
        if eff_range <= 1e-9:  # Handle zero or tiny range
            # All values are the same, pick a neutral color (e.g., middle index or default)
            spectrum_idx = n_colors // 2
        else:
            normalized = (eff - min_eff) / eff_range
            # Clamp normalized value between 0 and 1
            normalized = max(0.0, min(1.0, normalized))
            # Map normalized value to an index in our spectrum
            spectrum_idx_float = normalized * (n_colors - 1)
            spectrum_idx = int(round(spectrum_idx_float))
            # Ensure index is within bounds
            spectrum_idx = max(0, min(spectrum_idx, n_colors - 1))

        color_index = color_spectrum_indices[spectrum_idx]

        # --- Debug Print (Uncomment to see details) ---
        # print(f"  eff={eff:.4f}, norm={normalized:.4f}, float_idx={spectrum_idx_float:.2f}, idx={spectrum_idx}, color={color_index}")

        # Construct ANSI code for foreground color
        return f"\033[38;5;{color_index}m"

    # Print sorted pairs and their mean effects with color coding
    print(
        f"\n--- Mean Effect per Interaction Type (Color Spectrum: Green(low) <-> Red(high) | Min={min_eff:.4f}, Max={max_eff:.4f}) ---"
    )
    for pair, eff in sorted_pair_effects:
        giving, receiving = pair
        # Handle potential NaN effect values before getting color
        if np.isnan(eff):
            color_code = ""  # No color for NaN
            eff_str = "NaN"
        else:
            color_code = get_color_for_effect(eff)
            eff_str = f"{eff:.6f}"

        print(
            f"{color_code}Giving: {giving:<15} Receiving: {receiving:<15} -> Mean Eff: {eff_str}{COLOR_RESET}"
        )

    # --- Run Overall Regression ---
    # (Keep the quit() if you only want the colored list for now)
    # quit()

    # --- Run Overall Regression ---
    print("\n--- Overall Regression on Interactions ---")
    # Ensure 'pn' is treated as categorical if needed for regression
    df["pn"] = df["pn"].astype("category")
    # Optional: Treat giving/receiving as categorical too if not already strings
    df["giving"] = df["giving"].astype("category")
    df["receiving"] = df["receiving"].astype("category")

    # Add check for sufficient data before regression
    if len(df) > 1 and df["eff"].notna().sum() > (
        len(df.columns) - 1
    ):  # Check if enough non-NA data points
        try:
            do_itr_regression(df)  # Run the overall regression
        except Exception as e:
            print(f"Error during overall regression: {e}")
            print("Skipping regression.")
    else:
        print("Insufficient data or too many NaNs for overall regression.")


def get_uzay_scores(pn, pi, max_len, do_abs=False, uzay_t=0.5):
    uzay_mtx = get_importance_mtx(pn, old=False, t=uzay_t)
    if do_abs:
        uzay_mtx = np.abs(uzay_mtx)
    # print(uzay_mtx)

    uzay_mtx_ = np.full(uzay_mtx.shape, np.nan)
    for i in range(uzay_mtx.shape[0]):
        for j in range(uzay_mtx.shape[1]):
            uzay_mtx_[i, j] = uzay_mtx[j, i]
    uzay_mtx = uzay_mtx_

    if max_len is not None:
        uzay_mtx = uzay_mtx[:max_len, :max_len]
    vert_scores = get_attn_vert_scores(
        uzay_mtx,
        proximity_ignore=pi,
        control_depth=False,
        ignore_prompt=False,
        ignore_out=False,
        drop_first=0,
    )
    horz_scores = get_attn_horz_scores(
        uzay_mtx,
        proximity_ignore=pi,
        control_depth=False,
        ignore_prompt=False,
        ignore_out=False,
    )
    return vert_scores, horz_scores


def prepare_suppression_csv(
    graphs,
    valid_pns,
    problem_dir,
    only_pre_convergence,
    model_name="qwen-14b",
    proximity_ignore=(0, 1, 4, 8, 16),
    top_k=(8, 16, 32, 64),
    only_first=None,
    take_log=True,
    uzay_t=0.5,
):

    qwen_llama_str = "_qwen" if "qwen" in model_name else "_llama"
    problem_dir = os.path.join("target_problems", f"temperature_0.6_top_p_0.95{qwen_llama_str}")

    dfs_l = []
    for graph, pn in zip(graphs, valid_pns):
        # if "00" == str(pn)[-2:]:
        #     continue

        # sentence2ranges, problem = get_full_CoT_token_ranges(
        #     pn,
        #     problem_dir=os.path.join("target_problems", f"temperature_0.6_top_p_0.95{qwen_llama_str}"),
        #     verbose=False,
        #     only_pre_convergence=False,
        #     model_name=model_name,
        # )
        # quit()
        df_as_l = get_pn_df_info(pn, problem_dir, only_pre_convergence, None, model_name)

        # print(f'{graphs=}')
        if graph is not None:
            print(f"{graph.shape=}")

            entropy_l = get_sentence_entropies(
                problem_num=pn,
                p_nucleus=0.9999,
                model_name=model_name,
                only_pre_convergence=only_pre_convergence,
                problem_dir=problem_dir,
                only_first=only_first,
            )
            df_as_l["sentence_token_entropy"] = entropy_l

            for pi in proximity_ignore:
                vert_scores = get_attn_vert_scores(
                    graph,
                    proximity_ignore=pi,
                    control_depth=False,
                    ignore_prompt=False,
                    ignore_out=False,
                )
                df_as_l[f"v_supp_ignore_{pi}"] = vert_scores
                horz_scores = get_attn_horz_scores(
                    graph,
                    proximity_ignore=pi,
                    control_depth=False,
                    ignore_prompt=False,
                    ignore_out=False,
                )
                df_as_l[f"h_supp_ignore_{pi}"] = horz_scores
                diag_scores = get_attn_direction_scores(
                    graph,
                    distance=8,
                    control_depth=False,
                    ignore_prompt=False,
                    ignore_out=False,
                )
                df_as_l[f"d_supp_ignore_{pi}"] = diag_scores

            for pi in proximity_ignore:
                vert_scores, horz_scores = get_uzay_scores(
                    pn, pi, max_len=graph.shape[0], uzay_t=uzay_t
                )
                df_as_l[f"v_uzay_ignore_{pi}"] = vert_scores
                df_as_l[f"h_uzay_ignore_{pi}"] = horz_scores

                vert_scores, horz_scores = get_uzay_scores(pn, pi, max_len=None, do_abs=False)
                vert_scores = vert_scores[: graph.shape[0]]
                horz_scores = horz_scores[: graph.shape[0]]
                df_as_l[f"v_uzay_full_ignore_{pi}"] = vert_scores
                df_as_l[f"h_uzay_full_ignore_{pi}"] = horz_scores

                vert_scores, horz_scores = get_uzay_scores(
                    pn, pi, max_len=graph.shape[0], do_abs=True
                )
                df_as_l[f"v_uzay_abs_ignore_{pi}"] = vert_scores
                df_as_l[f"h_uzay_abs_ignore_{pi}"] = horz_scores

                vert_scores, horz_scores = get_uzay_scores(pn, pi, max_len=None, do_abs=True)
                vert_scores = vert_scores[: graph.shape[0]]
                horz_scores = horz_scores[: graph.shape[0]]
                df_as_l[f"v_uzay_full_abs_ignore_{pi}"] = vert_scores
                df_as_l[f"h_uzay_full_abs_ignore_{pi}"] = horz_scores

        for pi in proximity_ignore:
            for tk in top_k:
                vert_weights = get_weights_scores(
                    pn,
                    top_k=tk,
                    model_name=model_name,
                    quantize_8bit=False,
                    quantize_4bit=False,
                    only_pre_convergence=only_pre_convergence,
                    only=None,
                    proximity_ignore=pi,
                    problem_dir=problem_dir,
                    drop_first=4,
                    drop_last=32,
                )
                diag_weights = get_weights_scores(
                    pn,
                    top_k=tk,
                    model_name=model_name,
                    quantize_8bit=False,
                    quantize_4bit=False,
                    only_pre_convergence=only_pre_convergence,
                    only=None,
                    proximity_ignore=pi,
                    problem_dir=problem_dir,
                )
                # print(f'{len(vert_weights)=} | {len(df_as_l["correct_q"])=} | {pn=}')
                # quit()

                vert_weights[-8:] = np.nan

                df_as_l[f"v_weight_ignore_{pi}_top_{tk}"] = vert_weights
                df_as_l[f"d_weight_ignore_{pi}_top_{tk}"] = diag_weights

        for key in df_as_l.keys():
            print(f"{key} | {len(df_as_l[key])=}")
            if "llama" not in model_name:
                if len(df_as_l[key]) != len(vert_scores):
                    print(f"BAD! {key}: {len(df_as_l[key])=}, {len(vert_scores)=}")

        df = pd.DataFrame(df_as_l)
        # print(list(df.columns))
        # quit()
        dfs_l.append(df)
    df = pd.concat(dfs_l)
    df.reset_index(drop=True, inplace=True)
    s = ""
    if not take_log:
        s += "_nolog"
    if only_first is not None:
        s += f"_of{only_first}"
    if model_name == "qwen-14b":
        df.to_csv(f"csvs/df_attn_R1-Qwen14_May14{s}.csv", index=False)
    elif model_name == "qwen-14b-base":
        df.to_csv(f"csvs/df_attn_base-Qwen14_May14{s}.csv", index=False)
    elif "llama8-base" in model_name:
        df.to_csv(f"csvs/df_attn_llama8-base_May14{s}.csv", index=False)
    elif "llama8" in model_name:
        df.to_csv(f"csvs/df_attn_llama8_May14{s}.csv", index=False)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def make_sentence_csv(model_name="qwen-14b"):
    # problem_dir = os.path.join("target_problems", "temperature_0.6_top_p_0.92")
    problem_dir = os.path.join("target_problems", "temperature_0.6_top_p_0.95")

    only = None
    only_pre_convergence = "semi"
    take_log = True
    only_first = None
    # only_pre_convergence = 'semi'
    # layers_to_mask = get_most_sensitive_heads_map(top_k=200, only_pre_convergence=True,
    #                                                problem_dir=problem_dir)
    problem_nums = get_problem_nums(only=only, only_pre_convergence=only_pre_convergence)
    layers_to_mask = {i: list(range(40)) for i in range(48)}
    if "qwen" in model_name:
        graphs, valid_pns = load_all_graphs(
            problem_nums,
            layers_to_mask,
            only_pre_convergence=only_pre_convergence,
            problem_dir=problem_dir,
            only_first=only_first,
            take_log=take_log,
            model_name=model_name,
        )
    else:
        graphs = [None] * len(problem_nums)
        valid_pns = problem_nums
    prepare_suppression_csv(
        graphs,
        valid_pns,
        problem_dir,
        only_pre_convergence,
        model_name=model_name,
        only_first=only_first,
        take_log=take_log,
        uzay_t=0.5,
    )


def make_matrix_csv(model_name="qwen-14b"):
    # problem_dir = os.path.join("target_problems", "temperature_0.6_top_p_0.92")
    problem_dir = os.path.join("target_problems", "temperature_0.6_top_p_0.95")

    only = None
    only_pre_convergence = "semi"
    take_log = True
    only_first = None
    # only_pre_convergence = 'semi'
    # layers_to_mask = get_most_sensitive_heads_map(top_k=200, only_pre_convergence=True,
    #                                                problem_dir=problem_dir)
    problem_nums = get_problem_nums(only=only, only_pre_convergence=only_pre_convergence)
    layers_to_mask = {i: list(range(40)) for i in range(48)}
    graphs, valid_pns = load_all_graphs(
        problem_nums,
        layers_to_mask,
        only_pre_convergence=only_pre_convergence,
        problem_dir=problem_dir,
        only_first=only_first,
        take_log=take_log,
        model_name=model_name,
    )

    make_interaction_csv(
        graphs,
        valid_pns,
        problem_dir,
        only_pre_convergence,  # thresh=-5
        model_name=model_name,
        take_log=take_log,
        only_first=only_first,
        uzay_t=0.5,
    )


if __name__ == "__main__":
    make_sentence_csv(model_name="llama8")
    make_sentence_csv(model_name="llama8-base")
