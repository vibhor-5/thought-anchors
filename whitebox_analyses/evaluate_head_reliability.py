import os
from matplotlib import pyplot as plt
import numpy as np
from pkld import pkld
from scipy import stats

import sys
import numpy as np

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from run_target_problems import (
    get_3d_ar_kurtosis,
    get_all_heads_vert_data,
    get_problem_kurtosis,
    get_problem_nums,
)


# @pkld(store="both")
def get_kurt_matrix(
    model_name="qwen-14b",
    quantize_8bit=True,
    quantize_4bit=False,
    only_pre_convergence=False,
    only=None,
    problem_dir=os.path.join("target_problems", "temperature_0.6_top_p_0.95"),
    drop_first=0,
    drop_last=0,
    proximity_ignore=20,
    vert_score_calc=None,
    pns=None,
):
    if pns is None:
        problem_nums = get_problem_nums(
            only=only,
            only_pre_convergence=only_pre_convergence,
            problem_dir=problem_dir,
        )
    else:
        problem_nums = pns

    all_mappings = {}

    kurts = []
    # problem_nums = problem_nums[::-1]

    for pn_i, pn in enumerate(problem_nums):

        try:
            kurt = get_problem_kurtosis(
                pn,
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
            all_mappings[pn] = kurt
        except np.exceptions.AxisError as e:
            print(f"Error on {pn}: {e}")
            continue

        # print(kurt)
        # quit()

        if kurt is None:
            continue
        p_nan = np.sum(np.isnan(kurt)) / kurt.size
        if p_nan > 0.5:
            continue
        # print(f"{kurt.shape=}")
        # print(f"{p_nan=:.1%}")
        # print(f"{np.var(kurt)=}")
        kurts.append(kurt)
    kurts = np.array(kurts)
    assert np.sum(np.isnan(kurts)) == 0
    kurts[:, 0, :] = np.nan  # ignore layer 0
    # kurt = np.mean(kurts, axis=0)
    return kurts


if __name__ == "__main__":
    drop_first = 4
    drop_last = 32
    proximity_ignore = 4
    model_name = "qwen-14b"
    quantize_8bit = False
    quantize_4bit = False
    only_pre_convergence = "semi"
    only = None
    problem_dir = os.path.join("target_problems", "temperature_0.6_top_p_0.95")

    kurts = get_kurt_matrix(
        model_name=model_name,
        quantize_8bit=quantize_8bit,
        quantize_4bit=False,
        only_pre_convergence="semi",
        only=None,
        problem_dir=os.path.join("target_problems", "temperature_0.6_top_p_0.95"),
        drop_first=drop_first,
        drop_last=drop_last,
        proximity_ignore=proximity_ignore,
    )
    print(kurts.shape)

    layers = np.arange(1, kurts.shape[1])
    heads = np.arange(kurts.shape[2])

    n_pn = kurts.shape[0]
    pn_cutoff = n_pn // 2

    kurts_first_half = kurts[::2, :, :]
    kurts_second_half = kurts[1::2, :, :]

    kurts_first_l = []
    kurts_second_l = []

    for layer in layers:
        # if layer < 20:
        #     continue
        for head in heads:
            kurt_first_half = kurts_first_half[:, layer, head]
            kurt_second_half = kurts_second_half[:, layer, head]

            kurts_first_l.append(np.mean(kurt_first_half))
            kurts_second_l.append(np.mean(kurt_second_half))

    kurts_first_l = np.array(kurts_first_l)
    kurts_second_l = np.array(kurts_second_l)

    r, p = stats.pearsonr(kurts_first_l, kurts_second_l)
    print(f"{r=}, {p=}")

    plt.rcParams["font.size"] = 11

    fig = plt.figure(figsize=(3.5, 3))

    plt.scatter(kurts_first_l, kurts_second_l, color="dodgerblue", alpha=0.25, s=20)

    # Add line of best fit - using fixed endpoints
    z = np.polyfit(kurts_first_l, kurts_second_l, 1)
    slope, intercept = z

    # Create line using the min and max of x-axis for consistent endpoints
    x_min, x_max = plt.xlim()
    x_line = np.array([x_min, x_max])
    y_line = slope * x_line + intercept

    plt.plot(x_line, y_line, "k--", alpha=0.7, linewidth=1.5)

    # Add correlation text
    plt.text(
        0.6,
        0.7,
        f"r = {r:.2f}",
        transform=plt.gca().transAxes,  # Use axes coordinates
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )

    # plt.hlines(0, 0, 42, color="k", linestyle="--")
    # plt.vlines(0, 0, 42, color="k", linestyle="--")
    plt.axis("square")
    plt.xlim(-1, 42)
    plt.ylim(-1, 42)
    plt.xticks(np.arange(0, 42, 10))
    plt.yticks(np.arange(0, 42, 10))
    plt.gca().spines[["top", "right"]].set_visible(False)

    plt.xlabel("First half kurtosis", labelpad=7)
    plt.ylabel("Second half kurtosis", labelpad=7)
    plt.title("Split-half reliability assessment", fontsize=12, pad=10)
    fp_out = f"plots/kurt_plots/reliability_{model_name}_d{drop_first}-{drop_last}_pi{proximity_ignore}.png"
    plt.subplots_adjust(bottom=0.17, top=0.8, left=0.1, right=0.95)
    plt.savefig(fp_out, dpi=300)
    plt.show()
