import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3"

from collections import defaultdict
import json
import glob
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pkld import pkld
from tqdm import tqdm
from model_read import analyze_text
from utils import model2layers_heads, tokens_to_clean, get_tokenizer, get_raw_tokens
from uzay_utils import (
    get_chunk_ranges,
    get_chunk_token_ranges,
    split_solution_into_chunks,
)
from scipy import stats
import statsmodels.formula.api as smf
from numba import njit
from pathlib import Path

# print(os.getcwd())
# quit()


def load_problem_json(
    problem_num,
    problem_dir=os.path.join("target_problems", "temperature_0.6_top_p_0.95"),
):
    fp_json = os.path.join(problem_dir, f"problem_{problem_num}.json")
    with open(fp_json, "r", encoding="utf-8") as f:
        problem = json.load(f)
    return problem


# @pkld(store="both",)
def get_full_CoT_token_ranges(
    problem_num,
    problem_dir=os.path.join("target_problems", "temperature_0.6_top_p_0.95"),
    verbose=False,
    only_pre_convergence=False,
    model_name="qwen-14b",
):
    print(f"CoT thing: {problem_num=}")
    problem = load_problem_json(problem_num, problem_dir)

    sentences = [s["sentence"] for s in problem["sentences"]]
    solution = problem["base_solution"]["solution"]
    char_chunk_ranges = get_chunk_ranges(solution, sentences)
    token_chunk_ranges = get_chunk_token_ranges(
        solution,
        char_chunk_ranges,
        get_tokenizer(model_name),
    )

    full_text = problem["base_solution"]["full_cot"]
    full_text_tokens = get_raw_tokens(full_text, model_name)

    # assert len(full_text_tokens) == len(full_text_tokens_R1) + 1, 'Misalignment related to reasoning-model tokens'

    print(f"{problem_num=} | {len(full_text_tokens)=}")
    if only_pre_convergence == "semi" and len(full_text_tokens) > 4000:
        convergence_point = None
        for i, chunk_range in enumerate(token_chunk_ranges):
            if chunk_range[1] > (4000 - 256):  # buffer for the +256 added to the end
                convergence_point = i - 1
                break
        if convergence_point is None:
            if token_chunk_ranges[-1][1] < (4000 - 256):
                convergence_point = len(token_chunk_ranges) - 1

        if convergence_point is not None:
            problem["sentences"] = problem["sentences"][: convergence_point + 1]
            print(f"Limiting to first 4000 tokens for {problem_num=}")
        else:
            print("No convergence point found for ", problem_num)
            print(f"{len(full_text_tokens)=}")
            print(f"{token_chunk_ranges=}")
            quit()
    elif only_pre_convergence:
        convergence_point = None
        for i, s in enumerate(problem["sentences"]):
            if s["is_convergence"]:
                convergence_point = i
                break
        if convergence_point is None:
            print("No convergence point found for ", problem_num)
            # quit()
            convergence_point = len(problem["sentences"]) - 1
        assert convergence_point is not None, f"No convergence point found for {problem_num}"
        problem["sentences"] = problem["sentences"][: convergence_point + 1]

    assert len(sentences) == len(token_chunk_ranges)

    prompt = problem["base_solution"]["prompt"]
    len_prompt_str = len(prompt)
    # print(f'{len_prompt_str=}')

    sentences_ds = problem["sentences"]
    sentence2full_tokens = {}
    for i, sentence_d in enumerate(sentences_ds):
        s = sentence_d["sentence"]
        tag = sentence_d["tag"]
        # print
        solution_char_st = char_chunk_ranges[i][0] - 1
        if i == 0:
            solution_char_st += 1
        solution_char_end = char_chunk_ranges[i][1] - 1

        s_tokens = get_raw_tokens(s, model_name)
        if "%." in s_tokens:
            solution_char_end += 1
            if i < len(char_chunk_ranges) - 1:
                char_chunk_ranges[i + 1] = (
                    char_chunk_ranges[i + 1][0] + 1,
                    char_chunk_ranges[i + 1][1],
                )

        if solution_char_end - solution_char_st <= 1:
            # print(f'{model_name=}')
            # quit()
            # raise ValueError(f'{model_name=}')
            assert "llama" in model_name
            # continue
        #
        full_text_range = full_text[
            len_prompt_str + solution_char_st : len_prompt_str + solution_char_end
        ]
        if s not in full_text_range and verbose:
            pass
        full_text_to_start = full_text[: len_prompt_str + solution_char_st]
        full_text_to_end = full_text[: len_prompt_str + solution_char_end]
        if verbose:
            print(f"{i} ({tag}): {s}")
        # print(f'{s=}')
        # print(f'{i} | {full_text_range=}')

        full_text_to_start_tokens = len(get_raw_tokens(full_text_to_start, model_name))
        full_text_to_end_tokens = len(get_raw_tokens(full_text_to_end, model_name))
        sentence2full_tokens[i] = (full_text_to_start_tokens, full_text_to_end_tokens)
        # print(full_text_tokens[full_text_to_start_tokens:full_text_to_end_tokens])
        if only_pre_convergence and convergence_point is not None and i >= convergence_point:
            print("found convergence point")
            problem["full_cot_truncated"] = full_text[: len_prompt_str + solution_char_end + 256]
            break
        # if only_pre_convergence and model_name == "qwen-14b-base":
        # assert full_text_tokens[char_chunk_ranges[i][1] - 1] == full_text_tokens_R1[char_chunk_ranges[i][1] - 1], f"Found mismatch related to reasoning-model tokens ()"
    if only_pre_convergence == "semi" and convergence_point is None:
        problem["full_cot_truncated"] = full_text
    if only_pre_convergence and model_name == "qwen-14b-base":
        print("Found no mismatches!")
    for i in list(sentence2full_tokens.keys()):
        if sentence2full_tokens[i][1] - sentence2full_tokens[i][0] <= 1:
            del sentence2full_tokens[i]
    # print(sentence2full_tokens)
    # quit()

    return sentence2full_tokens, problem


def avg_matrix_by_chunk(matrix, sentence2ranges):
    end_idx = matrix.shape[-1]
    max_sentence = max(sentence2ranges.keys())
    idxs = [0] + [sentence2ranges[i][0] for i in range(max_sentence + 1)]
    idxs += [sentence2ranges[max_sentence][1], end_idx]

    # Ensure matrix is float32 for Numba compatibility
    if matrix.dtype == np.float16:
        matrix = matrix.astype(np.float32)
    avg_mat = create_averaged_matrix_safe(matrix, idxs)

    return avg_mat


# Numba optimized version - much faster for large matrices
@njit(cache=True, fastmath=True, debug=True)
def create_averaged_matrix(matrix, indices, only_bottom_triangle=False):
    # Sort indices to ensure proper ranges
    indices = np.sort(np.array(indices))
    n = len(indices)

    # Initialize result matrix
    result = np.zeros((n - 1, n - 1), dtype=np.float32)

    # For each region, calculate the average
    for i in range(n - 1):
        row_start = indices[i]
        row_end = indices[i + 1]

        for j in range(n - 1):
            if only_bottom_triangle and j > i:
                break
            col_start = indices[j]
            col_end = indices[j + 1]

            # Compute sum and count directly instead of calling mean()
            total = 0.0
            count = 0

            for ri in range(row_start, row_end + 1):
                for ci in range(col_start, col_end + 1):
                    total += matrix[ri, ci]
                    count += 1

            if count > 0:
                result[i, j] = total / count

    return result


def create_averaged_matrix_safe(matrix, indices, only_bottom_triangle=True):
    """Safer version without Numba that won't segfault"""
    indices = np.sort(np.array(indices))
    n = len(indices)
    result = np.zeros((n - 1, n - 1), dtype=np.float32)

    # Process each region pair
    for i in range(n - 1):
        row_start = min(indices[i], matrix.shape[0] - 1)  # Ensure within bounds
        row_end = min(indices[i + 1], matrix.shape[0] - 1)

        # Skip invalid regions
        if row_start >= row_end:
            continue

        for j in range(n - 1):
            if only_bottom_triangle and j > i:
                continue
            col_start = min(indices[j], matrix.shape[1] - 1)
            col_end = min(indices[j + 1], matrix.shape[1] - 1)

            # Skip invalid regions
            if col_start >= col_end:
                continue

            # Extract and compute mean directly with numpy
            region = matrix[row_start:row_end, col_start:col_end]
            if region.size > 0:
                result[i, j] = np.mean(region)

    return result


def clean_avg_mat(avg_mat, control_depth=True):
    n = avg_mat.shape[0]
    trius = np.triu_indices_from(avg_mat, k=1)
    avg_mat[trius] = np.nan
    if control_depth:
        avg_mat *= np.arange(n)[:, None]
    return avg_mat


def get_attn_horz_scores(
    avg_mat,
    proximity_ignore=20,
    control_depth=True,
    ignore_prompt=True,
    ignore_out=True,
):
    avg_mat = clean_avg_mat(avg_mat, control_depth)
    if ignore_prompt:
        avg_mat = avg_mat[1:, 1:]
    if ignore_out:
        avg_mat = avg_mat[:-1, :-1]
    horz_scores = []

    for i in range(avg_mat.shape[0]):
        upto = max(0, i - proximity_ignore)
        horz_lines = avg_mat[i, :upto]
        # print(f"{i=}, {horz_lines=}")
        horz_score = np.nanmean(horz_lines)
        horz_scores.append(horz_score)
    horz_scores = np.array(horz_scores)
    return horz_scores


def get_attn_vert_scores(
    avg_mat,
    proximity_ignore=20,
    ignore_out=True,
    control_depth=True,
    ignore_prompt=True,
    take_exp=False,
    max_proximity=None,
    drop_first=10,
    vert_score_calc=None,
):
    avg_mat = clean_avg_mat(avg_mat, control_depth)
    if ignore_prompt:
        avg_mat = avg_mat[1:, 1:]
    if ignore_out:
        avg_mat = avg_mat[:-1, :-1]
    n = avg_mat.shape[-1]
    vert_scores = []
    total_sum = 0
    for i in range(n):
        if max_proximity is not None:
            vert_lines = avg_mat[i + proximity_ignore : i + max_proximity, i]
        else:
            vert_lines = avg_mat[i + proximity_ignore :, i]
        if take_exp:
            vert_lines = np.exp(vert_lines)
        if vert_score_calc is None:
            vert_score = np.nanmean(vert_lines)
        elif vert_score_calc == "median":
            vert_score = np.nanmedian(vert_lines)
        else:
            raise ValueError(f"Unknown vert_score_calc: {vert_score_calc}")

        vert_scores.append(vert_score)
        total_sum += vert_score
    vert_scores = np.array(vert_scores)
    if drop_first > 0:
        vert_scores[:drop_first] = np.nan
    if drop_first > 0:
        vert_scores[-drop_first:] = np.nan
    return vert_scores


def get_attn_direction_scores(
    avg_mat,
    distance=10,
    ignore_out=True,
    ignore_prompt=True,
    control_depth=False,
    pad_nan=False,
):
    # TODO: address oscilatting cases
    avg_mat = clean_avg_mat(avg_mat, control_depth)
    diag_scores = []
    if ignore_prompt:
        avg_mat = avg_mat[1:, 1:]
        if pad_nan:
            diag_scores.append(np.nan)
    if ignore_out:
        avg_mat = avg_mat[:-1, :-1]
    n = avg_mat.shape[-1]
    for i in range(n):
        if (i < distance) or (i >= n - distance):
            diag_scores.append(np.nan)
            continue

        prev_attns = []
        for dist in range(1, distance + 1):
            prev_attns.append(avg_mat[i, i - dist])
        prev_attns = np.array(prev_attns)
        next_attns = []
        for dist in range(1, distance + 1):
            next_attns.append(avg_mat[i + dist, i])
        next_attns = np.array(next_attns)
        prev_avg = np.mean(prev_attns)
        next_avg = np.mean(next_attns)
        # print(f"{prev_avg=}, {next_avg=}")
        dif = next_avg - prev_avg
        norm_dif = dif / (prev_avg + next_avg)
        diag_scores.append(norm_dif)
    if ignore_out and pad_nan:
        diag_scores.append(np.nan)
    return diag_scores


def get_3d_ar_kurtosis(all_layer_scores):
    kurt = stats.kurtosis(
        all_layer_scores, axis=2, fisher=True, bias=True, nan_policy="omit"
    )  # NaNs from the proximity ignorance
    return kurt


@pkld
def get_all_heads_vert_data(
    problem_num,
    model_name,
    problem_dir,
    quantize_8bit,
    quantize_4bit,
    only_pre_convergence,
    drop_first=0,
    drop_last=0,
    proximity_ignore=20,
    vert_score_calc=None,
):
    all_head_vert_scores = []
    n_layers, n_heads = model2layers_heads(model_name)
    for layer in range(n_layers):
        head_scores = []
        for head in range(n_heads):
            avg_matrix = get_problem_attn_avg(
                problem_num,
                layer,
                head,
                problem_dir=problem_dir,
                only_pre_convergence=only_pre_convergence,
                model_name=model_name,
                quantize_8bit=quantize_8bit,
                quantize_4bit=quantize_4bit,
            )
            if avg_matrix is None:
                return None
            vert_scores = get_attn_vert_scores(
                avg_matrix,
                drop_first=drop_first,
                proximity_ignore=proximity_ignore,
                vert_score_calc=vert_score_calc,
            )
            head_scores.append(vert_scores)

        all_head_vert_scores.append(head_scores)
    all_head_vert_scores = np.array(all_head_vert_scores)
    return all_head_vert_scores


@pkld
def get_problem_kurtosis(
    problem_num,
    model_name="qwen-14b",
    problem_dir=os.path.join("target_problems", "temperature_0.6_top_p_0.95"),
    quantize_8bit=True,
    quantize_4bit=False,
    only_pre_convergence=False,
    drop_first=0,
    drop_last=0,
    proximity_ignore=20,
    vert_score_calc=None,
):
    all_head_vert_scores = get_all_heads_vert_data(
        problem_num,
        model_name,
        problem_dir,
        quantize_8bit,
        quantize_4bit,
        only_pre_convergence,
        drop_first,
        drop_last,
        proximity_ignore,
        vert_score_calc,
    )
    if all_head_vert_scores is None:
        return None
    if np.sum(~np.isnan(all_head_vert_scores)) == 0:
        print(
            f"All NaN for {problem_num} ({drop_first=}, {drop_last=}; {all_head_vert_scores.shape=})"
        )
        return None
    kurt = get_3d_ar_kurtosis(all_head_vert_scores)
    return kurt


def save_all_sentence_chunks(
    problem_num,
    problem_dir=os.path.join("target_problems", "temperature_0.6_top_p_0.95"),
    as_dict=True,
    model_name="qwen-15b",
    quantize_8bit=True,
    quantize_4bit=False,
    only_pre_convergence=False,
):
    print(f"Saving all sentence chunks: {problem_num=}, {only_pre_convergence=}")
    problem_dir_final = problem_dir.split("/")[-1]
    if "target_problems" in problem_dir_final:
        problem_dir_final = problem_dir_final.split("\\")[1]
    #     dir_matrices = r"P:\CoT_test0\CoT_test0\avg_matrices"

    # else:
    dir_matrices = "avg_matrices"

    n_layers, n_heads = model2layers_heads(model_name)

    all_exist = True
    for layer in range(n_layers):
        for head in range(n_heads):
            fp_out = rf"{dir_matrices}/{model_name}/{problem_dir_final}/{problem_num}/{layer}_{head}_{only_pre_convergence}.npy"
            # fp_out = fp_out.replace("/", "\\")

            if not os.path.exists(fp_out):
                all_exist = False
                print(f"Missing: {fp_out}")
                # raise ValueError(f"Missing: {fp_out}")
                # quit()
                break
    if all_exist:
        print(f"All matrices for {problem_num} already exist")
        return True
    print(f"\tStarting new computation...")

    try:
        sentence2ranges, problem = get_full_CoT_token_ranges(
            problem_num,
            problem_dir,
            only_pre_convergence=only_pre_convergence,
            model_name=model_name,
        )
    except AssertionError as e:
        print(f"No convergence on {problem_num}: {e}")
        # assert "p_0.92" in problem_dir
        return False

    text = (
        problem["full_cot_truncated"]
        if only_pre_convergence
        else problem["base_solution"]["full_cot"]
    )
    token_texts = get_raw_tokens(text, model_name)
    # print(f'{len(token_texts)=}')
    # print(f'{only_pre_convergence=}')
    # quit()
    if len(token_texts) > 4200:
        print("Just too long sadly")
        device_map = "cpu"
    else:
        device_map = "auto"
    device_map = "auto"

    if os.name == "nt":
        print("Running on PC. Using CPU")
        device_map = "cpu"
    # device_map = "auto"

    # print(f"{len(token_texts)=}")
    # quit()

    # raise ValueError("Trying to run model on PC")
    result = analyze_text(
        text,
        model_name=model_name,
        verbose=True,
        float32=model_name == "qwen-15b",  # Use fp16 as default since we're using quantization
        quantize_8bit=quantize_8bit,
        quantize_4bit=quantize_4bit,
        attn_layers=None,  # [6, 17],
        do_layers=[],  # None#[0],
        return_logits=False,
        device_map=device_map,
    )

    if len(result["attention_weights"]) == 0:
        print(f"No attention weights for {problem_num}")
        return False

    for layer in tqdm(range(n_layers), desc=f"Saving all sentence chunks ({problem_num=})"):
        for head in range(n_heads):
            # fp_out = rf"avg_matrices/{model_name}/{problem_dir_final}/{problem_num}/{layer}_{head}_{only_pre_convergence}.npy"

            fp_out = f"{dir_matrices}/{model_name}/{problem_dir_final}/{problem_num}/{layer}_{head}_{only_pre_convergence}.npy"
            Path(fp_out).parent.mkdir(parents=True, exist_ok=True)
            if os.path.exists(fp_out):
                continue
            # print(f"{sentence2ranges=}")
            matrix = result["attention_weights"][layer][0, head].numpy().astype(np.float32)
            # print(f"{matrix.shape=}")
            avg_matrix = avg_matrix_by_chunk(matrix, sentence2ranges)
            np.save(fp_out, avg_matrix)
    return True


# @pkld
def get_problem_attn_avg(
    problem_num,
    layer,
    head,
    problem_dir=os.path.join("target_problems", "temperature_0.6_top_p_0.95"),
    only_pre_convergence=False,
    model_name="qwen-14b",
    quantize_8bit=True,
    quantize_4bit=False,
):
    problem_dir_final = problem_dir.split("/")[-1]
    if "target_problems" in problem_dir_final:
        problem_dir_final = problem_dir_final.split("\\")[1]  # Windows
    fp_in = f"avg_matrices/{model_name}/{problem_dir_final}/{problem_num}/{layer}_{head}_{only_pre_convergence}.npy"
    if not os.path.exists(fp_in):
        success = save_all_sentence_chunks(
            problem_num,
            problem_dir=problem_dir,
            as_dict=True,
            model_name=model_name,
            quantize_8bit=quantize_8bit,
            quantize_4bit=quantize_4bit,
            only_pre_convergence=only_pre_convergence,
        )
        if not success:
            print(f"Failed to save matrix for {problem_num} {layer} {head}")
            return None
    avg_matrix = np.load(fp_in)
    return avg_matrix


def get_problem_nums(
    problem_dir=os.path.join("target_problems", "temperature_0.6_top_p_0.95"),
    only=None,
    only_pre_convergence=False,
):
    jsons = glob.glob(os.path.join(problem_dir, "*.json"))
    problem_nums = [int(os.path.basename(json).split("_")[-1].split(".")[0]) for json in jsons]
    if only == "correct":
        problem_nums_ = []
        for pn in problem_nums:
            pn_str = str(pn)
            if len(pn_str) > 4 and pn_str[-2:] == "01":
                problem_nums_.append(pn)
        problem_nums = problem_nums_
    elif only == "incorrect":
        problem_nums_ = []
        for pn in problem_nums:
            pn_str = str(pn)
            if len(pn_str) > 4 and pn_str[-2:] == "00":
                problem_nums_.append(pn)
        problem_nums = problem_nums_
    else:
        pass

    return problem_nums


@pkld(store="both")
def get_most_sensitive_layer_heads(
    top_k=20,
    model_name="qwen-14b",
    quantize_8bit=True,
    quantize_4bit=False,
    only_pre_convergence=False,
    only=None,
    problem_dir=os.path.join("target_problems", "temperature_0.6_top_p_0.95"),
    drop_first=0,
    drop_last=0,
    proximity_ignore=20,
    pool_before=False,
    vert_score_calc=None,
    pns=None,
):
    if pns is None:
        problem_nums = get_problem_nums(
            only=only,
            problem_dir=problem_dir,
        )
    else:
        problem_nums = pns

    all_mappings = {}
    kurts = []

    if pool_before:
        all_head_vert_scores_l = []
        for pn in problem_nums:
            all_head_vert_scores = get_all_heads_vert_data(
                pn,
                model_name,
                problem_dir,
                quantize_8bit,
                quantize_4bit,
                only_pre_convergence,
                drop_first,
                drop_last,
                proximity_ignore,
                vert_score_calc,
            )
            if all_head_vert_scores is None:
                continue
            all_head_vert_scores_l.append(all_head_vert_scores)
            print(f"{all_head_vert_scores.shape=}")
        all_head_vert_scores_l = np.concatenate(all_head_vert_scores_l, axis=-1)
        kurt = get_3d_ar_kurtosis(all_head_vert_scores_l)
    else:
        for pn_i, pn in enumerate(problem_nums):
            # if pn_i < 2:
            #     continue
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
        # print(f"{kurts.shape=}")
        # quit()
        kurts[:, 0, :] = np.nan  # ignore layer 0
        kurt = np.mean(kurts, axis=0)

    layer_l = []
    kurt_l = []
    for layer in range(kurt.shape[0]):
        for j in range(kurt.shape[1]):
            if np.isnan(kurt[layer, j]):
                continue
            kurt_l.append(kurt[layer, j])
            layer_l.append(layer)
    layer_l = np.array(layer_l)
    kurt_l = np.array(kurt_l)
    plt.scatter(layer_l, kurt_l, color="dodgerblue", alpha=0.5)
    plt.xlim(0, 48)
    plt.xlabel("Layer")
    plt.ylabel("Kurtosis")
    plt.title("Kurtosis of each attention head's vertical score")
    if drop_first > 0 or drop_last > 0:
        drop_str = f"_drop_{drop_first}-{drop_last}"
    else:
        drop_str = ""
    if pool_before:
        pool_str = "_pooled"

    else:
        pool_str = ""
        plt.ylim(0, 40)

    pi_str = f"_pi{proximity_ignore}"
    fp_out = f"plots/kurt_layer_scatter_{model_name}_{only_pre_convergence}_{only}{drop_str}{pool_str}{pi_str}.png"
    Path(fp_out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fp_out, dpi=300)
    # plt.show()
    plt.close()

    # Flatten the matrix
    flat = kurt.flatten()
    plt.rcParams["font.size"] = 12
    if pool_before:
        plt.hist(flat + 3, bins=80, color="dodgerblue")
    else:
        plt.hist(flat + 3, bins=80, color="dodgerblue", range=(0, 40))
    plt.vlines(3, 0, 100, color="k", linestyle="--")
    # print(np.nanmin(flat), np.nanmax(flat))
    # quit()
    plt.xlim(0, None)
    plt.title("Histogram of each attention head's vertical score kurtosis")
    # plt.title(f"Kurtosis: {model_name=}, {only_pre_convergence=}, {only=}")
    plt.ylabel("Count")
    plt.xlabel("Kurtosis")

    fp_plot = f"plots/kurt_hist_{model_name}_{only_pre_convergence}_{only}{drop_str}{pool_str}{pi_str}.png"
    Path(fp_plot).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fp_plot, dpi=300)
    plt.close()
    # plt.show()

    # Mask out NaNs
    valid_indices = np.where(~np.isnan(flat))[0]  # indices where it's not NaN
    valid_values = flat[valid_indices]

    # Find the indices of the top 20 valid values
    top_k = min(top_k, len(valid_values))  # in case fewer than 20 valid values
    top_indices_in_valid = np.argpartition(valid_values, -top_k)[-top_k:]
    # print(f"{top_indices_in_valid=}")
    # quit()

    # Sort them by value (optional)
    top_indices_in_valid = top_indices_in_valid[np.argsort(-valid_values[top_indices_in_valid])]
    # print(f"{top_indices_in_valid=}")
    # quit()

    # Map back to original flat indices
    top_flat_indices = valid_indices[top_indices_in_valid]

    # Convert flat indices back to (row, col) coordinates
    coords = np.array(np.unravel_index(top_flat_indices, kurt.shape)).T
    coords = coords.astype(int)
    print(f"{coords=}")
    # quit()
    print(f"Got most sensitive heads: {coords.shape}")
    # print(f"{coords=}")
    return coords


@pkld
def get_vert_scores_for_pn(
    pn,
    coords,
    model_name="qwen-15b",
    proximity_ignore=20,
    quantize_8bit=True,
    quantize_4bit=False,
    only_pre_convergence=False,
    problem_dir=os.path.join("target_problems", "temperature_0.6_top_p_0.95"),
    control_depth=False,
    drop_first=0,
):
    all_mappings = {}
    for layer, head in tqdm(coords, desc=f"Computing vert scores ({pn=})"):
        # print(f"{layer=}, {head=}")
        # print(f'{model_name=}')
        # quit()
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
        if avg_matrix is None:
            break
        if quantize_4bit == "diag":
            vert_scores = get_attn_direction_scores(
                avg_matrix,
                distance=proximity_ignore,
            )
        else:
            vert_scores = get_attn_vert_scores(
                avg_matrix,
                proximity_ignore=proximity_ignore,
                control_depth=control_depth,
                drop_first=drop_first,
            )
        all_mappings[(layer, head, pn)] = vert_scores
    return all_mappings


# @pkld
def get_vert_scores_for_heads(
    coords,
    problem_nums,
    proximity_ignore=20,
    model_name="qwen-15b",
    quantize_8bit=True,
    quantize_4bit=False,
    only_pre_convergence=False,
    control_depth=False,
    drop_first=0,
    problem_dir=os.path.join("target_problems", "temperature_0.6_top_p_0.95_qwen"),
):
    # print(f"Get vert scores: {len(problem_nums)=}")
    # quit()
    all_mappings = {}
    for pn in tqdm(problem_nums, desc="Computing vert scores"):
        # print(f'{model_name=}')
        # quit()
        pn_mappings = get_vert_scores_for_pn(
            pn,
            coords,
            model_name,
            proximity_ignore,
            quantize_8bit=quantize_8bit,
            quantize_4bit=quantize_4bit,
            only_pre_convergence=only_pre_convergence,
            control_depth=control_depth,
            drop_first=drop_first,
            problem_dir=problem_dir,
        )
        for (layer, head, pn), vert_scores in pn_mappings.items():
            all_mappings[(layer, head, pn)] = vert_scores
    return all_mappings


# @pkld
def get_pn_df_info(pn, problem_dir, only_pre_convergence, vert_score, model_name):
    print(f"Grabbing json for: {pn}")
    problem = load_problem_json(pn, problem_dir)
    df_as_l = defaultdict(list)
    qwen_llama_str = "_qwen" if "qwen" in model_name else "_llama"

    sentence2ranges, problem = get_full_CoT_token_ranges(
        pn,
        problem_dir=os.path.join("target_problems", f"temperature_0.6_top_p_0.95{qwen_llama_str}"),
        verbose=False,
        only_pre_convergence=False,
        model_name=model_name,
    )

    if only_pre_convergence == "semi":
        problem = load_problem_json(pn, problem_dir)

        sentences = [s["sentence"] for s in problem["sentences"]]
        solution = problem["base_solution"]["solution"]
        char_chunk_ranges = get_chunk_ranges(solution, sentences)
        token_chunk_ranges = get_chunk_token_ranges(
            solution,
            char_chunk_ranges,
            get_tokenizer(model_name),
        )

        full_text = problem["base_solution"]["full_cot"]
        full_text_tokens = get_raw_tokens(full_text, model_name)

    if only_pre_convergence == "semi" and len(full_text_tokens) > 4000:
        convergence_point = None
        for i, chunk_range in enumerate(token_chunk_ranges):
            if chunk_range[1] > (4000 - 256):  # buffer for the +256 added to the end
                convergence_point = i - 1
                break

        if convergence_point is None:
            if token_chunk_ranges[-1][1] < (4000 - 256):
                convergence_point = len(token_chunk_ranges) - 1

        if convergence_point is not None:
            problem["sentences"] = problem["sentences"][: convergence_point + 1]
            # print(f"Limiting to first 4000 tokens for {pn=}")
            chunks = problem["sentences"][: convergence_point + 1]
        else:
            chunks = problem["sentences"]
    elif only_pre_convergence:
        convergence_point = None
        for i, chunk in enumerate(problem["sentences"]):
            if chunk["is_convergence"]:
                convergence_point = i
                break
        if convergence_point is None:
            convergence_point = len(problem["sentences"]) - 1
        assert convergence_point is not None, f"No convergence point found for {pn}"
        chunks = problem["sentences"][: convergence_point + 1]
    else:
        chunks = problem["sentences"]
    if vert_score is not None:
        assert len(chunks) == len(
            vert_score
        ), f"Misaligned ({pn=}): {len(chunks)=}, {len(vert_score)=}"
        print("Matching len(chunks) == len(vert_score)")
    keep_keys = [
        "sentence_idx",
        "tag",
        # "kl_divergence",
        "jaccard_entropy",
        "math_density",
        "sentence_length",
        "is_convergence",
        "accuracy_gap",
        "next_unique_answers",
    ]

    if str(pn)[-2:] == "01":
        correct_q = 1
    else:
        correct_q = 0

    keep_keys = list(chunks[0].keys())
    delta_length = None
    for i, chunk in enumerate(chunks):
        if i not in sentence2ranges:
            # print(f'skip? A ({i})')
            assert "llama" in model_name
            continue
        chunk_range = sentence2ranges[i]
        if chunk_range[1] - chunk_range[0] <= 1:
            # print(f'skip? B ({i})')
            assert "llama" in model_name
            continue

        chunk_tokens = full_text_tokens[chunk_range[0] : chunk_range[1]]
        # print(f'{chunk_tokens=}')

        # print(f'{i} {convergence_point=} | {len(token_chunk_ranges)=} | {chunk["sentence"]=}')
        single_linebreak = False
        double_linebreak = False
        for token in chunk_tokens:
            if "ĊĊ" in token:
                double_linebreak = True
            elif "Ċ" in token:
                single_linebreak = True
        # print(f'{chunk["sentence"]=}')
        # print(f"{chunk_tokens=}")
        # print()
        # print(f'{chunk_range=}')
        clean_text = tokens_to_clean(chunk_tokens, model_name)
        df_as_l["text_targeted"].append(clean_text)
        df_as_l["tokens_targeted"].append(chunk_tokens)
        df_as_l["single_linebreak"].append(single_linebreak)
        df_as_l["double_linebreak"].append(double_linebreak)
        # removed_text = chunk["sentence"]

        df_as_l["correct_q"].append(correct_q)
        if i < len(chunks) - 1:
            delta_length = chunks[i + 1]["length"] - chunk["length"]
            df_as_l["delta_output_length"].append(delta_length)
        else:
            df_as_l["delta_output_length"].append(np.nan)
        for key in keep_keys:
            if key == "problem":
                continue
            if key == "num_resamples":
                continue
            if "has_" in key:
                continue
            df_as_l[key].append(chunk[key])
    # quit()
    # df_as_l["delta_length"] = delta_length
    df_as_l["mean_output_length"] = df_as_l["length"]
    del df_as_l["length"]
    df_as_l["pn"] = [str(pn)[:-2]] * len(df_as_l["text_targeted"])
    df_as_l["pn_ci"] = [str(pn)] * len(df_as_l["text_targeted"])
    return df_as_l


def analyze_top_heads(
    top_k=20,
    proximity_ignore=20,
    problem_dir=os.path.join("target_problems", "temperature_0.6_top_p_0.95"),
    model_name="qwen-14b",
    quantize_8bit=True,
    quantize_4bit=False,
    only_pre_convergence=False,
    only=None,
    drop_first=0,
    drop_last=0,
):
    coords = get_most_sensitive_layer_heads(
        top_k,
        model_name=model_name,
        quantize_8bit=quantize_8bit,
        quantize_4bit=False,
        only_pre_convergence=only_pre_convergence,
        only=only,
        problem_dir=problem_dir,
        drop_first=drop_first,
        drop_last=drop_last,
        proximity_ignore=proximity_ignore,
    )
    return
    # print(f"{coords=}")
    # quit()
    problem_nums = get_problem_nums(
        only=only, only_pre_convergence=only_pre_convergence, problem_dir=problem_dir
    )
    if quantize_4bit == "diag":
        do_diag = True
        proximity_ignore = proximity_ignore // 2
    else:
        do_diag = False
    all_mappings = get_vert_scores_for_heads(
        coords,
        problem_nums,
        proximity_ignore,
        model_name=model_name,
        quantize_8bit=quantize_8bit,
        quantize_4bit=quantize_4bit,
        only_pre_convergence=only_pre_convergence,
        problem_dir=problem_dir,
    )

    pn2vert_scores_l = defaultdict(list)
    for (layer, head, pn), vert_scores in all_mappings.items():
        pn2vert_scores_l[pn].append(vert_scores)

    pn2vert_score = {}
    for pn, vert_scores in pn2vert_scores_l.items():
        vert_scores_ar = np.array(vert_scores)
        print(vert_scores_ar.shape)
        if vert_scores_ar.shape[1] < 20:
            continue
        vert_scores_M = np.mean(vert_scores_ar, axis=0)
        print(f"{vert_scores_M=}")
        if do_diag:
            pass
        else:
            assert (
                np.sum(np.isnan(vert_scores_M[:-proximity_ignore])) == 0
            ), f"{vert_scores_M[:-proximity_ignore]=}"
        pn2vert_score[pn] = vert_scores_M
    print(f"Problems with vert scores: {len(pn2vert_score)}")

    # ['temperature', 'top_p', 'problem', 'sentence_idx', 'sentence', 'tag', 'accuracy',
    # 'length', 'entropy', 'kl_divergence', 'accuracy_gap', 'is_convergence',
    # 'normalized_position', 'jaccard_similarity', 'math_density', 'sentence_length',
    #  'token_count', 'acc_acceleration', 'next_unique_answers', 'jaccard_entropy',
    # 'has_wait', 'has_but', 'has_given', 'has_might', 'has_i', 'has_them', 'has_not',
    # 'has_alternatively', 'has_perhaps', 'has_mistake', 'has_break']

    dfs_l = []
    for pn, vert_score in pn2vert_score.items():
        df_as_l = get_pn_df_info(pn, problem_dir, only_pre_convergence, vert_score)
        df_as_l["vert_score"] = vert_score

        # print(f"{pn=}")
        df = pd.DataFrame(df_as_l)
        convergence_point = df["is_convergence"].idxmax()
        print(f"{pn} | {convergence_point=}")
        df["pn"] = str(pn)
        dfs_l.append(df)
    df = pd.concat(dfs_l)

    # # Convert categorical variables to dummy variables if needed
    # if "tag" in df.columns:
    #     df = pd.get_dummies(df, columns=["tag"], drop_first=True)

    # Create formula string with all variables as predictors of vert_score
    predictors = [col for col in df.columns if col != "vert_score" and col != "is_convergence"]
    formula = "vert_score ~ " + " + ".join(predictors)

    for predictor in predictors:
        r, p = stats.spearmanr(df[predictor], df["vert_score"], nan_policy="omit")
        if p < 0.05:
            print(f"{predictor=} | {r=:.3f} | {p=:.4f}")
    # quit()

    # formula = "vert_score ~ pn + tag + sentence_idx + "
    # formula = "vert_score ~ pn + correct_q + tag + kl_no_smooth + sentence_idx + normalized_position + sentence_token_len"

    # Fit linear regression model
    model = smf.ols(formula=formula, data=df).fit()

    # Print summary
    print(model.summary())

    return pn2vert_score


if __name__ == "__main__":

    problem_dir = os.path.join("target_problems", "temperature_0.6_top_p_0.95_llama")
    only_pre_convergence = "semi"
    drop_first = 4
    drop_last = 32
    proximity_ignore = 4
    top_k = 200

    # Default to 8-bit quantization
    analyze_top_heads(
        model_name="llama8-base",
        quantize_8bit=False,
        quantize_4bit=False,
        only_pre_convergence=only_pre_convergence,
        only=None,
        top_k=top_k,
        problem_dir=problem_dir,
        drop_first=drop_first,
        drop_last=drop_last,
        proximity_ignore=proximity_ignore,
    )

    # TODO: Look into how the tokenization of llama8 may differ from llama8-base?

    # analyze_top_heads(
    #     model_name="qwen-14b",
    #     quantize_8bit=False,
    #     quantize_4bit=False,
    #     only_pre_convergence=only_pre_convergence,
    #     only=None,
    #     top_k=top_k,
    #     problem_dir=problem_dir,
    #     drop_first=drop_first,
    #     drop_last=drop_last,
    #     proximity_ignore=proximity_ignore,
    # )
