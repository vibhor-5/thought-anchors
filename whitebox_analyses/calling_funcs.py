from run_target_problems import (
    get_most_sensitive_layer_heads,
    get_problem_nums,
    get_vert_scores_for_heads,
)


import numpy as np
from pkld import pkld


import os


@pkld(store="both")
def get_vert_scores_for_heads_cache(
    coords,
    problem_nums,
    proximity_ignore,
    model_name="qwen-14b",
    quantize_8bit=False,
    quantize_4bit=False,
    only_pre_convergence=False,
    problem_dir=os.path.join("target_problems", "temperature_0.6_top_p_0.95"),
):
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
    return all_mappings


@pkld(store="both")
def get_weights_scores(
    pn,
    top_k=200,
    model_name="qwen-14b",
    quantize_8bit=False,
    quantize_4bit=False,
    only_pre_convergence=False,
    only=None,
    proximity_ignore=20,
    problem_dir=os.path.join("target_problems", "temperature_0.6_top_p_0.95"),
    drop_first=10,
    drop_last=30,
):
    coords = get_most_sensitive_layer_heads(
        top_k,
        model_name=model_name,
        quantize_8bit=quantize_8bit,
        quantize_4bit=False,
        only_pre_convergence="semi",
        only=only,
        problem_dir=problem_dir,
        drop_first=drop_first,
        drop_last=drop_last,
    )
    problem_nums = get_problem_nums(only=only, only_pre_convergence=only_pre_convergence)

    # print(f'{model_name=}')
    # quit()

    coords = list(coords)
    all_mappings = get_vert_scores_for_heads_cache(
        coords,
        problem_nums,
        proximity_ignore,
        model_name=model_name,
        quantize_8bit=quantize_8bit,
        quantize_4bit=quantize_4bit,
        only_pre_convergence=only_pre_convergence,
        problem_dir=problem_dir,
    )
    vert_scores_l = []
    for layer, head in coords:
        key = (layer, head, pn)
        vert_scores_l.append(all_mappings[key])
    vert_scores_M = np.mean(vert_scores_l, axis=0)
    return vert_scores_M


if __name__ == "__main__":
    coords = [[21, 10]]
    problem_nums = [159100]
    proximity_ignore = 4
    model_name = "llama8"
    quantize_8bit = False
    quantize_4bit = False
    only_pre_convergence = "semi"
    problem_dir = os.path.join("target_problems", "temperature_0.6_top_p_0.95_llama")

    all_mappings = get_vert_scores_for_heads_cache(
        coords,
        problem_nums,
        proximity_ignore,
        model_name=model_name,
        quantize_8bit=quantize_8bit,
        quantize_4bit=quantize_4bit,
        only_pre_convergence=only_pre_convergence,
        problem_dir=problem_dir,
    )
