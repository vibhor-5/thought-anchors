import os
import json
import re
from pkld import pkld
import matplotlib.pyplot as plt
import re


def extract_solution_text(full_solution):
    # Extract text between </think> and DONE
    match = re.search(r"</think>(.*?)DONE", full_solution, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


# @pkld
def load_all_responses():
    problems = os.listdir("cots")
    solutions = []
    for problem in problems:
        problem_path = f"cots/{problem}/solutions.json"
        if os.path.exists(problem_path):
            with open(problem_path, "r") as f:
                problem_data = json.load(f)

                # If the data is a list of entries
                # if isinstance(problem_data, list):
                # print(f"Problem {problem}:")
                for entry in problem_data:
                    solution_text = entry["solution"]
                    assert "<think>" in solution_text
                    solution_text = solution_text.split("<think>")[1]
                    if "</think>" in solution_text:
                        solution_text = solution_text.split("</think>")[0]
                    if solution_text:
                        solutions.append(solution_text)
    return solutions


def get_word_idxs(solution, target_words):
    solution_split = solution.split(" ")
    # solution_split = re.split(r"\. |.\n", solution)
    word2idxs = {word: [] for word in target_words}
    num_words = len(solution_split)

    for i, word in enumerate(solution_split):
        for target_word in target_words:
            if target_word.lower() in word.lower():
                word2idxs[target_word].append(i)
    return word2idxs, num_words


if __name__ == "__main__":
    target_words = [
        "wait",
        # "however",
        # "contradict",
        # "alternatively",
        # "mistake",
    ]  # "think", "therefore", "answer", "yes", "no"]
    solutions = load_all_responses()
    word2idxs_all = {word: [] for word in target_words}
    word2idxs_weights = {word: [] for word in target_words}
    highest = 0
    for solution in solutions:
        word_idxs, num_words = get_word_idxs(solution, target_words)
        # if num_words < 1500:
        #     continue
        # if num_words > 2000:
        # continue
        for word in target_words:
            word2idxs_all[word].extend(word_idxs[word])
            word2idxs_weights[word].extend([num_words] * len(word_idxs[word]))
            if len(word_idxs[word]) and max(word_idxs[word]) > highest:
                highest = max(word_idxs[word])

    # Create stacked histogram
    # data = [word2idxs_all[word] for word in target_words]
    # labels = target_words
    # plt.hist(data, bins=100, histtype="bar", label=labels, stacked=True)
    # plt.title("Distribution of keywords in solutions")
    # plt.xlabel("Index")
    # plt.ylabel("Frequency")
    # plt.legend()
    # plt.show()

    for word, idxs in word2idxs_all.items():
        weights = word2idxs_weights[word]
        plt.hist(
            idxs,
            bins=range(0, highest, 25),
            # bins=100,
            # weights=weights,
            histtype="bar",
            # stacked=True,
        )
    plt.title(target_words)
    plt.xlabel("Index")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

    # Create stacked histogram
    data = [word2idxs_all[word] for word in target_words]
    weights = [word2idxs_weights[word] for word in target_words]
    labels = target_words
    plt.hist(
        data, bins=100, histtype="bar", label=labels, stacked=True  # weights=weights,
    )
    plt.title("Distribution of keywords in solutions (weighted)")
    plt.xlabel("Sentence Index")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()
