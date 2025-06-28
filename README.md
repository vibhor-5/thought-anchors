# Thought Anchors ⚓

We introduce a framework for interpreting the reasoning of large language models by attributing importance to individual sentences in their chain-of-thought. Using black-box, attention-based, and causal methods, we identify key reasoning steps, which we call **thought anchors**, that disproportionately influence downstream reasoning. These anchors are typically planning or backtracking sentences. Our work offers new tools and insights for understanding multi-step reasoning in language models.

See more:
* Paper: https://arxiv.org/abs/2506.19143
* Interactive demo: https://www.thought-anchors.com/
* Repository for the interactive demo: https://github.com/interp-reasoning/thought-anchors.com
* Dataset here: https://huggingface.co/datasets/uzaymacar/math-rollouts

## Get Started

Here's a quick rundown of the main scripts in this repository and what they do:

1. `generate_rollouts.py`: This is the main script for generating reasoning rollouts. Our [dataset](https://huggingface.co/datasets/uzaymacar/math-rollouts) was created with this.
2. `analyze_rollouts.py`: This script processes the generated rollouts and adds `chunks_labeled.json` and other metadata for each reasoning trace. It calculates metrics like **forced answer importance**, **resampling importance**, and **counterfactual importance**.
3. `step_attribution.py`: This script computes the sentence-to-sentence counterfactual importance score for all sentences in all reasoning traces.
4. `plots.py`: This script generates figures (e.g., the ones you see in the paper).

Here is what other files do:
* `selected_problems.json`: This is a list of problems identified in the 25% - 75% accuracy range (i.e., challenging problems). It is sorted in increasing order by average length of sentences (NOTE: We use *chunks*, *steps*, and *sentences* interchangeably through the code).
* `prompts.py`: This includes auto-labeler LLM prompts we used throughout this project. `DAG_PROMPT` is the one we used to generate labels (i.e., function tags or categories, e.g., uncertainty management) for each sentence.
* `utils.py`: Includes utility and helper functions for reasoning trace analysis.
* `misc-experiments/`: This folder includes miscellaneous experiment scripts. Some of them are ongoing work. 

## Contact

For any questions, thoughts, or feedback, please reach out to [uzaymacar@gmail.com](mailto:uzaymacar@gmail.com) and [paulcbogdan@gmail.com](mailto:paulcbogdan@gmail.com).

