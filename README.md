# Mechanistic Interpretability for Reasoning Models

This repository contains several basic analyses and mech-interp experiments on reasoning / thinking models.

We tried to build on top of [Understanding Reasoning in Thinking Language Models via Steering Vectors](https://openreview.net/pdf?id=OwhVWNOBcz).

Most of the experiments are:
1. Performed on [DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B)
2. Using a synthetic reasoning dataset curated on the [following topics and subtopics](./topics_subtopics.json) which can be founder under [`cots/`](./cots/)
3. Using auto-labeled sentences as the atomic unit of analysis, see example under [`analysis/problem_1/seed_0/chunks.json`](./analysis/problem_1/seed_0/chunks.json)
4. Performed on the residual stream activations of the last layer

## Taxonomy for Reasoning Steps in Chain-of-Thought (CoT)

![Distribution of Reasoning Categories](./final_figures/distribution_of_reasoning_categories.png)

## Transition Probabilities between Reasoning Categories

![Transition Probabilities](./final_figures/transition_probabilities.png)

## Distribution of Relative Category Positions

[./final_figures/distribution_of_category_positions.png]

## Average Correlation and Attention

[./final_figures/average_correlation_and_attention.png]

## Clustering Sentences by Categories

[./final_figures/clustering.png]

[./final_figures/clustering_with_categories.png]

## Uncertainty Estimation Analysis

[./final_figures/aggregated_uncertainty_entropy_by_category.png]

[./final_figures/aggregated_uncertainty_entropy.png]