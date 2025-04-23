og_categories = """
1. Initializing: Rephrasing the given task and stating initial thoughts at the beginning of reasoning.
2. Deduction: Deriving conclusions based on current approach and assumptions.
3. Adding Knowledge: Incorporating external knowledge to refine reasoning.
4. Example Testing: Generating examples or scenarios to test hypotheses.
5. Uncertainty Estimation: Explicitly stating confidence or uncertainty regarding reasoning.
6. Backtracking: Abandoning the current approach and exploring alternative strategies.
7. Unknown: The chunk does not fit into any of the above categories.
"""

et_categories = """
1. Initializing: Rephrasing the given task and stating initial thoughts at the beginning of reasoning.
2. Deduction: Deriving conclusions based on current approach and assumptions.
3. Adding Knowledge: Incorporating external knowledge to refine reasoning.
4. Example Testing: Generating examples or scenarios to test hypotheses.
5. Uncertainty Estimation: Explicitly stating confidence or uncertainty regarding reasoning.
6. Backtracking: Abandoning the current approach and exploring alternative strategies.
7. Comparative Analysis: Comparing multiple approaches or options side by side.
8. Question Posing: Asking self-directed questions to guide reasoning.
9. Summary: Summarizing or recapping thinking, often near the end of reasoning.
10. Metaphorical Thinking: Using metaphors or analogies to understand the problem.
11. Final Answer: Providing the final answer to the problem.
12. Unknown: The chunk does not fit into any of the above categories.

If you're unsure, then it's better to assign it to "Unknown".
"""

etm_categories = """
1. Prompt: The initial prompt that the model received. It is up until the ocurrence of the first <think> token. Everything after should be one of the below categories.
2. Initialization: Rephrasing the given task and stating initial thoughts at the beginning of reasoning.
3. Planning and Strategy: Laying out an approach or plan before solving.
4. Recall: Recalling math facts, definitions, or formulas from memory.
5. Substitution: Rewriting or re-encoding expressions (e.g., defining variables, changing forms).
6. Intermediate Step: Performing intermediate calculations or algebra.
7. Verification: Re-checking previous steps for correctness or coherence.
8. Alternative Strategy and Backtracking: Proposing a second method, reframing the problem, or abandoning the current approach.
9. Summary and Preparation: Summarizing reasoning just before giving the final answer.
10. Final Answer: Stating or formatting the final boxed answer. If "Final Answer" is assigned to a chunk, it should be the first category in the list.
11. Unknown: The chunk does not fit into any of the above categories.
"""

pm_categories = """
1. Prompt: The initial prompt that the model received. It is up until the ocurrence of the first <think> token. Everything after should be one of the below categories.
2. Initialization: Rephrasing the given task and stating initial thoughts at the beginning of reasoning.
3. Planning and Strategy: Laying out an approach or plan before solving.
4. Recall: Recalling math facts, definitions, or formulas from memory.
6. Intermediate Step: Performing intermediate calculations or algebra.
7. Final Answer: Stating or formatting the final boxed answer. If "Final Answer" is assigned to a chunk, it should be the first category in the list.
8. Unknown: The chunk does not fit into any of the above categories.

If you're unsure, then it's better to assign it to "Unknown".
"""

sm_categories = """
1. Prompt: The initial prompt that the model received. It is up until the ocurrence of the first <think> token. Everything after should be one of the below categories.
2. Intermediate Step: Performing intermediate calculations, i.e., intermediate steps, that are later used to derive the final answer.
3. Final Answer: Stating or formatting the final boxed answer. If "Final Answer" is assigned to a chunk, it should be the first category in the list.
4. Unknown: The chunk does not fit into any of the above categories.

We call "Intermediate Step" chunks with calculations or computations that are later directly used to derive the final answer. 
If a chunk doesn't contain any calculations or computations, then it's not an "Intermediate Step".
You need to assess if "Intermediate Step" chunks are causally important for the final answer or not.
If you're unsure, then it's better to assign it to "Unknown".
"""

ss_categories = """
1. Prompt:
   This is the model’s initial prompt (e.g., task description, problem statement) before the reasoning begins.
   This chunk **must always be labeled Prompt if it appears**, and typically ends at the first <think> token.

2. Initialization:
   Rephrasing the problem, unpacking definitions, or stating general intentions at the start of reasoning.
   No equations or plan details yet. Just framing and initial engagement.

3. Planning and Strategy:
   Laying out a specific plan, method, or approach the model intends to follow.
   Often includes conditionals (“I’ll try X first...”) or descriptions of potential steps.

4. Recall or Memory or Fact Retrieval:
   Recalling known math facts, identities, theorems, formulas, definitions, or number facts (e.g., “3^5 = 243”).
   Includes explicit retrieval from memory, not derivation.

5. Substitution or Rewriting:
   Rewriting or representing an expression in a more useful form (e.g., “243 = 3^5” or “Let x = ...”).
   Includes algebraic rearrangement, defining variables, or transforming the expression’s structure.

6. Intermediate Step:
   Performing actual computations, algebraic manipulations, simplifications, or solving sub-parts of the problem.
   These steps **directly contribute to the final answer** (e.g., computing 6000 / 750 = 8).

7. Verification or Consistency Check:
   Reviewing or re-computing a previous result to confirm correctness.
   Often introduced with phrases like “let me double check” or “just to be sure…”

8. Alternative Strategy or Backtracking:
   Proposing a new approach, re-analyzing earlier assumptions, or expressing doubt in the current direction.
   Includes backtracking, starting over, or switching methods (“Alternatively, I could...”)

9. Comparison or Meta-Analysis:
   Comparing multiple solution paths, strategies, or results side-by-side.
   Includes explicit assessment of which approach is more efficient, elegant, or robust.

10. Uncertainty or Confidence Estimation:
    Explicitly expressing confidence or uncertainty in the reasoning or result.
    E.g., “I’m not sure if this is right...” or “I feel confident in this answer.”

11. Final Answer Preparation or Summary:
    A short chunk that leads directly into the answer — summarizing key steps, restating conclusions, or rephrasing in final form before answer emission.

12. Final Answer:
    The explicit answer — boxed result, final numerical or symbolic output.
    If present, this chunk must include the boxed or final answer format.

13. Metaphorical or Conceptual Thinking:
    Using analogy, metaphor, or conceptual reframing to understand the problem (e.g., “Like in a handshake problem…”)
    Rare but important for open-ended or transfer tasks.

14. Question Posing or Self-Directed Inquiry:
    The model asks itself questions to guide reasoning (“What does the exponent mean?”, “Is this number a square?”).
    Different from Planning — it’s introspective rather than directive.

15. Unknown:
    The chunk does not fit into any of the above categories.
    Use this when the chunk is ambiguous, off-topic, irrelevant, or purely stylistic.
    
If you're unsure, then it's better to assign it to "Unknown".
"""

CATEGORIES = {
    "og": og_categories,
    "et": et_categories,
    "etm": etm_categories,
    "pm": pm_categories,
    "sm": sm_categories,
    "ss": ss_categories
}