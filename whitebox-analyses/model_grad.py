import os
import random
import sys
import torch
import time
import warnings
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from functools import cache, partial
from pkld import pkld
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
from numpy.linalg import norm

# Assume get_deepseek_r1 is available (copied or imported from model.py)
# If not, copy the function here. For brevity, I'll assume it exists.
from model import get_deepseek_r1


# --- Hook Function ---
def save_sliced_grad_hook(grad_dict, token_pos, layer_idx, tensor_name, grad):
    """Saves the relevant slice of the gradient flowing back into an attention tensor."""
    if grad is not None:
        # Captures grads for Q_t -> K_{<=t} for all heads
        grad_slice = grad[0, :, token_pos, : token_pos + 1].clone().detach().cpu()
        grad_dict[token_pos][layer_idx] = grad_slice


# --- Activation Hook Function ---
def save_activation_hook(activation_dict, token_pos, layer_idx, module, input):
    """Saves the input activation (residual stream) for a layer at a specific token position."""
    if input and isinstance(input, tuple) and len(input) > 0:
        # If processing multiple tokens (e.g., initial prompt pass), save activation for the specific token_pos
        # If processing one token at a time (gradient loop), token_pos matches the last element.
        activation = (
            input[0][:, token_pos, :].clone().detach().cpu()
        )  # Activation FOR token_pos
        if layer_idx not in activation_dict:
            activation_dict[layer_idx] = {}
        activation_dict[layer_idx][token_pos] = activation  # Store by layer then token


def generate_response_with_grads(
    model,
    tokenizer,
    prompt,
    max_new_tokens=20,
    temperature=0.7,
    verbose=False,
    target_layers=None,  # For gradients
    activation_layers=None,  # For activations & attention weights
    make_square=False,
):
    """
    Generates response, computes sliced attention grads, collects activations & full attn weights.
    Outputs grads in {layer: (head, query, key)} format. Includes prompt activations.
    """
    if model.dtype != torch.float32:
        warnings.warn("Model not in float32.", stacklevel=2)

    # --- Step 1: Tokenize & Get Config ---
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids
    prompt_len = input_ids.shape[1]
    cot_prompt_tokens = input_ids[0].tolist()
    cot_prompt_token_texts = tokenizer.convert_ids_to_tokens(cot_prompt_tokens)
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    hidden_dim = model.config.hidden_size

    if verbose:
        print(f"Prompt: '{prompt}'\nEncoded to {prompt_len} tokens")
    if verbose:
        print(
            f"Model layers: {num_layers}, Heads: {num_heads}, HiddenDim: {hidden_dim}"
        )

    target_grad_layers = (
        set(target_layers) if target_layers is not None else set(range(num_layers))
    )
    target_act_layers = (
        set(activation_layers)
        if activation_layers is not None
        else set(range(num_layers))
    )

    # --- Step 2: Initial Pass for Prompt Activations ---
    prompt_activations = defaultdict(dict)  # {layer_idx: {token_pos: tensor}}
    layer_modules_for_act_hooks = {}
    if target_act_layers:
        # Find modules ONCE
        for name, module in model.named_modules():
            if ".layers." in name:
                parts = name.split(".layers.")
                if len(parts) > 1 and parts[1].split(".")[0].isdigit():
                    layer_idx = int(parts[1].split(".")[0])
                    if layer_idx in target_act_layers and name.endswith(
                        f"layers.{layer_idx}"
                    ):
                        layer_modules_for_act_hooks[layer_idx] = module
        # Register hooks for prompt pass
        prompt_act_hooks = []
        for layer_idx, module in layer_modules_for_act_hooks.items():
            # We need activations for each prompt token
            for token_pos in range(prompt_len):
                hook_fn_act = partial(
                    save_activation_hook, prompt_activations, token_pos, layer_idx
                )
                h_act = module.register_forward_pre_hook(hook_fn_act)
                prompt_act_hooks.append(h_act)

    if verbose:
        print(
            f"Running initial pass for prompt activations (Layers: {target_act_layers})..."
        )
    with torch.no_grad():
        try:
            _ = model(input_ids=input_ids, attention_mask=inputs.attention_mask)
        finally:
            # Ensure hooks are removed even if error occurs
            for h in prompt_act_hooks:
                h.remove()
    if verbose:
        print("Prompt activation pass complete.")

    # --- Step 3: Generate Full Sequence ---
    if verbose:
        print(f"Generating response (max_new_tokens={max_new_tokens})...")
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
        )
    seq_len = output_ids.shape[1]
    response_start_idx = prompt_len
    response_len = seq_len - response_start_idx
    if verbose:
        print(f"Generated sequence length: {seq_len} tokens (Response: {response_len})")

    # --- Step 4: Get Full Attention Weights ---
    full_attn_weights = {}
    full_attention_mask = torch.ones_like(output_ids)  # Mask for full sequence
    if verbose:
        print(
            f"Running pass for full attention weights (Layers: {target_act_layers})..."
        )
    with torch.no_grad():
        full_outputs = model(
            input_ids=output_ids,  # Use full sequence
            attention_mask=full_attention_mask,
            output_attentions=True,
            return_dict=True,
        )
        if full_outputs.attentions:
            for layer_idx, attn in enumerate(full_outputs.attentions):
                # Use target_act_layers to decide which weights to save
                if layer_idx in target_act_layers:
                    if attn is not None:
                        full_attn_weights[layer_idx] = attn.detach().cpu()
        del full_outputs  # Free memory
    if verbose:
        print("Full attention weight pass complete.")

    # --- Step 5: Prep for Gradient & Response Activation Calculation ---
    model.zero_grad()
    captured_gradients_sliced = defaultdict(lambda: defaultdict(lambda: None))
    # This dict will now ONLY store response activations
    response_activations = defaultdict(dict)  # {layer_idx: {token_pos: tensor}}

    # --- Decode response ---
    response = tokenizer.decode(
        output_ids[0][response_start_idx:], skip_special_tokens=True
    )
    print(f"Generated response: {response}")

    # Enable gradient checkpointing to save memory during backward passes
    gradient_checkpointing_enabled = False
    if hasattr(model, "gradient_checkpointing_enable"):
        print("Enabling gradient checkpointing...")
        model.gradient_checkpointing_enable()
        # Some models might switch to train() mode internally when checkpointing is enabled
        # Ensure we stay in eval mode for consistent behavior during generation logic parts
        model.eval()
        gradient_checkpointing_enabled = True
    else:
        warnings.warn("Model does not support gradient_checkpointing_enable(). Memory usage might be high.", stacklevel=2)

    # --- Step 6: Iterate, compute gradients, collect response activations ---
    print(
        f"Starting gradient & response activation collection ({response_len} tokens)..."
    )
    try: # Use try...finally to ensure checkpointing is disabled
        for t in tqdm(range(response_start_idx, seq_len - 1), desc="Collecting Grads/Acts"):
            model.zero_grad()
            grad_hooks = []
            act_hooks = []  # noqa
            current_input_ids = output_ids[:, : t + 1]
            current_mask = full_attention_mask[:, : t + 1]

            # Forward pass needs gradients enabled for hooks AND backward pass
            with torch.enable_grad():
                # Register Response Activation Hooks BEFORE forward pass
                for layer_idx, module in layer_modules_for_act_hooks.items():
                    hook_fn_act = partial(
                        save_activation_hook, response_activations, t, layer_idx
                    )
                    h_act = module.register_forward_pre_hook(hook_fn_act)
                    act_hooks.append(h_act)

                # --- Forward pass to get attentions and logits ---
                # Important: Ensure model is in eval mode even with checkpointing if dropout/batchnorm matters
                # model.eval() # Already set after enabling checkpointing
                outputs_grad = model(
                    input_ids=current_input_ids,
                    attention_mask=current_mask,
                    output_attentions=True, # Request attentions
                    return_dict=True,
                    use_cache=False # Important when using gradient checkpointing
                )
                # Remove activation hooks immediately after forward pass
                for h_act in act_hooks:
                    h_act.remove()
                act_hooks = [] # Clear hooks list

                # Check if attentions were returned (might not be if model config changed)
                if not outputs_grad.attentions:
                    warnings.warn(f"No attentions found at token {t}, skipping gradient calculation.", stacklevel=2)
                    continue

                # Register Gradient Hooks AFTER forward pass, before backward
                current_attentions = outputs_grad.attentions
                for layer_idx, layer_attn in enumerate(current_attentions):
                    if layer_idx not in target_grad_layers:
                        continue
                    if layer_attn is not None and layer_attn.requires_grad: # Check if grad is required
                         hook_fn_grad = partial(
                            save_sliced_grad_hook,
                            captured_gradients_sliced,
                            t, # Use absolute position t
                            layer_idx,
                            f"attn_layer_{layer_idx}",
                        )
                         h_grad = layer_attn.register_hook(hook_fn_grad)
                         grad_hooks.append(h_grad)
                    # else:
                    #     warnings.warn(f"Layer {layer_idx} attention tensor at token {t} does not require grad.", stacklevel=2)


                # # --- Backward pass ---
                # target_logits = outputs_grad.logits[0, -1, :] # Logits for the last token t
                # scalar_target = target_logits.sum() # Example target scalar

                # loss_fn = torch.nn.CrossEntropyLoss()
                # target = torch.tensor([next_token_id], device=outputs_grad.logits.device)
                # loss = loss_fn(outputs_grad.logits[0, -1:], target)

                # if grad_hooks: # Only run backward if hooks were actually registered
                #     scalar_target.backward()

                # If you know the actual next token
                next_token_id = output_ids[0, t+1] if t+1 < seq_len else None
                if next_token_id is not None:
                    # Use cross-entropy loss for a cleaner signal
                    loss_fn = torch.nn.CrossEntropyLoss()
                    target = torch.tensor([next_token_id], device=outputs_grad.logits.device)
                    loss = loss_fn(outputs_grad.logits[0, -1:], target)
                    loss.backward()
                else:
                    # pass
                    raise ValueError(f"No next token ID found for token {t}.")

          

                # Remove gradient hooks immediately after backward pass
                for h_grad in grad_hooks:
                    h_grad.remove()
                grad_hooks = [] # Clear hooks list

            # --- Cleanup ---
            # del outputs_grad, current_attentions, target_logits, scalar_target
            # # Note: Avoid deleting current_input_ids/current_mask if they are slices pointing to original tensor
            # torch.cuda.empty_cache()

    finally:
        # Disable gradient checkpointing after the loop finishes or if an error occurs
        if gradient_checkpointing_enabled and hasattr(model, "gradient_checkpointing_disable"):
            print("Disabling gradient checkpointing...")
            model.gradient_checkpointing_disable()
        # Ensure model is back in eval mode definitively
        model.eval()

    if verbose:
        print("Gradient/Response Activation collection loop finished.")


    all_tokens = output_ids[0].tolist()
    token_texts = tokenizer.convert_ids_to_tokens(all_tokens)

    # --- Step 7: Restructure Gradients ---
    final_layer_gradients = {}
    max_len_in_result = seq_len
    for layer_idx in target_grad_layers:
        head_matrices = []
        found_grads_for_layer = any(
            layer_idx in captured_gradients_sliced.get(t, {})
            for t in range(response_start_idx, seq_len)
        )  # noqa
        if not found_grads_for_layer:
            continue

        for head_idx in range(num_heads):
            padded_vectors = []
            for t in range(response_start_idx, seq_len):
                grad_slice_all_heads = captured_gradients_sliced.get(t, {}).get(
                    layer_idx
                )
                if grad_slice_all_heads is not None:
                    # Check if head_idx is within bounds for this tensor
                    if head_idx < grad_slice_all_heads.shape[0]:
                        vec = grad_slice_all_heads[head_idx, :].cpu().numpy()
                        pad_width = max_len_in_result - len(vec)
                        padded_vec = np.pad(
                            vec, (0, pad_width), "constant", constant_values=np.nan
                        )  # noqa
                    else:  # Should not happen if num_heads is correct
                        padded_vec = np.full((max_len_in_result,), np.nan)
                else:
                    padded_vec = np.full(
                        (max_len_in_result,), np.nan
                    )  # Pad missing token grads
                padded_vectors.append(padded_vec)
            # Ensure we actually collected vectors before stacking
            if padded_vectors:
                head_matrices.append(
                    np.stack(padded_vectors, axis=0)
                )  # (response_len, seq_len)

        if not head_matrices:  # Check if any head matrix was created
            continue
        layer_tensor = np.stack(
            head_matrices, axis=0
        )  # (num_heads, response_len, seq_len)

        if make_square:
            square_tensor = np.full(
                (num_heads, seq_len, seq_len), np.nan, dtype=layer_tensor.dtype
            )
            square_tensor[:, response_start_idx:, :] = layer_tensor
            final_layer_gradients[layer_idx] = square_tensor
        else:
            final_layer_gradients[layer_idx] = layer_tensor

    # --- Step 8: Combine and Restructure Activations ---
    final_activations = {}
    for layer_idx in target_act_layers:
        full_layer_acts = []
        # Add prompt activations
        for t in range(prompt_len):
            act = prompt_activations.get(layer_idx, {}).get(t)
            if act is not None:
                full_layer_acts.append(act.squeeze(0).cpu().numpy())
            else:
                full_layer_acts.append(
                    np.full((hidden_dim,), np.nan)
                )  # Pad missing prompt acts

        # Add response activations
        for t in range(response_start_idx, seq_len):
            # Activations collected during grad loop are stored by layer -> token_pos
            act = response_activations.get(layer_idx, {}).get(t)
            if act is not None:
                full_layer_acts.append(act.squeeze(0).cpu().numpy())
            else:
                full_layer_acts.append(
                    np.full((hidden_dim,), np.nan)
                )  # Pad missing response acts

        if full_layer_acts:
            # Shape: (seq_len, hidden_dim)
            final_activations[layer_idx] = np.stack(full_layer_acts, axis=0)

    # --- Final Result Dict ---
    result = {
        "cot_prompt": prompt,
        "cot_prompt_tokens": cot_prompt_tokens,
        "cot_prompt_token_texts": cot_prompt_token_texts,
        "response": response,
        "response_start": response_start_idx,
        "tokens": all_tokens,
        "token_texts": token_texts,
        "input_length": prompt_len,
        "activations": final_activations,  # Combined prompt/response activations
        "attention_weights": {
            k: v.cpu().numpy() for k, v in full_attn_weights.items()
        },  # Full sequence weights
        "attention_gradients": final_layer_gradients,
        "is_square_padded": make_square,
    }

    return result


@pkld#(overwrite=False)  # Set overwrite=True if you change code and want to rerun
def test_prompt_grad(
    prompt,
    max_new_tokens=20,
    seed=0,
    model_name="qwen-14b",
    temperature=0.7,
    float32=True,
    target_layers=None,
    activation_layers=None,
    make_square=False,
):
    """Wrapper function to load model and run gradient/activation analysis."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)  # noqa
    if not float32:
        warnings.warn("float32=False not recommended.", stacklevel=2)
        float32 = True  # noqa

    effective_act_layers = (
        activation_layers if activation_layers is not None else target_layers
    )
    # Ensure target_layers for grads and effective_act_layers for acts/weights are sets or None
    target_grad_layers_set = set(target_layers) if target_layers is not None else None
    effective_act_layers_set = (
        set(effective_act_layers) if effective_act_layers is not None else None
    )

    model, tokenizer = get_deepseek_r1(model_name, float32=float32)
    print(
        f"Starting analysis: Model={model_name}, MaxNew={max_new_tokens}, Square={make_square}, GradLayers={target_grad_layers_set or 'All'}, ActLayers={effective_act_layers_set or 'All'}"
    )
    t_start = time.time()
    results = generate_response_with_grads(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        verbose=True,
        target_layers=target_grad_layers_set,
        activation_layers=effective_act_layers_set,
        make_square=make_square,
    )
    t_end = time.time()
    print(f"Analysis finished in {t_end - t_start:.2f} seconds.")
    del model
    torch.cuda.empty_cache()  # noqa
    return results


if __name__ == "__main__":
    context = "If the farmer waters his crops daily, they will grow strong and healthy. If there is a drought, then the crops will dry up. Either the farmer waters the crops routinely, or the crops did not dry up. If the crops grow strong, there will be a bountiful harvest. If there is a large harvest, the farmer will make good profits."
    question = "If there was a drought this season, will the farmer make good profits?"
    prompt = f"Context: {context}\n\nQuestion: {question}\n\n<think>"

    # --- Configuration ---
    MAX_TOKENS_TO_GENERATE = 5
    MODEL_NAME = "gpt2_medium"  # Use gpt2 or ensure get_deepseek_r1 handles your model
    GRAD_TARGET_LAYERS = [0, 1]
    ACT_TARGET_LAYERS = [0, 1]  # Collect acts/attn weights for same layers
    MAKE_SQUARE_MATRIX = False  # Keep False for testing plots initially

    # --- Layers and Heads to Plot ---
    LAYER_TO_PLOT = 0
    HEAD_TO_PLOT = 0

    print("=" * 80)
    print(f" RUNNING GRADIENT & ACTIVATION ANALYSIS (SQUARE={MAKE_SQUARE_MATRIX}) ")
    print(f" Model: {MODEL_NAME}, Max New Tokens: {MAX_TOKENS_TO_GENERATE}")
    print(
        f" Grad Layers: {GRAD_TARGET_LAYERS or 'All'}, Act/Attn Layers: {ACT_TARGET_LAYERS or 'All'}"
    )
    print("=" * 80)
    if MAX_TOKENS_TO_GENERATE > 20:
        print("\n!!! WARNING: max_new_tokens is high. VERY slow. !!!\n")
        time.sleep(5)  # noqa

    results = test_prompt_grad(
        prompt=prompt,
        max_new_tokens=MAX_TOKENS_TO_GENERATE,
        model_name=MODEL_NAME,
        float32=True,
        target_layers=GRAD_TARGET_LAYERS,
        activation_layers=ACT_TARGET_LAYERS,
        make_square=MAKE_SQUARE_MATRIX,
        seed=42,
    )

    print("\n--- Analysis Results Summary ---")
    print(f"Generated Response:\n{results['response']}")
    print(f"Prompt Length: {results['input_length']}")
    print(f"Total Tokens: {len(results['tokens'])}")

    # --- Check All Fields ---
    print("\nChecking returned fields:")
    expected_keys = [
        "cot_prompt",
        "cot_prompt_tokens",
        "cot_prompt_token_texts",
        "response",
        "response_start",
        "tokens",
        "token_texts",
        "input_length",
        "activations",
        "attention_weights",
        "attention_gradients",
        "is_square_padded",
    ]
    for key in expected_keys:
        print(f"  '{key}': Present - {key in results}, Type - {type(results.get(key))}")

    # --- Plotting ---
    print(f"\n--- Generating Plots for Layer {LAYER_TO_PLOT}, Head {HEAD_TO_PLOT} ---")

    # 1. Plot Attention Gradients
    gradients_by_layer = results["attention_gradients"]
    if LAYER_TO_PLOT in gradients_by_layer:
        grad_tensor = gradients_by_layer[LAYER_TO_PLOT]  # Shape (head, query, key)
        if HEAD_TO_PLOT < grad_tensor.shape[0]:
            plt.figure(figsize=(10, 8))
            im = plt.imshow(
                grad_tensor[HEAD_TO_PLOT, :, :],  # Select head
                aspect="auto",
                cmap="viridis",
                interpolation="none",
            )
            plt.colorbar(im, label="Gradient Value")
            plt.title(
                f"Attention Gradients: Layer {LAYER_TO_PLOT}, Head {HEAD_TO_PLOT}"
            )
            plt.xlabel("Key Token Index (Absolute)")
            ylabel = (
                "Query Token Index (Absolute)"
                if results["is_square_padded"]
                else "Query Token Index (Relative to Response Start)"
            )
            plt.ylabel(ylabel)
            if results["is_square_padded"]:
                plt.axvline(results["response_start"] - 0.5, color="r", ls="--")
                plt.axhline(results["response_start"] - 0.5, color="r", ls="--")  # noqa
            plt.tight_layout()
            plot_filename_grad = f'attn_grads_L{LAYER_TO_PLOT}_H{HEAD_TO_PLOT}_sq{results["is_square_padded"]}.png'
            plt.savefig(plot_filename_grad)
            print(f"Saved gradient plot: {plot_filename_grad}")
            plt.close()  # Close figure to free memory
        else:
            print(
                f"  Head {HEAD_TO_PLOT} not available in gradient tensor for Layer {LAYER_TO_PLOT}."
            )
    else:
        print(f"  No gradients found for Layer {LAYER_TO_PLOT}.")

    # 2. Plot Activation Norms
    activations_by_layer = results["activations"]
    if LAYER_TO_PLOT in activations_by_layer:
        act_tensor = activations_by_layer[LAYER_TO_PLOT]  # Shape (seq_len, hidden_dim)
        # Calculate L2 norm across hidden dimension
        act_norms = norm(act_tensor, axis=1)
        plt.figure(figsize=(12, 6))
        plt.plot(act_norms, marker=".", linestyle="-")
        plt.title(f"L2 Norm of Residual Stream Activations: Layer {LAYER_TO_PLOT}")
        plt.xlabel("Token Index (Absolute)")
        plt.ylabel("L2 Norm")
        plt.axvline(
            results["response_start"] - 0.5, color="r", ls="--", label="Response Start"
        )
        plt.grid(True, alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plot_filename_act = f"activation_norm_L{LAYER_TO_PLOT}.png"
        plt.savefig(plot_filename_act)
        print(f"Saved activation norm plot: {plot_filename_act}")
        plt.close()
    else:
        print(f"  No activations found for Layer {LAYER_TO_PLOT}.")

    # 3. Plot Attention Weights
    attn_weights_by_layer = results["attention_weights"]
    if LAYER_TO_PLOT in attn_weights_by_layer:
        # Shape: (1, n_heads, seq_len, seq_len) - from the full sequence pass
        attn_tensor = attn_weights_by_layer[LAYER_TO_PLOT][0]  # Remove batch dim
        if HEAD_TO_PLOT < attn_tensor.shape[0]:
            plt.figure(figsize=(10, 8))
            im = plt.imshow(
                attn_tensor[HEAD_TO_PLOT, :, :],  # Select head, convert to numpy
                aspect="auto",
                cmap="viridis",
                interpolation="none",
            )
            plt.colorbar(im, label="Attention Weight")
            plt.title(f"Attention Weights: Layer {LAYER_TO_PLOT}, Head {HEAD_TO_PLOT}")
            plt.xlabel("Key Token Index (Absolute)")
            plt.ylabel("Query Token Index (Absolute)")
            plt.axvline(
                results["response_start"] - 0.5,
                color="r",
                ls="--",
                label="Response Start",
            )
            plt.axhline(results["response_start"] - 0.5, color="r", ls="--")
            plt.tight_layout()
            plot_filename_attn = f"attn_weights_L{LAYER_TO_PLOT}_H{HEAD_TO_PLOT}.png"
            plt.savefig(plot_filename_attn)
            print(f"Saved attention weights plot: {plot_filename_attn}")
            plt.close()
        else:
            print(
                f"  Head {HEAD_TO_PLOT} not available in attention weights tensor for Layer {LAYER_TO_PLOT}."
            )
    else:
        print(f"  No attention weights found for Layer {LAYER_TO_PLOT}.")

    print("\nAnalysis and plotting complete.")
