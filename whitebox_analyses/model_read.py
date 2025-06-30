import os


import random
import sys
import time
from collections import defaultdict
from functools import cache
import warnings
import numpy as np
from tqdm import tqdm
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from pkld import pkld
import math
import torch.nn as nn
from types import MethodType
import matplotlib.pyplot as plt

from utils import print_gpu_memory_summary


# --- Utility Functions ---
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.
    (Copied from transformers/models/qwen2/modeling_qwen2.py for standalone use)
    """
    # The latest version unsqueezes the cos/sin tensors needed for broadcast comparisons.
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Equivalent to torch.repeat_interleave(x, dim=1, repeats=n_rep).
    (Copied from transformers/models/qwen2/modeling_qwen2.py for standalone use)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


@cache
def get_deepseek_r1(
    model_name="qwen-14b",
    float32=True,
    quantize_8bit=False,
    quantize_4bit=False,
    device_map="auto",
    do_flash_attn=False,
):
    if model_name == "qwen-14b":
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    elif model_name == "qwen-14b-base":
        model_name = "Qwen/Qwen2.5-14B"
    elif model_name == "qwen-15b":
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    elif model_name == "qwen-32b":
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    elif model_name == "it_qwen-14b":
        model_name = "Qwen/Qwen2.5-14B-Instruct"
    elif model_name == "llama8":
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    elif model_name == "llama8-base":
        model_name = r"meta-llama/Llama-3.1-8B"
    elif model_name == "gpt2_medium":
        model_name = "gpt2-medium"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True)
        return model, tokenizer
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Suppress specific warnings related to tokenizer and attention mechanisms
    warnings.filterwarnings(
        "ignore", message="Sliding Window Attention is enabled but not implemented"
    )
    warnings.filterwarnings("ignore", message="Setting `pad_token_id` to `eos_token_id`")

    # Load tokenizer with correct pad token settings
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # If tokenizer doesn't have a pad token, explicitly set it
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Configure model loading based on quantization setting
    model_kwargs = {
        "device_map": device_map,
        "sliding_window": None,
        "force_download": False,
    }

    if do_flash_attn:
        import flash_attn

        model_kwargs["attn_implementation"] = "flash_attention_2"
        print("Enabling Flash Attention!")

    if "Llama" in model_name:
        del model_kwargs["sliding_window"]
        # print('DELETEING SLIDING')
    # print(f'{model_name=}')
    # print(f'{model_kwargs=}')

    # Import bitsandbytes for quantization if needed
    if quantize_8bit or quantize_4bit:
        import bitsandbytes as bnb
        from transformers import BitsAndBytesConfig

    # Handle quantization options (prioritize 4-bit if both are set)
    if quantize_4bit:
        assert not quantize_8bit, "Cannot use both 8-bit and 4-bit quantization"
        assert not float32, "Cannot use float32 with 4-bit quantization"
        print("Doing 4-bit quantization!")
        # Configure 4-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model_kwargs["quantization_config"] = quantization_config
    elif quantize_8bit:
        assert not quantize_4bit, "Cannot use both 8-bit and 4-bit quantization"
        assert not float32, "Cannot use float32 with 8-bit quantization"
        print("Doing 8-bit quantization!")
        # Configure 8-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16,
            bnb_8bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = quantization_config
    else:
        print("No 4/8-bit quantization!")
        # Use regular precision settings
        model_kwargs["torch_dtype"] = torch.float32 if float32 else torch.float16

    # Beginning of the function
    print_gpu_memory_summary("Before model loading")

    # print(f'{model_kwargs=}')
    # quit()

    # Load model with the configured settings
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    # # after pip-installing xformers
    # model = model.to("cuda")
    # # this method *is* on most Transformer models once xFormers is available:
    # model.enable_xformers_memory_efficient_attention()

    print_gpu_memory_summary("After model loading")

    # Check if quantization worked
    linear_modules = [
        name
        for name, module in model.named_modules()
        if "Linear4bit" in str(type(module)) or "Linear8bit" in str(type(module))
    ]
    if quantize_8bit or quantize_4bit:
        print(f"Found {len(linear_modules)} quantized linear modules")
        if len(linear_modules) == 0:
            print("WARNING: No quantized modules found! Quantization may have failed.")

    return model, tokenizer


class ActivationCollector:
    def __init__(self, verbose=False, do_all=False, do_layers=None):
        self.activations = {
            "residual_stream": {},  # Input to each layer
            "self_attn_output": {},  # Output from self-attention
            "mlp_output": {},  # Output from MLP
        }
        if do_all == "mlp":
            del self.activations["self_attn_output"]
        elif do_all == "attn":
            del self.activations["mlp_output"]
        elif not do_all:
            del self.activations["mlp_output"]
            del self.activations["self_attn_output"]

        self.attn_weights = {}  # Attention weights stored separately
        self.hooks = []
        self.verbose = verbose
        self.do_all = do_all
        if do_layers is None:
            self.do_layers = None
        else:
            self.do_layers = set(do_layers)

    def collect_layer_input(self, layer_idx):
        """Collect the residual stream (input to transformer layer)"""

        def hook_fn(module, input):
            if input and isinstance(input, tuple) and len(input) > 0:
                # Convert directly to numpy instead of keeping as CPU tensor
                self.activations["residual_stream"][layer_idx] = (
                    input[0].detach().cpu()
                )  # .detach().cpu().numpy()
            return input

        return hook_fn

    def collect_self_attn_output(self, layer_idx):
        """Collect self-attention output"""

        def hook_fn(module, input, output):
            # print(f'{input=}')
            # print(f'{output=}')
            # quit()
            # print('TOAST')
            if isinstance(output, tuple) and len(output) > 0:
                # output[0][0, 2:4, ]
                self.activations["self_attn_output"][layer_idx] = (
                    output[0].detach().cpu()
                )  # .half().detach().cpu().numpy()
                # attn_weights = output[1]
                # output[1][0, :, :, :] = -1e6
                # print(f'TEST 0: {output[1].shape=}')
            else:
                self.activations["self_attn_output"][
                    layer_idx
                ] = output.detach().cpu()  # half().detach().cpu().numpy()
                # print(f'TEST 1: {output.shape=}')
            return output

        # print('HOOK SANITY')
        return hook_fn

    def collect_mlp_output(self, layer_idx):
        """Collect MLP output"""

        def hook_fn(module, input, output):
            # Convert to half precision before moving to CPU
            self.activations["mlp_output"][
                layer_idx
            ] = output.detach().cpu()  # .half().detach().cpu().numpy()
            return output

        return hook_fn

    def register_hooks(self, model):
        # Clear previous hooks if any
        self.remove_hooks()

        # Reset activations
        for key in self.activations:
            self.activations[key] = {}
        self.attn_weights = {}

        # Find the main decoder layers, attention modules, and MLP modules
        decoder_layers = {}
        attention_modules = {}
        mlp_modules = {}

        for name, module in model.named_modules():
            if ".layers." in name:
                parts = name.split(".layers.")
                if len(parts) > 1 and parts[1]:
                    layer_idx_str = parts[1].split(".")[0]
                    try:
                        layer_idx = int(layer_idx_str)
                        if self.do_layers is not None and layer_idx not in self.do_layers:
                            continue

                        # Collect decoder layers
                        if name.endswith(f"layers.{layer_idx}"):
                            decoder_layers[layer_idx] = module
                        if self.do_all:
                            if self.do_all == "attn":
                                if "self_attn" in name and name.endswith("self_attn"):
                                    attention_modules[layer_idx] = module
                            else:
                                raise ValueError(f"Unknown do_all: {self.do_all}")
                            # if "mlp" in name and name.endswith("mlp"):
                            #     mlp_modules[layer_idx] = module
                            # if self.do_all != "mlp":
                            #     # Collect attention modules
                            #     if "self_attn" in name and name.endswith("self_attn"):
                            #         attention_modules[layer_idx] = module

                    except ValueError:
                        pass
        # quit()

        if self.verbose:
            print(f"Found {len(decoder_layers)} decoder layers")
            print(f"Found {len(attention_modules)} attention modules")
            print(f"Found {len(mlp_modules)} MLP modules")

        # # Register hooks for layer inputs (residual stream)
        # for layer_idx, layer_module in decoder_layers.items():
        #     print(f'ADD HOOK: {layer_idx}/{layer_module}')
        #     pre_hook = layer_module.register_forward_pre_hook(
        #         self.collect_layer_input(layer_idx)
        #     )
        #     self.hooks.append(pre_hook)

        if self.do_all:
            if self.do_all == "attn":
                # Register hooks for self-attention outputs
                for layer_idx, attn_module in attention_modules.items():
                    # print(f'ADD HOOK: {layer_idx}/{attn_module}')
                    hook = attn_module.register_forward_hook(
                        self.collect_self_attn_output(layer_idx)
                    )
                    self.hooks.append(hook)
            else:
                raise ValueError(f"Unknown do_all: {self.do_all}")

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def run_activation_extraction(
    model,
    tokenizer,
    text,
    verbose=False,
    do_layers=None,
    do_all=False,
    pos_embedding_scale=None,
    return_logits=True,
    attn_layers=None,
    token_range_to_mask=None,
    mask_layers=None,
):
    """
    Analyze a text by running a forward pass and collecting internal activations

    Args:
        model: The language model
        tokenizer: The tokenizer
        text: Input text to analyze
        verbose: Whether to print debug information
        do_layers: Optional list of specific layers to collect activations from
        do_all: Whether to collect all activations (True) or just residual stream (False)
        pos_embedding_scale: Scaling factor for positional embeddings (0.0-1.0)
        return_logits: Whether to return the output logits
        attn_layers: Optional list of specific layers to collect attention weights from
        token_range_to_mask: Optional token range to mask attention
        mask_layers: Optional layer indices to apply attention mask
    """

    # Scale positional embeddings if requested
    position_embeddings_backup = {}
    method_backups = {}

    if pos_embedding_scale is not None and pos_embedding_scale is not False:
        try:
            alpha = float(pos_embedding_scale)
            if not (0.0 <= alpha <= 1.0):
                print(f"Warning: pos_embedding_scale should be between 0.0 and 1.0. Got {alpha}.")
                alpha = max(0.0, min(alpha, 1.0))  # Clamp between 0 and 1

            if verbose:
                action = "Scaling" if alpha > 0 else "Zeroing"
                print(f"{action} positional embeddings by factor {alpha}...")

            with torch.no_grad():
                # Try different possible locations for positional embeddings in various model architectures
                pos_embedding_found = False

                # Check for wpe (word position embeddings) in transformer models
                if hasattr(model, "transformer") and hasattr(model.transformer, "wpe"):
                    position_embeddings_backup["transformer.wpe"] = (
                        model.transformer.wpe.weight.clone()
                    )
                    model.transformer.wpe.weight.mul_(alpha)
                    pos_embedding_found = True
                    if verbose:
                        print(f"Scaled transformer.wpe positional embeddings by {alpha}")

                # Check for rotary embeddings in the attention modules
                for name, module in model.named_modules():
                    # For RoPE-based models
                    if "rotary_emb" in name or "rotary_pos_emb" in name:
                        # Store the original implementation
                        if hasattr(module, "forward"):
                            original_forward = module.forward

                            # Define a scaled forward function
                            def scaled_rotary_forward(*args, **kwargs):
                                # Call original but scale the result
                                result = original_forward(*args, **kwargs)
                                if isinstance(result, tuple):
                                    # If it returns multiple tensors, scale them all
                                    return tuple(x * alpha for x in result)
                                else:
                                    # If it's a single tensor
                                    return result * alpha

                            # Replace with scaled implementation
                            method_backups[name] = (module, original_forward)
                            module.forward = scaled_rotary_forward
                            pos_embedding_found = True
                            if verbose:
                                print(f"Scaled rotary positional embeddings in {name} by {alpha}")

                    # For absolute positional embeddings
                    elif any(
                        pe_name in name
                        for pe_name in [
                            "pos_embedding",
                            "position_embedding",
                            "embed_positions",
                        ]
                    ):
                        if hasattr(module, "weight"):
                            position_embeddings_backup[name] = module.weight.clone()
                            module.weight.mul_(alpha)
                            pos_embedding_found = True
                            if verbose:
                                print(f"Scaled positional embeddings in {name} by {alpha}")

                if not pos_embedding_found and verbose:
                    print("Warning: Could not find positional embeddings to scale")
        except Exception as e:
            print(f"Error while scaling positional embeddings: {e}")
            print("Continuing with unmodified positional embeddings")

    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    if verbose:
        print(f"Text: '{text}'")
        print(f"Encoded to {input_ids.shape[1]} tokens")

    # Create a fresh collector
    collector = ActivationCollector(verbose=verbose, do_all=do_all, do_layers=do_layers)
    collector.register_hooks(model)

    # Run forward pass to collect activations
    logits = None
    try:
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.float16):
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=".*does not support `output_attentions=True`.*",
                    )
                    # Apply our custom attention masking hooks if requested
                    hooks_applied = False
                    if token_range_to_mask and mask_layers:
                        apply_qwen_attn_mask_hooks(
                            model,
                            token_range_to_mask,
                            layer_2_heads_suppress=mask_layers,
                        )
                        hooks_applied = True

                    # --- Make the model call ---
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_attentions=attn_layers is None
                        or len(attn_layers) > 0,  # Request attentions
                        return_dict=True,  # Request dictionary output
                        attn_implementation="eager",  # Ensure eager attention for hooks
                        use_cache=False,  # <<< *** Explicitly disable KV cache ***
                        output_hidden_states=False,
                    )
                    # --- End model call ---

            # Store activations (already collected via separate collector hooks)
            activations = collector.activations
            if token_range_to_mask is None:
                # Store attention weights from the model output
                if hasattr(outputs, "attentions") and outputs.attentions is not None:
                    for layer_idx, attn_weights in enumerate(outputs.attentions):
                        if attn_layers is not None and layer_idx not in attn_layers:
                            continue
                        collector.attn_weights[layer_idx] = attn_weights.detach().cpu()
                else:
                    # Add a warning if attentions were expected but not found
                    if attn_layers is not None:
                        print(
                            "Warning: Requested attention layers but 'outputs.attentions' not found or is None."
                        )

            # Store logits if requested
            if return_logits and hasattr(outputs, "logits"):
                logits = outputs.logits.detach().cpu()
            else:
                # Add a warning if logits were expected but not found
                if return_logits:
                    print("Warning: Requested logits but 'outputs.logits' not found.")

    except Exception as e:
        print(f"WARNING: Error during data collection: {e}")
        import traceback

        traceback.print_exc()  # Print full traceback for better debugging
        activations = collector.activations  # Use whatever we managed to collect

    finally:
        # Remove our custom attention masking hooks if they were applied
        if hooks_applied:  # Use the flag to ensure removal only if applied
            remove_qwen_attn_mask_hooks(model)
        # Also remove collector hooks
    collector.remove_hooks()

    # Restore original positional embeddings
    if pos_embedding_scale is not None and pos_embedding_scale is not False:
        with torch.no_grad():
            for name, original_weight in position_embeddings_backup.items():
                parts = name.split(".")
                module = model
                for part in parts[:-1]:
                    if hasattr(module, part):
                        module = getattr(module, part)
                if hasattr(module, parts[-1]):
                    getattr(module, parts[-1]).weight.copy_(original_weight)

            for name, (module, original_forward) in method_backups.items():
                module.forward = original_forward

    # Clean up GPU memory
    if logits is not None:
        logits = logits.detach().cpu().numpy()  # Ensure it's detached numpy before returning
    torch.cuda.empty_cache()

    # Get token texts for all tokens
    all_tokens = input_ids[0].tolist()
    token_texts = tokenizer.convert_ids_to_tokens(all_tokens)

    # Organize the results
    result = {
        "text": text,
        "tokens": all_tokens,
        "token_texts": token_texts,
        "input_length": len(all_tokens),
        "activations": activations,  # From collector
        "attention_weights": collector.attn_weights,  # From collector
        "pos_embedding_scale": (pos_embedding_scale if pos_embedding_scale is not None else 1.0),
    }

    if logits is not None:
        result["logits"] = logits

    torch.cuda.empty_cache()
    return result


def get_token_logits_for_word(logits, word, model_name="qwen-14b"):
    """Get logits for a specific word token"""
    if "qwen" in model_name:
        _, tokenizer = get_deepseek_r1(model_name, float32=False)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Tokenize the word
    tokens = tokenizer.encode(word, add_special_tokens=False)
    assert len(tokens) == 1, f"Word {word} has {len(tokens)} tokens"
    word_logits = logits[0, :, tokens[0]]
    return word_logits


def analyze_text(
    text,
    model_name="qwen-14b",
    seed=0,
    float32=False,
    quantize_8bit=False,
    quantize_4bit=False,
    pos_embedding_scale=None,
    do_layers=None,
    return_logits=False,
    attn_layers=None,
    verbose=False,
    token_range_to_mask=None,
    layers_to_mask=None,
    device_map="auto",
):
    """Analyze a text using a model's forward pass"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    assert not (quantize_8bit and quantize_4bit), "Cannot use both 8-bit and 4-bit quantization"
    assert not (
        float32 and (quantize_8bit or quantize_4bit)
    ), "Cannot use 8-bit or 4-bit quantization with float32"

    if device_map == "cpu":
        print("Running on CPU!")

    if "qwen" in model_name or "llama" in model_name:
        model, tokenizer = get_deepseek_r1(
            model_name,
            float32=float32,
            quantize_8bit=quantize_8bit,
            quantize_4bit=quantize_4bit,
            device_map=device_map,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    if verbose:
        print(f"Analyzing text: {text}")
    else:
        print(f"Analyzing text: {text[:100]}...")
    t_st = time.time()

    result = run_activation_extraction(
        model,
        tokenizer,
        text,
        verbose=verbose,
        do_layers=do_layers,
        do_all="attn",  # Collect all activations
        pos_embedding_scale=pos_embedding_scale,
        return_logits=return_logits,
        attn_layers=attn_layers,
        token_range_to_mask=token_range_to_mask,
        mask_layers=layers_to_mask,
    )

    print_gpu_memory_summary("After forward pass")

    t_end = time.time()
    num_tokens = len(result["token_texts"])
    print(f"Time taken: {t_end - t_st:.2f} seconds (for {num_tokens} tokens)")

    if attn_layers and any(layer in result["attention_weights"] for layer in attn_layers):
        layer = next(layer for layer in attn_layers if layer in result["attention_weights"])
        attention_data = result["attention_weights"][layer]
        num_nans = np.isnan(attention_data).sum()
        p_nan = num_nans / attention_data.numel()
        assert ~np.isnan(
            attention_data
        ).any(), f"Attention has NaNs{attention_data.shape=} ({p_nan=:.1%})"

    print(f"\t*** Analysis complete! ({float32=}, {quantize_8bit=}, {quantize_4bit=}) ***")

    # Free up GPU memory
    del model
    torch.cuda.empty_cache()

    return result

    # Can uncomment for full details when needed
    # print(torch.cuda.memory_summary())


# --- Store original methods globally (or pass via a class) ---
original_qwen_forward_methods = {}


def apply_qwen_attn_mask_hooks(model, token_range, layer_2_heads_suppress=None):
    """
    Applies hooks to Qwen2-style attention modules to mask attention computation.
    (Corrected RoPE handling based on module structure)
    """
    global original_qwen_forward_methods
    original_qwen_forward_methods = {}

    if token_range is None:
        print("No token range specified for masking.")
        return
    assert isinstance(token_range, list), f"Bad token_range should be list of lists: {token_range=}"
    if isinstance(token_range[0], int):
        token_range = [token_range]

    target_modules = []
    module_prefix = "model.layers"
    attn_suffix = "self_attn"
    rotary_emb_module = None
    if hasattr(model, "model") and hasattr(model.model, "rotary_emb"):
        rotary_emb_module = model.model.rotary_emb
        print(f"Found rotary_emb at model.model.rotary_emb")
        if not callable(rotary_emb_module):
            print(f"Warning: Found rotary_emb module, but it is not callable. RoPE might fail.")
    else:
        print("Warning: Could not automatically find the main rotary_emb module.")

    for name, module in model.named_modules():
        if name.startswith(module_prefix) and name.endswith(attn_suffix):
            try:
                layer_idx_str = name.split(".")[2]
                layer_idx = int(layer_idx_str)
                if layer_2_heads_suppress is None or layer_idx in layer_2_heads_suppress:
                    if (
                        hasattr(module, "config")
                        and hasattr(module, "q_proj")
                        and hasattr(module, "k_proj")
                        and hasattr(module, "v_proj")
                        and hasattr(module, "o_proj")
                    ):
                        target_modules.append((name, module, layer_idx))
                    else:
                        missing = [
                            p
                            for p in ["config", "q_proj", "k_proj", "v_proj", "o_proj"]
                            if not hasattr(module, p)
                        ]
                        print(f"Warning: Module {name} missing attributes: {missing}. Skipping.")
            except (IndexError, ValueError):
                print(f"Warning: Could not parse layer index from module name: {name}. Skipping.")

    if not target_modules:
        print("Error: No suitable Qwen2-style attention modules found.")
        return

    print(f"Found {len(target_modules)} Qwen2 attention modules to patch.")

    # --- Define the modified forward function ---
    def create_masked_forward(
        original_forward_func, current_layer_idx, rotary_module_ref, heads_mask=None
    ):
        def masked_forward(
            self,  # self is the Qwen2Attention instance
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.LongTensor | None = None,
            past_key_value: tuple[torch.Tensor] | None = None,
            output_attentions: bool = False,  # This is passed from the original call stack
            use_cache: bool = False,  # Passed False externally
            cache_position: torch.LongTensor | None = None,
            **kwargs,
        ) -> tuple[torch.Tensor, torch.Tensor | None]:  # Adjusted return type hint

            bsz, q_len, _ = hidden_states.size()
            config = self.config
            device = hidden_states.device  # Get the target device from hidden_states

            num_heads = config.num_attention_heads
            head_dim = config.hidden_size // num_heads
            num_key_value_heads = config.num_key_value_heads
            num_key_value_groups = num_heads // num_key_value_heads
            hidden_size = config.hidden_size

            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(
                1, 2
            )

            # --- RoPE Embeddings ---
            if position_ids is None:
                position_ids = torch.arange(
                    0, q_len, dtype=torch.long, device=device
                )  # Ensure device
                position_ids = position_ids.unsqueeze(
                    0
                )  # No need for .view(-1, q_len) if bsz=1, but this is safer
            else:
                # Ensure provided position_ids are on the correct device
                position_ids = position_ids.to(device)

            if rotary_module_ref is not None and callable(rotary_module_ref):
                try:
                    # Ensure value_states (used for dtype/device hint) is on the target device
                    # It *should* be, as it derives from hidden_states, but check defensively if errors persist
                    # print(f"Layer {current_layer_idx} devices: hidden={device}, rotary={rotary_module_ref.inv_freq.device}, pos_ids={position_ids.device}") # Debug device placement
                    cos, sin = rotary_module_ref(
                        value_states.to(device), position_ids=position_ids
                    )  # Ensure input to rotary is on correct device

                    # Ensure cos/sin are on the correct device before applying
                    cos = cos.to(device)
                    sin = sin.to(device)

                    # Apply RoPE using the globally defined function
                    # Make sure q/k are on the correct device (should be inherited from hidden_states)
                    query_states, key_states = apply_rotary_pos_emb(
                        query_states, key_states, cos, sin, position_ids
                    )

                except Exception as e:
                    print(
                        f"Warning: Layer {current_layer_idx} - Error during RoPE application: {e}."
                    )
            else:
                print(
                    f"Warning: Layer {current_layer_idx} - No valid rotary embedding module reference. Skipping RoPE."
                )

            # --- KV Cache Handling ---
            kv_seq_len = q_len
            # This block should not run with use_cache=False, but keep for robustness
            if past_key_value is not None:
                print(f"Warning Layer {current_layer_idx}: past_key_value provided unexpectedly.")
                kv_seq_len += past_key_value[0].shape[-2]
                # Ensure tensors are on the same device before cat
                key_states = torch.cat([past_key_value[0].to(device), key_states], dim=2)
                value_states = torch.cat([past_key_value[1].to(device), value_states], dim=2)
            # past_key_value = None # Return value is handled below based on use_cache

            # --- GQA Handling ---
            key_states = repeat_kv(key_states, num_key_value_groups)
            value_states = repeat_kv(value_states, num_key_value_groups)

            # --- Attention Score Calculation ---
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
                head_dim
            )

            for token_range_ in token_range:
                assert isinstance(
                    token_range_, list
                ), f"Bad token_range should be list of lists: {token_range=}"

                # --- Masking Logic (Custom + Causal from input) ---
                effective_end_pos = min(token_range_[1], kv_seq_len)
                effective_start_pos = min(token_range_[0], effective_end_pos)

                # Apply custom mask
                if effective_start_pos < effective_end_pos:
                    mask_value = torch.finfo(attn_weights.dtype).min  # Use min float value
                    if heads_mask is None:
                        attn_weights[..., effective_start_pos:effective_end_pos] = (
                            mask_value  # <--- UNCOMMENT THIS LINE
                        )
                    else:
                        attn_weights[:, heads_mask, :, effective_start_pos:effective_end_pos] = (
                            mask_value
                        )

            # Apply causal mask from input `attention_mask`
            if attention_mask is not None:
                # Ensure attention_mask is correctly shaped and on the right device
                attention_mask = attention_mask.to(device)
                expected_mask_shape = (bsz, 1, q_len, kv_seq_len)
                # ... (rest of attention_mask reshaping/broadcasting logic as before) ...
                if attention_mask.shape != expected_mask_shape:
                    if attention_mask.ndim == 2 and attention_mask.shape == (
                        bsz,
                        kv_seq_len,
                    ):
                        attention_mask = attention_mask[:, None, None, :]
                    elif (
                        attention_mask.ndim == 4
                        and attention_mask.shape[1] == 1
                        and attention_mask.shape[2] == 1
                    ):
                        # Allow broadcasting from (bsz, 1, 1, kv_seq_len)
                        pass
                    elif attention_mask.shape[2] == 1 and q_len > 1:
                        attention_mask = attention_mask.expand(bsz, 1, q_len, kv_seq_len)
                    else:
                        print(
                            f"Warning Layer {current_layer_idx}: Mismatch attn mask shape {attention_mask.shape} vs {expected_mask_shape}."
                        )
                        # Avoid applying if mismatch is unresolvable
                        attention_mask = None  # Skip applying the mask

                if attention_mask is not None:
                    # Additive mask application
                    if attention_mask.dtype == torch.bool:
                        attention_mask_float = torch.where(
                            attention_mask, 0.0, torch.finfo(attn_weights.dtype).min
                        ).to(attn_weights.dtype)
                    else:
                        attention_mask_float = attention_mask.to(attn_weights.dtype)

                    try:
                        attn_weights = attn_weights + attention_mask_float
                    except RuntimeError as e:
                        print(
                            f"Error Layer {current_layer_idx}: Cannot add attention mask. Shape mismatch: {attn_weights.shape} vs {attention_mask_float.shape}. Error: {e}"
                        )

            # --- Softmax ---
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
                query_states.dtype
            )
            # Use self.attention_dropout if it exists, otherwise use config value directly
            dropout_p = (
                self.attention_dropout
                if hasattr(self, "attention_dropout")
                else config.attention_dropout
            )
            # Apply dropout only during training
            if self.training:
                attn_weights = nn.functional.dropout(attn_weights, p=dropout_p, training=True)

            # --- Final Weighted Sum ---
            attn_output = torch.matmul(attn_weights, value_states)

            # --- Reshape and Output Projection ---
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, hidden_size)
            attn_output = self.o_proj(attn_output)

            # --- *** Corrected Return Signature *** ---
            # Match the original Qwen2Attention return based on output_attentions
            # Since output_attentions=True is passed externally, return weights.
            # Return None for past_key_value since use_cache=False was passed.
            if not output_attentions:
                attn_weights = None  # Ensure attn_weights is None if not requested
            # attn_weights = None

            # The calling layer expects (hidden_states, attn_weights_or_None)
            return attn_output, attn_weights

        return masked_forward

    # --- Patch the forward methods ---
    for name, attn_module, layer_idx in target_modules:
        if name in original_qwen_forward_methods:
            print(f"Warning: Module {name} seems already patched. Skipping.")
            continue
        original_qwen_forward_methods[name] = attn_module.forward
        # Pass rotary_emb_module reference
        heads_mask = (
            layer_2_heads_suppress[layer_idx] if layer_2_heads_suppress is not None else None
        )
        attn_module.forward = MethodType(
            create_masked_forward(
                attn_module.forward, layer_idx, rotary_emb_module, heads_mask=heads_mask
            ),
            attn_module,
        )
        # print(f"Applied mask hook to: {name} (Layer {layer_idx})")


def remove_qwen_attn_mask_hooks(model):
    """Restores the original forward methods for Qwen attention modules."""
    global original_qwen_forward_methods
    if not original_qwen_forward_methods:
        print("No Qwen hooks seem to be applied or stored.")
        return

    restored_count = 0
    module_prefix = "model.layers"
    attn_suffix = "self_attn"
    for name, module in model.named_modules():
        if name.startswith(module_prefix) and name.endswith(attn_suffix):
            if name in original_qwen_forward_methods:
                module.forward = original_qwen_forward_methods[name]
                restored_count += 1

    if restored_count != len(original_qwen_forward_methods):
        print(
            f"Warning: Attempted to restore {len(original_qwen_forward_methods)} methods, but only restored {restored_count}."
        )
    else:
        print(f"Restored {restored_count} original Qwen forward methods.")

    original_qwen_forward_methods = {}


if __name__ == "__main__":
    # Test with a simple sentence
    text = "The quick brown fox jumps over the"
    model_id = "Qwen/Qwen2.5-14B-Instruct"  # Or your specific qwen model
    model_id = "qwen-14b"
    # Use a token range relevant to the text and tokenizer
    # Example: " brown fox " might be tokens 4-8 for some tokenizers
    token_range_to_mask = [2, 6]
    # layers_to_mask = {0, 1, 2, 3} # <<< Increased from {0, 1}
    layers_to_mask = {i: list(range(40)) for i in range(48)}

    # --- Baseline Analysis ---
    print("\n--- Running Baseline Analysis ---")
    # Keep baseline call the same (it now runs without our hook active)
    do_layers = list(range(40))
    results_normal = analyze_text(
        text=text,
        model_name=model_id,
        float32=False,
        quantize_4bit=True,
        return_logits=True,
        # We run baseline *without* applying our hooks, so token_range/mask_layers args here don't matter for baseline
        # But we still might want to collect attn weights/activations from these layers for inspection
        do_layers=do_layers,  # list(layers_to_mask) + [47], # Collect from same layers
        attn_layers=do_layers,  # list(layers_to_mask) + [47], # Collect from same layers
        verbose=False,
        # token_range_to_mask=None, # Explicitly None for baseline
        # mask_layers=None,         # Explicitly None for baseline
    )
    print("\nBaseline Analysis Results:")
    print(f"Tokens: {results_normal['token_texts']}")
    idx_target = 7
    if "logits" in results_normal:
        normal_logits_last = results_normal["logits"][0, idx_target]
        print(f"Baseline Logits shape (last token): {normal_logits_last.shape}")

    # --- Masked Analysis ---
    print("\n--- Running Masked Analysis ---")
    print(f"Applying mask to token range {token_range_to_mask} in layers {layers_to_mask}")
    # mask_layers = {6: [0, 10]}
    # layers_to_mask = {i: [1, 3] for i in range(48)}
    results_masked = analyze_text(
        text=text,
        model_name=model_id,
        float32=False,
        quantize_4bit=True,
        return_logits=True,
        do_layers=do_layers,  # list(layers_to_mask) + [47], # Collect from same layers
        attn_layers=do_layers,  # list(layers_to_mask) + [47], # Collect from same layers
        verbose=False,
        token_range_to_mask=token_range_to_mask,  # Apply mask range
        layers_to_mask=layers_to_mask,  # Apply to specified layers
    )
    print("\nMasked Analysis Results:")
    print(f"Tokens: {results_masked['token_texts']}")  # Should be same as baseline
    if "logits" in results_masked:
        masked_logits_last = results_masked["logits"][0, idx_target]
        print(f"Masked Logits shape (last token): {masked_logits_last.shape}")

    # --- Compare Logits ---
    print("\n--- Logit Comparison (Last Token Prediction) ---")
    if "logits" in results_normal and "logits" in results_masked:
        logit_diff = normal_logits_last - masked_logits_last
        abs_diff = np.abs(logit_diff)
        max_diff = np.max(abs_diff)
        mean_diff = np.mean(abs_diff)
        print(f"Max absolute logit difference: {max_diff:.6f}")
        print(f"Mean absolute logit difference: {mean_diff:.6f}")

        # Find top differences
        top_indices = np.argsort(abs_diff)[-10:][::-1]  # Top 10 differences

        # Reload tokenizer just for decoding output tokens if needed
        _, tokenizer = get_deepseek_r1(model_name=model_id)
        # tokenizer = AutoTokenizer.from_pretrained(model_id)

        print("\nTop 10 Logit Differences:")
        for idx in top_indices:
            token = tokenizer.decode([idx])
            norm_logit = normal_logits_last[idx]
            mask_logit = masked_logits_last[idx]
            d = abs_diff[idx]
            print(
                f"Token='{token}' ({idx}): Normal={norm_logit:.4f}, Masked={mask_logit:.4f}, Diff={d:.4f}"
            )

        # Compare top predicted token
        pred_idx_normal = np.argmax(normal_logits_last)
        pred_idx_masked = np.argmax(masked_logits_last)
        pred_token_normal = tokenizer.decode([pred_idx_normal])
        pred_token_masked = tokenizer.decode([pred_idx_masked])

        print(
            f"\nBaseline Top Prediction: '{pred_token_normal}' ({pred_idx_normal}) - Logit: {normal_logits_last[pred_idx_normal]:.4f}"
        )
        print(
            f"Masked Top Prediction:   '{pred_token_masked}' ({pred_idx_masked}) - Logit: {masked_logits_last[pred_idx_masked]:.4f}"
        )

        if pred_idx_normal != pred_idx_masked:
            print(">>> Prediction CHANGED! <<<")
        else:
            print(">>> Prediction did NOT change. <<<")

    else:
        print("Logits not found in one or both results for comparison.")
