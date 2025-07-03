import os
import random
import sys

from tqdm import tqdm

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
from collections import defaultdict
from functools import cache
import warnings
import numpy as np
from pkld import pkld


@cache
def get_deepseek_r1(model_name="qwen-14b", float32=True):
    if model_name == "qwen-14b":
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    elif model_name == "qwen-15b":
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    elif model_name == "qwen-32b":
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    elif model_name == "it_qwen-14b":
        model_name = "Qwen/Qwen2.5-14B-Instruct"
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
    warnings.filterwarnings(
        "ignore", message="Setting `pad_token_id` to `eos_token_id`"
    )

    # Load tokenizer with correct pad token settings
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # If tokenizer doesn't have a pad token, explicitly set it
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with specific config to disable sliding window attention
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32 if float32 else torch.float16,
        device_map="auto",
        # Disable sliding window attention
        sliding_window=None,
        force_download=False,  # Forces a fresh download
    )

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
                # Important: just get a copy, don't modify
                self.activations["residual_stream"][layer_idx] = input[0].detach().cpu()
            return input  # Return input unchanged

        return hook_fn

    def collect_self_attn_output(self, layer_idx):
        """Collect self-attention output"""

        def hook_fn(module, input, output):
            # Important: just get a copy, don't modify
            if isinstance(output, tuple) and len(output) > 0:
                self.activations["self_attn_output"][layer_idx] = (
                    output[0].detach().cpu()
                )
            else:
                self.activations["self_attn_output"][layer_idx] = output.detach().cpu()
            return output  # Return output unchanged

        return hook_fn

    def collect_mlp_output(self, layer_idx):
        """Collect MLP output"""

        def hook_fn(module, input, output):
            # Important: just get a copy, don't modify
            self.activations["mlp_output"][layer_idx] = output.detach().cpu()
            return output  # Return output unchanged

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
            # print(f'{name=}')
            # Extract layer number if possible
            if ".layers." in name:
                parts = name.split(".layers.")
                if len(parts) > 1 and parts[1]:
                    layer_idx_str = parts[1].split(".")[0]
                    try:
                        layer_idx = int(layer_idx_str)
                        if (
                            self.do_layers is not None
                            and layer_idx not in self.do_layers
                        ):
                            continue
                        # print(f'{name} | {layer_idx=}')

                        # Collect decoder layers
                        if name.endswith(f"layers.{layer_idx}"):
                            # print('DECODER LAYER!')
                            decoder_layers[layer_idx] = module
                        if self.do_all:
                            if "mlp" in name and name.endswith("mlp"):
                                mlp_modules[layer_idx] = module
                            if self.do_all != "mlp":
                                # Collect attention modules
                                if "self_attn" in name and name.endswith("self_attn"):
                                    attention_modules[layer_idx] = module

                    except ValueError:
                        pass

        if self.verbose:
            print(f"Found {len(decoder_layers)} decoder layers")
            print(f"Found {len(attention_modules)} attention modules")
            print(f"Found {len(mlp_modules)} MLP modules")
        # print(f'{decoder_layers.keys()=}')

        # Register hooks for layer inputs (residual stream)
        for layer_idx, layer_module in decoder_layers.items():
            pre_hook = layer_module.register_forward_pre_hook(
                self.collect_layer_input(layer_idx)
            )
            self.hooks.append(pre_hook)

        if self.do_all:
            if self.do_all != "mlp":
                # Register hooks for self-attention outputs
                for layer_idx, attn_module in attention_modules.items():
                    hook = attn_module.register_forward_hook(
                        self.collect_self_attn_output(layer_idx)
                    )
                    self.hooks.append(hook)

            # Register hooks for MLP outputs
            for layer_idx, mlp_module in mlp_modules.items():
                hook = mlp_module.register_forward_hook(
                    self.collect_mlp_output(layer_idx)
                )
                self.hooks.append(hook)

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


class SteeringVectorApplier:
    def __init__(
        self,
        steering_vectors,
        response_start_idx=None,
        target_positions=None,
        verbose=False,
    ):
        """
        Initialize with steering vectors to apply during generation

        Args:
            steering_vectors: Dict mapping layer indices to steering vectors {layer_idx: vector}
            response_start_idx: Token index where the response starts (to apply steering only to response tokens)
            target_positions: Optional list of token positions to apply steering to (if None, apply to all positions)
            verbose: Whether to print debug information
        """
        self.steering_vectors = steering_vectors
        self.response_start_idx = response_start_idx
        self.target_positions = target_positions
        self.hooks = []
        self.verbose = verbose
        self.applied_count = defaultdict(int)
        self.generation_started = False
        self.current_position = 0

    def steering_hook(self, layer_idx):
        """Hook function that applies steering vector to the residual stream at specified layer"""

        def hook_fn(module, input_tuple):
            if (
                not input_tuple
                or not isinstance(input_tuple, tuple)
                or len(input_tuple) == 0
            ):
                return input_tuple

            hidden_states = input_tuple[0]
            if layer_idx not in self.steering_vectors:
                return input_tuple

            # Get steering vector and ensure correct format/device
            steering_vector = self.steering_vectors[layer_idx]
            if isinstance(steering_vector, np.ndarray):
                steering_vector = torch.tensor(
                    steering_vector, dtype=hidden_states.dtype
                )
            if steering_vector.device != hidden_states.device:
                steering_vector = steering_vector.to(hidden_states.device)

            # Verify dimensions match
            if steering_vector.shape[0] != hidden_states.shape[-1]:
                if self.verbose:
                    print(
                        f"Warning: Steering vector dimension mismatch: {steering_vector.shape[0]} vs {hidden_states.shape[-1]}"
                    )
                return input_tuple

            modified_hidden_states = hidden_states.clone()
            batch_size, seq_len, hidden_dim = hidden_states.shape

            if seq_len == 1:  # We're in generation mode
                if not hasattr(self, "absolute_position"):
                    # Initialize position counter at prompt length
                    self.absolute_position = self.response_start_idx

                # Apply steering only if we're at or past the response start
                if self.absolute_position >= self.response_start_idx:
                    modified_hidden_states += steering_vector
                    self.applied_count[layer_idx] += 1

                    if self.verbose and self.applied_count[layer_idx] <= 5:
                        print(
                            f"Applied steering at layer {layer_idx}, absolute position {self.absolute_position}"
                        )

                # Increment position counter
                self.absolute_position += 1

            else:  # We're doing a full forward pass
                if self.response_start_idx is not None:
                    # Apply to all positions from response_start onwards
                    response_positions = slice(self.response_start_idx, seq_len)
                    modified_hidden_states[:, response_positions, :] += steering_vector
                    self.applied_count[layer_idx] += seq_len - self.response_start_idx

                    if self.verbose and self.applied_count[layer_idx] <= 3:
                        print(
                            f"Applied steering at layer {layer_idx} to positions {self.response_start_idx}-{seq_len-1}"
                        )

            return (modified_hidden_states,) + input_tuple[1:]

        return hook_fn

    def register_hooks(self, model):
        """Register hooks to apply steering vectors at specific layers"""
        self.remove_hooks()
        self.applied_count = defaultdict(int)
        if hasattr(self, "absolute_position"):
            delattr(self, "absolute_position")

        decoder_layers = {}
        for name, module in model.named_modules():
            if ".layers." in name:
                parts = name.split(".layers.")
                if len(parts) > 1 and parts[1]:
                    layer_idx_str = parts[1].split(".")[0]
                    try:
                        layer_idx = int(layer_idx_str)
                        if layer_idx in self.steering_vectors and name.endswith(
                            f"layers.{layer_idx}"
                        ):
                            decoder_layers[layer_idx] = module
                    except ValueError:
                        pass

        if self.verbose:
            print(
                f"Registered steering hooks for {len(decoder_layers)} layers: {sorted(decoder_layers.keys())}"
            )

        for layer_idx, layer_module in decoder_layers.items():
            pre_hook = layer_module.register_forward_pre_hook(
                self.steering_hook(layer_idx)
            )
            self.hooks.append(pre_hook)

    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

        if self.verbose and sum(self.applied_count.values()) > 0:
            print(f"Steering vector application summary:")
            for layer_idx, count in sorted(self.applied_count.items()):
                print(f"  Layer {layer_idx}: Applied {count} times")


def generate_response_with_activations(
    model,
    tokenizer,
    cot_prompt,
    max_new_tokens=100,
    steering_vectors=None,
    verbose=False,
    temperature=0.6,
    do_layers=None,
    do_all=False,
    pos_embedding_scale=None,
    return_logits=False,
    attn_layers=None,
):
    """
    Generate a response while collecting activations and optionally applying steering vectors

    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum number of new tokens to generate
        steering_vectors: Optional dict mapping layer indices to steering vectors {layer_idx: vector}
        verbose: Whether to print debug information
        temperature: Temperature for sampling
        do_layers: Optional list of specific layers to collect activations from
        do_all: Whether to collect all activations (True) or just residual stream (False)
        pos_embedding_scale: Scaling factor for positional embeddings (0.0-1.0).
                           If None or False, positional embeddings are unchanged.
                           If 0.0, positional embeddings are zeroed out.
                           If a value between 0.0 and 1.0, embeddings are scaled by that factor.
        return_logits: Whether to return the output logits
    """

    # Scale positional embeddings if requested
    if pos_embedding_scale is not None and pos_embedding_scale is not False:
        try:
            alpha = float(pos_embedding_scale)
            if not (0.0 <= alpha <= 1.0):
                print(
                    f"Warning: pos_embedding_scale should be between 0.0 and 1.0. Got {alpha}."
                )
                alpha = max(0.0, min(alpha, 1.0))  # Clamp between 0 and 1

            if verbose:
                action = "Scaling" if alpha > 0 else "Zeroing"
                print(f"{action} positional embeddings by factor {alpha}...")

            # Save original positions to restore later
            position_embeddings_backup = {}
            method_backups = {}  # Separate backup for methods

            with torch.no_grad():
                # Try different possible locations for positional embeddings in various model architectures
                # For transformer-based models like GPT, DeepSeek, etc.
                pos_embedding_found = False

                # Check for wpe (word position embeddings) in transformer models
                if hasattr(model, "transformer") and hasattr(model.transformer, "wpe"):
                    position_embeddings_backup["transformer.wpe"] = (
                        model.transformer.wpe.weight.clone()
                    )
                    model.transformer.wpe.weight.mul_(alpha)
                    pos_embedding_found = True
                    if verbose:
                        print(
                            f"Scaled transformer.wpe positional embeddings by {alpha}"
                        )

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
                            method_backups[name] = (
                                module,
                                original_forward,
                            )  # Store module reference and original method
                            module.forward = scaled_rotary_forward
                            pos_embedding_found = True
                            if verbose:
                                print(
                                    f"Scaled rotary positional embeddings in {name} by {alpha}"
                                )

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
                                print(
                                    f"Scaled positional embeddings in {name} by {alpha}"
                                )

                if not pos_embedding_found and verbose:
                    print("Warning: Could not find positional embeddings to scale")
                    raise ValueError("Could not find positional embeddings to scale")
        except Exception as e:
            print(f"Error while scaling positional embeddings: {e}")
            print("Continuing with unmodified positional embeddings")
            position_embeddings_backup = {}
            method_backups = {}

    # Tokenize input
    inputs = tokenizer(cot_prompt, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    if verbose:
        print(f"Prompt: '{cot_prompt}'")
        print(f"Encoded to {input_ids.shape[1]} tokens")

    # Set up the response start index for steering
    response_start = input_ids.shape[1]

    # Initialize the steering vector applier if steering vectors are provided
    steerer = None
    if steering_vectors:
        if verbose:
            print(f"Using steering vectors for {len(steering_vectors)} layers")
        steerer = SteeringVectorApplier(
            steering_vectors=steering_vectors,
            response_start_idx=response_start,
            verbose=verbose,
        )
        steerer.register_hooks(model)

    # Generate the full response first (with steering if enabled)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*token_type_ids.*")
        warnings.filterwarnings("ignore", message=".*Sliding Window Attention.*")
        warnings.filterwarnings(
            "ignore", message=".*Setting `pad_token_id` to `eos_token_id`.*"
        )

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id,  # Explicitly set pad_token_id
            )

    # Clean up steering hooks if they were used
    if steerer:
        steerer.remove_hooks()

    # Decode response
    response = tokenizer.decode(
        output_ids[0][response_start:], skip_special_tokens=True
    )

    # Now collect activations for the full sequence, but with a clean model state
    # VERY IMPORTANT: Reset any cached states in the model
    for module in model.modules():
        if hasattr(module, "cache_present"):
            module.cache_present = None
        if hasattr(module, "past_key_values"):
            module.past_key_values = None

    # Set up attention mask for the full sequence
    full_attention_mask = torch.ones_like(output_ids)

    # Create a fresh collector
    collector = ActivationCollector(verbose=verbose, do_all=do_all, do_layers=do_layers)
    collector.register_hooks(model)

    # Use a single model forward call to collect both activations and attention weights
    logits = None
    try:
        with torch.no_grad():
            # Handle warning about scaled_dot_product_attention not supporting output_attentions
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message=".*does not support `output_attentions=True`.*"
                )
                outputs = model(
                    input_ids=output_ids,
                    attention_mask=full_attention_mask,
                    output_attentions=True,  # Enable attention weights collection
                    return_dict=True,
                    attn_implementation="eager",  # Force eager implementation for attention
                )

            # Store activations (already collected via hooks)
            activations = collector.activations

            # Store attention weights from the output
            if hasattr(outputs, "attentions") and outputs.attentions is not None:
                for layer_idx, attn_weights in enumerate(outputs.attentions):
                    if attn_layers is not None and layer_idx not in attn_layers:
                        continue
                    collector.attn_weights[layer_idx] = attn_weights.detach().cpu()
            # quit()
            # Store logits if requested
            if return_logits and hasattr(outputs, "logits"):
                logits = outputs.logits.detach().cpu()

    except Exception as e:
        print(f"WARNING: Error during data collection: {e}")
        # Try with a single token as fallback
        if verbose:
            print("Trying with just the first token...")
        with torch.no_grad():
            try:
                single_token = output_ids[:, :1]
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=".*does not support `output_attentions=True`.*",
                    )
                    outputs = model(
                        single_token,
                        output_attentions=True,
                        return_dict=True,
                        attn_implementation="eager",
                    )

                # Store whatever activations we got
                activations = collector.activations

                # Try to get attention weights if available
                if hasattr(outputs, "attentions") and outputs.attentions is not None:
                    for layer_idx, attn_weights in enumerate(outputs.attentions):
                        collector.attn_weights[layer_idx] = attn_weights.detach().cpu()

                # Try to get logits if requested and available
                if return_logits and hasattr(outputs, "logits"):
                    logits = outputs.logits.detach().cpu()

            except Exception as e:
                print(f"WARNING: Fallback also failed: {e}")
                activations = (
                    collector.activations
                )  # Use whatever we managed to collect

    # Clean up hooks after collecting everything
    collector.remove_hooks()

    # Clean up GPU memory after collecting activations
    if logits is not None:
        logits = logits.detach().cpu().numpy()
    torch.cuda.empty_cache()

    # Get token texts for all tokens
    all_tokens = output_ids[0].tolist()
    token_texts = tokenizer.convert_ids_to_tokens(all_tokens)

    cot_prompt_tokens = inputs.input_ids[0].tolist()
    cot_prompt_token_texts = tokenizer.convert_ids_to_tokens(cot_prompt_tokens)

    # Organize the results
    input_length = input_ids.shape[1]

    
    result = {
        "cot_prompt": cot_prompt, # input question as a plain string
        "cot_prompt_tokens": cot_prompt_tokens, # input prompt tokenized
        "cot_prompt_token_texts": cot_prompt_token_texts, # input prompt tokenized, converted to clean text (e.g., "Ġ" -> " ")
        "response": response, # output response as a plain string
        "response_start": response_start, # index of the first token in the response
        "tokens": all_tokens, # all tokens in the prompt + response concatenated
        "token_texts": token_texts, # all tokens in the prompt + response concatenated, converted to text (length is seq_len)
        "input_length": input_length, # length of the input question (number of tokens)
        "activations": activations, # residual stream activity. dict of {layer_idx: np.ndarray(batch_size, seq_len, hidden_dim)} # I think
                                    # I've only ever used batch size = 1, so that dimension is always 1
        "attention_weights": collector.attn_weights, # attention weights. dict of {layer_idx: np.ndarray(batch_size, num_heads, seq_len, seq_len)}
        "steering_applied": steering_vectors is not None, # related to optional steering
        "pos_embedding_scale": ( # related to optional positional embedding scaling (>1) or suppression (<1)
            pos_embedding_scale if pos_embedding_scale is not None else 1.0
        ),
    }

    # Add logits to result if available
    if logits is not None:
        result["logits"] = logits # logits. np.ndarray(batch_size, seq_len, vocab_size)

    # Final memory cleanup
    torch.cuda.empty_cache()

    return result


def get_token_logits_for_word(logits, word, model_name="qwen-14b"):  # no context
    """
    Get logits for each token in a word

    Args:
        model: The language model
        tokenizer: The tokenizer
        word: The word to get logits for
        context: Optional context to use for prediction

    Returns:
        A dictionary with token IDs, token texts, and corresponding logits
    """

    if "qwen" in model_name:
        _, tokenizer = get_deepseek_r1(model_name, float32=False)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Tokenize the word
    tokens = tokenizer.encode(word, add_special_tokens=False)
    # print(f'{word=}')
    assert len(tokens) == 1, f"Word {word} has {len(tokens)} tokens"
    word_logits = logits[0, :, tokens[0]]
    return word_logits


@pkld
def test_prompt(
    prompt,
    max_new_tokens=2000,
    seed=0,
    model_name="qwen-14b",
    temperature=0.6,
    steering_vectors=None,
    float32=False,
    pos_embedding_scale=None,
    do_layers=None,
    return_logits=True,
    attn_layers=None,
):
    # extra_prompt='Please end your response with: "ANSWER: <your percentage here>\n<think>\n'):
    np.random.seed(seed)
    torch.manual_seed(seed)  # Affects PyTorch CPU operations
    torch.cuda.manual_seed(seed)  # Affects PyTorch CUDA operations
    random.seed(seed)

    if "qwen" in model_name:
        model, tokenizer = get_deepseek_r1(model_name, float32=float32)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    print(f"Testing prompt: {prompt=}")
    t_st = time.time()
    result = generate_response_with_activations(
        model,
        tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        steering_vectors=steering_vectors,
        pos_embedding_scale=pos_embedding_scale,
        do_layers=do_layers,
        return_logits=return_logits,
        attn_layers=attn_layers,
    )
    t_end = time.time()
    num_tokens = len(result["token_texts"])
    print(f"Time taken: {t_end - t_st:.2f} seconds (for {num_tokens} tokens)")
    attention_data = result["attention_weights"][0]
    num_nans = np.isnan(attention_data).sum()
    p_nan = num_nans / attention_data.numel()
    assert ~np.isnan(
        attention_data
    ).any(), f"Attention has NaNs{attention_data.shape=} ({p_nan=:.1%})"
    print("\t*** Got result! ***")

    # Free up GPU memory
    del model
    torch.cuda.empty_cache()

    return result


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Create prompt from the provided context and question
    context = "If the farmer waters his crops daily, they will grow strong and healthy. If there is a drought, then the crops will dry up. Either the farmer waters the crops routinely, or the crops did not dry up. If the crops grow strong, there will be a bountiful harvest. If there is a large harvest, the farmer will make good profits."
    question = "If there was a drought this season, will the farmer make good profits?"

    prompt = f"Context: {context}\n\nQuestion: {question}\n\n<think>"

    # Run the model with logits
    results = test_prompt(
        prompt=prompt,
        max_new_tokens=3000,
        temperature=0.7,
        model_name="qwen-14b",
        return_logits=True,
        float32=True,
        do_layers=[0, 15, 47],
        attn_layers=[0, 15, 47],
    )

    # Extract the logits and token information
    logits = results["logits"]
    # print(f"{logits.shape=}")
    # quit()
    # print(logits)
    # quit()
    word_for_analysis = "wait"
    # word_for_analysis = " wait"
    token_texts = results["token_texts"]
    response_start = results["response_start"]
    wait_logits = get_token_logits_for_word(
        logits, word_for_analysis, model_name="qwen-15b"
    )

    print(results["response"])
    # quit()
    print(f"{wait_logits=}")
    # quit()

    # Plot the logits for "wait" over the response
    plt.figure(figsize=(12, 6))

    # Use numeric indices for x-axis
    x_indices = list(range(len(wait_logits)))
    plt.plot(x_indices, wait_logits, marker="o")

    plt.title(f'Logit values for "{word_for_analysis}" throughout response')
    plt.xlabel("Token Position Index")
    plt.ylabel("Logit Value")

    # Add a grid for better readability
    plt.grid(True, linestyle="--", alpha=0.7)

    # Annotate special tokens where "wait" appears
    for i, token in enumerate(token_texts):
        if token.lower() in ["ġwait", "wait"]:
            plt.axvline(x=i, color="r", linestyle="--", alpha=0.5)
            plt.annotate(
                "'wait' token",
                xy=(i, wait_logits[i]),
                xytext=(i + 5, wait_logits[i] + 1),
                arrowprops=dict(arrowstyle="->", color="red"),
            )

    plt.tight_layout()

    # Save the plot
    plt.savefig("wait_logits_plot.png")
    print(f"Plot saved as wait_logits_plot.png")

    # Optionally show the plot
    plt.show()
