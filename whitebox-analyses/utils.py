from functools import cache
import numpy as np

import torch
import torch.nn.functional as F
import hashlib
import json


def compute_autocorrelation(vector, max_lag):
    """
    Efficiently compute autocorrelation for lags 1 to max_lag.

    Args:
        vector: 1D numpy array or list containing the time series data
        max_lag: Maximum lag to compute autocorrelation for

    Returns:
        numpy array of autocorrelation values for lags 1 to max_lag
    """
    # Convert input to numpy array if it's not already
    x = np.asarray(vector)

    # Remove NaN values
    x = x[~np.isnan(x)]

    # Get length of array
    n = len(x)

    # Ensure max_lag is within valid range
    max_lag = min(max_lag, n - 1)

    # Center the data by subtracting the mean
    x_centered = x - np.mean(x)

    # Compute denominator (variance * n)
    denominator = np.sum(x_centered**2)

    # Initialize array to store results
    autocorr = np.zeros(max_lag)

    # Compute autocorrelation for each lag
    for lag in range(1, max_lag + 1):
        # Compute dot product of x[:-lag] and x[lag:]
        numerator = np.sum(x_centered[:-lag] * x_centered[lag:])
        # r, p = stats.spearmanr(x_centered[:-lag], x_centered[lag:], nan_policy='omit')
        autocorr[lag - 1] = numerator / denominator

    return autocorr


def compute_autocorrelation_fft(vector, max_lag):
    """
    Compute autocorrelation using FFT for improved efficiency with large arrays.

    Args:
        vector: 1D numpy array or list containing the time series data
        max_lag: Maximum lag to compute autocorrelation for

    Returns:
        numpy array of autocorrelation values for lags 1 to max_lag
    """
    # Convert input to numpy array if it's not already
    x = np.asarray(vector)

    # Remove NaN values
    x = x[~np.isnan(x)]

    # Get length of array
    n = len(x)

    # Ensure max_lag is within valid range
    max_lag = min(max_lag, n - 1)

    # Center the data by subtracting the mean
    x_centered = x - np.mean(x)

    # Compute autocorrelation using FFT
    # Pad with zeros to avoid circular correlation
    fft = np.fft.fft(np.concatenate([x_centered, np.zeros_like(x_centered)]))
    acf = np.fft.ifft(fft * np.conjugate(fft))

    # Normalize by the variance
    acf = acf[:n] / np.sum(x_centered**2)

    # Return real part of the result for lags 1 to max_lag
    return np.real(acf[1 : max_lag + 1])


@cache
def get_qwen_tokenizer(base_model):
    from transformers import AutoTokenizer

    if base_model:
        model_name = "Qwen/Qwen2.5-14B"
    else:
        model_name = r"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


@cache
def get_llama_tokenizer():
    from transformers import AutoTokenizer

    model_name = r"deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def get_tokenizer(model_name):
    if "qwen" in model_name:
        return get_qwen_tokenizer(model_name == "qwen-14b-base")
    elif "llama" in model_name.lower():
        return get_llama_tokenizer()
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_qwen_raw_tokens_(text, base_model):
    tokenizer = get_qwen_tokenizer(base_model)
    tokens_int = tokenizer.encode(text)
    tokens_words = tokenizer.convert_ids_to_tokens(tokens_int)
    return tokens_words


def get_llama_raw_tokens(text):
    tokenizer = get_llama_tokenizer()
    tokens_int = tokenizer.encode(text)
    tokens_words = tokenizer.convert_ids_to_tokens(tokens_int)
    return tokens_words


def get_raw_tokens(text, model_name):
    if "qwen" in model_name:
        return get_qwen_raw_tokens_(text, base_model=model_name == "qwen-14b-base")
    elif "llama" in model_name.lower():
        return get_llama_raw_tokens(text)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def qwen_tokens_to_clean(tokens, base_model):
    tokenizer = get_qwen_tokenizer(base_model)
    if isinstance(tokens[0], str):
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
    else:
        token_ids = tokens
    # Decode back to text
    clean_text = tokenizer.decode(token_ids)
    return clean_text


def llama_tokens_to_clean(tokens):
    tokenizer = get_llama_tokenizer()
    if isinstance(tokens[0], str):
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
    else:
        token_ids = tokens
    # Decode back to text
    clean_text = tokenizer.decode(token_ids)
    return clean_text


def tokens_to_clean(tokens, model_name):
    if "qwen" in model_name:
        return qwen_tokens_to_clean(tokens, model_name == "qwen-14b-base")
    elif "llama" in model_name.lower():
        return llama_tokens_to_clean(tokens)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_top_p_logits(logits_tensor, p):
    """
    Returns indices and logits for the top-p nucleus.
    Assumes logits_tensor is a 1D tensor.
    """
    probs = F.softmax(logits_tensor.float(), dim=-1)  # Use float32 for softmax stability
    probs_sorted, indices_sorted = torch.sort(probs, descending=True)
    probs_sum_cumulative = torch.cumsum(probs_sorted, dim=-1)

    # Find the index + 1 of the smallest set >= p
    # Using right=True finds the first index strictly > p
    # We add 1 to include that index itself.
    nucleus_index_plus_1 = torch.searchsorted(probs_sum_cumulative, p, right=False) + 1

    # Ensure at least one token is selected
    nucleus_index_plus_1 = max(1, nucleus_index_plus_1)

    indices_nucleus = indices_sorted[:nucleus_index_plus_1]
    logits_nucleus = logits_tensor[indices_nucleus]  # Get original logits

    return indices_nucleus, logits_nucleus


def hash_dict(dictionary):
    """
    Convert a dictionary to a deterministic hash string of 16 characters.

    Args:
        dictionary (dict): The dictionary to hash

    Returns:
        str: A 16-character hexadecimal hash string representation of the dictionary
    """
    # Sort the dictionary by keys to ensure deterministic output
    # Convert to a JSON string with sorted keys
    d_clean = {}
    for k, v in dictionary.items():
        if isinstance(v, list):
            d_clean[int(k)] = [int(i) for i in v]
        elif isinstance(v, np.ndarray):
            d_clean[int(k)] = [int(i) for i in v.tolist()]
        else:
            d_clean[int(k)] = v

    # print(d_clean)
    # quit()
    dict_str = json.dumps(d_clean, sort_keys=True)

    # Create a hash object using MD5 (faster than SHA-256 and we only need 16 chars)
    hash_obj = hashlib.md5(dict_str.encode("utf-8"))

    # Return the first 16 characters of the hexadecimal representation
    return hash_obj.hexdigest()[:16]


def print_gpu_memory_summary(label="Memory Usage"):
    """Print a concise summary of GPU memory usage"""
    if not torch.cuda.is_available():
        print(f"{label}: CUDA not available")
        return

    # Get statistics
    allocated = torch.cuda.memory_allocated() / 1024**3  # GB
    reserved = torch.cuda.memory_reserved() / 1024**3  # GB
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB

    # Print concise summary
    print(f"=== {label} ===")
    print(f"  Allocated: {allocated:.2f} GB")
    print(f"  Reserved:  {reserved:.2f} GB")
    print(f"  Max Used:  {max_allocated:.2f} GB")
    print(f"  Available: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")


@cache
def get_qwen_14b_tokenizer():
    from transformers import AutoTokenizer

    model_name = r"deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def get_qwen_14b_tokens_lower(text, lower=True):
    tokenizer = get_qwen_14b_tokenizer()
    tokens = tokenizer.encode(text)
    raw_words = tokenizer.convert_ids_to_tokens(tokens)
    # Clean the tokens
    cleaned_words = []
    for word in raw_words:
        if word.startswith("Ġ"):
            cleaned_word = word[1:]  # Remove the leading 'Ġ'
        # Add more cleaning rules if needed for other special tokens like 'Ċ' or others
        # elif word == 'Ċ':
        #     cleaned_word = '\\n' # Represent newline explicitly if desired
        else:
            cleaned_word = word
        # Handle potential empty strings after cleaning (if 'Ġ' was the whole token)
        if lower:
            cleaned_word = cleaned_word.lower()
        if cleaned_word:
            cleaned_words.append(cleaned_word)
        elif word == " ":  # Keep actual spaces if they are tokenized separately
            cleaned_words.append(" ")

    return cleaned_words


def model2layers_heads(model_name):
    if model_name == "qwen-14b":
        return 48, 40
    elif model_name == "qwen-14b-base":
        return 48, 40
    elif model_name == "qwen-15b":
        return 28, 12
    elif model_name == "llama8":
        return 32, 32
    elif model_name == "llama8-base":
        return 32, 32
    else:
        raise ValueError(f"Unknown model: {model_name}")
