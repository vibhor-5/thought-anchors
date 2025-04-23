import re
from typing import List, Tuple
from transformers import AutoTokenizer

def get_chunk_ranges(full_text: str, chunks: List[str]) -> List[Tuple[int, int]]:    
    # Get character ranges for each chunk in the full text
    chunk_ranges = []
    current_pos = 0
    
    for chunk in chunks:
        # Normalize the chunk for comparison (preserve length but standardize whitespace)
        normalized_chunk = re.sub(r'\s+', ' ', chunk).strip()
        
        # Try to find the chunk in the full text
        chunk_start = -1
        
        # First try exact match from current position
        exact_match_pos = full_text.find(chunk, current_pos)
        if exact_match_pos != -1:
            chunk_start = exact_match_pos
        else:
            # If exact match fails, try with normalized text
            chunk_words = normalized_chunk.split()
            
            # Search for the sequence of words, allowing for different whitespace
            for i in range(current_pos, len(full_text) - len(normalized_chunk)):
                # Check if this could be the start of our chunk
                text_window = full_text[i:i+len(normalized_chunk) + 20]  # Add some buffer
                normalized_window = re.sub(r'\s+', ' ', text_window).strip()
                
                if normalized_window.startswith(normalized_chunk):
                    chunk_start = i
                    break
                
                # If not found with window, try word by word matching
                if i == current_pos + 100:  # Limit detailed search to avoid performance issues
                    for j in range(current_pos, len(full_text) - 10):
                        # Try to match first word
                        if re.match(r'\b' + re.escape(chunk_words[0]) + r'\b', full_text[j:j+len(chunk_words[0])+5]):
                            # Check if subsequent words match
                            match_text = full_text[j:j+len(normalized_chunk)+30]
                            normalized_match = re.sub(r'\s+', ' ', match_text).strip()
                            if normalized_match.startswith(normalized_chunk):
                                chunk_start = j
                                break
                    break
        
        if chunk_start == -1:
            print(f"Warning: Chunk not found in full text: {chunk[:50]}...")
            continue
            
        # For the end position, find where the content of the chunk ends in the full text
        chunk_content = re.sub(r'\s+', '', chunk)  # Remove all whitespace
        full_text_from_start = full_text[chunk_start:]
        full_text_content = re.sub(r'\s+', '', full_text_from_start[:len(chunk) + 50])  # Remove all whitespace
        
        # Find how many characters of content match
        content_match_len = 0
        for i in range(min(len(chunk_content), len(full_text_content))):
            if chunk_content[i] == full_text_content[i]:
                content_match_len += 1
            else:
                break
        
        # Map content length back to original text with whitespace
        chunk_end = chunk_start
        content_chars_matched = 0
        for i in range(len(full_text_from_start)):
            if chunk_end + i >= len(full_text):
                break
            if not full_text[chunk_start + i].isspace():
                content_chars_matched += 1
            if content_chars_matched > content_match_len:
                break
            chunk_end = chunk_start + i
        
        chunk_end += 1  # Include the last character
        current_pos = chunk_end
        
        chunk_ranges.append((chunk_start, chunk_end))
        
    return chunk_ranges

def get_chunk_token_ranges(text: str, chunk_ranges: List[Tuple[int, int]], tokenizer: AutoTokenizer) -> List[Tuple[int, int]]:
    """Convert character positions to token indices"""
    chunk_token_ranges = []
    
    for (chunk_start, chunk_end) in chunk_ranges:        
        chunk_start_token = tokenizer.encode(text[:chunk_start], add_special_tokens=False)
        chunk_start_token_idx = len(chunk_start_token)
        chunk_end_token = tokenizer.encode(text[:chunk_end], add_special_tokens=False)
        chunk_end_token_idx = len(chunk_end_token)
        chunk_token_ranges.append((chunk_start_token_idx, chunk_end_token_idx))
        
    return chunk_token_ranges

def extract_boxed_answers(text: str) -> List[str]:
    """
    Extract answers enclosed in \boxed{} from the text with improved handling
    of nested braces and complex LaTeX expressions.
    
    Args:
        text: The text to extract boxed answers from
        
    Returns:
        List of extracted boxed answers
    """
    # Find all occurrences of \boxed{
    boxed_starts = [m.start() for m in re.finditer(r'\\boxed\{', text)]
    
    if not boxed_starts:
        return ['']
    
    answers = []
    
    for start_idx in boxed_starts:
        # Start after \boxed{
        idx = start_idx + 7
        brace_count = 1  # We've already opened one brace
        answer = ""
        
        # Parse until we find the matching closing brace
        while idx < len(text) and brace_count > 0:
            char = text[idx]
            
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                
                # Skip the closing brace of \boxed{}
                if brace_count == 0:
                    break
            
            if brace_count > 0:  # Only add if we're still inside the boxed content
                answer += char
            
            idx += 1
        
        if answer:
            answers.append(answer)
    
    return answers if answers else ['']

def check_answer(answer: str, gt_answer: str) -> bool:
    """
    Check if the generated answer matches the ground truth answer
    after normalizing LaTeX formatting.
    
    Args:
        answer: The generated answer to check
        gt_answer: The ground truth answer to compare against
        
    Returns:
        True if the answers match after normalization, False otherwise
    """
    # Normalize both answers
    normalized_answer = normalize_latex(answer)
    normalized_gt_answer = normalize_latex(gt_answer)
    
    return normalized_answer == normalized_gt_answer

def normalize_latex(latex_str: str) -> str:
    """
    Normalize LaTeX string by applying various transformations.
    
    Args:
        latex_str: The LaTeX string to normalize
        
    Returns:
        Normalized LaTeX string
    """
    normalized = latex_str.strip().lower()
    
    # Replace different fraction notations
    normalized = normalized.replace("dfrac", "frac")
    normalized = normalized.replace("tfrac", "frac")
    
    # Normalize spaces
    normalized = re.sub(r'\s+', '', normalized)
    
    # Normalize funny commas
    normalized = normalized.replace("{,}", "")
    
    # Normalize common mathematical notations
    normalized = normalized.replace("\\times", "*")
    normalized = normalized.replace("\\cdot", "*")
    
    # Normalize decimal representation
    normalized = re.sub(r'(\d+)[\.,](\d+)', r'\1.\2', normalized)
    
    # Remove unnecessary braces in simple expressions
    normalized = re.sub(r'{([^{}]+)}', r'\1', normalized)
    
    # Normalize common constants
    normalized = normalized.replace("\\pi", "pi")
    
    return normalized