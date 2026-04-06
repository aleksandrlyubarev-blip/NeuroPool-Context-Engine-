"""Token counting utilities for context management."""

import re


def count_tokens_approximate(text: str) -> int:
    """Approximate token count using whitespace and punctuation splitting.

    This is a rough heuristic: ~1 token per 4 characters or per word,
    whichever gives a higher count.

    Args:
        text: The text to count tokens for.

    Returns:
        Approximate number of tokens.

    # TODO: Replace with tiktoken-based exact counting for OpenAI-compatible models.
    # TODO: Add support for model-specific tokenizers (e.g., cl100k_base for GPT-4).
    """
    if not text:
        return 0

    # Word-based estimate
    words = len(re.findall(r"\S+", text))

    # Character-based estimate (GPT tokenizers average ~4 chars/token)
    char_estimate = len(text) // 4

    return max(words, char_estimate)


def count_tokens_by_model(text: str, model: str = "gpt-4") -> int:
    """Count tokens for a given model.

    # TODO: Implement model-specific tokenization using the tiktoken library.
    #       Supported models should include: gpt-4, gpt-3.5-turbo, claude-3, etc.

    Args:
        text: The text to count tokens for.
        model: The model name to use for tokenization.

    Returns:
        Token count for the given model.
    """
    # Fallback to approximate counting until model-specific tokenizers are added
    return count_tokens_approximate(text)


def truncate_to_token_limit(text: str, max_tokens: int, model: str = "gpt-4") -> str:
    """Truncate text to fit within a token limit.

    # TODO: Implement smarter truncation that preserves sentence boundaries.

    Args:
        text: The text to truncate.
        max_tokens: Maximum allowed token count.
        model: The model name for token counting.

    Returns:
        Truncated text that fits within the token limit.
    """
    current_tokens = count_tokens_by_model(text, model)
    if current_tokens <= max_tokens:
        return text

    # Estimate the fraction to keep
    ratio = max_tokens / current_tokens
    cutoff = int(len(text) * ratio)
    return text[:cutoff]
