"""Token counting utilities for context management."""

import re
from typing import Union


def count_tokens_approx(text: str) -> int:
    """Approximate token count using word/punctuation splitting.

    Uses a heuristic: ~4 characters per token on average, with special handling
    for whitespace and punctuation boundaries.

    Args:
        text: The input text to count tokens for.

    Returns:
        Estimated token count.
    """
    if not text:
        return 0

    # Split on whitespace and punctuation boundaries
    tokens = re.findall(r"\w+|[^\w\s]", text)
    # Account for sub-word tokenization: long words are split more aggressively
    count = 0
    for token in tokens:
        if len(token) <= 4:
            count += 1
        else:
            # Roughly one token per 4 chars for longer words
            count += max(1, (len(token) + 3) // 4)
    return count


def truncate_to_token_limit(text: str, max_tokens: int) -> str:
    """Truncate text to fit within a token limit.

    Truncates at word boundaries to avoid cutting mid-word.

    Args:
        text: The input text to truncate.
        max_tokens: Maximum allowed token count.

    Returns:
        Truncated text that fits within the token limit.

    # TODO: Implement smart truncation that preserves the most relevant
    # sections (beginning and end) rather than just truncating at the end.
    # This is useful for long documents where context at both ends matters.
    """
    if count_tokens_approx(text) <= max_tokens:
        return text

    words = text.split()
    result = []
    current_tokens = 0

    for word in words:
        word_tokens = count_tokens_approx(word + " ")
        if current_tokens + word_tokens > max_tokens:
            break
        result.append(word)
        current_tokens += word_tokens

    return " ".join(result)


def split_into_chunks(text: str, chunk_size: int, overlap: int = 0) -> list[str]:
    """Split text into overlapping token-based chunks.

    Args:
        text: The input text to split.
        chunk_size: Maximum tokens per chunk.
        overlap: Number of tokens to overlap between consecutive chunks.

    Returns:
        List of text chunks.

    # TODO: Add support for splitting at sentence boundaries instead of
    # token boundaries to avoid cutting sentences in half.
    """
    if not text:
        return []

    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        chunk_words = []
        token_count = 0

        for i in range(start, len(words)):
            word_tokens = count_tokens_approx(words[i] + " ")
            if token_count + word_tokens > chunk_size and chunk_words:
                break
            chunk_words.append(words[i])
            token_count += word_tokens

        chunks.append(" ".join(chunk_words))

        if not chunk_words:
            break

        # Advance start, stepping back by overlap tokens
        advance = len(chunk_words)
        if overlap > 0:
            overlap_words = 0
            overlap_tokens = 0
            for word in reversed(chunk_words):
                overlap_tokens += count_tokens_approx(word + " ")
                if overlap_tokens >= overlap:
                    break
                overlap_words += 1
            advance = max(1, len(chunk_words) - overlap_words)

        start += advance

    return chunks
