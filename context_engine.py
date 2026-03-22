"""
NeuroPool Context Engine
CLI agent for more effective token use.

Manages context windows intelligently by scoring, pruning, and summarizing
conversation history to maximize useful information within token limits.
"""

import json
import math
import re
import argparse
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

try:
    import tiktoken
    _encoder = tiktoken.get_encoding("cl100k_base")
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False


@dataclass
class Message:
    role: str
    content: str
    token_count: int = 0
    importance_score: float = 1.0

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "token_count": self.token_count,
            "importance_score": self.importance_score,
        }


@dataclass
class ContextWindow:
    max_tokens: int
    messages: list[Message] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return sum(m.token_count for m in self.messages)

    @property
    def remaining_tokens(self) -> int:
        return self.max_tokens - self.total_tokens

    def add_message(self, message: Message) -> bool:
        """Add a message if it fits within the token budget. Returns True if added."""
        if message.token_count <= self.remaining_tokens:
            self.messages.append(message)
            return True
        return False

    def to_dict(self) -> dict:
        return {
            "max_tokens": self.max_tokens,
            "total_tokens": self.total_tokens,
            "remaining_tokens": self.remaining_tokens,
            "messages": [m.to_dict() for m in self.messages],
        }


def estimate_tokens(text: str) -> int:
    """Count tokens in a string using tiktoken (cl100k_base encoding).

    Falls back to the ~4 chars/token heuristic when tiktoken is not installed.
    cl100k_base is used by GPT-4 and Claude-compatible tokenizers and gives
    accurate counts for most Latin-script text.
    """
    if _TIKTOKEN_AVAILABLE:
        return len(_encoder.encode(text))
    return max(1, len(text) // 4)


_STOP_WORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would could should may might must shall can i you he she it "
    "we they them their this that these those of in on at to for with "
    "by from up about into through during before after above below and "
    "or but if so then than because as until while nor not no yes".split()
)


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, remove stop words."""
    words = re.findall(r"[a-z]+", text.lower())
    return [w for w in words if w not in _STOP_WORDS and len(w) > 1]


def _tfidf_vector(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    tf = Counter(tokens)
    total = max(len(tokens), 1)
    return {t: (count / total) * idf.get(t, 1.0) for t, count in tf.items()}


def _cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(a.get(t, 0.0) * b.get(t, 0.0) for t in b)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def build_idf(all_tokens: list[list[str]]) -> dict[str, float]:
    """Compute IDF weights across a corpus of token lists."""
    N = len(all_tokens)
    df: Counter = Counter()
    for tokens in all_tokens:
        df.update(set(tokens))
    return {term: math.log((N + 1) / (count + 1)) + 1.0 for term, count in df.items()}


def score_message_importance(
    message: Message,
    position: int,
    total: int,
    query_vector: Optional[dict[str, float]] = None,
    idf: Optional[dict[str, float]] = None,
) -> float:
    """
    Score message importance on a 0-1 scale.

    Factors considered:
    - Recency: more recent messages score higher
    - Role: system messages are most important, then assistant, then user
    - Position: first message (often system prompt) gets a boost
    - Semantic relevance: TF-IDF cosine similarity to the most recent user query
    """
    recency_score = position / max(total - 1, 1)

    role_weights = {"system": 1.0, "assistant": 0.8, "user": 0.6}
    role_score = role_weights.get(message.role, 0.5)

    first_message_boost = 0.2 if position == 0 else 0.0

    if query_vector and idf:
        tokens = _tokenize(message.content)
        msg_vector = _tfidf_vector(tokens, idf)
        relevance_score = _cosine_similarity(msg_vector, query_vector)
    else:
        relevance_score = 0.0

    return min(
        1.0,
        (recency_score * 0.35)
        + (role_score * 0.25)
        + (relevance_score * 0.2)
        + first_message_boost
        + 0.1,
    )


def _extractive_summary(messages: list[Message], max_summary_tokens: int) -> Optional[Message]:
    """
    Build an extractive summary of *messages* that fits within max_summary_tokens.

    Strategy: rank sentences across all messages by their TF-IDF score against
    the joint corpus, then greedily add the top sentences until the budget is full.
    The result is injected as a single synthetic 'system' message so the model
    understands it represents compressed prior context.
    """
    if not messages:
        return None

    # Collect all sentences with their source role
    sentences: list[tuple[str, str]] = []
    for msg in messages:
        for sent in re.split(r"(?<=[.!?])\s+", msg.content.strip()):
            sent = sent.strip()
            if sent:
                sentences.append((sent, msg.role))

    if not sentences:
        return None

    # Build IDF over sentence-level corpus
    sent_tokens = [_tokenize(s) for s, _ in sentences]
    idf = build_idf(sent_tokens)

    # Score each sentence by its mean TF-IDF weight (higher = more informative)
    scored_sents: list[tuple[float, str]] = []
    for tokens, (sent, _) in zip(sent_tokens, sentences):
        if not tokens:
            scored_sents.append((0.0, sent))
            continue
        vec = _tfidf_vector(tokens, idf)
        score = sum(vec.values()) / len(tokens)
        scored_sents.append((score, sent))

    scored_sents.sort(key=lambda x: x[0], reverse=True)

    # Greedily fill the summary budget
    chosen: list[str] = []
    used = estimate_tokens("[Summary of earlier context] ")
    for _, sent in scored_sents:
        t = estimate_tokens(sent + " ")
        if used + t <= max_summary_tokens:
            chosen.append(sent)
            used += t

    if not chosen:
        return None

    summary_text = "[Prior context] " + " ".join(chosen)
    return Message(
        role="system",
        content=summary_text,
        token_count=estimate_tokens(summary_text),
        importance_score=1.0,
    )


def prune_context(
    window: ContextWindow,
    target_tokens: int,
    summarize: bool = False,
    summary_ratio: float = 0.15,
) -> ContextWindow:
    """
    Prune the context window to fit within target_tokens.

    Importance is scored using recency, role weight, and TF-IDF cosine
    similarity to the most recent user message as the relevance query.

    If summarize=True, dropped messages are extractively summarized and
    injected as a synthetic system message instead of being discarded.
    summary_ratio controls what fraction of the target budget the summary
    may consume (default 15%).

    Returns a new ContextWindow with pruned (and optionally summarized) messages.
    """
    if window.total_tokens <= target_tokens:
        return window

    pruned = ContextWindow(max_tokens=window.max_tokens)
    msgs = window.messages
    total = len(msgs)

    # Build IDF over the entire conversation corpus
    all_tokens = [_tokenize(m.content) for m in msgs]
    idf = build_idf(all_tokens)

    # Use the most recent user message as the relevance query
    query_vector: Optional[dict[str, float]] = None
    for msg in reversed(msgs):
        if msg.role == "user":
            tokens = _tokenize(msg.content)
            query_vector = _tfidf_vector(tokens, idf)
            break

    # Score all messages and write scores back onto the objects
    scored = []
    for i, msg in enumerate(msgs):
        s = score_message_importance(msg, i, total, query_vector, idf)
        msg.importance_score = s
        scored.append((s, msg))

    # When summarizing, reserve a slice of the budget for the summary upfront
    # so kept messages don't crowd it out.
    summary_budget = max(25, int(target_tokens * summary_ratio)) if summarize else 0
    msg_budget = target_tokens - summary_budget

    # Keep highest-importance messages that fit within the (possibly reduced) budget
    scored.sort(key=lambda x: x[0], reverse=True)
    kept: set[int] = set()
    for score, msg in scored:
        if pruned.total_tokens + msg.token_count <= msg_budget:
            pruned.messages.append(msg)
            kept.add(id(msg))

    # Build and inject summary of dropped messages
    if summarize:
        dropped = [m for m in msgs if id(m) not in kept and m.role != "system"]
        if dropped:
            summary_msg = _extractive_summary(dropped, summary_budget)
            if summary_msg:
                pruned.messages.append(summary_msg)

    # Restore original conversation order (summary goes at position 1, after system)
    system_msgs = [m for m in pruned.messages if m.role == "system" and m.content.startswith("[Prior context]")]
    other_msgs = [m for m in pruned.messages if m not in system_msgs]
    other_msgs.sort(key=lambda m: msgs.index(m) if m in msgs else -1)

    first_system = next((m for m in other_msgs if m.role == "system"), None)
    if system_msgs and first_system:
        insert_at = other_msgs.index(first_system) + 1
        other_msgs[insert_at:insert_at] = system_msgs
        pruned.messages = other_msgs
    else:
        pruned.messages = system_msgs + other_msgs

    return pruned


def build_context(
    messages: list[dict],
    max_tokens: int,
    reserve_tokens: int = 500,
    summarize: bool = False,
) -> ContextWindow:
    """
    Build a context window from a list of message dicts, respecting the token budget.

    All messages are scored first; if the total exceeds the budget the
    importance-based pruner selects which to keep (rather than naively
    dropping messages that happen to arrive last).

    Args:
        messages: List of {"role": str, "content": str} dicts
        max_tokens: Maximum tokens for the context window
        reserve_tokens: Tokens to reserve for the model's response
        summarize: If True, inject an extractive summary of dropped messages
    """
    usable_tokens = max_tokens - reserve_tokens

    # Build a full window ignoring the budget so pruner can score everything
    full_window = ContextWindow(max_tokens=usable_tokens)
    for msg in messages:
        content = msg.get("content", "")
        role = msg.get("role", "user")
        token_count = estimate_tokens(content)
        full_window.messages.append(Message(role=role, content=content, token_count=token_count))

    return prune_context(full_window, usable_tokens, summarize=summarize)


def main():
    parser = argparse.ArgumentParser(
        description="NeuroPool Context Engine — manage token context windows effectively"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum token budget for the context window (default: 4096)",
    )
    parser.add_argument(
        "--reserve",
        type=int,
        default=500,
        help="Tokens to reserve for model response (default: 500)",
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Path to JSON file containing messages array",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to write pruned context JSON (default: stdout)",
    )
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Inject an extractive summary of dropped messages instead of discarding them",
    )
    args = parser.parse_args()

    if args.input:
        with open(args.input) as f:
            data = json.load(f)
        messages = data if isinstance(data, list) else data.get("messages", [])
    else:
        # Demo with sample messages
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, can you help me with Python?"},
            {"role": "assistant", "content": "Of course! What would you like to know about Python?"},
            {"role": "user", "content": "How do I read a file?"},
            {"role": "assistant", "content": "You can read a file using open(): with open('file.txt') as f: content = f.read()"},
        ]

    window = build_context(messages, args.max_tokens, args.reserve, summarize=args.summarize)
    result = window.to_dict()

    output_json = json.dumps(result, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output_json)
        print(f"Context written to {args.output}")
    else:
        print(output_json)


if __name__ == "__main__":
    main()
