"""Core context engine for managing and compressing conversation context."""

from dataclasses import dataclass, field
from typing import List, Optional

from .token_counter import count_tokens_by_model, truncate_to_token_limit


@dataclass
class ContextItem:
    """A single item in the context window."""

    role: str
    content: str
    token_count: int = 0
    priority: float = 1.0

    def __post_init__(self):
        if self.token_count == 0:
            self.token_count = count_tokens_by_model(self.content)


@dataclass
class ContextEngine:
    """Manages a context window with token budget awareness."""

    max_tokens: int = 8192
    model: str = "gpt-4"
    items: List[ContextItem] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return sum(item.token_count for item in self.items)

    @property
    def remaining_tokens(self) -> int:
        return self.max_tokens - self.total_tokens

    def add(self, role: str, content: str, priority: float = 1.0) -> ContextItem:
        """Add a new item to the context.

        If adding the item would exceed the token budget, items with the
        lowest priority are evicted first.

        Args:
            role: The role of the speaker (e.g., "user", "assistant", "system").
            content: The content of the message.
            priority: Priority score; higher values are kept longer.

        Returns:
            The created ContextItem.
        """
        item = ContextItem(role=role, content=content, priority=priority,
                           token_count=count_tokens_by_model(content, self.model))

        # Evict low-priority items if over budget
        while self.items and self.total_tokens + item.token_count > self.max_tokens:
            self._evict_lowest_priority()

        # If a single item is still too large, truncate it
        if item.token_count > self.max_tokens:
            item.content = truncate_to_token_limit(item.content, self.max_tokens, self.model)
            item.token_count = count_tokens_by_model(item.content, self.model)

        self.items.append(item)
        return item

    def _evict_lowest_priority(self) -> Optional[ContextItem]:
        """Remove the item with the lowest priority score.

        When two items share the same priority, the oldest one (lowest index)
        is evicted first (FIFO order).

        # TODO: Add a 'protected' flag to prevent system prompts from eviction.
        """
        if not self.items:
            return None
        # Use (priority, insertion_index) so ties are broken by age (oldest first)
        lowest = min(
            enumerate(self.items),
            key=lambda idx_item: (idx_item[1].priority, idx_item[0]),
        )
        idx, item = lowest
        del self.items[idx]
        return item

    def to_messages(self) -> List[dict]:
        """Return the context as a list of chat message dicts.

        # TODO: Add support for collapsing consecutive same-role messages.
        """
        return [{"role": item.role, "content": item.content} for item in self.items]

    def summarize(self) -> str:
        """Return a brief summary of the current context state."""
        return (
            f"ContextEngine(model={self.model}, "
            f"items={len(self.items)}, "
            f"tokens={self.total_tokens}/{self.max_tokens})"
        )
