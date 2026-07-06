"""PII masking before any logging (ТЗ п.5 — архитектурный инвариант).

Everything that goes to a logger passes through mask_pii(). Masks:
- Israeli ID numbers (ת"ז): 9 digits validated by the official checksum;
- phone numbers (Israeli formats and +972);
- email addresses.

Sender names are never logged at all — the log schema only contains
request_id, doc_type, latency_ms, tokens, verdicts, refusal_flag.
"""

import re

_ID_CANDIDATE_RE = re.compile(r"(?<!\d)\d{9}(?!\d)")
_PHONE_RE = re.compile(
    r"""(
        \+?972[-\s]?\d(?:[-\s]?\d){7,8}
        | 0\d{1,2}[-\s]?\d{3}[-\s]?\d{4}
        | 0\d{2}[-\s]?\d{7}
        | 1[-\s]?[78]00[-\s]?\d{2,3}[-\s]?\d{2,4}
        | \*\d{3,6}
    )""",
    re.VERBOSE,
)
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

MASK = "***"


def is_valid_israeli_id(digits: str) -> bool:
    """Official ת"ז checksum: alternate weights 1/2, sum digits of products,
    total must be divisible by 10."""
    if len(digits) != 9 or not digits.isdigit():
        return False
    total = 0
    for i, ch in enumerate(digits):
        product = int(ch) * (1 if i % 2 == 0 else 2)
        total += product if product < 10 else product - 9
    return total % 10 == 0


def mask_pii(text: str) -> str:
    if not text:
        return text
    masked = _EMAIL_RE.sub(MASK, text)
    masked = _PHONE_RE.sub(MASK, masked)

    def _mask_id(m: re.Match) -> str:
        return MASK if is_valid_israeli_id(m.group(0)) else m.group(0)

    return _ID_CANDIDATE_RE.sub(_mask_id, masked)
