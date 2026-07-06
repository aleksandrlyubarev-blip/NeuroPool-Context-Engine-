"""Value Locking (ТЗ п.4.6).

After Pass B the values {amount, deadline, sender} are frozen constants.
Every number/date appearing in the model's free text is checked against the
locked set. A mismatch triggers one regeneration of the field; a repeated
mismatch triggers a refusal. Card numbers/dates are rendered ONLY from the
verified JSON, never parsed out of free text.
"""

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

# Phone-like sequences are the only numbers allowed to differ from the locked
# set (letters legitimately tell you whom to call). Matched and removed before
# the numeric scan.
_PHONE_RE = re.compile(
    r"""(
        \*\d{3,6}                                   # short star numbers (*6050)
        | \+?972[-\s]?\d(?:[-\s]?\d){7,8}           # +972 numbers
        | 0\d{1,2}[-\s]?\d{3}[-\s]?\d{4}            # 02-123-4567 / 03 1234567
        | 0\d{2}[-\s]?\d{7}                         # 050-1234567
        | 1[-\s]?[78]00[-\s]?\d{2,3}[-\s]?\d{2,4}   # 1-700/1-800 hotlines
    )""",
    re.VERBOSE,
)

_DATE_PATTERNS = [
    re.compile(r"(?<!\d)(\d{1,2})[./](\d{1,2})[./](\d{4})(?!\d)"),  # DD.MM.YYYY
    re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)"),  # ISO YYYY-MM-DD
]

_RU_MONTHS = {
    "январ": 1, "феврал": 2, "март": 3, "апрел": 4, "ма": 5, "июн": 6,
    "июл": 7, "август": 8, "сентябр": 9, "октябр": 10, "ноябр": 11, "декабр": 12,
}
_RU_DATE_RE = re.compile(
    r"(?<!\d)(\d{1,2})\s+(январ[яе]|феврал[яе]|марта?|апрел[яе]|ма[яе]|июн[яе]|"
    r"июл[яе]|августа?|сентябр[яе]|октябр[яе]|ноябр[яе]|декабр[яе])\s*(\d{4})?",
    re.IGNORECASE,
)

_NUMBER_RE = re.compile(r"(?<![\d.,])(\d{1,3}(?:[ ,]\d{3})+|\d+)(?:[.,](\d{1,2}))?(?![\d])")


@dataclass(frozen=True)
class LockedValues:
    """Frozen constants after Pass B."""

    sender_ru: Optional[str]
    amount_value: Optional[float]
    deadline: Optional[str]  # ISO YYYY-MM-DD or None


@dataclass
class LockCheckResult:
    ok: bool
    violations: list[str] = field(default_factory=list)


def _parse_ru_month(word: str) -> Optional[int]:
    w = word.lower()
    for stem, month in _RU_MONTHS.items():
        if w.startswith(stem):
            return month
    return None


def extract_dates(text: str) -> set[str]:
    """All dates found in text, normalized to ISO YYYY-MM-DD."""
    found: set[str] = set()
    for m in _DATE_PATTERNS[0].finditer(text):
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            found.add(date(y, mo, d).isoformat())
        except ValueError:
            pass
    for m in _DATE_PATTERNS[1].finditer(text):
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            found.add(date(y, mo, d).isoformat())
        except ValueError:
            pass
    for m in _RU_DATE_RE.finditer(text):
        month = _parse_ru_month(m.group(2))
        if month is None or not m.group(3):
            continue
        try:
            found.add(date(int(m.group(3)), month, int(m.group(1))).isoformat())
        except ValueError:
            pass
    return found


def extract_numbers(text: str) -> set[float]:
    """All standalone numbers in text, with phone-like sequences and date
    expressions removed first."""
    cleaned = _PHONE_RE.sub(" ", text)
    for pat in _DATE_PATTERNS:
        cleaned = pat.sub(" ", cleaned)
    cleaned = _RU_DATE_RE.sub(" ", cleaned)
    numbers: set[float] = set()
    for m in _NUMBER_RE.finditer(cleaned):
        whole = m.group(1).replace(" ", "").replace(",", "")
        frac = m.group(2) or ""
        try:
            numbers.add(float(f"{whole}.{frac}" if frac else whole))
        except ValueError:
            pass
    return numbers


def _allowed_numbers(locked: LockedValues) -> set[float]:
    allowed: set[float] = set()
    if locked.amount_value is not None:
        allowed.add(float(locked.amount_value))
        # both "150" and "150.00" renderings of an integral amount are fine
        if float(locked.amount_value).is_integer():
            allowed.add(float(int(locked.amount_value)))
    if locked.deadline:
        try:
            y, mo, d = (int(p) for p in locked.deadline.split("-"))
            allowed.update({float(y), float(mo), float(d)})
        except ValueError:
            pass
    return allowed


def check_free_text(texts: list[str], locked: LockedValues) -> LockCheckResult:
    """Verify that free text contains no numbers/dates outside the locked set.

    Numbers 0-4 are allowed (list ordinals like 'шаг 2', 'в 2 экземплярах' are
    harmless); everything else must match the locked amount or deadline parts.
    """
    violations: list[str] = []
    allowed = _allowed_numbers(locked)
    for text in texts:
        if not text:
            continue
        for num in extract_numbers(text):
            if num in allowed or num < 5:
                continue
            violations.append(f"number {num:g} not in locked set")
        for iso in extract_dates(text):
            if locked.deadline and iso == locked.deadline:
                continue
            violations.append(f"date {iso} not in locked set")
    return LockCheckResult(ok=not violations, violations=violations)
