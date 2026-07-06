"""Output sanitizer for the LoFrayer invariant (ТЗ п.0, п.6.4).

The product explains letters and gives everyday steps. It NEVER suggests
legal remedies. The stop-list below is enforced in code, not only in the
prompt: any hit in recommended_actions_ru / demand_summary_ru is replaced
with a neutral step and flagged in metrics.
"""

import re
from dataclasses import dataclass, field

# Roots of forbidden formulations (п.6.4): «обжалуйте», «подайте ерур /
# ходатайство / возражение», «есть основания оспорить», «юридически» и однокоренные.
_STOP_ROOTS = [
    r"обжал\w*",        # обжаловать, обжалуйте, обжалование
    r"ерур\w*",         # ерур (ערר)
    r"ходатайств\w*",   # ходатайство, ходатайствовать
    r"возражени\w*",    # возражение, возражений
    r"оспор\w*",        # оспорить, оспаривание
    r"оспарива\w*",
    r"юридическ\w*",    # юридически, юридический
    r"апелляци\w*",     # апелляция — та же граница
    r"иск[ауеом]?\b",   # подать иск
]

_STOP_RE = re.compile("|".join(_STOP_ROOTS), re.IGNORECASE | re.UNICODE)

NEUTRAL_ACTION_RU = (
    "Уточните детали в приёмной организации-отправителя или по телефону, "
    "указанному в письме."
)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class SanitizeResult:
    demand_summary_ru: str
    recommended_actions_ru: list[str]
    flagged: bool = False
    hits: list[str] = field(default_factory=list)


def contains_stop_words(text: str) -> bool:
    return bool(text) and bool(_STOP_RE.search(text))


def sanitize_card_texts(
    demand_summary_ru: str, recommended_actions_ru: list[str]
) -> SanitizeResult:
    """Apply the stop-list. Offending action items are replaced with a neutral
    step; offending sentences in the summary are dropped (or the whole summary
    replaced if nothing safe remains)."""
    hits: list[str] = []

    actions: list[str] = []
    neutral_added = False
    for item in recommended_actions_ru:
        if contains_stop_words(item):
            hits.append(item)
            if not neutral_added:
                actions.append(NEUTRAL_ACTION_RU)
                neutral_added = True
        else:
            actions.append(item)

    summary = demand_summary_ru or ""
    if contains_stop_words(summary):
        kept = [
            s for s in _SENTENCE_SPLIT_RE.split(summary) if not contains_stop_words(s)
        ]
        hits.append(summary)
        summary = " ".join(kept).strip() or (
            "Отправитель обращается к вам по официальному вопросу — подробности "
            "уточните у отправителя."
        )

    return SanitizeResult(
        demand_summary_ru=summary,
        recommended_actions_ru=actions,
        flagged=bool(hits),
        hits=hits,
    )
