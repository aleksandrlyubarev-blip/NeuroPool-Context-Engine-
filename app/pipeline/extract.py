"""Two-pass extraction pipeline (ТЗ п.4): Pass A generator → Pass B critic →
decision policy → value locking → output sanitizer.

The MC ensemble (п.4.5) is intentionally NOT implemented in v1.0 — it is a
contingent layer with an explicit trigger (golden-set thresholds failing or
refusal_rate > 25%). See docs/implementation-notes.md.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Union

from app.config import get_settings
from app.pipeline import prompts
from app.pipeline.locking import LockedValues, check_free_text
from app.pipeline.sanitizer import sanitize_card_texts
from app.pipeline.schemas import (
    Card,
    CriticResult,
    FreeTextResult,
    PassAResult,
    Refusal,
    Verdict,
)

logger = logging.getLogger("polingvo.pipeline")

REFUSAL_UNRELIABLE_RU = (
    "Не удалось надёжно извлечь ключевые данные. Покажите письмо человеку — "
    "не полагайтесь на автоматический разбор."
)
REFUSAL_NOT_A_LETTER_RU = (
    "Похоже, это не официальное письмо. Загрузите фото письма от организации "
    "(ирия, битуах леуми, банк, купат холим и т.п.)."
)
REFUSAL_UNREADABLE_RU = (
    "Текст на фото не читается. Сфотографируйте письмо при хорошем свете, "
    "целиком и без бликов, и попробуйте ещё раз."
)

NOT_FOUND_RU = "в письме не указано"

CRITICAL_FIELDS = ("sender", "amount", "deadline")


@dataclass
class PipelineOutcome:
    result: Union[Card, Refusal]
    verdicts: dict[str, str]
    doc_type_ru: Optional[str]
    tokens_in: int = 0
    tokens_out: int = 0
    lock_regenerated: bool = False
    sanitizer_flagged: bool = False

    @property
    def is_refusal(self) -> bool:
        return isinstance(self.result, Refusal)


def decide(
    pass_a: PassAResult, critic: CriticResult, confidence_threshold: float = 0.7
) -> Optional[Refusal]:
    """Decision policy (ТЗ п.4.3). Returns a Refusal or None (→ card).

    - mismatch on amount or deadline → refusal, no guessing;
    - mismatch on sender → refusal (the card is meaningless with a wrong sender);
    - not_found → the field is rendered as «в письме не указано», not a refusal;
    - confirm with confidence < threshold on a critical field → refusal.
    """
    checks = {"sender": critic.sender, "amount": critic.amount, "deadline": critic.deadline}
    confidences = {
        "sender": pass_a.confidence_sender,
        "amount": pass_a.confidence_amount,
        "deadline": pass_a.confidence_deadline,
    }
    for name, check in checks.items():
        if check.verdict == Verdict.MISMATCH:
            return Refusal(
                reason_code=f"mismatch_{name}", reason_ru=REFUSAL_UNRELIABLE_RU
            )
        if check.verdict == Verdict.CONFIRM and confidences[name] < confidence_threshold:
            return Refusal(
                reason_code=f"low_confidence_{name}", reason_ru=REFUSAL_UNRELIABLE_RU
            )
    return None


def analyze_letter(image_bytes: bytes, mime_type: str, client=None) -> PipelineOutcome:
    """Run the full pipeline over an in-memory image. The bytes are never
    written to disk (ТЗ п.5)."""
    settings = get_settings()
    if client is None:
        from app.services.gemini import GeminiClient

        client = GeminiClient()

    tokens_in = tokens_out = 0

    def _track():
        nonlocal tokens_in, tokens_out
        usage = getattr(client, "last_usage", {}) or {}
        tokens_in += usage.get("tokens_in", 0)
        tokens_out += usage.get("tokens_out", 0)

    # ---- Pass A: generator (structured output, T=0) ----
    pass_a: PassAResult = client.generate_structured(
        system=prompts.PASS_A_SYSTEM,
        schema=PassAResult,
        image_bytes=image_bytes,
        image_mime=mime_type,
        temperature=0.0,
    )
    _track()

    if not pass_a.is_official_letter:
        return PipelineOutcome(
            result=Refusal(reason_code="not_a_letter", reason_ru=REFUSAL_NOT_A_LETTER_RU),
            verdicts={}, doc_type_ru=None, tokens_in=tokens_in, tokens_out=tokens_out,
        )
    if not pass_a.readable:
        return PipelineOutcome(
            result=Refusal(reason_code="unreadable", reason_ru=REFUSAL_UNREADABLE_RU),
            verdicts={}, doc_type_ru=pass_a.doc_type_ru,
            tokens_in=tokens_in, tokens_out=tokens_out,
        )

    # ---- Pass B: critic (independent call, T=0) ----
    critic: CriticResult = client.generate_structured(
        system=prompts.PASS_B_SYSTEM,
        schema=CriticResult,
        text=prompts.pass_b_user_text(pass_a.model_dump_json()),
        image_bytes=image_bytes,
        image_mime=mime_type,
        temperature=0.0,
    )
    _track()

    verdicts = {
        "sender": critic.sender.verdict.value,
        "amount": critic.amount.verdict.value,
        "deadline": critic.deadline.verdict.value,
    }

    refusal = decide(pass_a, critic, settings.confidence_threshold)
    if refusal is not None:
        return PipelineOutcome(
            result=refusal, verdicts=verdicts, doc_type_ru=pass_a.doc_type_ru,
            tokens_in=tokens_in, tokens_out=tokens_out,
        )

    # ---- Value locking (п.4.6): freeze constants, render only from JSON ----
    not_found = []
    sender_ru = pass_a.sender_ru
    amount_value = pass_a.amount_value
    deadline = pass_a.deadline
    if critic.sender.verdict == Verdict.NOT_FOUND:
        sender_ru, not_found = None, not_found + ["sender"]
    if critic.amount.verdict == Verdict.NOT_FOUND:
        amount_value = None
        not_found.append("amount")
    if critic.deadline.verdict == Verdict.NOT_FOUND:
        deadline = None
        not_found.append("deadline")

    locked = LockedValues(
        sender_ru=sender_ru, amount_value=amount_value, deadline=deadline
    )

    summary = pass_a.demand_summary_ru or ""
    consequences = pass_a.consequences_ru or ""
    actions = list(pass_a.recommended_actions_ru)
    lock_regenerated = False

    check = check_free_text([summary, consequences, *actions], locked)
    if not check.ok:
        # One regeneration of the free-text fields with frozen constants in
        # XML tags; a second violation → refusal (п.4.6).
        logger.info("value-lock violation, regenerating free text: %s", check.violations)
        lock_regenerated = True
        regen: FreeTextResult = client.generate_structured(
            system=prompts.FREE_TEXT_SYSTEM,
            schema=FreeTextResult,
            text=prompts.free_text_user_text(
                sender_ru=sender_ru or NOT_FOUND_RU,
                amount=(f"{amount_value:g} ILS" if amount_value is not None else NOT_FOUND_RU),
                deadline=deadline or NOT_FOUND_RU,
                doc_type_ru=pass_a.doc_type_ru or "официальное письмо",
                context_summary=pass_a.demand_summary_ru or "",
            ),
            temperature=0.0,
        )
        _track()
        summary, consequences, actions = (
            regen.demand_summary_ru,
            regen.consequences_ru,
            regen.recommended_actions_ru,
        )
        recheck = check_free_text([summary, consequences, *actions], locked)
        if not recheck.ok:
            return PipelineOutcome(
                result=Refusal(
                    reason_code="value_lock_violation", reason_ru=REFUSAL_UNRELIABLE_RU
                ),
                verdicts=verdicts, doc_type_ru=pass_a.doc_type_ru,
                tokens_in=tokens_in, tokens_out=tokens_out, lock_regenerated=True,
            )

    # ---- Output sanitizer (п.6.4): LoFrayer invariant enforced in code ----
    sanitized = sanitize_card_texts(summary, actions)

    card = Card(
        sender_ru=sender_ru or NOT_FOUND_RU,
        sender_he=pass_a.sender_he,
        doc_type_ru=pass_a.doc_type_ru or "официальное письмо",
        demand_summary_ru=sanitized.demand_summary_ru,
        amount_value=amount_value,
        amount_currency=pass_a.amount_currency if amount_value is not None else None,
        deadline=deadline,
        consequences_ru=consequences or None,
        recommended_actions_ru=sanitized.recommended_actions_ru,
        confidence={
            "sender": pass_a.confidence_sender,
            "amount": pass_a.confidence_amount,
            "deadline": pass_a.confidence_deadline,
        },
        not_found_fields=not_found,
    )
    return PipelineOutcome(
        result=card, verdicts=verdicts, doc_type_ru=pass_a.doc_type_ru,
        tokens_in=tokens_in, tokens_out=tokens_out,
        lock_regenerated=lock_regenerated, sanitizer_flagged=sanitized.flagged,
    )
