"""Pydantic schemas for the two-pass extraction pipeline (ТЗ п.4).

Pass A (Generator) extracts a structured card draft from the letter image.
Pass B (Critic) verifies only the critical fields: sender, amount, deadline.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class PassAResult(BaseModel):
    """Structured output of Pass A. temperature=0, response_schema enforced."""

    is_official_letter: bool = Field(
        description="True if the image is an official letter/notice from an "
        "organization (municipality, Bituach Leumi, bank, kupat holim, utility, "
        "etc.). False for photos of unrelated objects."
    )
    readable: bool = Field(
        description="True if the text is legible enough to extract data reliably."
    )
    sender_he: Optional[str] = Field(
        default=None, description="Sender organization name exactly as printed (Hebrew)."
    )
    sender_ru: Optional[str] = Field(
        default=None, description="Sender organization name translated to Russian."
    )
    doc_type_ru: Optional[str] = Field(
        default=None,
        description="Document type in Russian, e.g. 'требование об оплате', "
        "'уведомление', 'напоминание о задолженности'.",
    )
    demand_summary_ru: Optional[str] = Field(
        default=None,
        description="1-3 sentences in Russian: what exactly is demanded/communicated.",
    )
    amount_value: Optional[float] = Field(
        default=None,
        description="The main amount demanded, as a number. null if no amount in the letter.",
    )
    amount_currency: Optional[str] = Field(
        default=None, description="Currency code, normally ILS. null if no amount."
    )
    deadline: Optional[str] = Field(
        default=None,
        description="The main deadline in ISO format YYYY-MM-DD. null if no deadline in the letter.",
    )
    consequences_ru: Optional[str] = Field(
        default=None,
        description="What the letter says will happen if ignored, in Russian. "
        "Only what is stated in the letter, no speculation.",
    )
    recommended_actions_ru: list[str] = Field(
        default_factory=list,
        description="2-4 practical everyday steps in Russian: where to go, whom to "
        "call, what to bring. NEVER legal advice, NEVER drafting appeals/objections.",
    )
    confidence_sender: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Self-assessed confidence for sender."
    )
    confidence_amount: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Self-assessed confidence for amount."
    )
    confidence_deadline: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Self-assessed confidence for deadline."
    )


class Verdict(str, Enum):
    CONFIRM = "confirm"
    MISMATCH = "mismatch"
    NOT_FOUND = "not_found"


class FieldCheck(BaseModel):
    verdict: Verdict = Field(
        description="confirm = the value matches the letter; mismatch = the letter "
        "contains a different value; not_found = the letter does not contain this field."
    )
    corrected_value: Optional[str] = Field(
        default=None,
        description="If verdict is mismatch: the correct value as read from the letter "
        "(amount as plain number, deadline as YYYY-MM-DD, sender as printed).",
    )


class CriticResult(BaseModel):
    """Structured output of Pass B: verification of the three critical fields."""

    sender: FieldCheck
    amount: FieldCheck
    deadline: FieldCheck


class FreeTextResult(BaseModel):
    """Structured output for (re)generation of free-text fields with locked values."""

    demand_summary_ru: str
    consequences_ru: str
    recommended_actions_ru: list[str]


class Card(BaseModel):
    """The final result card rendered to the user (ТЗ п.2)."""

    sender_ru: str
    sender_he: Optional[str] = None
    doc_type_ru: str
    demand_summary_ru: str
    amount_value: Optional[float] = None
    amount_currency: Optional[str] = None
    deadline: Optional[str] = None
    consequences_ru: Optional[str] = None
    recommended_actions_ru: list[str] = Field(default_factory=list)
    confidence: dict[str, float] = Field(default_factory=dict)
    not_found_fields: list[str] = Field(default_factory=list)


class Refusal(BaseModel):
    reason_code: str
    reason_ru: str
