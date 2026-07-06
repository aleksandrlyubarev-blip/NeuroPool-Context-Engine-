"""ТЗ п.4.3: политика решения два прохода → карточка / refusal."""

from app.pipeline.extract import decide
from app.pipeline.schemas import CriticResult, FieldCheck, PassAResult, Verdict


def _pass_a(**over) -> PassAResult:
    base = dict(
        is_official_letter=True, readable=True,
        sender_he="עיריית חיפה", sender_ru="Ирия Хайфы",
        doc_type_ru="требование об оплате",
        demand_summary_ru="Оплатить долг по арноне.",
        amount_value=150.0, amount_currency="ILS", deadline="2026-08-15",
        consequences_ru="Начислят пеню.",
        recommended_actions_ru=["Оплатите на сайте ирии"],
        confidence_sender=0.95, confidence_amount=0.9, confidence_deadline=0.9,
    )
    base.update(over)
    return PassAResult(**base)


def _critic(sender="confirm", amount="confirm", deadline="confirm") -> CriticResult:
    return CriticResult(
        sender=FieldCheck(verdict=Verdict(sender)),
        amount=FieldCheck(verdict=Verdict(amount)),
        deadline=FieldCheck(verdict=Verdict(deadline)),
    )


def test_all_confirm_gives_card():
    assert decide(_pass_a(), _critic()) is None


def test_amount_mismatch_refuses():
    refusal = decide(_pass_a(), _critic(amount="mismatch"))
    assert refusal is not None and refusal.reason_code == "mismatch_amount"


def test_deadline_mismatch_refuses():
    assert decide(_pass_a(), _critic(deadline="mismatch")) is not None


def test_sender_mismatch_refuses():
    assert decide(_pass_a(), _critic(sender="mismatch")) is not None


def test_not_found_is_not_refusal():
    assert decide(_pass_a(amount_value=None), _critic(amount="not_found")) is None
    assert decide(_pass_a(deadline=None), _critic(deadline="not_found")) is None


def test_low_confidence_on_confirmed_field_refuses():
    refusal = decide(_pass_a(confidence_amount=0.5), _critic())
    assert refusal is not None and refusal.reason_code == "low_confidence_amount"


def test_low_confidence_on_not_found_field_is_ok():
    # confidence threshold applies only to confirmed fields
    assert decide(
        _pass_a(amount_value=None, confidence_amount=0.0),
        _critic(amount="not_found"),
    ) is None
