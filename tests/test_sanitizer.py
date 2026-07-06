"""ТЗ п.6.4: стоп-лист юридических формулировок — обязательный тест."""

from app.pipeline.sanitizer import (
    NEUTRAL_ACTION_RU,
    contains_stop_words,
    sanitize_card_texts,
)


def test_stop_words_detected():
    for phrase in [
        "Обжалуйте это решение в течение 30 дней",
        "Подайте ерур в битуах леуми",
        "подайте ходатайство",
        "напишите возражение",
        "есть основания оспорить начисление",
        "юридически это спорно",
        "можно оспаривать сумму",
        "подайте апелляцию",
    ]:
        assert contains_stop_words(phrase), phrase


def test_safe_phrases_pass():
    for phrase in [
        "Позвоните в приёмную ирии",
        "Придите в отделение битуах леуми с письмом и теудат-зеут",
        "Оплатите сумму на сайте до срока",
        "Уточните детали по телефону, указанному в письме",
    ]:
        assert not contains_stop_words(phrase), phrase


def test_offending_action_replaced_with_neutral():
    result = sanitize_card_texts(
        "Ирия требует оплатить долг.",
        ["Оплатите на сайте ирии", "Обжалуйте начисление через ерур"],
    )
    assert result.flagged
    assert result.recommended_actions_ru == ["Оплатите на сайте ирии", NEUTRAL_ACTION_RU]


def test_offending_summary_sentence_dropped():
    result = sanitize_card_texts(
        "Ирия требует оплатить долг. Вы можете обжаловать это решение.",
        [],
    )
    assert result.flagged
    assert "обжаловать" not in result.demand_summary_ru.lower()
    assert "оплатить долг" in result.demand_summary_ru


def test_clean_texts_not_flagged():
    result = sanitize_card_texts(
        "Банк сообщает об остатке.", ["Позвоните в банк"]
    )
    assert not result.flagged
    assert result.demand_summary_ru == "Банк сообщает об остатке."
