"""ТЗ п.4.6 (value locking): подсадное расхождение числа/даты в свободном
тексте должно ловиться санитайзером."""

from app.pipeline.locking import (
    LockedValues,
    check_free_text,
    extract_dates,
    extract_numbers,
)

LOCKED = LockedValues(sender_ru="Ирия Хайфы", amount_value=150.0, deadline="2026-08-15")


def test_planted_wrong_amount_caught():
    result = check_free_text(["Оплатите 180 шекелей в кассе ирии"], LOCKED)
    assert not result.ok


def test_planted_wrong_date_caught():
    result = check_free_text(["Оплатите до 20.09.2026"], LOCKED)
    assert not result.ok


def test_correct_values_pass():
    result = check_free_text(
        ["Оплатите 150 шекелей до 15.08.2026", "Сумма: 150.00 ₪"], LOCKED
    )
    assert result.ok, result.violations


def test_russian_date_format_checked():
    assert check_free_text(["до 15 августа 2026 года"], LOCKED).ok
    assert not check_free_text(["до 20 сентября 2026 года"], LOCKED).ok


def test_phone_numbers_allowed():
    result = check_free_text(
        ["Позвоните по телефону *6050 или 03-123-4567, звонок бесплатный"], LOCKED
    )
    assert result.ok, result.violations


def test_deadline_components_allowed():
    # "2026" or "15" alone (parts of the locked deadline) are not violations
    assert check_free_text(["до 15 числа, в 2026 году"], LOCKED).ok


def test_small_ordinals_allowed():
    assert check_free_text(["Шаг 2: возьмите 2 экземпляра"], LOCKED).ok


def test_no_locked_amount_any_number_is_violation():
    locked = LockedValues(sender_ru="Банк", amount_value=None, deadline=None)
    assert not check_free_text(["Оплатите 500 шекелей"], locked).ok


def test_extract_numbers_handles_thousands():
    assert 1500.0 in extract_numbers("сумма 1,500 шекелей")
    assert 1500.5 in extract_numbers("сумма 1500.50")


def test_extract_dates_iso_and_dotted():
    assert extract_dates("до 15.08.2026 или 2026-08-15") == {"2026-08-15"}
