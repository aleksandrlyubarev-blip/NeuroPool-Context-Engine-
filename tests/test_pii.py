"""ТЗ п.5: логи без PII — маскер для ת"ז (с чек-суммой), телефонов, email."""

from app.privacy.pii import is_valid_israeli_id, mask_pii


def test_valid_israeli_id_checksum():
    # Known-valid IDs (checksum-correct)
    assert is_valid_israeli_id("123456782")
    assert is_valid_israeli_id("000000000")
    assert not is_valid_israeli_id("123456789")
    assert not is_valid_israeli_id("12345678")   # 8 digits
    assert not is_valid_israeli_id("1234567890")


def test_masks_valid_id_only():
    text = "ת\"ז 123456782 и случайное число 123456789"
    masked = mask_pii(text)
    assert "123456782" not in masked
    assert "***" in masked
    # invalid checksum → not an ID → left as-is (e.g. case numbers)
    assert "123456789" in masked


def test_masks_phones():
    for phone in ["050-1234567", "03-123-4567", "+972-50-1234567", "*6050", "1-700-505-202"]:
        assert phone not in mask_pii(f"позвоните {phone} сегодня"), phone


def test_masks_email():
    masked = mask_pii("обратитесь на olim@example.co.il за помощью")
    assert "olim@example.co.il" not in masked
    assert "***" in masked


def test_keeps_ordinary_text():
    text = "требование об оплате 150 шекелей до 15.08.2026"
    assert mask_pii(text) == text
