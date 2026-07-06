# Golden set (ТЗ п.7)

15 real **anonymized** Israeli official letters + 2 negative cases
(unreadable blur, not-a-letter). Each case is a directory:

```
data/golden/arnona_haifa_01/
  letter.jpg            # anonymized scan/photo
  ground_truth.json
```

`ground_truth.json`:

```json
{
  "sender_contains": "ирия",       // substring of the expected sender_ru
  "amount": 150.0,                  // main amount, or omit if absent
  "deadline": "2026-08-15",        // ISO date, or omit if absent
  "expect": "card"                  // "refusal" for the two negative cases
}
```

Anonymization checklist before adding a letter (must ALL be true):
- [ ] name, address, ת"ז, account/case numbers blacked out or replaced
- [ ] barcode / QR / payment slip numbers removed
- [ ] the amount and deadline you keep are the ones in ground_truth.json

`letter.*` files are gitignored by default; after the checklist passes, add
deliberately: `git add -f data/golden/<case>/letter.jpg`.

Run the gate: `GEMINI_API_KEY=... python scripts/run_golden.py`
(appends results to docs/eval_log.md; exit 0 = release gate passed).
