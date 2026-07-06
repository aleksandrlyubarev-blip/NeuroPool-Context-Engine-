# PoLingvo — "What do they want from me?"

Русская версия: [README.ru.md](README.ru.md)

PoLingvo explains Israeli official letters (municipality, Bituach Leumi, banks,
kupat holim, electric company…) to Russian-speaking new immigrants. Upload a
photo or PDF of a letter in Hebrew — get a structured card in Russian: who
sent it, what they demand, the amount, the deadline, what happens if you
ignore it, and practical everyday steps.

Built for the **XPRIZE "Build with Gemini"** competition (Professional
Services category).

## How it works

```
photo/PDF ──► Pass A (Gemini, structured output, T=0)  — extract card draft
          ──► Pass B (Gemini critic, T=0)              — verify sender/amount/deadline
          ──► decision policy                          — card or honest refusal
          ──► value locking                            — numbers/dates rendered only
                                                          from verified JSON
          ──► output sanitizer                         — no legal advice, ever
```

Key properties, enforced **in code**, not just prompts:

- **Honest refusal.** If the two passes disagree on the amount or deadline, or
  confidence is low, the answer is "show this letter to a human" — never a guess.
  A refusal never consumes a paid credit.
- **Value locking.** Every number and date in the card comes from the verified
  JSON. Free text is scanned; a stray number triggers one regeneration with
  frozen constants in XML tags, then a refusal.
- **Product boundary.** PoLingvo explains letters and suggests everyday steps
  ("where to go, whom to call"). It never drafts appeals or objections and never
  gives legal advice — a code-level stop-list guards the output.
- **Privacy by architecture.** Letters are processed in request memory only:
  no disk, no buckets, no temp files. Logs pass a PII masker (Israeli ID
  checksum-validated, phones, emails). Firestore stores only credits,
  purchases (hashed email), anonymous metrics and ledger events.
- **Prompt-injection resistance.** Letter content is spotlighted as untrusted
  data in every system prompt, with an e2e injection test.

## Stack

FastAPI (Python 3.12) · Gemini API (`google-genai`, structured output) ·
Cloud Run (me-west1) · Firestore · Stripe Payment Link · single-page vanilla-JS
frontend (mobile-first, RTL support for Hebrew quotes).

## Run locally

```bash
pip install -r requirements-dev.txt
export GEMINI_API_KEY=...        # local dev uses the in-memory store
uvicorn app.main:app --reload
# open http://localhost:8000
```

Tests:

```bash
pytest                       # unit + API tests (no network)
pytest -m golden             # golden-set quality gate (needs API key + data/golden)
python scripts/run_golden.py # golden gate + docs/eval_log.md entry
```

## Deploy

Push a `v*` tag → GitHub Actions builds and deploys to Cloud Run
(`.github/workflows/deploy.yml`). Configuration lives in env vars
(`.env.example`); secrets in GCP Secret Manager.

## Quality gate

Release requires the golden-set thresholds (deadline ≥14/15, amount ≥14/15,
sender ≥13/15, both negatives → refusal, injection/sanitizer/value-locking
tests green). Every run is logged in [docs/eval_log.md](docs/eval_log.md).
Revenue evidence: [docs/ledger.md](docs/ledger.md).

## License

Proprietary — see [LICENSE](LICENSE). Read access for XPRIZE judging:
testing@devpost.com, judging@hacker.fund.
