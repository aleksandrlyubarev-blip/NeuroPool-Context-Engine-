# Implementation notes — deviations from ТЗ v1.0 and rationale

## Deviations / interpretations

1. **Sender mismatch → refusal** (ТЗ п.4.3 lists refusal explicitly only for
   amount/deadline). A card attributed to the wrong sender is misleading even
   with correct numbers, so a sender mismatch also refuses. `not_found` for
   sender still renders «в письме не указано» as specified.

2. **Value-locking regeneration is a third, text-only Gemini call.** ТЗ п.4.6
   requires free-text generation to receive frozen constants in XML tags. To
   keep the happy path at exactly two passes, free text comes from Pass A;
   the dedicated XML-constants call (`FREE_TEXT_SYSTEM`) runs only when the
   lock check finds a violation (one retry, then refusal). Numbers below 5 in
   free text are allowed (list ordinals, «в 2 экземплярах»); phone-like
   sequences are excluded from the numeric scan because letters legitimately
   contain hotline numbers.

3. **Default model ID is the `gemini-flash-latest` alias.** ТЗ п.3 says «дефолт
   — актуальный Flash; сверить ID при сборке». The alias tracks the current
   Flash release; pin an explicit ID via `GEMINI_MODEL` at deploy time
   (GitHub Actions variable `GEMINI_MODEL`). Upgrade remains a one-env change.

4. **«Первое письмо бесплатно»** is implemented per anonymous browser token
   (localStorage UUID), backed by the same IP rate limit (5/hour) to cap
   token-resetting abuse. No accounts in v1.0 by design.

5. **Rate limiter is in-memory.** Cloud Run at v1.0 traffic runs 0–1 instances;
   a distributed limiter (Firestore/Redis) is deliberately out of scope. If
   max-instances is ever raised above 2, revisit.

6. **Golden letters are gitignored by default** and added with `git add -f`
   only after the anonymization checklist (data/golden/README.md) passes —
   a safety default so a raw letter can't land in git by accident.

7. **MC-ансамбль (п.4.5) not implemented** — contingent layer; trigger
   (golden thresholds failing or refusal_rate > 25%) has not fired. The
   metric log already records `mode: "2-pass"` per request so the switch
   will be visible in eval_log.md.

8. **PDF page limit**: v1.0 sends the PDF to Gemini as-is (inline, ≤10 MB);
   multi-page letters are supported by the model natively.

8a. **Upload spooling**: Starlette's `UploadFile` buffers bodies larger than
   1 MB in an OS-level anonymous spooled temp file for the duration of the
   request; it is destroyed automatically when the request ends. Nothing
   survives the request and no named files are ever created, which satisfies
   the п.5 invariant («никаких temp-файлов, переживающих запрос»).

9. **Repo prerequisites for XPRIZE (п.1) need owner action**: repository
   rename (it currently carries a legacy name), private visibility check, and
   read access for testing@devpost.com / judging@hacker.fund. Code-side
   requirements (Gemini API call, Cloud Run + Firestore, EN/RU README,
   ledger) are in place.

## Operational checklist (DoD п.8 items that need the owner)

- Create GCP project, enable Cloud Run + Firestore (region me-west1),
  put `GEMINI_API_KEY` and `STRIPE_WEBHOOK_SECRET` in Secret Manager;
  set GitHub secrets `GCP_PROJECT_ID`, `GCP_WORKLOAD_IDENTITY_PROVIDER`,
  `GCP_SERVICE_ACCOUNT`, `STRIPE_PAYMENT_LINK_URL`.
- Create the Stripe Payment Link (10 ₪) and point its webhook
  (checkout.session.completed) at `https://<cloud-run-url>/api/stripe/webhook`.
- Collect and anonymize 15 golden letters + 2 negatives, run
  `python scripts/run_golden.py` until the gate passes.
- Tag `v1.0.0` to deploy; verify e2e photo→card ≤ 20 s; run one test and one
  live payment; check a day of prod logs with the PII regexes
  (`python - <<'PY' ...` over exported logs using `app/privacy/pii.py`).
