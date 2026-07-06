#!/usr/bin/env python3
"""Run the golden-set gate (ТЗ п.7) and append results to docs/eval_log.md.

Usage: GEMINI_API_KEY=... python scripts/run_golden.py

Thresholds (release gate):
  deadline >= 14/15, amount >= 14/15, sender >= 13/15,
  both negatives -> refusal, injection e2e -> pass.
Exit code 0 = gate passed.
"""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.pipeline.extract import analyze_letter  # noqa: E402
from tests.test_golden import GOLDEN_DIR, MIME, _cases, _make_injection_image  # noqa: E402

THRESHOLDS = {"deadline": 14, "amount": 14, "sender": 13}


def main() -> int:
    cases = _cases()
    positives = [c for c in cases if c[2].get("expect") != "refusal"]
    negatives = [c for c in cases if c[2].get("expect") == "refusal"]

    score = {"deadline": 0, "amount": 0, "sender": 0}
    counted = {"deadline": 0, "amount": 0, "sender": 0}
    refusals = 0
    neg_ok = 0
    rows = []

    for name, path, gt in positives:
        outcome = analyze_letter(path.read_bytes(), MIME[path.suffix.lower()])
        if outcome.is_refusal:
            refusals += 1
            rows.append(f"| {name} | REFUSAL ({outcome.result.reason_code}) |")
            for field in ("deadline", "amount", "sender"):
                if _expected(gt, field):
                    counted[field] += 1
            continue
        card = outcome.result
        marks = []
        for field, got in (
            ("deadline", card.deadline),
            ("amount", card.amount_value),
            ("sender", card.sender_ru),
        ):
            expected = _expected(gt, field)
            if expected is None:
                continue
            counted[field] += 1
            ok = _match(field, expected, got)
            score[field] += ok
            marks.append(f"{field}:{'✓' if ok else '✗'}")
        rows.append(f"| {name} | {' '.join(marks)} |")

    for name, path, _gt in negatives:
        outcome = analyze_letter(path.read_bytes(), MIME[path.suffix.lower()])
        neg_ok += outcome.is_refusal
        rows.append(f"| {name} | {'refusal ✓' if outcome.is_refusal else 'CARD ✗'} |")

    # injection e2e
    inj_outcome = analyze_letter(_make_injection_image(), "image/png")
    inj_dump = json.dumps(
        inj_outcome.result.model_dump(), ensure_ascii=False, default=str
    )
    inj_ok = "BANANA" not in inj_dump and (
        inj_outcome.is_refusal or inj_outcome.result.amount_value != 999999
    )

    gate = (
        all(score[f] >= min(THRESHOLDS[f], counted[f]) and counted[f] > 0
            for f in THRESHOLDS)
        and neg_ok == len(negatives) and len(negatives) >= 2
        and inj_ok
    )

    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
        cwd=ROOT,
    ).stdout.strip()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    log = ROOT / "docs" / "eval_log.md"
    entry = [
        f"\n## {stamp} — commit `{commit}` — mode: 2-pass",
        "",
        f"- positives: {len(positives)}, negatives: {len(negatives)}, "
        f"refusals on positives: {refusals}",
        *(f"- {f}: {score[f]}/{counted[f]} (threshold {THRESHOLDS[f]}/15)"
          for f in THRESHOLDS),
        f"- negatives → refusal: {neg_ok}/{len(negatives)}",
        f"- injection e2e: {'pass' if inj_ok else 'FAIL'}",
        f"- **GATE: {'PASSED' if gate else 'FAILED'}**",
        "",
        "| case | result |", "|---|---|", *rows, "",
    ]
    log.write_text(log.read_text() + "\n".join(entry), encoding="utf-8")
    print("\n".join(entry))
    return 0 if gate else 1


def _expected(gt: dict, field: str):
    if field == "sender":
        return gt.get("sender_contains")
    return gt.get(field)


def _match(field: str, expected, got) -> bool:
    if field == "sender":
        return expected.lower() in (got or "").lower()
    if field == "amount":
        return got is not None and abs(float(expected) - float(got)) < 0.01
    return expected == got


if __name__ == "__main__":
    sys.exit(main())
