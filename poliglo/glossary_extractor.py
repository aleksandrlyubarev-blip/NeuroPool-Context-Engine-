"""
GlossaryExtractor — scan text documents and suggest new glossary terms.

Reads plain-text documents (reports, manuals, alert logs) and asks the LLM to
identify domain-specific terms not yet in the RomeoFlexVision glossary.
Results include translations for ru / he so new terms can be added directly.

Typical workflow:
    extractor = GlossaryExtractor(poliglo_agent)

    # Scan a single document
    with open("robo_qc_manual.txt") as f:
        text = f.read()

    suggestions = extractor.extract(text)
    for s in suggestions:
        print(s.term_en, "→", s.translations)
    # → "throughput" → {"ru": "пропускная способность", "he": "תפוקה"}

    # Apply approved suggestions to glossary.py
    extractor.apply_to_glossary(approved_suggestions)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .agent import PoligloAgent, ChatMessage
from .glossary import GLOSSARY


@dataclass
class GlossarySuggestion:
    """
    A candidate glossary term extracted from a document.

    Fields:
        term_en:      Canonical English form of the term.
        translations: {"ru": "...", "he": "..."} from LLM.
        context:      Sentence(s) from the source text where the term appeared.
        confidence:   "high" | "medium" | "low" — LLM self-assessment.
        already_in_glossary: True if term_en is already covered.
    """
    term_en:              str
    translations:         dict[str, str]
    context:              str = ""
    confidence:           str = "medium"
    already_in_glossary:  bool = False

    def to_glossary_entry(self) -> dict[str, dict[str, str]]:
        """Return {term_en: {ru: ..., he: ..., en: ...}} ready for GLOSSARY."""
        return {
            self.term_en: {
                "ru": self.translations.get("ru", self.term_en),
                "he": self.translations.get("he", self.term_en),
                "en": self.term_en,
            }
        }


# Prompt template for LLM extraction
_EXTRACT_PROMPT = """\
You are a glossary curator for RomeoFlexVision, an industrial machine-vision company.

Analyse the following document excerpt and identify domain-specific technical terms
that are NOT already in the current glossary (listed below).

Focus on:
- Manufacturing / quality-control terminology
- Machine-vision / robotics / sensor terms
- Industry 4.0 / IIoT vocabulary
- RomeoFlexVision product-specific jargon

For each new term return a JSON array where each element has:
  "term_en"     : canonical English term (lowercase, singular)
  "ru"          : Russian translation
  "he"          : Hebrew translation
  "context"     : one sentence from the text showing usage
  "confidence"  : "high" | "medium" | "low"

Return ONLY the JSON array, no commentary.

EXISTING GLOSSARY KEYS (skip these):
{existing_keys}

DOCUMENT EXCERPT:
{text}
"""

_MAX_CHUNK = 3000   # characters per LLM call to stay within limits
_OVERLAP   = 200    # character overlap between chunks for continuity


class GlossaryExtractor:
    """
    Scans documents for domain terms not yet in the glossary.

    Works in chunks so arbitrarily long documents are handled gracefully.
    Duplicate suggestions across chunks are merged automatically.
    """

    def __init__(self, agent: PoligloAgent, glossary: dict = GLOSSARY):
        self._agent   = agent
        self._glossary = glossary

    # ── Main API ─────────────────────────────────────────────────────────────

    def extract(self, text: str) -> list[GlossarySuggestion]:
        """
        Extract glossary suggestions from *text*.

        Long texts are automatically chunked.  Duplicates are deduplicated
        (highest confidence kept).

        Returns a list of GlossarySuggestion sorted by confidence desc.
        """
        chunks = self._chunk(text)
        seen: dict[str, GlossarySuggestion] = {}

        for chunk in chunks:
            for sug in self._extract_chunk(chunk):
                key = sug.term_en.lower()
                if key not in seen or _confidence_rank(sug.confidence) > _confidence_rank(seen[key].confidence):
                    seen[key] = sug

        return sorted(seen.values(), key=lambda s: _confidence_rank(s.confidence), reverse=True)

    def extract_file(self, path: str | Path) -> list[GlossarySuggestion]:
        """Convenience wrapper — read a file then call extract()."""
        text = Path(path).read_text(encoding="utf-8")
        return self.extract(text)

    def apply_to_glossary(
        self,
        suggestions: list[GlossarySuggestion],
        glossary_path: str | Path = Path(__file__).parent / "glossary.py",
    ) -> int:
        """
        Append approved suggestions to glossary.py as Python dict entries.

        Args:
            suggestions:    Suggestions to apply (already_in_glossary ones skipped).
            glossary_path:  Path to glossary.py (default: this package's file).

        Returns:
            Number of entries actually written.
        """
        to_add = [s for s in suggestions if not s.already_in_glossary]
        if not to_add:
            return 0

        gp = Path(glossary_path)
        source = gp.read_text(encoding="utf-8")

        new_entries = []
        for s in to_add:
            entry = (
                f'    "{s.term_en}": {{\n'
                f'        "ru": "{s.translations.get("ru", s.term_en)}",\n'
                f'        "he": "{s.translations.get("he", s.term_en)}",\n'
                f'        "en": "{s.term_en}",\n'
                f'    }},\n'
            )
            new_entries.append(entry)

        # Insert before closing `}`
        insertion = "".join(new_entries)
        updated = re.sub(r'(\n\})\s*$', f'\n{insertion}}}', source)
        gp.write_text(updated, encoding="utf-8")

        # Update in-memory glossary too
        for s in to_add:
            self._glossary.update(s.to_glossary_entry())

        return len(to_add)

    # ── Internals ────────────────────────────────────────────────────────────

    def _extract_chunk(self, chunk: str) -> list[GlossarySuggestion]:
        existing_keys = ", ".join(f'"{k}"' for k in self._glossary)
        prompt = _EXTRACT_PROMPT.format(existing_keys=existing_keys, text=chunk)

        msg = ChatMessage(
            sender="glossary_extractor",
            text=prompt,
            target_lang="en",  # response in English JSON
            source_lang="auto",
            context="glossary extraction task — respond in JSON only",
        )
        reply = self._agent.handle_message(msg)

        if reply.error:
            return []

        return self._parse_response(reply.translated)

    def _parse_response(self, raw: str) -> list[GlossarySuggestion]:
        """Parse JSON array from LLM response, tolerating markdown fences."""
        import json

        # Strip ```json ... ``` fences if present
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)

        try:
            items = json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to extract the first [...] block
            match = re.search(r"\[.*\]", cleaned, re.DOTALL)
            if not match:
                return []
            try:
                items = json.loads(match.group())
            except json.JSONDecodeError:
                return []

        results = []
        existing_lower = {k.lower() for k in self._glossary}

        for item in items:
            if not isinstance(item, dict) or "term_en" not in item:
                continue
            term = item["term_en"].strip().lower()
            results.append(GlossarySuggestion(
                term_en=term,
                translations={"ru": item.get("ru", term), "he": item.get("he", term)},
                context=item.get("context", ""),
                confidence=item.get("confidence", "medium"),
                already_in_glossary=term in existing_lower,
            ))
        return results

    @staticmethod
    def _chunk(text: str) -> list[str]:
        """Split text into overlapping chunks of ~_MAX_CHUNK characters."""
        if len(text) <= _MAX_CHUNK:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + _MAX_CHUNK, len(text))
            chunks.append(text[start:end])
            start += _MAX_CHUNK - _OVERLAP
        return chunks


def _confidence_rank(confidence: str) -> int:
    return {"high": 2, "medium": 1, "low": 0}.get(confidence, 1)
