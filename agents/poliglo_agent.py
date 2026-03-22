"""
Popugai Poliglo — Translation Agent for RomeoFlexVision
Phase 1: ru ↔ he ↔ en (any pair, including ru ↔ he direct)
Primary: Amazon Translate | Refinement: Claude 3.5 Sonnet
"""
import boto3
import json
from typing import Optional
from botocore.exceptions import ClientError


SUPPORTED_LANGS = {"ru", "he", "en"}

# RomeoFlexVision glossary (English key → translations)
GLOSSARY = {
    "Robo QC": {
        "ru": "Robo QC",
        "he": "Robo QC",
        "en": "Robo QC",
    },
    "optical inspection": {
        "ru": "оптическая инспекция",
        "he": "בדיקה אופטית",
        "en": "optical inspection",
    },
    "defect classification": {
        "ru": "классификация дефектов",
        "he": "סיווג פגמים",
        "en": "defect classification",
    },
    "predictive maintenance": {
        "ru": "предиктивное обслуживание",
        "he": "תחזוקה חזויה",
        "en": "predictive maintenance",
    },
    "Industry 4.0": {
        "ru": "Промышленность 4.0",
        "he": "Industry 4.0",
        "en": "Industry 4.0",
    },
    "quality control": {
        "ru": "контроль качества",
        "he": "בקרת איכות",
        "en": "quality control",
    },
    # Add new terms here — they will be picked up automatically
}


class PoligloAgent:
    """
    Translate industrial text between Russian, Hebrew, and English.

    Usage:
        agent = PoligloAgent()
        result = agent.translate("optical inspection report", target_lang="ru")

    Compatible with Nemotron 3 Super as a tool — see tool_schemas() for schema.
    """

    def __init__(self, region: str = "us-west-2"):
        self.translate_client = boto3.client("translate", region_name=region)
        self.bedrock_client = boto3.client("bedrock-runtime", region_name=region)
        self.glossary = GLOSSARY

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def translate(
        self,
        text: str,
        target_lang: str,
        source_lang: str = "auto",
        context: Optional[str] = None,
    ) -> str:
        """
        Translate text to target_lang.

        Args:
            text:        Input text (max ~10 000 chars).
            target_lang: One of "ru", "he", "en".
            source_lang: One of "ru", "he", "en" or "auto" (default).
            context:     Optional domain hint sent to the LLM refinement step.

        Returns:
            Translated string.  RTL markers are preserved for Hebrew output.
        """
        if not text or not text.strip():
            return text

        if target_lang not in SUPPORTED_LANGS:
            raise ValueError(f"target_lang must be one of {SUPPORTED_LANGS}")

        if source_lang != "auto" and source_lang == target_lang:
            return text

        try:
            response = self.translate_client.translate_text(
                Text=text,
                SourceLanguageCode=source_lang if source_lang != "auto" else "auto",
                TargetLanguageCode=target_lang,
            )
            translated = response["TranslatedText"]
        except ClientError:
            # Amazon Translate unavailable — fall back to full LLM translation
            return self._llm_translate(text, source_lang, target_lang, context)

        # LLM refinement for Hebrew/Russian targets or long texts
        if target_lang in {"he", "ru"} or len(text) > 500:
            translated = self._llm_technical_refine(
                translated, source_lang, target_lang, context
            )

        return translated

    # ------------------------------------------------------------------
    # Nemotron / tool schema
    # ------------------------------------------------------------------

    @staticmethod
    def tool_schemas() -> list:
        """Return Nemotron-compatible tool schemas for all translation directions."""
        directions = [
            ("translate_to_russian",  "ru",  "Translate text to Russian"),
            ("translate_to_hebrew",   "he",  "Translate text to Hebrew"),
            ("translate_to_english",  "en",  "Translate text to English"),
        ]
        schemas = []
        for name, target, description in directions:
            schemas.append({
                "name": name,
                "description": (
                    f"{description}. "
                    "Preserves RomeoFlexVision industrial terminology. "
                    "Supports ru / he / en input with auto-detection."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Text to translate (max ~10 000 chars)",
                        },
                        "source_lang": {
                            "type": "string",
                            "enum": ["auto", "ru", "he", "en"],
                            "description": "Source language (default: auto-detect)",
                            "default": "auto",
                        },
                        "context": {
                            "type": "string",
                            "description": "Optional domain context hint",
                        },
                    },
                    "required": ["text"],
                },
                "_target_lang": target,  # internal hint — strip before sending to model
            })
        return schemas

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _llm_technical_refine(
        self,
        text: str,
        source: str,
        target: str,
        context: Optional[str],
    ) -> str:
        prompt = (
            "Ты — технический переводчик Industry 4.0 для израильских и "
            "русскоязычных заводов.\n"
            "Используй глоссарий ниже. "
            "Сохраняй точность терминов, естественность и RTL.\n\n"
            f"Глоссарий:\n{json.dumps(self.glossary, ensure_ascii=False, indent=2)}\n\n"
            f"Оригинал: {text}\n"
            f"Контекст: {context or 'Нет'}\n\n"
            f"Переведи/улучши на {target.upper()}. "
            "Ответь ТОЛЬКО переводом, без объяснений."
        )
        return self._call_claude(prompt)

    def _llm_translate(
        self,
        text: str,
        source: str,
        target: str,
        context: Optional[str],
    ) -> str:
        prompt = (
            f"Переведи текст с {source} на {target}.\n"
            f"Глоссарий RomeoFlexVision:\n"
            f"{json.dumps(self.glossary, ensure_ascii=False)}\n\n"
            f"Текст: {text}\n"
            f"Контекст: {context or 'Нет'}\n\n"
            "Ответь ТОЛЬКО переводом."
        )
        return self._call_claude(prompt)

    def _call_claude(self, prompt: str) -> str:
        try:
            response = self.bedrock_client.converse(
                modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
                messages=[
                    {"role": "user", "content": [{"text": prompt}]}
                ],
                inferenceConfig={"maxTokens": 2000, "temperature": 0.1},
            )
            return response["output"]["message"]["content"][0]["text"].strip()
        except Exception as e:
            return f"[ОШИБКА ПЕРЕВОДА: {e}]"


# ======================================================================
# Quick smoke-test  (python agents/poliglo_agent.py)
# ======================================================================
if __name__ == "__main__":
    poliglo = PoligloAgent()

    print("→ he → ru:")
    print(poliglo.translate(
        "התוצאות של בדיקת Robo QC מראות 3 פגמים מסוג predictive maintenance.",
        target_lang="ru",
        source_lang="he",
    ))

    print("\n→ ru → he:")
    print(poliglo.translate(
        "Результаты оптической инспекции показывают 3 дефекта, "
        "требующих предиктивного обслуживания.",
        target_lang="he",
        source_lang="ru",
    ))

    print("\n→ en → ru:")
    print(poliglo.translate(
        "The optical inspection unit requires predictive maintenance.",
        target_lang="ru",
    ))

    print("\n→ Tool schemas (Nemotron):")
    for schema in PoligloAgent.tool_schemas():
        print(f"  • {schema['name']}")
