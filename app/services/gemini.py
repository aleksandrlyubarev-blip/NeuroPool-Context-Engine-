"""Thin wrapper around the google-genai SDK (ТЗ п.3).

One place that talks to the Gemini API: structured output via response_schema,
temperature pinned by the caller. The model ID comes from GEMINI_MODEL.
"""

from typing import Optional, Type, TypeVar

from pydantic import BaseModel

from app.config import get_settings

T = TypeVar("T", bound=BaseModel)


class GeminiClient:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        settings = get_settings()
        self._model = model or settings.gemini_model
        # Imported lazily so unit tests can run without the SDK installed.
        from google import genai

        self._client = genai.Client(api_key=api_key or settings.gemini_api_key)
        self.last_usage: dict[str, int] = {}

    def generate_structured(
        self,
        *,
        system: str,
        schema: Type[T],
        text: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        image_mime: Optional[str] = None,
        temperature: float = 0.0,
    ) -> T:
        """One structured-output call. Image bytes stay in memory only."""
        from google.genai import types

        parts: list = []
        if image_bytes is not None:
            parts.append(types.Part.from_bytes(data=image_bytes, mime_type=image_mime))
        if text:
            parts.append(types.Part.from_text(text=text))

        response = self._client.models.generate_content(
            model=self._model,
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=temperature,
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )
        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            self.last_usage = {
                "tokens_in": getattr(usage, "prompt_token_count", 0) or 0,
                "tokens_out": getattr(usage, "candidates_token_count", 0) or 0,
            }
        parsed = response.parsed
        if isinstance(parsed, schema):
            return parsed
        # Fallback: SDK returned raw JSON text.
        return schema.model_validate_json(response.text)
