"""Lazy SDK loading and explicit model configuration for live and offline runs."""
from __future__ import annotations

import os
from typing import Any


def configured_model(model: str | None) -> str:
    return model or os.getenv("PAPER_LLM_MODEL") or os.getenv("OPENAI_MODEL") or ""


class LocalClientSetupError(RuntimeError):
    """SDK import/configuration failed before any API request could be sent."""
    request_not_sent = True


class StructuredOutputParseError(ValueError):
    """The server returned a response, but local structured validation failed."""
    def __init__(self, cause: Exception, response_receipt: dict[str, Any] | None):
        self.response_receipt = response_receipt
        super().__init__(f"Response received; structured output validation failed: "
                         f"{type(cause).__name__}: {cause}")


class LazyOpenAI:
    """SDK-shaped adapter. Constructing an index/workflow never spends tokens."""
    def __init__(self) -> None:
        self._client: Any = None

    @property
    def responses(self) -> "LazyOpenAI":
        return self

    def parse(self, **kwargs: Any) -> Any:
        if not kwargs.get("model"):
            raise ValueError("Set PAPER_LLM_MODEL or pass --model with a model your API account supports.")
        if self._client is None:
            from urllib.parse import urlparse
            if os.getenv("OPENAI_BASE_URL") and urlparse(os.environ["OPENAI_BASE_URL"]).hostname != "api.openai.com":
                from app.budget import BudgetStop
                raise BudgetStop("Budget guard has official OpenAI prices only; custom API endpoint not priced.")
            try:
                from openai import OpenAI
                self._client = OpenAI(timeout=60.0, max_retries=0)  # no invisible paid retries
            except Exception as exc:
                raise LocalClientSetupError(
                    f"API client setup failed ({type(exc).__name__}); check the SDK installation and OPENAI_API_KEY. No request sent."
                ) from exc
        # The raw wrapper makes ONE request. Parsing below is local; never
        # reissue a paid request merely to recover its usage receipt.
        raw = self._client.responses.with_raw_response.parse(**kwargs)
        try:
            return raw.parse()
        except Exception as exc:
            receipt = None
            try:
                body = raw.http_response.json()
                if isinstance(body, dict):
                    # Keep accounting/provenance metadata, not prompt text,
                    # base64 images, or the model's raw response content.
                    receipt = {key: body.get(key) for key in
                               ("id", "model", "usage", "service_tier", "status")}
            except Exception:
                pass  # Missing usage must remain unknown, never become $0.
            raise StructuredOutputParseError(exc, receipt) from exc
