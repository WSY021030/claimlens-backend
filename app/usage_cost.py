"""Best-effort USD cost estimates from actual OpenAI Responses API usage.

Prices below are standard online input/output rates per 1M tokens, checked 2026-10-21:
https://developers.openai.com/api/docs/models/gpt-4o-mini
https://developers.openai.com/api/docs/models/gpt-5-mini

This module does NOT read billing data. Never interpret missing usage or unknown
model pricing as a free API call. Prompt-cached tokens are part of input_tokens.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any
import re


USD_MILLION = Decimal("1000000")
PRICE_CHECKED_ON = "2026-10-24"
# (normal input, cached input, output), in US dollars per million tokens.
MODEL_PRICES = {
    "gpt-4o-mini": (Decimal("0.15"), Decimal("0.075"), Decimal("0.60")),
    "gpt-5-mini": (Decimal("0.25"), Decimal("0.025"), Decimal("2.00")),
    "gpt-4.1-mini": (Decimal("0.40"), Decimal("0.10"), Decimal("1.60")),
    "gpt-5.4-mini": (
        Decimal("0.75"), Decimal("0.075"), Decimal("4.50")
    ),
}
AGENT_BY_FORMAT = {
    "ClaimExtraction": "Claim",
    "EvidenceExtraction": "Evidence",
    "MappingExtraction": "Mapping",
    "CoordinatorPlan": "Coordinator",
    "CoordinatorReply": "Coordinator",
    "ClaimSelection": "Coordinator",
    "CollectionReview": "Coordinator",
    "SourceNotes": "Coordinator",
}  

def _get(obj: Any, attr: str, default: Any = None) -> Any:
    return obj.get(attr, default) if isinstance(obj, dict) else getattr(obj, attr, default)


def _nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _prices_for(model: str) -> tuple[Decimal, Decimal, Decimal] | None:
    # Exact alias or known dated snapshot only, not e.g. an unknown 'mini-plus'.
    for name, rates in MODEL_PRICES.items():
        if model == name or re.fullmatch(re.escape(name) + r"-\d{4}-\d{2}-\d{2}", model):
            return rates
    return None


@dataclass
class CostTracker:
    """One tracker per analysis invocation; never shared across user runs."""
    calls: list[dict[str, Any]] = field(default_factory=list)

    def add(self, requested_model: str, response: Any, text_format: Any = None,
            *, input_images: int = 0, call_status: str = "returned") -> None:
        usage = _get(response, "usage")
        actual_model = str(_get(response, "model") or requested_model or "unknown")
        role_name = getattr(text_format, "__name__", "")
        agent = AGENT_BY_FORMAT.get(role_name, role_name or "Other")
        has_usage = usage is not None and _get(usage, "input_tokens") is not None and _get(usage, "output_tokens") is not None
        input_tokens = _nonnegative_int(_get(usage, "input_tokens")) if has_usage else None
        output_tokens = _nonnegative_int(_get(usage, "output_tokens")) if has_usage else None
        details = _get(usage, "input_tokens_details")
        cached_tokens = min(input_tokens, _nonnegative_int(_get(details, "cached_tokens"))) if has_usage else None
        output_details = _get(usage, "output_tokens_details")
        reasoning_tokens = _nonnegative_int(_get(output_details, "reasoning_tokens")) if has_usage else None
        rates = _prices_for(actual_model)
        service_tier = _get(response, "service_tier")
        if service_tier not in {None, "default", "auto"}:
            rates = None  # Do not price Flex/Priority responses at standard rates.
        charge = None
        if rates and has_usage:
            normal, cached, output = rates
            charge = (
                Decimal(input_tokens - cached_tokens) * normal
                + Decimal(cached_tokens) * cached
                + Decimal(output_tokens) * output
            ) / USD_MILLION
        self.calls.append({
            "agent": agent, "model": actual_model,
            "response_id": _get(response, "id"), "call_status": call_status,
            "input_images": input_images, "reasoning_tokens": reasoning_tokens,
            "service_tier": service_tier,
            "requested_model": requested_model,
            "input_tokens": input_tokens,
            "cached_input_tokens": cached_tokens,
            "output_tokens": output_tokens,
            "estimated_cost_usd": float(charge) if charge is not None else None,
            "estimated_cost_exact": charge,  # not serialized; precision for sums
            "pricing_status": "priced" if charge is not None else ("usage_unavailable" if not has_usage else "model_price_unknown"),
        })

    def report(self) -> dict[str, Any]:
        total = sum((call["estimated_cost_exact"] for call in self.calls if call["estimated_cost_exact"] is not None), Decimal(0))
        complete = all(c["pricing_status"] == "priced" for c in self.calls)
        buckets: dict[str, dict[str, Any]] = defaultdict(lambda: {"llm_calls": 0, "input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0, "known_cost": Decimal(0), "complete": True})
        def aggregate(kind: str) -> list[dict[str, Any]]:
            buckets.clear()
            for call in self.calls:
                b = buckets[call[kind]]
                b["llm_calls"] += 1
                for token_key in ("input_tokens", "cached_input_tokens", "output_tokens"):
                    b[token_key] += call[token_key] or 0
                if call["estimated_cost_exact"] is None:
                    b["complete"] = False
                else:
                    b["known_cost"] += call["estimated_cost_exact"]
            return [{kind: name, "llm_calls": values["llm_calls"],
                     "input_tokens": values["input_tokens"],
                     "cached_input_tokens": values["cached_input_tokens"],
                     "output_tokens": values["output_tokens"],
                     "estimated_cost_usd": round(float(values["known_cost"]), 9) if values["complete"] else None,
                     "known_cost_usd": round(float(values["known_cost"]), 9)}
                    for name, values in sorted(buckets.items())]
        by_agent = aggregate("agent")
        by_model = aggregate("model")
        unpriced = sorted({call["model"] for call in self.calls if call["pricing_status"] != "priced"})
        return {
            "currency": "USD", "estimate_only": True,
            "price_checked_on": PRICE_CHECKED_ON,
            "pricing_basis": "OpenAI standard online rates applied to API-reported tokens (including image input/reasoning output, not billing data)",
            "accounting_note": "Image input is already included in reported input tokens. Reasoning is already included in output tokens. Neither is charged a second time. Hidden SDK retries/requests without usage receipts cannot be reconciled with the invoice.",
            "input_images": sum(c["input_images"] for c in self.calls),
            "reasoning_tokens": sum(c["reasoning_tokens"] or 0 for c in self.calls),
            "failed_attempts": sum(c["call_status"] != "returned" for c in self.calls),
            "llm_calls_recorded": len(self.calls),
            "input_tokens": sum(c["input_tokens"] or 0 for c in self.calls),
            "cached_input_tokens": sum(c["cached_input_tokens"] or 0 for c in self.calls),
            "output_tokens": sum(c["output_tokens"] or 0 for c in self.calls),
            "complete": complete,
            "estimated_total_usd": round(float(total), 9) if complete else None,
            "known_cost_usd": round(float(total), 9),
            "unpriced_models": unpriced,
            "by_agent": by_agent,
            "by_model": by_model,
            "calls": [{key: val for key, val in c.items() if key != "estimated_cost_exact"} for c in self.calls],
        }


class MeteredClient:
    """Drop-in wrapper for client.responses.parse, including injected test clients."""
    def __init__(self, client: Any, tracker: CostTracker, budget: Any = None) -> None:
        self._client = client
        self.tracker = tracker
        self.budget = budget

    @property
    def responses(self) -> "MeteredClient":
        return self

    def parse(self, **kwargs: Any) -> Any:
        # Count attachment parts, not base64 characters as text tokens. The API
        # usage receipt already includes image input and hidden reasoning output.
        images = 0
        items = kwargs.get("input", [])
        if isinstance(items, list):
            for item in items:
                content = item.get("content") if isinstance(item, dict) else None
                if isinstance(content, list):
                    images += sum(isinstance(p, dict) and p.get("type") == "input_image" for p in content)
        kwargs = dict(kwargs)
        role = AGENT_BY_FORMAT.get(getattr(kwargs.get("text_format"), "__name__", ""), "Other")
        cap = 4096 if role == "Claim" else (
            2048 if role == "Summary" else 4096
        )
        requested_cap = kwargs.get("max_output_tokens", cap)
        if isinstance(requested_cap, bool) or not isinstance(requested_cap, int) or requested_cap <= 0:
            raise ValueError("max_output_tokens must be a positive integer; no request sent")
        kwargs["max_output_tokens"] = min(requested_cap, cap)
        if re.fullmatch(r"gpt-5-mini(?:-\d{4}-\d{2}-\d{2})?", str(kwargs.get("model", ""))):
            kwargs.setdefault("reasoning", {"effort": "low"})
        reservation = None
        if self.budget is not None:
            from app.budget import prepare_request
            kwargs, allowance, input_allowance = prepare_request(kwargs)
            role = AGENT_BY_FORMAT.get(getattr(kwargs.get("text_format"), "__name__", ""), "Other")
            reservation = self.budget.reserve(allowance, model=str(kwargs.get("model")), agent=role)
        try:
            response = self._client.responses.parse(**kwargs)
        except (Exception, KeyboardInterrupt) as exc:
            if getattr(exc, "request_not_sent", False):
                if reservation is not None:
                    self.budget.settle(reservation, Decimal(0))
                raise
            # A failed request may have incurred cost. Missing receipts are NOT
            # represented as $0, and error bodies/secrets are not stored here.
            self.tracker.add(str(kwargs.get("model") or "unknown"),
                             getattr(exc, "response_receipt", None),
                             kwargs.get("text_format"), input_images=images,
                             call_status=type(exc).__name__)
            if reservation is not None:
                self.budget.settle(reservation, self.tracker.calls[-1]["estimated_cost_exact"])
            raise
        self.tracker.add(str(kwargs.get("model") or "unknown"), response,
                         kwargs.get("text_format"), input_images=images)
        if reservation is not None:
            self.budget.settle(reservation, self.tracker.calls[-1]["estimated_cost_exact"])
        # Do not let a truncated structured response silently trigger retries.
        if _get(response, "status") == "incomplete":
            self.tracker.calls[-1]["call_status"] = "incomplete"
            from app.llm import StructuredOutputParseError
            raise StructuredOutputParseError(ValueError("Output limit reached; partial structured output not used; no automatic retry."),
                {k: _get(response, k) for k in ("id", "model", "usage", "service_tier", "status")})
        return response
