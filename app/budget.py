"""Persistent per-question budget for the Reader-v3 pipeline.

Each user-question workflow has an independent configurable ceiling (default $0.50) with 10% headroom.
All paid Coordinator/Claim/Evidence/Mapping requests under one workflow share
the same question budget, including retries and workflow resume. Historical
per-paper records remain in SQLite for auditing but never fund new questions.
A reservation is persisted before an API request; this is an estimate guard,
NOT a guarantee of the provider invoice or protection against hidden retries.
"""
from __future__ import annotations

import base64
from contextlib import contextmanager
from decimal import Decimal, ROUND_CEILING
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import threading
from typing import Any
from uuid import uuid4

from app.usage_cost import AGENT_BY_FORMAT, MODEL_PRICES, _prices_for

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = PROJECT_ROOT / "output" / "paper_budget.sqlite3"
QUESTION_BUDGET_ENV = "READER_V3_QUESTION_BUDGET_ID"
DEFAULT_HISTORY = PROJECT_ROOT / "output" / "api_cost_history.jsonl"
NANO = Decimal("1000000000")
HARD_LIMIT = Decimal("0.50")
QUESTION_LIMIT_ENV = "READER_V3_QUESTION_LIMIT_USD"
# Includes hidden reasoning tokens, not just the visible JSON.
OUTPUT_LIMITS = {"Claim": 4096, "Evidence": 4096, "Summary": 4096, "Mapping": 4096, "Coordinator": 4096}


class BudgetStop(RuntimeError):
    """A preflight stop: no generation request was made for this attempt."""
    request_not_sent = True


def money(value: Any) -> Decimal:
    try:
        value = Decimal(str(value))
    except Exception as exc:
        raise ValueError("Invalid USD amount") from exc
    if not value.is_finite() or value < 0:
        raise ValueError("USD amount must be finite and nonnegative")
    return value


def nanos(value: Any) -> int:
    return int((money(value) * NANO).to_integral_value(rounding=ROUND_CEILING))


def usd(value: int) -> float:
    return float(Decimal(value) / NANO)


def model_family(model: str) -> str:
    for name in MODEL_PRICES:
        if model == name or re.fullmatch(re.escape(name) + r"-\d{4}-\d{2}-\d{2}", model):
            return name
    raise BudgetStop(f"Budget guard: no checked price for model {model!r}; no request sent.")


def prepare_request(kwargs: dict[str, Any]) -> tuple[dict[str, Any], Decimal, int]:
    """Conservative local input allowance + maximum output; assume NO cache hit.

    Text allowance uses UTF-8 bytes (not chars/4), plus double schema bytes and
    framing headroom. Inline images use an intentionally high token allowance.
    Unknown input types, remote images, paid tools and nonstandard tiers stop.
    This is an estimate under the checked prices; the provider invoice wins.
    """
    request = dict(kwargs)
    model = str(request.get("model") or "")
    family = model_family(model)
    role = AGENT_BY_FORMAT.get(getattr(request.get("text_format"), "__name__", ""), "Other")
    cap = OUTPUT_LIMITS.get(role, 4096)
    requested = request.get("max_output_tokens", cap)
    if isinstance(requested, bool) or not isinstance(requested, int) or requested <= 0:
        raise BudgetStop("Budget guard: invalid output token limit; no request sent.")
    request["max_output_tokens"] = min(requested, cap)
    if request.get("service_tier") not in (None, "default", "auto"):
        raise BudgetStop("Budget guard supports standard/default API pricing only.")
    request["service_tier"] = "default"
    for key in ("tools", "previous_response_id", "conversation", "background", "stream", "extra_body"):
        if request.get(key):
            raise BudgetStop(f"Budget guard cannot bound {key}; no request sent.")
    if family == "gpt-5-mini":
        # Use an explicit low effort rather than unbounded/default reasoning.
        request["reasoning"] = {"effort": "low"}

    image_tokens = 0
    def scrub(obj: Any) -> Any:
        nonlocal image_tokens
        if isinstance(obj, dict):
            kind = obj.get("type", "")
            if kind in {"input_file", "input_audio", "input_video"}:
                raise BudgetStop(f"Cannot estimate {kind} before sending.")
            if kind == "input_image":
                url = obj.get("image_url", "")
                if not isinstance(url, str) or not url.startswith("data:image/") or ";base64," not in url:
                    raise BudgetStop("Budget guard requires prepared local inline images.")
                try:
                    from PIL import Image
                    raw = base64.b64decode(url.split(",", 1)[1], validate=True)
                    with Image.open(io.BytesIO(raw)) as im:
                        width, height = im.size
                    if width <= 0 or height <= 0:
                        raise ValueError("Invalid dimensions")
                except Exception as exc:
                    raise BudgetStop("Cannot inspect image dimensions; no request sent.") from exc
                if family == "gpt-4o-mini":
                    # Conservative maximum for tile-based mini, independent of size.
                    image_tokens += 100000
                else:
                    # Unresized patches * 2 is above the published multipliers
                    # for the two supported patch models. 512 covers framing.
                    image_tokens += math.ceil(width / 32) * math.ceil(height / 32) * 2 + 512
                return {"type": "input_image", "detail": obj.get("detail", "high")}
            return {k: scrub(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [scrub(v) for v in obj]
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        raise BudgetStop("Unsupported request content; no request sent.")
    body = {k: v for k, v in request.items() if k != "text_format"}
    text_bytes = len(json.dumps(scrub(body), ensure_ascii=False).encode("utf-8"))
    # A conservative request-size safety brake. This is BYTES, not the actual
    # API token count; the latter is always taken from the usage receipt.
    try:
        max_text_bytes = int(os.environ.get("READER_V3_MAX_REQUEST_TEXT_BYTES", "110000"))
    except ValueError as exc:
        raise BudgetStop("Invalid request-text byte cap; no request sent.") from exc
    if max_text_bytes < 10000:
        raise BudgetStop("Request-text byte cap must be >= 10000; no request sent.")
    if text_bytes > max_text_bytes:
        raise BudgetStop(
            f"Input request is {text_bytes} bytes, exceeding the per-call "
            f"limit of {max_text_bytes} bytes. Compact source material before "
            "running this task; no API request was sent."
        )
    schema = request.get("text_format")
    schema_bytes = len(json.dumps(schema.model_json_schema(), ensure_ascii=False).encode("utf-8")) if schema is not None else 0
    input_allowance = text_bytes + 2 * schema_bytes + 2048 + image_tokens
    rates = _prices_for(model)
    allowance = (Decimal(input_allowance) * rates[0] +
                 Decimal(request["max_output_tokens"]) * rates[2]) / Decimal(1000000)
    return request, allowance, input_allowance


class PaperBudget:
    """One durable USD budget per (paper_id, question_budget_id).

    The legacy class name remains for compatibility with existing agents.
    The main workflow sets QUESTION_BUDGET_ENV before every paid request and
    child subprocess; stand-alone agent commands receive independent budget
    IDs for their own invocation unless an explicit ID is supplied.
    """

    def __init__(self, paper_id: str, *, question_id: str | None = None,
                 limit_usd: Any = None, db_path: str | Path | None = None,
                 history_paths: list[Path] | None = None):
        self.paper_id = str(paper_id)
        self.limit = money(
            limit_usd if limit_usd is not None
            else os.environ.get(QUESTION_LIMIT_ENV, str(HARD_LIMIT))
        )
        self.question_id = str(
            question_id or os.environ.get(QUESTION_BUDGET_ENV)
            or ('standalone-' + uuid4().hex)
        )
        if not self.paper_id or not self.question_id or self.limit <= 0:
            raise ValueError(
                "Question budget requires a paper ID, question ID and a "
                "positive explicit limit."
            )
        if len(self.question_id) > 128 or not re.fullmatch(r"[A-Za-z0-9_.:-]+", self.question_id):
            raise ValueError("Invalid question budget ID")
        self.stop_reason: str | None = None
        self.db_path = Path(db_path) if db_path is not None else DEFAULT_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(
            str(self.db_path), timeout=15, isolation_level=None,
            check_same_thread=False,
        )
        self._db.execute("PRAGMA synchronous=FULL")
        # DO NOT rename, drop, truncate or migrate the legacy `budget_entries`
        # table: its historical paper-level reservations are preserved intact.
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS question_budget_entries (
                entry_id TEXT PRIMARY KEY,
                paper_id TEXT NOT NULL,
                question_id TEXT NOT NULL,
                amount INTEGER NOT NULL,
                state TEXT NOT NULL,
                note TEXT NOT NULL
            )
        """)
        self._db.execute("""
            CREATE INDEX IF NOT EXISTS question_budget_scope
            ON question_budget_entries (paper_id, question_id)
        """)
        # Old api_cost_history.jsonl receipts have no reliable question ID,
        # therefore they must not be imported into a fresh question's budget.
        # Existing files and SQLite history remain untouched.

    def close(self) -> None:
        """Release the local SQLite handle; historical rows remain intact."""
        self._db.close()

    @contextmanager
    def transaction(self):
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                yield self._db
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise

    def _rows(self, db):
        return db.execute(
            "SELECT amount, state FROM question_budget_entries "
            "WHERE paper_id=? AND question_id=?",
            (self.paper_id, self.question_id),
        ).fetchall()

    def _legacy_paper_spending(self, db):
        # Information only. This amount never reduces this question's balance.
        exists = db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' "
            "AND name='budget_entries'"
        ).fetchone()
        if not exists:
            return 0
        return db.execute(
            "SELECT COALESCE(SUM(amount),0) FROM budget_entries "
            "WHERE paper_id=?", (self.paper_id,)
        ).fetchone()[0]

    def report(self) -> dict[str, Any]:
        with self._lock:
            rows = self._rows(self._db)
            historical = self._legacy_paper_spending(self._db)
        committed = sum(amount for amount, _ in rows)
        uncertain = sum(
            state in {"unknown", "reserved"} for _, state in rows
        )
        operating = nanos(self.limit * Decimal("0.90"))
        return {
            "paper_id": self.paper_id,
            "question_budget_id": self.question_id,
            "budget_policy": "per_question_v1",
            "limit_usd": float(self.limit),
            "operating_limit_usd": usd(operating),
            "safety_margin_usd": float(self.limit * Decimal("0.10")),
            "committed_usd": usd(committed),
            "remaining_usd": usd(max(0, operating - committed)),
            "uncertain_entries": uncertain,
            "legacy_paper_spending_usd": usd(historical),
            "blocked": bool(
                uncertain or committed >= operating or self.stop_reason
            ),
            "stop_reason": self.stop_reason,
            "ledger_path": str(self.db_path),
            "scope": (
                "One user-question workflow: planning, all agents, "
                "retries and resume share the same budget ID. "
                "Other questions do not consume this limit."
            ),
            "estimate_only": True,
        }

    def reserve(self, allowance: Decimal, *, model: str, agent: str) -> str:
        amount = nanos(allowance)
        entry_id = uuid4().hex
        with self.transaction() as db:
            rows = self._rows(db)
            if any(state in {"unknown", "reserved"} for _, state in rows):
                self.stop_reason = (
                    "Question budget paused: an unknown or in-flight request "
                    "exists for this question. No new request sent."
                )
                raise BudgetStop(self.stop_reason)
            total = sum(value for value, _ in rows)
            operating = nanos(self.limit * Decimal("0.90"))
            if total + amount > operating:
                self.stop_reason = (
                    f"Question budget stopped before {agent}: remaining "
                    f"${usd(max(0, operating - total)):.6f}, requested "
                    f"reservation ${usd(amount):.6f}. No request sent."
                )
                raise BudgetStop(self.stop_reason)
            db.execute(
                "INSERT INTO question_budget_entries "
                "(entry_id,paper_id,question_id,amount,state,note) "
                "VALUES (?,?,?,?,?,?)",
                (
                    entry_id, self.paper_id, self.question_id, amount,
                    "reserved", f"{agent}/{model}; persisted before request",
                ),
            )
        return entry_id

    def settle(self, entry_id: str, cost: Any | None) -> None:
        with self.transaction() as db:
            previous = db.execute(
                "SELECT amount FROM question_budget_entries "
                "WHERE entry_id=? AND paper_id=? AND question_id=?",
                (entry_id, self.paper_id, self.question_id),
            ).fetchone()
            if previous is None:
                raise RuntimeError("Missing question-budget reservation")
            if cost is None:
                # API may have charged a request without returning its usage.
                db.execute(
                    "UPDATE question_budget_entries SET state='unknown' "
                    "WHERE entry_id=?", (entry_id,),
                )
                self.stop_reason = (
                    "Usage receipt missing/unpriced; reservation retained "
                    "and further requests for this question blocked."
                )
            else:
                actual = nanos(cost)
                state = "settled" if actual <= previous[0] else "unknown"
                db.execute(
                    "UPDATE question_budget_entries SET amount=?, state=? "
                    "WHERE entry_id=?", (actual, state, entry_id),
                )
                if state == "unknown":
                    self.stop_reason = (
                        "Observed cost exceeded preflight allowance; "
                        "further requests for this question blocked."
                    )

    def acknowledge_unknown_reservation(self, entry_id: str) -> None:
        """Manual review retains the full reservation; never charges $0."""
        with self.transaction() as db:
            row = db.execute(
                "SELECT state,note FROM question_budget_entries "
                "WHERE entry_id=? AND paper_id=? AND question_id=?",
                (entry_id, self.paper_id, self.question_id),
            ).fetchone()
            if row is None or row[0] != "unknown":
                raise ValueError("Unknown reservation not found for this question")
            if not row[1].startswith("Coordinator/"):
                raise ValueError("Only Coordinator reservations support review")
            db.execute(
                "UPDATE question_budget_entries SET state='reviewed_unknown', "
                "note=? WHERE entry_id=?",
                (row[1] + "; manually reviewed; full allowance retained", entry_id),
            )
        self.stop_reason = None

    def reconcile_legacy(self, confirmed_upper_usd: Any) -> None:
        raise ValueError(
            "Legacy paper-level spending is preserved for audit and cannot "
            "be used to refill or reset a question budget."
        )


# Preferred name for new imports; PaperBudget remains a compatible alias.
QuestionBudget = PaperBudget
