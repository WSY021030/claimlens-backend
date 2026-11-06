"""Local FastAPI server for saved-result reads and question submission.

Development-only service. Do not expose it directly to the Internet.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path
from typing import Literal

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field

from app.reader_read_api import create_app as create_read_app
from app.core.request_contract import (
    ReaderRequest, SelectionReference, ConversationMessage,
    validate_selection_reference,
)
from app.models.document_v2 import PaperDocument


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPER_ID_RE = re.compile(r"^[A-Za-z0-9_-]{2,100}$")
TASK_ID_RE = re.compile(r"^[0-9a-f]{20}$")
WORKFLOW_ID_RE = re.compile(r"^[0-9a-f]{12}$")
REQUEST_ID_RE = r"^[A-Za-z0-9_-]{8,80}$"


class QuestionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    client_request_id: str = Field(pattern=REQUEST_ID_RE)
    question: str = Field(min_length=1, max_length=8000)
    language: Literal["en", "zh"] = "en"
    question_limit_usd: float = Field(default=0.50, gt=0, le=0.50)
    confirm_spend: bool = False
    selection_context: SelectionReference | None = None
    chat_history: list[ConversationMessage] = Field(
        default_factory=list, max_length=6
    )


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def create_app(project: Path | str = PROJECT_ROOT):
    root = Path(project).resolve()
    app = create_read_app(root)

    task_dir = root / "data" / "api_tasks"
    lock_dir = task_dir / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)

    mutex = threading.RLock()
    active_tasks: set[str] = set()

    def task_file(task_id: str) -> Path:
        if not TASK_ID_RE.fullmatch(task_id):
            raise HTTPException(422, "Invalid task ID.")
        return task_dir / f"{task_id}.json"

    def load_task(task_id: str) -> dict:
        path = task_file(task_id)

        if not path.is_file():
            raise HTTPException(404, "Task not found.")

        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise HTTPException(503, "Task state is unavailable.")

        if result["state"] in {"queued", "running"}:
            if task_id not in active_tasks:
                result["state"] = "needs_reconciliation"
                result["message"] = (
                    "The server may have restarted. "
                    "Check the original workflow before resuming."
                )

        return result

    def run_question(
        record: dict,
        paper_path: Path,
        request: QuestionRequest,
        lock_path: Path,
    ) -> None:
        task_id = record["task_id"]

        command = [
            sys.executable,
            "-m",
            "app.workflow_runner",
            "--paper",
            str(paper_path),
            "--question",
            request.question.strip(),
            "--language",
            request.language,
            "--question-limit-usd",
            str(request.question_limit_usd),
        ]

        request_file = None
        try:
            if request.selection_context is not None or request.chat_history:
                paper = PaperDocument.model_validate_json(
                    paper_path.read_text(encoding="utf-8-sig")
                )
                structured = ReaderRequest(
                    paper_id=paper.paper_id,
                    paper_revision=paper.revision,
                    question=request.question.strip(),
                    selection_context=request.selection_context,
                    chat_history=request.chat_history,
                )
                validate_selection_reference(
                    structured.selection_context, paper
                )
                request_file = task_dir / f"{task_id}.request.json"
                write_json(
                    request_file, structured.model_dump(mode="json")
                )
                command[5:7] = ["--request-file", str(request_file)]
            record["state"] = "running"
            write_json(task_file(task_id), record)

            workflow_id = None
            workflow_status = None

            with subprocess.Popen(
                command,
                cwd=root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                shell=False,
            ) as process:
                if process.stdout is not None:
                    for line in process.stdout:
                        line = line.strip()

                        if line.startswith("Workflow ID:"):
                            value = line.partition(":")[2].strip()
                            if WORKFLOW_ID_RE.fullmatch(value):
                                workflow_id = value
                                record["workflow_id"] = value
                                write_json(task_file(task_id), record)

                        elif line.startswith("Status:"):
                            workflow_status = (
                                line.partition(":")[2].strip().lower()
                            )

                exit_code = process.wait()

            record["exit_code"] = exit_code

            if exit_code != 0:
                record["state"] = "failed"
                record["error_code"] = "WORKFLOW_FAILED"
            elif workflow_status == "completed":
                record["state"] = "completed"
            elif workflow_status == "partial":
                record["state"] = "partial"
            else:
                record["state"] = "failed"
                record["error_code"] = "STATUS_UNCONFIRMED"

            if workflow_id:
                record["response_url"] = (
                    f"/api/papers/{record['paper_id']}/response"
                    f"?workflow_id={workflow_id}"
                )

        except Exception:
            record["state"] = "failed"
            record["error_code"] = "WORKER_ERROR"

        finally:
            if request_file is not None:
                try:
                    request_file.unlink(missing_ok=True)
                except OSError:
                    pass
            write_json(task_file(task_id), record)

            try:
                if lock_path.read_text(encoding="utf-8") == task_id:
                    lock_path.unlink()
            except OSError:
                pass

            with mutex:
                active_tasks.discard(task_id)

    @app.post("/api/papers/{paper_id}/questions", status_code=202)
    def submit_question(paper_id: str, request: QuestionRequest):
        if not PAPER_ID_RE.fullmatch(paper_id):
            raise HTTPException(422, "Invalid paper ID.")

        question = request.question.strip()

        if not question:
            raise HTTPException(422, "Question cannot be empty.")

        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "paper_id": paper_id,
                    "question": question,
                    "language": request.language,
                    "question_limit_usd": request.question_limit_usd,
                    "selection_context": (
                        request.selection_context.model_dump(mode="json")
                        if request.selection_context else None
                    ),
                    "chat_history": [
                        m.model_dump(mode="json")
                        for m in request.chat_history
                    ],
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

        task_id = hashlib.sha256(
            f"{paper_id}:{request.client_request_id}".encode("utf-8")
        ).hexdigest()[:20]

        with mutex:
            if task_file(task_id).exists():
                previous = load_task(task_id)

                if previous.get("fingerprint") != fingerprint:
                    raise HTTPException(
                        409, "Request ID already used with different inputs."
                    )

                return {"status": "accepted", "data": previous}

            if request.confirm_spend is not True:
                raise HTTPException(
                    409, "Explicit model spending confirmation is required."
                )

            if not os.environ.get("OPENAI_API_KEY"):
                raise HTTPException(
                    409, "Backend OPENAI_API_KEY is not configured."
                )

            paper_path = (
                root / "data" / "standardized" / paper_id / "paper.json"
            )

            if not paper_path.is_file() or paper_path.is_symlink():
                raise HTTPException(
                    404, "Standardized paper not found."
                )
            if request.selection_context is not None:
                try:
                    paper = PaperDocument.model_validate_json(
                        paper_path.read_text(encoding="utf-8-sig")
                    )
                    if paper.paper_id != paper_id:
                        raise ValueError("Paper identifier mismatch.")
                    validate_selection_reference(
                        request.selection_context, paper
                    )
                except (OSError, ValueError) as exc:
                    raise HTTPException(
                        422, f"Invalid selection context: {exc}"
                    ) from exc

            lock_path = lock_dir / f"{paper_id}.lock"

            try:
                with lock_path.open("x", encoding="utf-8") as stream:
                    stream.write(task_id)
            except FileExistsError:
                raise HTTPException(
                    409, "Another task may already be using this paper."
                )

            record = {
                "task_id": task_id,
                "paper_id": paper_id,
                "client_request_id": request.client_request_id,
                "fingerprint": fingerprint,
                "state": "queued",
                "workflow_id": None,
                "response_url": None,
                "exit_code": None,
                "error_code": None,
                "status_url": f"/api/tasks/{task_id}",
            }

            try:
                write_json(task_file(task_id), record)

                active_tasks.add(task_id)

                worker = threading.Thread(
                    target=run_question,
                    args=(record, paper_path, request, lock_path),
                    daemon=True,
                )
                worker.start()
            except Exception:
                active_tasks.discard(task_id)
                lock_path.unlink(missing_ok=True)
                raise

        return {"status": "accepted", "data": record}

    @app.get("/api/tasks/{task_id}")
    def get_task(task_id: str):
        return {
            "status": "success",
            "data": load_task(task_id),
        }

    return app


def main() -> None:
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()

    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535.")

    uvicorn.run(
        create_app(args.project),
        host="127.0.0.1",
        port=args.port,
    )


if __name__ == "__main__":
    main()