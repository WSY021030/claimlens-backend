"""Optional read-only HTTP adapter. Existing CLI analysis entry points are unchanged.

Start locally: python -m app.reader_read_api --project . --port 8010
Or mount: app.include_router(build_read_router(ReadRepository(project)))
Do not expose this single-user development service directly to the Internet.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import logging
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import APIRouter, FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.responses import FileResponse, JSONResponse, Response

from app.core.reader_queries import API_VERSION, ReadError, ReadRepository

LOG = logging.getLogger("reader.read_api")
MAX_RESPONSE_BYTES = 512 * 1024
MAX_SELECTION_BYTES = 128 * 1024
DEFAULT_ORIGINS = ("http://localhost:5173", "http://127.0.0.1:5173",
                   "http://localhost:3000", "http://127.0.0.1:3000")


def json_response(value, status=200):
    content = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if status == 200 and len(content) > MAX_RESPONSE_BYTES:
        return JSONResponse({"error": {"code": "VIEW_TOO_LARGE",
            "message": "This view exceeds the response limit. Reduce the page size or open one Evidence separately; no content was silently discarded.",
            "details": {"limit_bytes": MAX_RESPONSE_BYTES}}}, status_code=413)
    return Response(content, status_code=status, media_type="application/json",
                    headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
                             "X-Reader-API-Version": API_VERSION})


def call_read(request: Request, operation):
    try:
        result = {"status": "success", "data": operation()}
        response = json_response(result)
        if response.status_code == 200 and request.method == "GET":
            etag = '"' + sha256(response.body).hexdigest() + '"'
            response.headers["ETag"] = etag
            response.headers["Cache-Control"] = "private, no-cache"
            if request.headers.get("if-none-match") == etag:
                return Response(status_code=304, headers={"ETag": etag, "Cache-Control": "private, no-cache"})
        return response
    except ReadError as exc:
        return json_response(exc.envelope(), exc.status)
    except Exception:
        LOG.exception("Read-only result projection failed")
        return json_response({"error": {"code": "READ_ADAPTER_ERROR",
            "message": "The saved result could not be projected. Original analysis files were not changed.", "details": {}}}, 500)


def build_read_router(repository: ReadRepository) -> APIRouter:
    """Attach to an existing FastAPI app without replacing any routes or middleware.

    The parent app owns authentication/CORS/host policy when mounting this router.
    All paths here are read-only, including the POST selection lookup.
    """
    router = APIRouter(prefix="/api", tags=["Reader saved results (read-only)"])

    @router.get("/health")
    def health():
        return {"status": "success", "data": {"api_version": API_VERSION,
                "mode": "saved_results_read_only", "model_calls_enabled": False,
                "analysis_submission_enabled": False, "persistent_writes_enabled": False}}

    @router.get("/papers/{paper_id}")
    def overview(request: Request, paper_id: str, include_content: bool = False):
        if include_content:
            return json_response(ReadError("USE_SECTION_ENDPOINT", "Whole-paper content is not returned here. Load section elements on demand.").envelope(), 400)
        return call_read(request, lambda: repository.overview(paper_id))

    @router.get("/papers/{paper_id}/sections")
    def sections(request: Request, paper_id: str, offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=100)):
        return call_read(request, lambda: repository.sections(paper_id, offset, limit))

    @router.get("/papers/{paper_id}/sections/{section_id}/elements")
    def elements(request: Request, paper_id: str, section_id: str, offset: int = Query(0, ge=0),
                 limit: int = Query(20, ge=1, le=50), paper_revision: str | None = None):
        return call_read(request, lambda: repository.elements(paper_id, section_id, offset, limit, paper_revision))

    @router.get("/papers/{paper_id}/elements/{element_id}")
    def element(request: Request, paper_id: str, element_id: str, row_offset: int = Query(0, ge=0),
                row_limit: int = Query(20, ge=1, le=100), paper_revision: str | None = None):
        return call_read(request, lambda: repository.element(paper_id, element_id, row_offset, row_limit, paper_revision))

    @router.get("/papers/{paper_id}/claims")
    def claims(request: Request, paper_id: str, offset: int = Query(0, ge=0),
               limit: int = Query(50, ge=1, le=100), element_id: str | None = None,
               snapshot_id: str | None = None):
        return call_read(request, lambda: repository.claims(paper_id, offset, limit, element_id, snapshot_id))

    @router.post("/papers/{paper_id}/claims/selection")
    async def lookup_selection(request: Request, paper_id: str):
        """Filter saved Claims by exact spans. This endpoint does NOT start extraction."""
        if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
            return json_response(ReadError("JSON_REQUIRED", "Use application/json for this read-only selection lookup.", 415).envelope(), 415)
        received = bytearray()
        async for chunk in request.stream():
            received.extend(chunk)
            if len(received) > MAX_SELECTION_BYTES:
                return json_response(ReadError("SELECTION_TOO_LARGE", "The selection lookup body is too large.", 413).envelope(), 413)
        try:
            body = json.loads(received.decode("utf-8"))
        except (ValueError, UnicodeError):
            return json_response(ReadError("INVALID_JSON", "The request body is not valid JSON.").envelope(), 400)
        # This local read performs no asynchronous job submission or model call.
        from starlette.concurrency import run_in_threadpool
        return await run_in_threadpool(call_read, request, lambda: repository.selection_claims(paper_id, body))

    @router.get("/papers/{paper_id}/claims/{claim_id}")
    def claim(request: Request, paper_id: str, claim_id: str, mapping_id: str | None = None,
              evidence_offset: int = Query(0, ge=0), evidence_limit: int = Query(20, ge=1, le=100),
              snapshot_id: str | None = None):
        return call_read(request, lambda: repository.claim(paper_id, claim_id, mapping_id, evidence_offset, evidence_limit, snapshot_id))

    @router.get("/papers/{paper_id}/analysis")
    def focused_analysis(request: Request, paper_id: str, claim_id: str | None = None, mapping_id: str | None = None,
                         evidence_offset: int = Query(0, ge=0), evidence_limit: int = Query(20, ge=1, le=100),
                         snapshot_id: str | None = None):
        """Narrow replacement for an old draft whole-analysis endpoint."""
        if not claim_id:
            return json_response(ReadError("CLAIM_REQUIRED", "Specify claim_id; a whole analysis.json dump is not a public API.").envelope(), 400)
        return call_read(request, lambda: repository.claim(paper_id, claim_id, mapping_id, evidence_offset, evidence_limit, snapshot_id))

    @router.get("/papers/{paper_id}/evidence")
    def evidence_list(request: Request, paper_id: str, offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=100),
                      element_id: str | None = None, kind: str | None = None, snapshot_id: str | None = None):
        return call_read(request, lambda: repository.evidence_list(paper_id, offset, limit, element_id, kind, snapshot_id))

    @router.get("/papers/{paper_id}/evidence/{evidence_id}")
    def evidence(request: Request, paper_id: str, evidence_id: str, claim_id: str | None = None, mapping_id: str | None = None,
                 row_offset: int = Query(0, ge=0), row_limit: int = Query(20, ge=1, le=100), snapshot_id: str | None = None):
        return call_read(request, lambda: repository.evidence(paper_id, evidence_id, claim_id, mapping_id, row_offset, row_limit, snapshot_id))

    @router.get("/papers/{paper_id}/response")
    def response(request: Request, paper_id: str, workflow_id: str | None = None, snapshot_id: str | None = None):
        return call_read(request, lambda: repository.response(paper_id, workflow_id, snapshot_id))

    @router.get("/papers/{paper_id}/assets/{asset_id}")
    def asset(request: Request, paper_id: str, asset_id: str, paper_revision: str | None = None):
        try:
            paper = repository.paper(paper_id)
            if paper_revision and paper_revision != paper.revision:
                raise ReadError("REVISION_MISMATCH", "The image belongs to a different paper revision; refresh the reader.", 409)
            path, media_type = repository.asset(paper_id, asset_id)
            return FileResponse(path, media_type=media_type, headers={
                "Cache-Control": "private, no-cache", "X-Content-Type-Options": "nosniff",
                "Content-Security-Policy": "sandbox; default-src 'none'; style-src 'unsafe-inline'; img-src data:",
                "Cross-Origin-Resource-Policy": "cross-origin"})
        except ReadError as exc:
            return json_response(exc.envelope(), exc.status)
        except OSError:
            return json_response(ReadError("ASSET_READ_FAILED", "The registered asset is temporarily unavailable.", 503).envelope(), 503)

    return router


def create_app(project: str | Path | None = None, *, allowed_origins=None) -> FastAPI:
    root = Path(project) if project is not None else Path(__file__).resolve().parents[1]
    repository = ReadRepository(root)
    origins = list(DEFAULT_ORIGINS if allowed_origins is None else allowed_origins)
    for origin in origins:
        parsed = urlsplit(origin)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.path not in {"", "/"} or parsed.query or parsed.fragment or parsed.username:
            raise ValueError("Use an explicit HTTP(S) frontend origin without a path or credentials.")
        if origin.endswith("/"):
            raise ValueError("CORS origins must not have a trailing slash.")
    app = FastAPI(title="Reader-v3 On-demand Saved Results", version="1.0.0",
                  description="Read-only local adapter. No upload, analysis submission, chat generation, reset, or model calls.")
    app.include_router(build_read_router(repository))

    @app.exception_handler(RequestValidationError)
    async def bad_request(request: Request, exc: RequestValidationError):
        return json_response(ReadError("INVALID_PARAMETERS", "Query parameters have invalid types or ranges.", 422).envelope(), 422)

    @app.middleware("http")
    async def restrict_origin(request: Request, call_next):
        origin = request.headers.get("origin")
        if origin and origin not in origins:
            return json_response(ReadError("ORIGIN_NOT_ALLOWED", "This frontend origin is not allowed by the local read service.", 403).envelope(), 403)
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        return response

    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]"])
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=False,
                       allow_methods=["GET", "POST", "HEAD", "OPTIONS"],
                       allow_headers=["Content-Type", "If-None-Match"], expose_headers=["ETag", "X-Reader-API-Version"])
    return app


def main():
    parser = argparse.ArgumentParser(description="Local read-only Reader-v3 HTTP service")
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--host", choices=["127.0.0.1", "localhost", "::1"], default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--allow-origin", action="append", help="Allowed frontend origin; repeat as needed (replaces defaults)")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Port must be 1..65535.")
    import uvicorn
    print("Read-only result service. No source writes or model calls are enabled.", flush=True)
    print("Open /docs for the read API. Existing analysis commands are unchanged.", flush=True)
    uvicorn.run(create_app(args.project, allowed_origins=args.allow_origin), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
