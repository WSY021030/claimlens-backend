# Reader-v3: on-demand saved-result API

Version: `reader.read.v1`. Implementation delivered; **runtime integration has not been tested**.

## Boundary

This is an additive **read-only adapter**, not an agent/pipeline refactor. The installer
creates two Python modules and this handoff. It does not replace an existing source
file, change project dependencies, migrate saved data, delete records, or edit a
budget ledger. The current-state storage update is optional for this adapter: old
`reader.v3` analysis snapshots and compacted snapshots can both be read.

This document supersedes the earlier instruction to download and join the complete
`analysis.json`, `display.json`, and `response.json` in the browser. Those files are
internal storage/optional exports, not a public response contract. **This service
reads the authoritative analysis snapshot and builds the requested view in memory.**
It never reads `display.json` or `response.json`, and never creates a result JSON.

No read endpoint invokes an LLM, extraction, Mapping, automatic repair, a workflow,
or a new budget. Invalid/missing results are reported, not regenerated. Existing
CLI commands for Claim-only, independent Evidence, selection analysis, Mapping,
and questions remain unchanged. This release does NOT implement upload, submit
analysis, submit chat, job execution, or permanent conversation history. A POST to
the selection-lookup endpoint is still only a local read.

## Installation and startup

Put `reader_v3_on_demand_update.py` in the project root. Check/install:

```powershell
Set-Location "D:\新建文件夹\paper_comprehending"
uv run --no-project --isolated python .\reader_v3_on_demand_update.py --project "." --check
uv run --no-project --isolated python .\reader_v3_on_demand_update.py --project "." --apply
```

Start in a dedicated terminal. This uses an isolated optional environment rather
than editing `pyproject.toml`, `uv.lock`, or the existing project environment:

```powershell
uv run --no-project --isolated `
  --with "fastapi>=0.115,<1" `
  --with "uvicorn>=0.30,<1" `
  --with "pydantic>=2,<3" `
  python -m app.reader_read_api --project "." --port 8010
```

Open `http://127.0.0.1:8010/docs` or `http://127.0.0.1:8010/api/health`.
Keep this terminal open; Ctrl+C stops only the optional reader service. The
analysis CLIs do not import these new modules and continue to work independently.
No API key is required for this read service. Installing dependencies can contact
the package index, but reading results does not contact a model provider.

Default CORS origins are localhost/127.0.0.1 on ports 5173 and 3000. To replace them,
repeat `--allow-origin "http://localhost:YOUR_PORT"`. Do not use a wildcard.
The standalone server deliberately binds only to loopback. It is a local single-user
integration tool, **not a production multi-user server**. Before remote deployment,
provide authentication, authorization, HTTPS, request limits, and a suitable server
configuration. Do not expose the entire project directory as static files.

A developer with an existing FastAPI service may mount the router instead of
running a second process. Check for path conflicts first; this does not replace an
existing app or its security middleware:

```python
from pathlib import Path
from app.core.reader_queries import ReadRepository
from app.reader_read_api import build_read_router

repository = ReadRepository(Path("/path/to/project"))
app.include_router(build_read_router(repository))
```

The host app owns CORS, authentication, request-validation handling and host checks
when mounting. This router uses the `/api` prefix and returns root-relative `/api/...`
URLs; hosting it under an additional path prefix requires adapting URL generation.

## Read operations

Every JSON success response is `{"status":"success","data":{...}}`. Errors are
`{"error":{"code":"...","message":"...","details":{...}}}`. IDs are opaque;
never turn the trailing number of a Claim into an Evidence/Mapping ID.

| Request | Returned data | Not returned |
|---|---|---|
| `GET /api/papers/{paper_id}` | Metadata, counts, quality, route URLs | Full paper body, pools, execution histories |
| `GET /api/papers/{paper_id}/sections?offset=0&limit=50` | Section outline page | All section text |
| `GET /api/papers/{paper_id}/sections/{section_id}/elements?offset=0&limit=20` | Canonical paragraph text and element descriptors | Table grids and image bytes |
| `GET /api/papers/{paper_id}/elements/{element_id}` | One canonical source element; table rows paged | Unrelated elements |
| `GET /api/papers/{paper_id}/claims?offset=0&limit=50` | Lightweight Claim cards and current Mapping IDs | Mapping bodies, full Evidence pool |
| `POST /api/papers/{paper_id}/claims/selection` | Existing Claims inside exact paragraph spans | New Claim extraction or source writes |
| `GET /api/papers/{paper_id}/claims/{claim_id}` | One Claim, its exact Mapping, adopted Evidence cards, highlights and Graph | Other Claims' arguments or candidate Evidence |
| `GET /api/papers/{paper_id}/analysis?claim_id=...` | Alias of the focused Claim view | No unfiltered analysis dump |
| `GET /api/papers/{paper_id}/evidence?offset=0&limit=50` | Standalone saved Evidence cards | Logical support inferred from pool membership |
| `GET /api/papers/{paper_id}/evidence/{evidence_id}` | One saved Evidence; table rows paged | Other Evidence records |
| `GET /api/papers/{paper_id}/response` | Most recent recorded operation's answer/status/usage summary | Runs, prompts, API receipts, candidates, checkpoints |
| `GET /api/papers/{paper_id}/assets/{asset_id}` | Registered local image | Arbitrary paths, raw HTML, JSON, keys, budget DB |

`include_content=true` on the overview is rejected: use section-element pages.
`GET .../analysis` without `claim_id` is rejected rather than returning the pool.
The old v0.2 HTTP document was a draft; these are intentionally narrower **read**
contracts, not a claim that the remaining draft write routes now exist.

## Clicking a Claim

Example (IDs are from the user's experiment; the endpoint is not pre-tested):

```text
GET /api/papers/acm_10_1145_3653982/claims/claim_000002
```

The response contains `claim`, `mapping`, `evidence`, `display`, `evidence_page`,
`next_evidence_url`, `issues`, `warnings`, and identity/version fields. The Mapping
body occurs once. Evidence cards have short **explicitly marked previews** and
`detail_url`; complete passages/cells are loaded when opened, without a model call.

- A current Mapping pointer is authoritative. A broken/invalid active pointer
  yields unavailable; the API does not silently fall back to an old result.
- Optional `mapping_id` pins a specific saved version. A missing pin is an error.
- `display.evidence_ids`, highlights, nodes, and edges are filtered together from
  this Claim's adopted Evidence. There is no Link-based or whole-pool fallback.
- Every Graph edge retains its saved/projected relation and source-part metadata.
  A null relation remains null. `needs_origin_review` and `not_recorded` are not
  synonyms for `supports`. The existing origin guard is reused, not reimplemented.
- `completed` means processing completed, not scientific verification. Keep
  `material_status`, qualifications, provenance warnings, and review flags visible.
- This adapter preserves known model/citation mistakes. It does not certify a
  table comparison, invent missing evidence, or rewrite scientific explanations.

Default Evidence/Graph page size is 20, maximum 100. Larger used sets have
`next_evidence_url`; follow it and merge only pages with the same `snapshot_id` and
`mapping_id`. Do not label a partial Graph as complete. Single-Claim pagination is
presentation pagination, not a change to the saved Mapping's adopted Evidence set.

## Source detail and independent Evidence

Cards returned inside a Claim view have `detail_url` bound to `claim_id`,
`mapping_id`, and `snapshot_id`. Following that URL returns only components the
Mapping recorded as using: source text/context, caption, visual observation, cells,
or equation. It does not implicitly replace caption-only support with pixel data.

A direct unbound Evidence request is a distinct **standalone inspection** operation.
It can read extracted Evidence not used by any Mapping. This preserves independent
Evidence use without manufacturing a Claim relation. A bound read of an Evidence
not adopted by that Mapping is rejected. Historical unknown component usage is
reported explicitly and an optional standalone inspection URL is supplied.

Tables use `row_offset`/`row_limit` (default 20, max 100); these count stored rows,
not arbitrary cell slices. Original header cells and saved row-group context remain
available. `source_anchor.selector.cell_ids` defines the actual cited selection;
header/context cells are not extra logical evidence. Row-group labels retain their
original best-effort status. A record can cover only part of an original table.
No unit conversion, numeric rewriting, or dataset relabeling happens here.

Image URLs include the paper revision. Use images through `<img>`; never inject SVG
or publisher HTML through `innerHTML`. Mathematical markup is returned as source
text and must be sanitized with a frontend allowlist before rendering. Preserve
canonical paragraph text for selection; formatting is not permission to rewrite it.

## Exact selection lookup

This endpoint only returns already-extracted Claims fully inside the supplied
ranges. It does not verify that extraction previously covered every selected span.
An empty result therefore does not prove the selected text contains no Claims.

```json
{
  "paper_revision": "<revision from the paper response>",
  "segments": [
    {"element_id": "<paragraph ID>", "start": 0, "end": 42, "exact": "<exact canonical substring>"}
  ],
  "offset": 0,
  "limit": 50
}
```

Offsets are Unicode **code points**, end-exclusive, measured in canonical
`Paragraph.text`. JavaScript DOM/string indices are UTF-16; convert deliberately.
For multiple paragraphs send a segment per paragraph. The API rejects inconsistent
revisions or text rather than guessing a boundary. Adjacent/overlapping selected
ranges are merged only for the local containment lookup. Images/tables are not
mixed into this text selection.

## Versions, paging, and errors

`snapshot_id` is an opaque hash of the actual saved analysis file bytes. It is a
read-view identity, not necessarily the same string as an older internal export's
snapshot field. Send it on subsequent Claim/Evidence/selection pages and when
following Evidence details. A change produces HTTP 409 `SNAPSHOT_CHANGED`; reload
the current view. The API itself assembles a response from one analysis snapshot,
so the frontend never compares three disk exports.

Source-reader pages use `paper_revision`, independent of analysis progress. Pass
it on subsequent element pages and table detail calls. The server checks a supplied
revision. Missing analysis on a registered paper gives empty lists/not_analyzed,
not automatic extraction. A stale analysis revision returns HTTP 409.

The `response` endpoint defaults to the operation with the most recent recorded
activity, accounting for parent/child IDs. For a particular question use
`?workflow_id=<its ID>`, especially while polling. It never silently substitutes a
new task for a requested task that was pruned. It returns saved progress; this is
not a new task scheduler, chat-history service, or live model-token stream.

JSON responses have a 512 KiB guard. An oversized atomic object returns explicit
HTTP 413 instead of cutting scientific source text. Lists and table rows have
pagination. An unusually huge single record/header set may still require a future
specialized viewer; this implementation does not promise all inputs fit one page.
The local backend caches parsed files by file metadata and invalidates them on
change. It still reads the whole internal file on a cache miss, but the browser
receives only the requested projection. Cache limits are based on source bytes,
not a strict Python heap-memory guarantee.

## Review boundary and handoff

Only static source/syntax inspection was performed by the code author. No server,
project tests, migration, API calls, or LLM calls were executed. The supplied client
file is an integration helper, not a tested frontend application. Validate reads
locally without starting paid tasks. No full extraction rerun is needed for this
change. If the adapter fails, stop it; the original CLI and saved results are intact.

To remove the additive files, stop the service, then use the same installer with
`--remove`. Removal refuses to delete a file modified since installation.

Reference documentation: FastAPI APIRouter/CORS and uv isolated command execution.
https://fastapi.tiangolo.com/tutorial/bigger-applications/
https://fastapi.tiangolo.com/tutorial/cors/
https://docs.astral.sh/uv/reference/cli/#uv-run


## Candidate-association retirement

The overview uses `routes` for navigation URLs, never `links`. No public endpoint
returns a candidate `links` or `claim_evidence_links` collection, even an empty
one. Argument relationships remain in the selected Mapping's `evidence_relations`
and the corresponding Graph `edges`. Do not remove those. URL, caption, image,
and cell locators are also retained. Older source files may be read, but their
retired candidate-association fields are discarded by the data boundary.
