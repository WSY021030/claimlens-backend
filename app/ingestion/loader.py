"""Load a user-supplied HTML paper together with its browser-saved resources.

Stage 1 of ingestion, BEFORE publisher adapters:
- Works offline and never executes website JavaScript.
- Keeps raw HTML unchanged.
- Resolves local <img> references and marks unavailable/unsafe ones.
- Does not pretend that every page image is a paper figure/table.

Supported typical layout:
    Paper.html
    Paper_files/figure.gif

The adapter will later decide which images are paper content. Nothing here
changes app/models/paper.py or the existing LangGraph workflow.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import unquote, urlsplit

from bs4 import BeautifulSoup


ImageStatus = Literal["available", "missing", "remote", "embedded", "invalid"]


@dataclass(frozen=True)
class ImageReference:
    index: int
    src: str
    alt: str
    status: ImageStatus
    local_path: Path | None = None
    media_type: str | None = None
    html_id: str | None = None
    enclosing_href: str | None = None


@dataclass
class HTMLBundle:
    html_path: Path
    resource_dir: Path | None
    source_url: str | None
    html: str
    images: list[ImageReference]

    def image_counts(self) -> dict[str, int]:
        counts = {status: 0 for status in (
            "available", "missing", "remote", "embedded", "invalid"
        )}
        for img in self.images:
            counts[img.status] += 1
        return counts


class HTMLBundleError(ValueError):
    """HTML or resource-folder input is invalid."""


def _detect_resource_dir(html_path: Path) -> Path | None:
    """Use the directory produced by Edge/Chrome 'Webpage, Complete'."""
    expected = html_path.with_name(html_path.stem + "_files")
    return expected.resolve() if expected.is_dir() else None


def _source_url(soup: BeautifulSoup, html: str) -> str | None:
    canonical = soup.find("link", rel=lambda value: value and "canonical" in value)
    if canonical and canonical.get("href", "").startswith(("http://", "https://")):
        return canonical["href"]

    match = re.search(r"<!--\s*saved from url=\(\d+\)(https?://[^\s]+)\s*-->", html)
    return match.group(1) if match else None


def _sniff_media_type(path: Path) -> str | None:
    """Content-based detection; saved images sometimes have .txt extensions."""
    with path.open("rb") as f:
        start = f.read(512)
    if start.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if start.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if start.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if start.startswith(b"RIFF") and start[8:12] == b"WEBP":
        return "image/webp"
    if start.startswith(b"BM"):
        return "image/bmp"
    if start.startswith((b"II*\x00", b"MM\x00*")):
        return "image/tiff"
    trimmed = start.lstrip(b"\xef\xbb\xbf\r\n\t ")
    if trimmed.startswith(b"<svg") or (
        trimmed.startswith(b"<?xml") and b"<svg" in trimmed
    ):
        return "image/svg+xml"
    return None


def _best_img_src(tag) -> str:
    for key in ("src", "data-src", "data-original", "data-lazy-src"):
        value = tag.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    srcset = tag.get("srcset") or tag.get("data-srcset")
    if srcset:
        # Basic fallback for a normal comma-separated srcset, not for data URIs.
        return srcset.split(",", 1)[0].strip().split()[0]
    return ""


def _resolve_img(
    src: str,
    html_path: Path,
    resource_dir: Path | None,
) -> tuple[ImageStatus, Path | None, str | None]:
    if not src:
        return "missing", None, None
    if src.startswith("data:image/"):
        return "embedded", None, src.split(";", 1)[0][5:]
    if src.startswith("//"):
        return "remote", None, None

    parts = urlsplit(src)
    if parts.scheme in ("http", "https"):
        return "remote", None, None
    if parts.scheme or parts.netloc or not parts.path:
        return "invalid", None, None

    relative = unquote(parts.path).replace("\\", "/")
    if relative.startswith("/") or re.match(r"^[A-Za-z]:", relative):
        return "invalid", None, None
    components = relative.split("/")
    if any(component == ".." for component in components):
        return "invalid", None, None

    root = html_path.parent.resolve()
    expected_folder = html_path.stem + "_files"

    # Explicit folder may be elsewhere (e.g. UI uploaded HTML and folder
    # separately). A reference to <original-stem>_files/x is remapped there.
    stripped = [c for c in components if c not in ("", ".")]
    if resource_dir is not None and stripped and stripped[0] in (
        resource_dir.name, expected_folder
    ):
        local_root = resource_dir.resolve()
        candidate = local_root.joinpath(*stripped[1:]).resolve()
    else:
        local_root = root
        candidate = (root / relative).resolve()

    # Reject symlinks or paths that leave the declared upload root.
    if not candidate.is_relative_to(local_root):
        return "invalid", None, None
    if candidate.is_file():
        return "available", candidate, _sniff_media_type(candidate)
    return "missing", None, None


def load_html_bundle(
    html_path: str | Path,
    resource_dir: str | Path | None = None,
) -> HTMLBundle:
    """Load local HTML and inspect its image references without network calls.

    If resource_dir is omitted, auto-detect '<HTML filename stem>_files'.
    The original HTML file and any resources are only read, never modified.
    """
    path = Path(html_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"HTML file not found: {path}")
    if path.suffix.lower() not in (".html", ".htm"):
        raise HTMLBundleError("Input must be a .html or .htm file")

    folder = Path(resource_dir).expanduser().resolve() if resource_dir else _detect_resource_dir(path)
    if resource_dir is not None and not folder.is_dir():
        raise HTMLBundleError(f"Resource directory not found: {folder}")

    # UTF-8 works for these IEEE / ScienceDirect samples; detect other
    # document encodings from <meta> with BeautifulSoup UnicodeDammit.
    raw = path.read_bytes()
    soup = BeautifulSoup(raw, "lxml")
    # Preserve the saved HTML text; do not replace it with soup serialization.
    raw_text = soup.original_encoding or "utf-8"
    try:
        saved_html = raw.decode(raw_text)
    except (UnicodeDecodeError, LookupError):
        saved_html = raw.decode("utf-8", errors="replace")

    image_refs: list[ImageReference] = []
    for idx, tag in enumerate(soup.find_all("img"), start=1):
        src = _best_img_src(tag)
        status, local_path, media_type = _resolve_img(src, path, folder)
        anchor = tag.find_parent("a", href=True)
        image_refs.append(ImageReference(
            index=idx,
            src=src,
            alt=tag.get("alt", "").strip(),
            status=status,
            local_path=local_path,
            media_type=media_type,
            html_id=tag.get("id"),
            enclosing_href=anchor.get("href") if anchor else None,
        ))

    return HTMLBundle(
        html_path=path,
        resource_dir=folder,
        source_url=_source_url(soup, saved_html),
        html=saved_html,
        images=image_refs,
    )
