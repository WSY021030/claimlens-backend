"""Prepare source-grounded figure/table images for LLM input.

The immutable paper.json remains the authority for IDs and paths. An image
receipt checks provenance, not whether a model read a digit correctly.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET
import os
import shutil

from app.models.document_v2 import PaperDocument

MAX_IMAGE_BYTES = 12 * 1024 * 1024
MAX_IMAGE_PIXELS = 30_000_000


@dataclass(frozen=True)
class PreparedImage:
    element_id: str
    asset_id: str
    path: Path
    source_path: str
    source_sha256: str
    sent_sha256: str
    data_url: str
    width: int
    height: int
    cell_id: str | None = None
    conversion: str | None = None

    def receipt(self) -> dict[str, Any]:
        # Never persist data URLs / image bytes in analysis JSON or cost logs.
        return {"element_id": self.element_id, "asset_id": self.asset_id,
                "source_path": self.source_path, "source_sha256": self.source_sha256,
                "sent_sha256": self.sent_sha256, "width": self.width,
                "height": self.height, "cell_id": self.cell_id,
                "conversion": self.conversion}

    def input_part(self) -> dict[str, str]:
        return {"type": "input_image", "image_url": self.data_url, "detail": "high"}


class ImageInputs:
    """Per-analysis local asset resolver, preparation cache and submission ledger."""
    def __init__(self, document: PaperDocument | None, document_dir: Path | None = None,
                 source_root: Path | None = None, *, enabled: bool = True,
                 max_images_per_call: int = 2):
        self.document = document
        self.document_dir = Path(document_dir).resolve() if document_dir else None
        self.source_root = Path(source_root).resolve() if source_root else None
        self.enabled = enabled
        self.max_images_per_call = max_images_per_call
        self._prepared: dict[tuple[str, str], PreparedImage] = {}
        self._submitted: dict[tuple[str, str, str], dict[str, Any]] = {}
        self.events: list[dict[str, Any]] = []

    def asset_refs(self, element_id: str) -> list[tuple[str, str | None]]:
        if not self.document:
            return []
        element = self.document.elements.get(element_id)
        if element is None or element.kind not in {"table", "figure"}:
            return []
        refs: list[tuple[str, str | None]] = []
        # Prefer authoritative HTML cell text when it exists. Image-only and
        # hybrid tables may still need their figure; image-bearing cells also
        # carry a source asset and are not replaced with guessed text.
        need_whole = (element.kind == "figure" or not element.grid or not element.grid.cells
                      or element.representation in {"image_only", "hybrid"})
        if need_whole and element.image_asset_id:
            refs.append((element.image_asset_id, None))
        if element.kind == "table" and element.grid:
            refs.extend((cell.image_asset_id, cell.cell_id) for cell in element.grid.cells
                        if cell.image_asset_id)
        return list(dict.fromkeys(refs))

    def _resolve(self, asset_id: str) -> tuple[Path, str]:
        if not self.document or asset_id not in self.document.assets:
            raise ValueError(f"Unknown source image asset: {asset_id}")
        asset = self.document.assets[asset_id]
        candidates: list[tuple[Path, str]] = []
        if self.document_dir:
            if asset.normalized_relative_path:
                candidates.append((self.document_dir, asset.normalized_relative_path))
            candidates.append((self.document_dir, asset.original_relative_path))
        if self.source_root:
            candidates.append((self.source_root, asset.original_relative_path))
        if not candidates:
            raise FileNotFoundError("No document directory configured; pass the paper.json directory to analyze_document(document_dir=...).")
        tried: list[str] = []
        for root, relative in candidates:
            path = (root / relative).resolve()
            if not path.is_relative_to(root):
                raise ValueError(f"Image path escapes the supplied paper directory: {asset_id}")
            tried.append(str(path))
            if path.is_file():
                return path, relative
        raise FileNotFoundError("Image file not found. Checked: " + " | ".join(tried))

    def prepare(self, element_id: str, asset_id: str,
                cell_id: str | None = None) -> PreparedImage:
        if (asset_id, cell_id) not in self.asset_refs(element_id):
            raise ValueError(f"{asset_id} is not an image owned by {element_id} / {cell_id}")
        path, relative = self._resolve(asset_id)
        if path.stat().st_size > MAX_IMAGE_BYTES:
            raise ValueError("Image exceeds the 12 MiB per-image local limit.")
        raw = path.read_bytes()
        source_hash = sha256(raw).hexdigest()
        old = self._prepared.get((element_id, asset_id))
        if old and old.source_sha256 == source_hash and old.cell_id == cell_id:
            return old
        pixels, mime, width, height, conversion = _image_payload(raw, path.suffix.lower())
        if len(pixels) > MAX_IMAGE_BYTES:
            raise ValueError("Prepared image exceeds the 12 MiB per-image local limit.")
        image = PreparedImage(element_id, asset_id, path, relative, source_hash,
                              sha256(pixels).hexdigest(),
                              f"data:{mime};base64," + base64.b64encode(pixels).decode("ascii"),
                              width, height, cell_id, conversion)
        self._prepared[(element_id, asset_id)] = image
        return image

    def for_candidates(self, candidates: list[Any]) -> list[PreparedImage]:
        images: list[PreparedImage] = []
        seen: set[tuple[str, str]] = set()
        total_bytes = 0
        for candidate in candidates:
            for asset_id, cell_id in self.asset_refs(candidate.element_id):
                key = (candidate.element_id, asset_id)
                if key in seen:
                    continue
                seen.add(key)
                event = {"element_id": candidate.element_id, "asset_id": asset_id}
                if not self.enabled:
                    self.events.append({**event, "status": "disabled"})
                    continue
                if len(images) >= self.max_images_per_call:
                    self.events.append({**event, "status": "limit_reached"})
                    continue
                try:
                    image = self.prepare(candidate.element_id, asset_id, cell_id)
                    # Bound combined encoded input, not just each image.
                    if total_bytes + len(image.data_url) > 24 * 1024 * 1024:
                        self.events.append({**event, "status": "payload_limit_reached"})
                        continue
                    images.append(image)
                    total_bytes += len(image.data_url)
                except (ValueError, OSError, ImportError) as exc:
                    self.events.append({**event, "status": "unavailable", "message": str(exc)})
        return images

    def record_submission(self, images: list[PreparedImage], model: str, response: Any) -> None:
        actual = getattr(response, "model", None) or model
        request_id = getattr(response, "id", None)
        for image in images:
            receipt = {**image.receipt(), "model": actual, "request_id": request_id,
                       "status": "submitted_with_response"}
            self._submitted[(image.element_id, image.asset_id, image.sent_sha256)] = receipt
            self.events.append(receipt)

    def check_receipt(self, element_id: str, observation: Any) -> bool:
        receipt = self._submitted.get((element_id, observation.asset_id, observation.sent_sha256))
        if receipt is None:
            return False
        if (receipt["source_sha256"] != observation.source_sha256
                or receipt["source_path"] != observation.source_path
                or receipt["model"] != observation.model):
            return False
        try:
            path, _ = self._resolve(observation.asset_id)
            return sha256(path.read_bytes()).hexdigest() == observation.source_sha256
        except (ValueError, OSError):
            return False

    def inspect(self) -> list[dict[str, Any]]:
        """Local format/path preflight. No SDK initialization or API request."""
        if not self.document:
            return []
        rows = []
        for element_id in self.document.elements:
            for asset_id, cell_id in self.asset_refs(element_id):
                row = {"element_id": element_id, "asset_id": asset_id, "cell_id": cell_id}
                try:
                    image = self.prepare(element_id, asset_id, cell_id)
                    rows.append({**row, **image.receipt(), "status": "ready"})
                except (ValueError, OSError, ImportError) as exc:
                    rows.append({**row, "status": "unavailable", "message": str(exc)})
        return rows

    def report(self) -> dict[str, Any]:
        sent = [e for e in self.events if e["status"] == "submitted_with_response"]
        return {"mode": "image_assets_enabled" if self.enabled else "text_only",
                "attachments_sent": len(sent),
                "unique_images_sent": len({e["source_sha256"] for e in sent}),
                "events": self.events,
                "note": "An attachment receipt proves which image was sent, not that every digit or plot interpretation is correct."}


def _image_payload(raw: bytes, suffix: str) -> tuple[bytes, str, int, int, str | None]:
    """Lossless PNG conversion for GIF/BMP/TIFF, keeping academic text legible."""
    from PIL import Image, ImageOps, UnidentifiedImageError
    conversion = None
    if suffix == ".svg" or b"<svg" in raw[:500].lower():
        raw = _render_svg(raw)
        conversion = "svg_to_png"
    try:
        with Image.open(BytesIO(raw)) as source:
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise ValueError("Image exceeds the 30 megapixel local limit.")
            if getattr(source, "n_frames", 1) != 1:
                raise ValueError("Animated/multipage image is not silently reduced to one frame; export a static frame first.")
            image = ImageOps.exif_transpose(source)
            width, height = image.size
            # Keep originals where possible; no sharpening, OCR or invented pixels.
            if source.format in {"PNG", "JPEG", "WEBP"} and image.size == source.size and not source.getexif().get(274):
                return raw, {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}[source.format], width, height, conversion
            output = BytesIO()
            image.convert("RGBA" if "A" in image.getbands() or "transparency" in image.info else "RGB").save(output, format="PNG")
            return output.getvalue(), "image/png", width, height, conversion or f"{source.format.lower()}_to_png"
    except (UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise ValueError(f"Unreadable/unsupported image bytes: {type(exc).__name__}") from exc


def _render_svg(raw: bytes) -> bytes:
    """Render existing normalized static SVG with the project's Playwright.

    Embedded data images are allowed; external network/files are blocked.
    This supports the reader's arXiv SVG packaging without reparsing the HTML.
    """
    low = raw.lower()
    if b"<!doctype" in low or b"<!entity" in low:
        raise ValueError("SVG with DTD/entity definitions is not a static reader asset.")
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise ValueError("Invalid SVG image.") from exc
    for node in root.iter():
        tag = node.tag.rsplit("}", 1)[-1].lower()
        if tag in {"script", "iframe", "object", "embed"}:
            raise ValueError("Active content is not supported in an image asset.")
        for key, value in node.attrib.items():
            name = key.rsplit("}", 1)[-1].lower()
            if name.startswith("on"):
                raise ValueError("Active SVG event attribute is not supported.")
            if name in {"href", "src"} and not (value.startswith("#") or value.startswith("data:image/")):
                raise ValueError("SVG refers to an external resource; use the normalized standalone SVG.")
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            launch_args: dict[str, Any] = {"headless": True}
            if not Path(p.chromium.executable_path).is_file():
                # Reuse an already-installed browser instead of forcing a
                # second browser download just to rasterize one SVG.
                candidates = [shutil.which(name) for name in ("chromium", "chromium-browser", "google-chrome", "msedge")]
                for root_name in ("PROGRAMFILES(X86)", "PROGRAMFILES", "LOCALAPPDATA"):
                    root_dir = os.getenv(root_name)
                    if root_dir:
                        candidates += [str(Path(root_dir) / rel) for rel in
                            ("Microsoft/Edge/Application/msedge.exe", "Google/Chrome/Application/chrome.exe")]
                existing = next((x for x in candidates if x and Path(x).is_file()), None)
                if existing:
                    launch_args["executable_path"] = existing
            browser = p.chromium.launch(**launch_args)
            try:
                page = browser.new_page(viewport={"width": 1600, "height": 1200},
                                        device_scale_factor=1, java_script_enabled=False)
                page.route("**/*", lambda route: route.abort())
                data = base64.b64encode(raw).decode("ascii")
                page.set_content('<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; img-src data:; style-src \'unsafe-inline\'">'
                                 '<style>body{margin:0;background:white}img{display:block;max-width:2400px;max-height:6000px}</style>'
                                 f'<img id="paper_image" src="data:image/svg+xml;base64,{data}">', wait_until="load")
                return page.locator("#paper_image").screenshot(type="png", timeout=15000)
            finally:
                browser.close()
    except ImportError:
        raise ImportError("SVG conversion needs the project's existing Playwright dependency.")
    except Exception as exc:
        raise ValueError("SVG conversion failed. Check the existing Playwright Chromium installation; "
                         f"raster images do not require it. ({type(exc).__name__})") from exc
