
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from bs4 import BeautifulSoup


PublisherType = Literal[
    "arxiv",
    "acm",
    "ieee",
    "springer",
]


@dataclass
class DetectionResult:
    publisher: PublisherType
    source_url: str
    detected_by: str


class UnsupportedPublisherError(ValueError):
    """Raised when the publisher cannot be identified."""


def identify_publisher(url: str) -> PublisherType | None:
    """
    Identify the publisher by the URL hostname.
    """

    hostname = (
        urlparse(url).hostname or ""
    ).lower()

    if hostname in {
        "arxiv.org",
        "www.arxiv.org",
        "export.arxiv.org",
        "ar5iv.labs.arxiv.org",
    }:
        return "arxiv"

    if hostname == "dl.acm.org":
        return "acm"

    if hostname == "ieeexplore.ieee.org":
        return "ieee"

    if hostname == "link.springer.com":
        return "springer"

    return None


def detect_publisher(
    html_path: str | Path,
    source_url: str | None = None,
) -> DetectionResult:
    """
    Detect the publisher of a locally saved HTML file.

    Priority:
    1. Canonical URL
    2. Open Graph URL
    3. Original HTML URL metadata
    4. User-provided source URL

    DOI alone is not used because it identifies the
    publication, not necessarily the HTML page format.
    """

    path = Path(html_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"HTML file not found: {path}"
        )

    html = path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    soup = BeautifulSoup(html, "lxml")

    candidates = []

    for link in soup.find_all("link", href=True):
        rel = [
            str(value).lower()
            for value in link.get("rel", [])
        ]

        if "canonical" in rel:
            candidates.append(
                ("canonical", link["href"])
            )
            break

    for meta in soup.find_all("meta"):
        key = (
            meta.get("property")
            or meta.get("name")
            or ""
        ).lower()

        if key in {
            "og:url",
            "citation_fulltext_html_url",
        }:
            content = meta.get("content")

            if content:
                candidates.append(
                    (key, content)
                )

    if source_url:
        candidates.append(
            ("user_provided_url", source_url)
        )

    for detected_by, url in candidates:
        publisher = identify_publisher(url)

        if publisher is not None:
            return DetectionResult(
                publisher=publisher,
                source_url=url,
                detected_by=detected_by,
            )

        # A canonical URL identifies the actual page.
        # Do not override an unsupported site using
        # unrelated metadata pointing to another publisher.
        if detected_by == "canonical":
            raise UnsupportedPublisherError(
                f"Unsupported canonical URL: {url}"
            )

    raise UnsupportedPublisherError(
        "Cannot identify the HTML publisher. "
        "Supported: arXiv, ACM, IEEE, Springer. "
        "Provide the original source URL if needed."
    )
