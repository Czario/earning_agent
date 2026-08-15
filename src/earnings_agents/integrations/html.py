"""Shared HTML utilities — used by the agent pipeline."""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Tags that carry no earnings content
NOISE_TAGS = frozenset(
    {"script", "style", "nav", "footer", "header", "aside", "noscript", "svg", "form"}
)

# Minimum meaningful content length; below this we assume JS rendering is needed
MIN_CONTENT_CHARS = 300


def strip_sgml_wrapper(html: str) -> str:
    """Extract the HTML payload from an EDGAR SGML wrapper if present.

    EDGAR archive files are often wrapped in SGML::

        <DOCUMENT>
        <TYPE>EX-99.1
        ...
        <TEXT>
        <html>...</html>
        </TEXT>
        </DOCUMENT>

    This function returns the content after the ``<TEXT>`` tag so that
    BeautifulSoup only sees valid HTML.
    """
    if "<DOCUMENT>" not in html.upper():
        return html
    match = re.search(r"<TEXT>\s*(<html.*)", html, re.DOTALL | re.IGNORECASE)
    if match:
        payload = match.group(1)
        # Strip trailing </TEXT></DOCUMENT> closing tags
        payload = re.sub(r"</TEXT>\s*</DOCUMENT>\s*$", "", payload, flags=re.IGNORECASE | re.DOTALL)
        return payload
    return html
