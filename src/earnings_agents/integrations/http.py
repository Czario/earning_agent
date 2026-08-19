"""Thin HTTP client used by nodes that fetch raw documents.

Centralises headers, timeout handling, and error normalisation so nodes do not
import ``requests`` directly.  Two header presets are provided:

- ``SEC_HEADERS``  — programmatic User-Agent required by SEC EDGAR.
- ``BROWSER_HEADERS`` — generic browser headers for non-SEC HTML pages.
- ``PDF_HEADERS`` — minimal headers for static binary assets.

Public functions
----------------
head(url) -> requests.Response
    Issue a HEAD request and return the response.
get(url, *, sec=False) -> requests.Response
    Issue an HTML GET request, automatically choosing the correct header preset.
get_binary(url, *, sec=False) -> requests.Response
    Issue a static binary GET without spoofing a browser User-Agent.
"""
from __future__ import annotations

import requests

from earnings_agents.config import HTTP_TIMEOUT

# SEC EDGAR requires a descriptive, non-browser User-Agent with contact info.
SEC_HEADERS: dict[str, str] = {
    "User-Agent": "earning-agents data-pipeline@truegrids.com",
    "Accept-Encoding": "gzip, deflate",
}

# Generic browser User-Agent for non-SEC pages.
BROWSER_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Static assets should not be sent with a spoofed browser User-Agent.  Some
# CDNs treat a Chrome UA from a plain HTTP client differently from the browser
# TLS fingerprint it normally comes with and leave the request hanging.
PDF_HEADERS: dict[str, str] = {
    "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
}


def head(url: str) -> requests.Response:
    """Send a HEAD request with browser headers and return the response.

    Raises ``requests.RequestException`` on network errors.
    """
    return requests.head(
        url,
        headers=BROWSER_HEADERS,
        timeout=HTTP_TIMEOUT,
        allow_redirects=True,
    )


def get(url: str, *, sec: bool = False) -> requests.Response:
    """Send a GET request and return the response.

    Parameters
    ----------
    url:
        Target URL.
    sec:
        When ``True``, send SEC-compliant headers instead of browser headers.

    Raises ``requests.RequestException`` on network errors.
    """
    headers = SEC_HEADERS if sec else BROWSER_HEADERS
    return requests.get(url, headers=headers, timeout=HTTP_TIMEOUT)


def get_binary(url: str, *, sec: bool = False) -> requests.Response:
    """Fetch a static binary document with the smallest useful header set.

    HTML pages benefit from browser headers, but PDF downloads do not need
    JavaScript, cookies, or a spoofed browser identity.  Keeping this separate
    from :func:`get` also avoids changing the HTML fetching behavior.
    """
    headers = SEC_HEADERS if sec else PDF_HEADERS
    response = requests.get(
        url,
        headers=headers,
        timeout=HTTP_TIMEOUT,
        allow_redirects=True,
    )
    response.raise_for_status()
    return response
