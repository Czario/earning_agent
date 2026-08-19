from __future__ import annotations

import logging
import subprocess

from playwright.sync_api import TimeoutError as PlaywrightTimeout
from playwright.sync_api import sync_playwright

from earnings_agents.config import HTTP_TIMEOUT

logger = logging.getLogger(__name__)

_SEC_USER_AGENT = "earning-agents data-pipeline@truegrids.com"

_BROWSER_HEADERS = {
    "Accept-Language": "en-US,en;q=0.9",
}

_BINARY_HEADERS = {
    "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def fetch_page_js(url: str) -> str:
    """Fetch a JavaScript-rendered page using Playwright headless Chromium.

    Returns the fully rendered HTML, or an empty string on failure.
    """
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            # Keep Chromium's native User-Agent.  Overriding it with an old
            # Chrome version can make the UA and TLS/browser fingerprint
            # inconsistent to a CDN/WAF.
            context = browser.new_context(extra_http_headers=_BROWSER_HEADERS)
            page = context.new_page()
            page.set_extra_http_headers(_BROWSER_HEADERS)
            page.goto(url, timeout=HTTP_TIMEOUT * 1_000)
            page.wait_for_load_state("networkidle", timeout=HTTP_TIMEOUT * 1_000)
            html = page.content()
            browser.close()
            return html
    except PlaywrightTimeout:
        logger.warning("Playwright timeout for %s", url)
        return ""
    except Exception as exc:  # noqa: BLE001
        logger.warning("Playwright error for %s: %s", url, exc)
        return ""


def fetch_binary(url: str, *, referer: str | None = None) -> bytes:
    """Download a binary document through a headless browser context.

    Some investor-relations CDNs reject or stall ordinary requests but serve
    the same file to a browser.  Returning bytes keeps this useful for PDFs
    without trying to interpret the document in the browser.
    """
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                headless=True,
                args=[],
            )
            # Do not spoof a fixed browser version here.  Chromium's native
            # UA matches its actual network fingerprint, which is more useful
            # to CDNs than a hand-written Chrome/124 string.
            context = browser.new_context(extra_http_headers=_BINARY_HEADERS)
            page = context.new_page()
            try:
                goto_options = {
                    "wait_until": "commit",
                    "timeout": HTTP_TIMEOUT * 1_000,
                }
                if referer:
                    goto_options["referer"] = referer
                response = page.goto(url, **goto_options)
                if response is None or not response.ok:
                    logger.warning(
                        "Playwright binary fetch returned HTTP %s for %s",
                        response.status if response is not None else "no response",
                        url,
                    )
                    return b""
                return response.body()
            finally:
                # Keep browser cleanup inside the Playwright lifecycle.
                browser.close()
    except PlaywrightTimeout:
        logger.warning("Playwright binary fetch timeout for %s", url)
        return b""
    except Exception as exc:  # noqa: BLE001
        logger.warning("Playwright binary fetch error for %s: %s", url, exc)
        return b""


def fetch_binary_curl(url: str, *, sec: bool = False) -> bytes:
    """Download a binary document with the system curl transport.

    For non-SEC assets the curl User-Agent is intentionally left alone.  A
    manually supplied Chrome User-Agent is not equivalent to a browser TLS
    fingerprint and causes some CDNs to stall the response (Adobe is one
    example).  SEC still receives its required descriptive User-Agent.
    """
    try:
        command = [
            "curl",
            "--fail",
            "--verbose",
            "--location",
            "--http1.1",
            "--max-time",
            str(HTTP_TIMEOUT),
            "--header",
            "Accept: application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
            "--header",
            "Accept-Language: en-US,en;q=0.9",
        ]
        if sec:
            command.extend(["--user-agent", _SEC_USER_AGENT])
        command.append(url)
        result = subprocess.run(
            command,
            capture_output=True,
            timeout=HTTP_TIMEOUT + 5,
            check=False,
        )
        trace = result.stderr.decode(errors="replace").strip()
        if trace:
            logger.info("curl PDF fetch trace for %s:\n%s", url, trace)
        if result.returncode != 0:
            logger.warning("curl binary fetch failed for %s", url)
            return b""
        return result.stdout
    except Exception as exc:  # noqa: BLE001
        logger.warning("curl binary fetch error for %s: %s", url, exc)
        return b""
