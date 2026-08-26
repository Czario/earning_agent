"""Per-run log file mirroring the admin-panel event stream.

Every line the worker publishes to the admin panel (Redis ``sec:worker:events``
→ SSE) is ALSO appended to a dated log file under the project's ``Logs/``
directory — one file per run, named by the run's start date+time::

    Logs/2026-08-26_11-45-30.log

The file log is best-effort only: any failure to create/write the file is
logged at WARNING and never interrupts the pipeline.  Heartbeat events are
skipped, matching the admin panel which filters them from the visible log.
"""
from __future__ import annotations

import logging
import re
import threading
from datetime import datetime
from pathlib import Path

from earnings_agents import config

logger = logging.getLogger(__name__)

_HEARTBEAT_KIND = "heartbeat"


class RunLogFile:
    """Append-only log file capturing one pipeline run's event lines."""

    def __init__(
        self,
        ticker: str,
        load_request_id: str | None,
        log_dir: str | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._handle = None
        self._path: Path | None = None
        try:
            directory = Path(log_dir or config.RUN_LOGS_DIR)
            directory.mkdir(parents=True, exist_ok=True)
            # Machine-local time — the same clock the operator sees (e.g. IST
            # on this machine; whatever the machine TZ is wherever it runs).
            now = datetime.now()
            stamp = now.strftime("%Y-%m-%d_%H-%M-%S")
            safe_ticker = re.sub(r"[^A-Za-z0-9]+", "_", (ticker or "").upper()).strip("_")
            # <TICKER>_<date>_<time>.log (falls back to <date>_<time>.log when no ticker)
            base = f"{safe_ticker}_{stamp}" if safe_ticker else stamp
            path = directory / f"{base}.log"
            suffix = 2
            while path.exists():  # two runs started in the same second
                path = directory / f"{base}_{suffix}.log"
                suffix += 1
            self._handle = open(path, "a", encoding="utf-8")
            self._path = path
            self.write(
                f"── run started {now.isoformat(timespec='seconds')}  "
                f"ticker={ticker or '?'}  "
                f"load_request_id={load_request_id or '-'}"
            )
            logger.info("Run log file: %s", path)
        except Exception as exc:  # noqa: BLE001 — logging must never break the pipeline
            logger.warning("RunLogFile: could not open log file: %s", exc)

    def write(self, message: str, kind: str | None = None) -> None:
        """Append one event line (same text the admin panel displays)."""
        if kind == _HEARTBEAT_KIND:
            return
        if self._handle is None:
            return
        try:
            ts = datetime.now().strftime("%H:%M:%S")
            with self._lock:
                self._handle.write(f"{ts}  {message}\n")
                self._handle.flush()
        except Exception as exc:  # noqa: BLE001
            logger.warning("RunLogFile: write failed: %s", exc)

    def close(self) -> None:
        if self._handle is None:
            return
        try:
            with self._lock:
                self._handle.write(
                    f"── run ended {datetime.now().isoformat(timespec='seconds')}\n"
                )
                self._handle.flush()
                self._handle.close()
        except Exception:  # noqa: BLE001
            pass
        finally:
            self._handle = None

    @property
    def path(self) -> str | None:
        return str(self._path) if self._path is not None else None
