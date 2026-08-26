"""Tests for the per-run log file (Logs/<date-time>.log) that mirrors the
admin-panel event stream."""
from __future__ import annotations

import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from earnings_agents.filelog import RunLogFile
from earnings_agents.progress import WorkerProgressPublisher


class TestRunLogFile(unittest.TestCase):
    def test_creates_dated_file_with_header_footer_and_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = RunLogFile("PYPL", "req-1", log_dir=tmp)
            try:
                self.assertIsNotNone(log.path)
                name = Path(log.path).name
                # Named TICKER_date_time: PYPL_YYYY-MM-DD_HH-MM-SS.log
                self.assertRegex(
                    name, r"^PYPL_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.log$"
                )
                log.write("[EDGAR] resolving pinned accession ACC-1...")
                log.write("✓ PYPL_2026_latest saved  (14 LLM calls)", kind="summary")
            finally:
                log.close()

            content = Path(log.path).read_text(encoding="utf-8")
            self.assertIn("── run started", content)
            self.assertIn("ticker=PYPL  load_request_id=req-1", content)
            self.assertIn("[EDGAR] resolving pinned accession ACC-1...", content)
            self.assertIn("✓ PYPL_2026_latest saved  (14 LLM calls)", content)
            self.assertIn("── run ended", content)
            # Each line is prefixed with a wall-clock timestamp
            lines = [l for l in content.splitlines() if l.startswith("──") is False]
            for line in lines:
                self.assertRegex(line, r"^\d{2}:\d{2}:\d{2}  ")

    def test_same_second_collision_gets_unique_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = RunLogFile("PYPL", None, log_dir=tmp)
            b = RunLogFile("PYPL", None, log_dir=tmp)  # same ticker+second → suffix
            try:
                self.assertNotEqual(a.path, b.path)
                self.assertRegex(Path(b.path).name, r"_2\.log$")
            finally:
                a.close()
                b.close()

    def test_no_ticker_falls_back_to_date_only_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = RunLogFile("", None, log_dir=tmp)
            try:
                name = Path(log.path).name
                # No ticker → plain date-time name
                self.assertRegex(name, r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.log$")
            finally:
                log.close()

    def test_heartbeat_kind_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = RunLogFile("A", None, log_dir=tmp)
            try:
                log.write("♥ alive  (60s)", kind="heartbeat")
                log.write("real line")
            finally:
                log.close()
            content = Path(log.path).read_text(encoding="utf-8")
            self.assertNotIn("alive", content)
            self.assertIn("real line", content)

    def test_missing_log_dir_is_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            nested = Path(tmp) / "a" / "b"
            log = RunLogFile("A", None, log_dir=str(nested))
            try:
                self.assertTrue(nested.is_dir())
                self.assertTrue(Path(log.path).is_file())
            finally:
                log.close()

    def test_recreates_deleted_log_dir_automatically(self):
        """If the Logs dir is deleted at any time, the next run re-creates it."""
        with tempfile.TemporaryDirectory() as tmp:
            log = RunLogFile("PYPL", None, log_dir=tmp)
            log.close()

            # Operator deletes the log dir while things are running
            for f in Path(tmp).glob("*.log"):
                f.unlink()
            Path(tmp).rmdir()
            self.assertFalse(Path(tmp).exists())

            # Next run must auto-create the dir + file
            log2 = RunLogFile("PYPL", None, log_dir=tmp)
            try:
                self.assertTrue(Path(tmp).is_dir())
                self.assertTrue(Path(log2.path).is_file())
                self.assertIn("── run started", Path(log2.path).read_text(encoding="utf-8"))
            finally:
                log2.close()


class TestPublisherWritesToFile(unittest.TestCase):
    def test_publish_writes_lines_even_when_redis_unreachable(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("earnings_agents.config.RUN_LOGS_DIR", tmp), \
                 patch("earnings_agents.config.RUN_LOGS_ENABLED", True):
                pub = WorkerProgressPublisher(
                    "redis://127.0.0.1:1", "PYPL", "req-1"
                )  # unreachable Redis — file logging must still work
                try:
                    pub.publish("call", "[llm] chunk 1/1 → calling llm  (deepseek)",
                                kind="call_llm")
                    pub.publish("mongodb_save_node", "✓ saved", kind="summary")
                finally:
                    pub.close()

            files = list(Path(tmp).glob("*.log"))
            self.assertEqual(len(files), 1)
            content = files[0].read_text(encoding="utf-8")
            self.assertIn("[llm] chunk 1/1 → calling llm  (deepseek)", content)
            self.assertIn("✓ saved", content)
            self.assertIn("ticker=PYPL  load_request_id=req-1", content)

    def test_disabled_via_config_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("earnings_agents.config.RUN_LOGS_DIR", tmp), \
                 patch("earnings_agents.config.RUN_LOGS_ENABLED", False):
                pub = WorkerProgressPublisher("redis://127.0.0.1:1", "PYPL", None)
                pub.publish("call", "nothing to see")
                pub.close()
            self.assertEqual(list(Path(tmp).glob("*.log")), [])


if __name__ == "__main__":
    unittest.main()
