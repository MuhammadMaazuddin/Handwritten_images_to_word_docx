"""
Integration tests for memory persistence.
Requirements: 7.1, 7.2, 7.4, 18.4
"""
import json
import os
import tempfile
import uuid
import pytest

from src.agents.memory_manager import MemoryManager


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMemoryPersistence:
    """Tests that verify LTM and STM behaviour across sessions."""

    def test_ltm_updated_across_sessions(self, tmp_path, monkeypatch):
        """
        LTM must accumulate records across multiple update calls.

        After two updates for the same image_type/strategy, attempts must equal 2
        and avg_confidence must be a valid positive float.

        Requirements: 7.1, 7.2
        """
        # Isolate to a fresh temp directory so prior LTM data doesn't interfere
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()

        ltm_path = memory_dir / "long_term_memory.json"
        monkeypatch.setattr(MemoryManager, "LTM_PATH", str(ltm_path))

        original_makedirs = os.makedirs

        def patched_makedirs(path, **kwargs):
            if path == "memory":
                return original_makedirs(str(memory_dir), **kwargs)
            if path == "logs":
                return original_makedirs(str(logs_dir), **kwargs)
            return original_makedirs(path, **kwargs)

        monkeypatch.setattr(os, "makedirs", patched_makedirs)

        mm = MemoryManager()

        mm.update_long_term_memory(
            "dense_handwriting",
            {"name": "groq_api"},
            0.8,
            True,
        )
        mm.update_long_term_memory(
            "dense_handwriting",
            {"name": "groq_api"},
            0.9,
            True,
        )

        ltm = mm.get_long_term_memory()
        record = ltm["strategy_performance"]["dense_handwriting"]["groq_api"]

        assert record["attempts"] == 2, (
            f"Expected 2 attempts, got {record['attempts']}"
        )
        assert record["avg_confidence"] > 0.0, (
            f"Expected avg_confidence > 0.0, got {record['avg_confidence']}"
        )

    def test_corrupted_ltm_initialises_fresh(self, tmp_path, monkeypatch):
        """
        A corrupted LTM file must not crash MemoryManager; it should initialise
        a fresh empty store instead.

        Requirements: 7.4, 18.4
        """
        # Point MemoryManager at a temp directory so we don't pollute the real one
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()

        ltm_path = memory_dir / "long_term_memory.json"
        ltm_path.write_text("not valid json {{{", encoding="utf-8")

        # Patch the class-level constant so the new instance uses our temp path
        monkeypatch.setattr(MemoryManager, "LTM_PATH", str(ltm_path))
        # Also patch os.makedirs calls to use our temp dirs (they already exist)
        original_makedirs = os.makedirs

        def patched_makedirs(path, **kwargs):
            # Redirect memory/ and logs/ to our temp dirs
            if path == "memory":
                return original_makedirs(str(memory_dir), **kwargs)
            if path == "logs":
                return original_makedirs(str(logs_dir), **kwargs)
            return original_makedirs(path, **kwargs)

        monkeypatch.setattr(os, "makedirs", patched_makedirs)

        # Should not raise
        mm = MemoryManager()

        assert mm.get_long_term_memory() == {"strategy_performance": {}}, (
            f"Expected fresh LTM after corruption, got: {mm.get_long_term_memory()}"
        )

    def test_stm_cleared_after_end_session(self, tmp_path, monkeypatch):
        """
        After end_session(), the STM entry for that session must be cleared
        (get_session_log returns {}).

        Requirements: 7.1, 7.4
        """
        # Redirect logs/ writes to tmp_path so we don't litter the workspace
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()

        ltm_path = memory_dir / "long_term_memory.json"
        monkeypatch.setattr(MemoryManager, "LTM_PATH", str(ltm_path))

        original_makedirs = os.makedirs

        def patched_makedirs(path, **kwargs):
            if path == "memory":
                return original_makedirs(str(memory_dir), **kwargs)
            if path == "logs":
                return original_makedirs(str(logs_dir), **kwargs)
            return original_makedirs(path, **kwargs)

        monkeypatch.setattr(os, "makedirs", patched_makedirs)

        # Patch open() for the session log write so it goes to tmp_path
        original_open = open

        def patched_open(path, *args, **kwargs):
            if isinstance(path, str) and path.startswith("logs/session_"):
                filename = os.path.basename(path)
                return original_open(str(logs_dir / filename), *args, **kwargs)
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr("builtins.open", patched_open)

        mm = MemoryManager()
        session_id = str(uuid.uuid4())

        mm.start_session(session_id, "test.jpg")
        mm.record_action({"agent": "test", "action": "test_action"})

        # STM should have data before end_session
        log_before = mm.get_session_log(session_id)
        assert log_before != {}, (
            "Expected non-empty session log after start_session + record_action"
        )

        mm.end_session(session_id)

        # STM should be cleared after end_session
        log_after = mm.get_session_log(session_id)
        assert log_after == {}, (
            f"Expected empty session log after end_session, got: {log_after}"
        )
