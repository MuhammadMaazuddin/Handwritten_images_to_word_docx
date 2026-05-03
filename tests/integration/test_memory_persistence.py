"""
Integration tests for long-term memory persistence.
Requirements: 7.1, 7.2, 7.4, 18.4
"""
import json
import os
import tempfile
import uuid
import pytest
from src.agents.memory_manager import MemoryManager


def test_ltm_updated_across_two_sessions(tmp_path, monkeypatch):
    """LTM is updated with records from two sequential sessions. Req 7.1, 7.2"""
    monkeypatch.setattr(MemoryManager, "LTM_PATH", str(tmp_path / "ltm.json"))
    monkeypatch.setattr("src.agents.memory_manager.MemoryManager.LTM_PATH",
                        str(tmp_path / "ltm.json"))

    mm = MemoryManager()
    mm.LTM_PATH = str(tmp_path / "ltm.json")

    # Session 1
    sid1 = str(uuid.uuid4())
    mm.start_session(sid1, "img1.jpg")
    mm.update_long_term_memory("dense_handwriting", {"name": "got_ocr_local"}, 0.75, True)
    mm.end_session(sid1)

    # Session 2
    sid2 = str(uuid.uuid4())
    mm.start_session(sid2, "img2.jpg")
    mm.update_long_term_memory("dense_handwriting", {"name": "got_ocr_local"}, 0.80, True)
    mm.end_session(sid2)

    # Reload from disk
    mm2 = MemoryManager()
    mm2.LTM_PATH = str(tmp_path / "ltm.json")
    mm2._load_long_term_memory()

    stats = mm2.get_ltm_stats()
    assert "dense_handwriting" in stats
    assert "got_ocr_local" in stats["dense_handwriting"]
    assert stats["dense_handwriting"]["got_ocr_local"]["attempts"] == 2


def test_corrupted_ltm_initialises_fresh(tmp_path):
    """Corrupted LTM file → fresh init without crash. Req 7.4, 18.4"""
    ltm_path = str(tmp_path / "ltm.json")
    # Write corrupted JSON
    with open(ltm_path, "w") as f:
        f.write("{this is not valid json!!!")

    mm = MemoryManager()
    mm.LTM_PATH = ltm_path
    mm._load_long_term_memory()  # Should not raise

    ltm = mm.get_long_term_memory()
    assert ltm == {"strategy_performance": {}}


def test_ltm_persists_to_disk(tmp_path):
    """LTM is written to disk after update. Req 7.1"""
    mm = MemoryManager()
    mm.LTM_PATH = str(tmp_path / "ltm.json")

    mm.update_long_term_memory("printed_text", {"name": "florence_local"}, 0.9, True)

    assert os.path.exists(mm.LTM_PATH)
    with open(mm.LTM_PATH) as f:
        data = json.load(f)
    assert "printed_text" in data["strategy_performance"]
