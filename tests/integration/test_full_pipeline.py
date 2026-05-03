"""
Integration tests for the full agentic pipeline.
Requirements: 1.1, 1.6, 4.7, 17.1
"""
import time
import uuid
from unittest.mock import MagicMock, patch
import pytest
from src.agents import create_agentic_system
from src.agents.memory_manager import MemoryManager


def _make_orchestrator_with_mock_tools():
    """Create orchestrator with all tools mocked to return success."""
    from src.agents.orchestrator_agent import OrchestratorAgent
    mm = MemoryManager()
    mock_ocr_tool = MagicMock()
    mock_ocr_tool.invoke.return_value = {
        "status": "success",
        "output": {"text": " ".join(["word"] * 100), "confidence": 0.85, "method": "mock"},
        "execution_time_seconds": 0.1
    }
    tools = {
        "ocr_tool": mock_ocr_tool,
        "preprocessor_tool": MagicMock(),
        "diagram_tool": MagicMock(),
        "corrector_tool": MagicMock(),
        "document_tool": MagicMock(),
    }
    return OrchestratorAgent(mm, tools), mm


def test_full_pipeline_phases_completed():
    """All four phases complete on a successful run. Req 1.1, 1.6"""
    orch, _ = _make_orchestrator_with_mock_tools()
    # Mock image analysis to return success
    orch.image_analysis_agent.execute = MagicMock(return_value={
        "status": "success",
        "output": {
            "image_type": "dense_handwriting",
            "resolution_dpi": 200.0,
            "noise_level": 0.1,
            "contrast_ratio": 5.0,
            "has_diagrams": False,
            "image_dimensions": {"width": 800, "height": 1100},
            "recommended_enhancements": []
        },
        "execution_time_seconds": 0.1
    })
    session_id = str(uuid.uuid4())
    result = orch.execute({"image_path": "fake.jpg", "session_id": session_id})
    assert result["status"] == "success"
    assert result["phases_completed"] == ["perceive", "decide", "act", "learn"]


def test_full_pipeline_execution_time():
    """Full loop completes within 120 seconds. Req 4.7, 17.1"""
    orch, _ = _make_orchestrator_with_mock_tools()
    orch.image_analysis_agent.execute = MagicMock(return_value={
        "status": "success",
        "output": {
            "image_type": "printed_text",
            "resolution_dpi": 300.0,
            "noise_level": 0.05,
            "contrast_ratio": 10.0,
            "has_diagrams": False,
            "image_dimensions": {"width": 800, "height": 1100},
            "recommended_enhancements": []
        },
        "execution_time_seconds": 0.05
    })
    session_id = str(uuid.uuid4())
    start = time.time()
    result = orch.execute({"image_path": "fake.jpg", "session_id": session_id})
    elapsed = time.time() - start
    assert elapsed <= 120, f"Pipeline took {elapsed:.1f}s, expected <= 120s"
    assert result["status"] == "success"
