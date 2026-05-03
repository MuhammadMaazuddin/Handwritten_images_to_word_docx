"""
Integration tests for API fallback behaviour.
Requirements: 3.5, 18.1, 18.2
"""
import os
import uuid
from unittest.mock import MagicMock, patch
import pytest
from src.agents.memory_manager import MemoryManager
from src.agents.orchestrator_agent import OrchestratorAgent
from src.agents.strategy_agent import StrategySelectionAgent, STRATEGY_REGISTRY


def _make_orch(api_available=True):
    mm = MemoryManager()
    mock_ocr_tool = MagicMock()
    mock_ocr_tool.invoke.return_value = {
        "status": "success",
        "output": {"text": " ".join(["word"] * 80), "confidence": 0.7, "method": "mock"},
        "execution_time_seconds": 0.1
    }
    tools = {"ocr_tool": mock_ocr_tool}
    orch = OrchestratorAgent(mm, tools)
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
    # Patch strategy agent to respect api_available
    original_execute = orch.strategy_agent.execute
    def patched_execute(inputs):
        inputs["api_available"] = api_available
        return original_execute(inputs)
    orch.strategy_agent.execute = patched_execute
    return orch, mm


def test_no_groq_api_key_excludes_groq_strategy():
    """When api_available=False, groq_api is not in candidates. Req 3.5, 18.1"""
    mm = MemoryManager()
    agent = StrategySelectionAgent(mm)
    result = agent.execute({
        "analysis_report": {"image_type": "dense_handwriting"},
        "session_id": "test",
        "api_available": False,
    })
    assert result["status"] == "success"
    candidate_names = [s["name"] for s in result["output"]["candidate_strategies"]]
    assert "groq_api" not in candidate_names, "groq_api should be excluded when API unavailable"


def test_local_strategy_used_when_api_unavailable():
    """When API key is empty, a local strategy is selected. Req 3.5, 18.1"""
    mm = MemoryManager()
    agent = StrategySelectionAgent(mm)
    result = agent.execute({
        "analysis_report": {"image_type": "dense_handwriting"},
        "session_id": "test",
        "api_available": False,
    })
    assert result["status"] == "success"
    selected = result["output"]["selected_strategy"]
    assert selected["requires_api"] is False


def test_easyocr_available_as_fallback():
    """easyocr_local is always in the registry as last-resort fallback. Req 18.2"""
    assert "easyocr_local" in STRATEGY_REGISTRY
    assert STRATEGY_REGISTRY["easyocr_local"]["requires_api"] is False
