"""
Integration tests for API fallback behaviour.
Requirements: 3.5, 18.1, 18.2
"""
import os
import uuid
import pytest
from unittest.mock import patch, MagicMock

from src.agents import create_agentic_system
from src.agents.strategy_agent import StrategySelectionAgent, STRATEGY_REGISTRY
from src.agents.memory_manager import MemoryManager


# ---------------------------------------------------------------------------
# Shared analysis report used across tests
# ---------------------------------------------------------------------------

_ANALYSIS_REPORT = {
    "image_type": "dense_handwriting",
    "resolution_dpi": 200.0,
    "noise_level": 0.3,
    "contrast_ratio": 5.0,
    "has_diagrams": False,
    "image_dimensions": {"width": 800, "height": 1000},
    "recommended_enhancements": [],
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestApiFallback:
    """Tests that verify correct strategy selection when the API is unavailable."""

    def test_no_api_key_excludes_groq(self):
        """
        When api_available=False, groq_api must not appear in candidate strategies.

        Requirements: 3.5, 18.1
        """
        mm = MemoryManager()
        agent = StrategySelectionAgent(mm)

        result = agent.execute({
            "analysis_report": _ANALYSIS_REPORT,
            "session_id": "test-no-api",
            "api_available": False,
        })

        assert result["status"] == "success", (
            f"Strategy agent returned error: {result.get('error_message')}"
        )

        candidate_names = [s["name"] for s in result["output"]["candidate_strategies"]]
        assert "groq_api" not in candidate_names, (
            f"groq_api should be excluded when api_available=False, "
            f"but candidates were: {candidate_names}"
        )

    def test_local_strategy_used_when_api_unavailable(self):
        """
        When api_available=False, at least one local (non-API) strategy must be
        present in the candidates and the call must succeed.

        Requirements: 3.5, 18.1, 18.2
        """
        mm = MemoryManager()
        agent = StrategySelectionAgent(mm)

        result = agent.execute({
            "analysis_report": _ANALYSIS_REPORT,
            "session_id": "test-local-fallback",
            "api_available": False,
        })

        assert result["status"] == "success", (
            f"Strategy agent returned error: {result.get('error_message')}"
        )

        # Determine which strategies are local (do not require API)
        local_strategy_names = {
            name for name, info in STRATEGY_REGISTRY.items()
            if not info.get("requires_api", False)
        }

        candidate_names = [s["name"] for s in result["output"]["candidate_strategies"]]
        local_candidates = [n for n in candidate_names if n in local_strategy_names]

        assert len(local_candidates) >= 1, (
            f"Expected at least one local strategy in candidates when api_available=False, "
            f"but candidates were: {candidate_names}"
        )
