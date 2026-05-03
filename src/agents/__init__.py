"""
Agentic OCR Pipeline — agents package.

This module exposes the public API for the agentic system.  The primary
entry-point is :func:`create_agentic_system`, which wires together the
:class:`MemoryManager`, all :class:`AgentTool` wrappers, and the
:class:`OrchestratorAgent` into a ready-to-use orchestrator instance.

Usage::

    from src.agents import create_agentic_system

    orchestrator = create_agentic_system()
    result = orchestrator.execute({
        "image_path": "path/to/image.jpg",
        "session_id": "my-session-001",
    })
"""

from .memory_manager import MemoryManager
from .orchestrator_agent import OrchestratorAgent
from .tools.corrector_tool import CorrectorTool
from .tools.diagram_tool import DiagramTool
from .tools.document_tool import DocumentTool
from .tools.ocr_tool import OCRTool
from .tools.preprocessor_tool import PreprocessorTool


def create_agentic_system() -> OrchestratorAgent:
    """
    Instantiate and wire together the full agentic OCR system.

    Creates a :class:`MemoryManager`, all five :class:`AgentTool` wrappers,
    and an :class:`OrchestratorAgent` that owns them.  The orchestrator is
    returned ready to accept :meth:`~OrchestratorAgent.execute` calls.

    Returns:
        OrchestratorAgent: Fully configured orchestrator instance.

    Requirements: 19.3, 19.4
    """
    memory_manager = MemoryManager()

    tools = {
        "preprocessor_tool": PreprocessorTool(),
        "diagram_tool": DiagramTool(),
        "ocr_tool": OCRTool(),
        "corrector_tool": CorrectorTool(),
        "document_tool": DocumentTool(),
    }

    orchestrator = OrchestratorAgent(memory_manager, tools)
    return orchestrator


__all__ = [
    "create_agentic_system",
    "OrchestratorAgent",
    "MemoryManager",
    "PreprocessorTool",
    "DiagramTool",
    "OCRTool",
    "CorrectorTool",
    "DocumentTool",
]
