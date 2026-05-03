"""
Data model type definitions for the Agentic OCR Pipeline.

All models are defined as TypedDicts for type safety and documentation clarity.
"""

from typing import TypedDict, List, Dict, Literal, Optional


# ---------------------------------------------------------------------------
# Literal type for image classification
# ---------------------------------------------------------------------------

ImageType = Literal[
    "dense_handwriting",
    "mixed_text_diagram",
    "formula_heavy",
    "low_quality",
    "printed_text"
]


# ---------------------------------------------------------------------------
# ImageAnalysisReport
# ---------------------------------------------------------------------------

class ImageAnalysisReport(TypedDict):
    """Structured report produced by the Image Analysis Agent."""

    image_type: ImageType                   # Classified image type
    resolution_dpi: float                   # Estimated DPI
    noise_level: float                      # 0.0 (clean) to 1.0 (very noisy)
    contrast_ratio: float                   # Max/min pixel ratio
    has_diagrams: bool                      # Whether diagrams were detected
    image_dimensions: dict                  # {"width": int, "height": int}
    recommended_enhancements: List[str]     # Actionable suggestions


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class Strategy(TypedDict):
    """Represents a single OCR strategy in the strategy registry."""

    name: str                               # Unique strategy identifier
    tool_name: str                          # Which AgentTool to invoke
    engine: str                             # OCR engine: "api", "florence", "got", "easyocr"
    requires_api: bool                      # Whether Groq API is needed
    description: str                        # Human-readable description
    best_for: List[ImageType]               # Image types this strategy excels at


# ---------------------------------------------------------------------------
# OCRResult
# ---------------------------------------------------------------------------

class OCRResult(TypedDict, total=False):
    """Result returned by an OCR engine."""

    text: str                               # Extracted text
    confidence: float                       # Engine-reported confidence [0.0, 1.0]
    engine: str                             # Which engine produced this result
    execution_time_seconds: float           # Time taken by the OCR engine
    error: str                              # Error message if failed (optional)


# ---------------------------------------------------------------------------
# QualityReport
# ---------------------------------------------------------------------------

class QualityReport(TypedDict):
    """Quality assessment report produced by the Quality Assessment Agent."""

    confidence_score: float                 # Computed confidence [0.0, 1.0]
    classification: Literal[               # Quality tier
        "high_quality",
        "acceptable",
        "low_quality"
    ]
    rationale: str                          # Plain-English explanation


# ---------------------------------------------------------------------------
# DecisionLog
# ---------------------------------------------------------------------------

class DecisionLog(TypedDict):
    """A single entry in the session decision log."""

    session_id: str                         # Unique session identifier
    timestamp: str                          # ISO 8601 timestamp
    agent: str                              # Name of the agent that made the decision
    decision_type: str                      # Type of decision (e.g. "strategy_selection")
    inputs: dict                            # Inputs that led to the decision
    output: dict                            # Decision output / result
    rationale: str                          # Plain-English explanation


# ---------------------------------------------------------------------------
# ShortTermMemory
# ---------------------------------------------------------------------------

class ShortTermMemory(TypedDict, total=False):
    """In-memory session state, keyed by session_id in MemoryManager."""

    session_id: str                         # Session identifier
    image_path: str                         # Full path to the image being processed
    actions: List[dict]                     # All agent actions with timestamps
    decisions: List[dict]                   # All agent decisions
    tried_strategies: List[str]             # Strategies attempted this session
    start_time: str                         # ISO 8601 session start time


# ---------------------------------------------------------------------------
# LongTermMemory helpers
# ---------------------------------------------------------------------------

class StrategyPerformance(TypedDict):
    """Rolling performance record for one strategy / image-type combination."""

    attempts: int                           # Total invocations
    successes: int                          # Successful invocations (confidence >= 0.5)
    avg_confidence: float                   # Rolling average confidence
    recent_confidences: List[float]         # Last 50 confidence scores


class LongTermMemory(TypedDict):
    """
    Persistent long-term memory structure.

    Outer key: image_type  (e.g. "dense_handwriting")
    Inner key: strategy name (e.g. "groq_api")
    """

    strategy_performance: Dict[
        str,                                # image_type
        Dict[str, StrategyPerformance]      # strategy_name → performance
    ]
