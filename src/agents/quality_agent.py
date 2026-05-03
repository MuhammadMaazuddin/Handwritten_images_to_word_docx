"""
Quality Assessment Agent for the Agentic OCR Pipeline.

Evaluates OCR output quality using multiple signals:
1. Engine-reported confidence
2. Text coherence (word-length heuristic)
3. Word count plausibility
4. Garbled character sequence detection
"""

import re
import time
from typing import Any, Dict, Optional

from .base_agent import BaseAgent


class QualityAssessmentAgent(BaseAgent):
    """
    Evaluates OCR output quality using multiple signals.

    Confidence score computation uses:
    1. Engine-reported confidence (if available)
    2. Text coherence (word-length heuristic)
    3. Expected vs actual word count
    4. Garbled character sequence detection

    The final confidence score is a weighted average of the four signals,
    clamped to [0.0, 1.0], and classified into:
    - "high_quality"  (score >= 0.8)
    - "acceptable"    (score >= 0.5)
    - "low_quality"   (score <  0.5)
    """

    HIGH_QUALITY_THRESHOLD = 0.8
    ACCEPTABLE_THRESHOLD = 0.5

    # Signal weights (must sum to 1.0)
    WEIGHTS = {
        "engine_confidence": 0.35,
        "text_coherence": 0.25,
        "word_count": 0.20,
        "garbled_sequences": 0.20,
    }

    def __init__(self, memory_manager: "MemoryManager") -> None:
        super().__init__("quality_assessment", memory_manager)

    # ------------------------------------------------------------------
    # Public Interface
    # ------------------------------------------------------------------

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate OCR result quality.

        Args:
            inputs: {
                "ocr_result": {
                    "text": str,
                    "confidence": float | None  (engine-reported, optional)
                },
                "strategy": dict  (strategy that produced the result)
            }

        Returns: {
            "status": "success" | "error",
            "output": {
                "confidence_score": float,   # clamped [0.0, 1.0]
                "classification": str,       # "high_quality" | "acceptable" | "low_quality"
                "rationale": str
            },
            "execution_time_seconds": float
        }
        """
        start_time = time.time()

        try:
            ocr_result = inputs["ocr_result"]
            text: str = ocr_result.get("text", "") or ""
            engine_confidence: Optional[float] = ocr_result.get("confidence", None)

            # Compute individual signals
            signal_engine = self._signal_engine_confidence(engine_confidence)
            signal_coherence = self._signal_text_coherence(text)
            signal_word_count = self._signal_word_count(text)
            signal_garbled = self._signal_garbled_sequences(text)

            signals = {
                "engine_confidence": signal_engine,
                "text_coherence": signal_coherence,
                "word_count": signal_word_count,
                "garbled_sequences": signal_garbled,
            }

            # Weighted average, clamped to [0.0, 1.0]
            score = self._compute_confidence(signals)
            score = max(0.0, min(1.0, score))

            classification = self._classify(score)
            rationale = self._build_rationale(signals, score, classification)

            # Record to memory
            self.memory_manager.record_action({
                "agent": self.name,
                "action": "quality_assessment",
                "result": {
                    "confidence_score": score,
                    "classification": classification,
                },
            })

            execution_time = time.time() - start_time

            return {
                "status": "success",
                "output": {
                    "confidence_score": score,
                    "classification": classification,
                    "rationale": rationale,
                },
                "execution_time_seconds": execution_time,
            }

        except Exception as e:
            return self.handle_error(e, inputs)

    # ------------------------------------------------------------------
    # Signal Computation
    # ------------------------------------------------------------------

    def _signal_engine_confidence(self, engine_confidence: Any) -> float:
        """
        Return the engine-reported confidence as a signal.

        If engine_confidence is None or not a number, return 0.5 (neutral).
        Otherwise return float(engine_confidence) clamped to [0.0, 1.0].
        """
        if engine_confidence is None:
            return 0.5
        try:
            value = float(engine_confidence)
        except (TypeError, ValueError):
            return 0.5
        return max(0.0, min(1.0, value))

    def _signal_text_coherence(self, text: str) -> float:
        """
        Estimate text coherence using an average word-length heuristic.

        - No words → 0.0
        - avg_word_len in [3, 10] → 0.9  (plausible natural language)
        - avg_word_len < 3         → 0.3  (likely garbled / single chars)
        - avg_word_len > 10        → 0.5  (possibly concatenated tokens)
        """
        words = text.split() if text else []
        if not words:
            return 0.0

        avg_word_len = sum(len(w) for w in words) / len(words)

        if 3 <= avg_word_len <= 10:
            return 0.9
        elif avg_word_len < 3:
            return 0.3
        else:  # avg_word_len > 10
            return 0.5

    def _signal_word_count(self, text: str) -> float:
        """
        Check whether the word count is plausible for a scanned A4 page.

        Ranges:
        - 50–600 words  → 0.9  (plausible A4 page)
        - 10–49 words   → 0.6
        - 601–1000 words → 0.7
        - 0–9 words     → 0.1
        - >1000 words   → 0.5
        """
        words = text.split() if text else []
        count = len(words)

        if 50 <= count <= 600:
            return 0.9
        elif 10 <= count <= 49:
            return 0.6
        elif 601 <= count <= 1000:
            return 0.7
        elif count <= 9:
            return 0.1
        else:  # count > 1000
            return 0.5

    def _signal_garbled_sequences(self, text: str) -> float:
        """
        Detect garbled character sequences using regex patterns.

        Patterns:
        - Consonant clusters: 4+ consecutive non-vowel, non-whitespace, non-digit chars
        - Symbol clusters:    3+ consecutive non-alphanumeric, non-whitespace chars

        Scoring:
        - 0 matches   → 1.0
        - 1–2 matches → 0.8
        - 3–5 matches → 0.5
        - >5 matches  → 0.2
        """
        if not text:
            return 1.0

        consonant_cluster_pattern = r'[^aeiouAEIOU\s\d]{4,}'
        symbol_cluster_pattern = r'[^a-zA-Z0-9\s]{3,}'

        matches = (
            len(re.findall(consonant_cluster_pattern, text))
            + len(re.findall(symbol_cluster_pattern, text))
        )

        if matches == 0:
            return 1.0
        elif matches <= 2:
            return 0.8
        elif matches <= 5:
            return 0.5
        else:
            return 0.2

    # ------------------------------------------------------------------
    # Aggregation and Classification
    # ------------------------------------------------------------------

    def _compute_confidence(self, signals: Dict[str, float]) -> float:
        """
        Compute a weighted average of the four quality signals.

        Weights:
        - engine_confidence: 0.35
        - text_coherence:    0.25
        - word_count:        0.20
        - garbled_sequences: 0.20

        Returns the weighted sum (weights already sum to 1.0).
        """
        return (
            self.WEIGHTS["engine_confidence"] * signals["engine_confidence"]
            + self.WEIGHTS["text_coherence"] * signals["text_coherence"]
            + self.WEIGHTS["word_count"] * signals["word_count"]
            + self.WEIGHTS["garbled_sequences"] * signals["garbled_sequences"]
        )

    def _classify(self, confidence_score: float) -> str:
        """
        Classify a confidence score into a quality tier.

        - >= 0.8 → "high_quality"
        - >= 0.5 → "acceptable"
        - <  0.5 → "low_quality"
        """
        if confidence_score >= self.HIGH_QUALITY_THRESHOLD:
            return "high_quality"
        elif confidence_score >= self.ACCEPTABLE_THRESHOLD:
            return "acceptable"
        else:
            return "low_quality"

    def _build_rationale(
        self, signals: Dict[str, float], score: float, classification: str
    ) -> str:
        """
        Build a plain-English rationale string summarising the assessment.

        Format:
        "Quality assessment: <classification> (score: X.XX). Engine confidence: X.XX.
         Text coherence: X.XX. Word count signal: X.XX. Garbled sequence check: X.XX."
        """
        return (
            f"Quality assessment: {classification} (score: {score:.2f}). "
            f"Engine confidence: {signals['engine_confidence']:.2f}. "
            f"Text coherence: {signals['text_coherence']:.2f}. "
            f"Word count signal: {signals['word_count']:.2f}. "
            f"Garbled sequence check: {signals['garbled_sequences']:.2f}."
        )
