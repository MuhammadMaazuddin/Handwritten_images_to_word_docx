"""
Strategy Selection Agent for the Agentic OCR Pipeline.

Selects the optimal OCR strategy based on image analysis and historical performance.

Decision factors:
1. Image type compatibility (base score)
2. Long-term memory success rates (>=40% weight when >=5 records)
3. API availability
4. User preferences
"""

from typing import Dict, Any, List, Optional
import time

from .base_agent import BaseAgent

# Strategy registry: all available strategies
STRATEGY_REGISTRY = {
    "groq_api": {
        "name": "groq_api",
        "tool_name": "ocr_tool",
        "engine": "api",
        "requires_api": True,
        "description": "Groq Llama Vision API - highest quality, requires internet",
        "best_for": ["dense_handwriting", "formula_heavy", "mixed_text_diagram"]
    },
    "florence_local": {
        "name": "florence_local",
        "tool_name": "ocr_tool",
        "engine": "florence",
        "requires_api": False,
        "description": "Florence-2 local model - good for mixed content",
        "best_for": ["mixed_text_diagram", "printed_text"]
    },
    "got_ocr_local": {
        "name": "got_ocr_local",
        "tool_name": "ocr_tool",
        "engine": "got",
        "requires_api": False,
        "description": "GOT-OCR 2.0 local model - best for handwriting and formulas",
        "best_for": ["dense_handwriting", "formula_heavy"]
    },
    "easyocr_local": {
        "name": "easyocr_local",
        "tool_name": "ocr_tool",
        "engine": "easyocr",
        "requires_api": False,
        "description": "EasyOCR - lightweight fallback",
        "best_for": ["printed_text", "low_quality"]
    }
}

MEMORY_WEIGHT_THRESHOLD = 5   # Min records before memory influences selection
MEMORY_WEIGHT = 0.4           # Memory contributes 40% when threshold met


class StrategySelectionAgent(BaseAgent):
    """
    Selects the optimal OCR strategy based on image analysis and historical performance.

    Decision factors:
    1. Image type compatibility (base score)
    2. Long-term memory success rates (>=40% weight when >=5 records)
    3. API availability
    4. User preferences
    """

    def __init__(self, memory_manager: "MemoryManager") -> None:
        super().__init__("strategy_selection", memory_manager)

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select optimal strategy for the given image analysis report.

        Args:
            inputs: {
                "analysis_report": ImageAnalysisReport dict,
                "session_id": str,
                "api_available": bool (optional, default True),
                "user_preference": str (optional, engine name)
            }

        Returns: {
            "status": "success" | "error",
            "output": {
                "selected_strategy": Strategy dict,
                "candidate_strategies": List[Strategy],
                "explainability_report": str,
                "selection_rationale": List[str]
            },
            "execution_time_seconds": float
        }
        """
        start_time = time.time()

        try:
            analysis_report = inputs["analysis_report"]
            image_type = analysis_report["image_type"]
            api_available = inputs.get("api_available", True)
            user_preference = inputs.get("user_preference")

            # Get available strategies (filter by API availability)
            available_strategies = self._get_available_strategies(api_available)

            # Score each strategy
            scored_strategies = self._score_strategies(
                available_strategies, image_type, user_preference
            )

            # Sort by score descending
            scored_strategies.sort(key=lambda x: x["score"], reverse=True)

            # Always return all candidates (minimum 3)
            candidate_strategies = scored_strategies[: max(3, len(scored_strategies))]
            selected_strategy = candidate_strategies[0]["strategy"]

            # Generate explainability report
            explainability = self._generate_explainability(
                selected_strategy, candidate_strategies, image_type, user_preference
            )

            # Record decision
            self.log_decision(
                decision_type="strategy_selection",
                inputs={"image_type": image_type, "api_available": api_available},
                output={"selected": selected_strategy["name"]},
                rationale=explainability,
            )

            execution_time = time.time() - start_time

            return {
                "status": "success",
                "output": {
                    "selected_strategy": selected_strategy,
                    "candidate_strategies": [s["strategy"] for s in candidate_strategies],
                    "explainability_report": explainability,
                    "selection_rationale": [s["rationale"] for s in candidate_strategies[:3]],
                },
                "execution_time_seconds": execution_time,
            }

        except Exception as e:
            return self.handle_error(e, inputs)

    def _get_available_strategies(self, api_available: bool) -> List[Dict[str, Any]]:
        """
        Filter strategies by API availability.

        Args:
            api_available: Whether the external API is reachable.

        Returns:
            List of strategy dicts from STRATEGY_REGISTRY that can be used.
        """
        return [
            s for s in STRATEGY_REGISTRY.values()
            if not s["requires_api"] or api_available
        ]

    def _score_strategies(
        self,
        strategies: List[Dict[str, Any]],
        image_type: str,
        user_preference: Optional[str],
    ) -> List[Dict[str, Any]]:
        """
        Score each strategy using base compatibility + memory data.

        Scoring formula:
        - base_score: 1.0 if image_type in strategy["best_for"], else 0.5
        - memory_score: avg_confidence from LTM (default 0.5 equal prior)
        - preference_bonus: 0.1 if strategy matches user_preference
        - When sufficient memory: final = (1 - MEMORY_WEIGHT) * base + MEMORY_WEIGHT * memory + bonus
        - When insufficient memory: final = base + bonus

        Args:
            strategies: List of strategy dicts to score.
            image_type: Classified image type from analysis report.
            user_preference: Optional preferred strategy name.

        Returns:
            List of dicts with keys: "strategy", "score", "rationale".
        """
        ltm = self.memory_manager.get_long_term_memory()
        image_type_records = ltm.get("strategy_performance", {}).get(image_type, {})
        has_sufficient_memory = (
            sum(v.get("attempts", 0) for v in image_type_records.values())
            >= MEMORY_WEIGHT_THRESHOLD
        )

        scored = []
        for strategy in strategies:
            # Base score: compatibility with image type
            base_score = 1.0 if image_type in strategy.get("best_for", []) else 0.5

            # Memory score (default equal prior)
            memory_score = 0.5
            if has_sufficient_memory:
                perf = image_type_records.get(strategy["name"], {})
                if perf.get("attempts", 0) > 0:
                    memory_score = perf.get("avg_confidence", 0.5)

            # User preference bonus
            preference_bonus = 0.1 if strategy["name"] == user_preference else 0.0

            # Combine scores
            if has_sufficient_memory:
                final_score = (
                    (1 - MEMORY_WEIGHT) * base_score
                    + MEMORY_WEIGHT * memory_score
                    + preference_bonus
                )
            else:
                final_score = base_score + preference_bonus

            rationale = self._build_rationale(
                strategy, base_score, memory_score, has_sufficient_memory, user_preference
            )

            scored.append({
                "strategy": strategy,
                "score": final_score,
                "rationale": rationale,
            })

        return scored

    def _build_rationale(
        self,
        strategy: Dict[str, Any],
        base_score: float,
        memory_score: float,
        has_memory: bool,
        user_preference: Optional[str],
    ) -> str:
        """
        Build a plain-English rationale string for a strategy's score.

        Args:
            strategy: The strategy dict.
            base_score: Compatibility score (1.0 = well-suited, 0.5 = general).
            memory_score: Historical avg_confidence (0.5 default when no data).
            has_memory: Whether sufficient LTM records exist.
            user_preference: Optional preferred strategy name.

        Returns:
            Human-readable rationale string.
        """
        parts = []

        if base_score >= 1.0:
            parts.append(f"{strategy['name']} is well-suited for this image type")
        else:
            parts.append(f"{strategy['name']} is a general-purpose option")

        if has_memory:
            parts.append(f"historical success rate: {memory_score:.0%}")
        else:
            parts.append("no historical data yet (equal probability applied)")

        if strategy["name"] == user_preference:
            parts.append("matches your preferred engine")

        return "; ".join(parts)

    def _generate_explainability(
        self,
        selected: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        image_type: str,
        user_preference: Optional[str],
    ) -> str:
        """
        Generate a multi-line human-readable explainability report.

        Includes bias disclosure as required by ethical AI design principles.

        Args:
            selected: The chosen strategy dict.
            candidates: All scored candidate dicts (sorted by score descending).
            image_type: Classified image type.
            user_preference: Optional preferred strategy name.

        Returns:
            Multi-line explainability string.
        """
        lines = []

        # Line 1: selected strategy and image type
        lines.append(
            f"Selected strategy: {selected['name']} for image type '{image_type}'."
        )

        # Line 2: top 3 reasons
        reason_parts = []
        if len(candidates) >= 1:
            reason_parts.append(candidates[0]["rationale"])
        if len(candidates) >= 2:
            reason_parts.append(candidates[1]["rationale"])
        if len(candidates) >= 3:
            reason_parts.append(candidates[2]["rationale"])
        lines.append(f"Top 3 reasons: {'; '.join(reason_parts)}")

        # Line 3: user preference note (only when provided)
        if user_preference:
            lines.append(f"User preference noted: {user_preference}.")

        # Line 4: bias disclosure
        lines.append(
            "Bias disclosure: Strategy selection is based solely on image characteristics "
            "and historical performance data. No demographic, linguistic, or cultural "
            "attributes are used."
        )

        # Line 5: ranked candidates
        ranked = ", ".join(
            f"{s['strategy']['name']} (score: {s['score']:.2f})" for s in candidates
        )
        lines.append(f"Candidate strategies (ranked): {ranked}")

        return "\n".join(lines)
