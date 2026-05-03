"""
Orchestrator Agent for the Agentic OCR Pipeline.

Central coordinator that manages the full Perceive → Decide → Act → Learn loop.

Responsibilities:
- Coordinate all other agents
- Manage retry logic (max 3 retries)
- Trigger human-in-the-loop when needed
- Ensure all phases execute in order
- Record every retry attempt, strategy, and confidence in the Decision Log

Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 18.5
"""

import time
from typing import Any, Dict, List, Optional

from .base_agent import BaseAgent
from .image_analysis_agent import ImageAnalysisAgent
from .memory_manager import MemoryManager
from .quality_agent import QualityAssessmentAgent
from .strategy_agent import StrategySelectionAgent
from .tools.base_tool import AgentTool


class OrchestratorAgent(BaseAgent):
    """
    Central coordinator that manages the full Perceive → Decide → Act → Learn loop.

    Responsibilities:
    - Coordinate all other agents (ImageAnalysisAgent, StrategySelectionAgent,
      QualityAssessmentAgent)
    - Manage retry logic (max RETRY_BUDGET=3 retries)
    - Trigger human-in-the-loop when all retries are exhausted and confidence < 0.5
    - Ensure all phases execute in order: perceive → decide → act → learn
    - Record every retry attempt, strategy, and confidence in the Decision Log
    - Catch all unhandled exceptions from tools/agents, log them, and proceed
      to the next strategy without crashing (Req 4.6)
    - Invoke Phase 1 components exclusively through Agent_Tool wrappers (Req 8.4)
    """

    RETRY_BUDGET = 3
    MIN_ACCEPTABLE_CONFIDENCE = 0.5

    def __init__(self, memory_manager: MemoryManager, tools: Dict[str, AgentTool]) -> None:
        """
        Initialise the OrchestratorAgent.

        Args:
            memory_manager: Shared MemoryManager instance for STM/LTM access.
            tools: Dict mapping tool names (str) to AgentTool instances.
        """
        super().__init__("orchestrator", memory_manager)
        self.tools = tools

        # Instantiate sub-agents
        self.image_analysis_agent = ImageAnalysisAgent(memory_manager)
        self.strategy_agent = StrategySelectionAgent(memory_manager)
        self.quality_agent = QualityAssessmentAgent(memory_manager)

    # ------------------------------------------------------------------
    # Public Interface
    # ------------------------------------------------------------------

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the full agentic loop for an image.

        Args:
            inputs: {
                "image_path": str,
                "session_id": str,
                "user_override": dict | None  — manual strategy override
            }

        Returns: {
            "status": "success" | "error",
            "final_result": dict | None,
            "confidence_score": float,
            "retry_count": int,
            "phases_completed": list[str],
            "decision_log": dict,
            "explainability_report": str,
            "hitl_triggered": bool,
            "execution_time_seconds": float
        }
        """
        start_time = time.time()
        image_path: str = inputs["image_path"]
        session_id: str = inputs["session_id"]
        user_override: Optional[Dict[str, Any]] = inputs.get("user_override")

        phases_completed: List[str] = []
        retry_count: int = 0
        hitl_triggered: bool = False

        try:
            # Initialise session memory
            self.memory_manager.start_session(session_id, image_path)

            # ----------------------------------------------------------
            # PHASE 1: PERCEIVE
            # ----------------------------------------------------------
            self.logger.info("[%s] PHASE 1: PERCEIVE", session_id)
            analysis_result = self.image_analysis_agent.execute({"image_path": image_path})

            if analysis_result["status"] == "error":
                return self._build_error_result("Perception phase failed", analysis_result)

            analysis_report: Dict[str, Any] = analysis_result["output"]
            phases_completed.append("perceive")

            # ----------------------------------------------------------
            # PHASE 2: DECIDE
            # ----------------------------------------------------------
            self.logger.info("[%s] PHASE 2: DECIDE", session_id)

            if user_override:
                # User manually specified strategy — bypass strategy agent
                strategy: Dict[str, Any] = user_override
                explainability: str = "User manually overrode strategy selection"
                candidate_strategies: List[Dict[str, Any]] = [strategy]
            else:
                strategy_result = self.strategy_agent.execute({
                    "analysis_report": analysis_report,
                    "session_id": session_id,
                })

                if strategy_result["status"] == "error":
                    return self._build_error_result("Decision phase failed", strategy_result)

                strategy = strategy_result["output"]["selected_strategy"]
                explainability = strategy_result["output"]["explainability_report"]
                candidate_strategies = strategy_result["output"]["candidate_strategies"]

            phases_completed.append("decide")

            # ----------------------------------------------------------
            # PHASE 3: ACT (with retry loop)
            # ----------------------------------------------------------
            self.logger.info("[%s] PHASE 3: ACT", session_id)

            best_result: Optional[Dict[str, Any]] = None
            best_confidence: float = 0.0

            for attempt in range(self.RETRY_BUDGET + 1):  # initial attempt + up to 3 retries
                if attempt > 0:
                    retry_count += 1
                    self.logger.info(
                        "[%s] Retry %d/%d", session_id, retry_count, self.RETRY_BUDGET
                    )

                    # Select next strategy from candidates
                    if retry_count <= len(candidate_strategies) - 1:
                        strategy = candidate_strategies[retry_count]
                    else:
                        # Exhausted all candidates
                        break

                # Record that this strategy is being tried (deduplication guard)
                self.memory_manager.record_tried_strategy(
                    session_id, strategy.get("name", "unknown")
                )

                # Execute OCR with selected strategy
                try:
                    ocr_result = self._execute_strategy(strategy, image_path)
                except Exception as tool_exc:  # pragma: no cover — belt-and-suspenders
                    self.logger.warning(
                        "[%s] Unexpected exception from _execute_strategy: %s",
                        session_id,
                        tool_exc,
                        exc_info=True,
                    )
                    self.memory_manager.record_action({
                        "agent": "orchestrator",
                        "action": "strategy_exception",
                        "attempt": attempt,
                        "strategy": strategy.get("name"),
                        "error": str(tool_exc),
                    })
                    continue

                if ocr_result["status"] == "error":
                    self.logger.warning(
                        "[%s] Strategy %s failed: %s",
                        session_id,
                        strategy.get("name"),
                        ocr_result.get("error_message"),
                    )
                    self.memory_manager.record_action({
                        "agent": "orchestrator",
                        "action": "strategy_error",
                        "attempt": attempt,
                        "strategy": strategy.get("name"),
                        "error": ocr_result.get("error_message"),
                    })
                    continue

                # Assess quality of OCR output
                quality_result = self.quality_agent.execute({
                    "ocr_result": ocr_result["output"],
                    "strategy": strategy,
                })

                confidence_score: float = quality_result["output"]["confidence_score"]

                # Record attempt to memory (Req 4.5)
                self.memory_manager.record_action({
                    "agent": "orchestrator",
                    "action": "retry_attempt",
                    "attempt": attempt,
                    "strategy": strategy.get("name"),
                    "confidence": confidence_score,
                })

                # Track best result across all attempts
                if confidence_score > best_confidence:
                    best_result = ocr_result["output"]
                    best_confidence = confidence_score

                # Break early if quality is acceptable (Req 4.2)
                if confidence_score >= self.MIN_ACCEPTABLE_CONFIDENCE:
                    self.logger.info(
                        "[%s] Acceptable quality achieved: %.2f", session_id, confidence_score
                    )
                    break

            phases_completed.append("act")

            # ----------------------------------------------------------
            # HITL check (Req 4.4)
            # ----------------------------------------------------------
            if best_confidence < self.MIN_ACCEPTABLE_CONFIDENCE:
                self.logger.warning(
                    "[%s] Quality below threshold after %d retries — triggering HITL",
                    session_id,
                    retry_count,
                )
                hitl_triggered = True

            # ----------------------------------------------------------
            # PHASE 4: LEARN
            # ----------------------------------------------------------
            self.logger.info("[%s] PHASE 4: LEARN", session_id)
            self.memory_manager.update_long_term_memory(
                image_type=analysis_report["image_type"],
                strategy=strategy,
                confidence_score=best_confidence,
                success=(best_confidence >= self.MIN_ACCEPTABLE_CONFIDENCE),
            )
            phases_completed.append("learn")

            # ----------------------------------------------------------
            # Build final result
            # ----------------------------------------------------------
            execution_time = time.time() - start_time

            return {
                "status": "success",
                "final_result": best_result,
                "confidence_score": best_confidence,
                "retry_count": retry_count,
                "phases_completed": phases_completed,
                "decision_log": self.memory_manager.get_session_log(session_id),
                "explainability_report": explainability,
                "hitl_triggered": hitl_triggered,
                "execution_time_seconds": execution_time,
            }

        except Exception as e:
            return self.handle_error(
                e,
                {
                    "image_path": image_path,
                    "session_id": session_id,
                    "phases_completed": phases_completed,
                },
            )

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _execute_strategy(self, strategy: Dict[str, Any], image_path: str) -> Dict[str, Any]:
        """
        Execute a strategy by looking up and invoking the appropriate Agent_Tool.

        Args:
            strategy: Strategy dict containing at least "tool_name" and "name".
            image_path: Path to the image to process.

        Returns:
            Tool invocation result dict (status/output/execution_time_seconds).
            Returns an error dict if the tool is not found.
        """
        tool_name: str = strategy.get("tool_name", "ocr_tool")
        tool: Optional[AgentTool] = self.tools.get(tool_name)

        if tool is None:
            return {
                "status": "error",
                "error_message": f"Tool {tool_name} not found",
            }

        return tool.invoke({
            "image_path": image_path,
            "strategy_params": strategy,
        })

    def _build_error_result(self, message: str, error_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build a standardised error result dict.

        Args:
            message: Human-readable error description.
            error_details: The raw error dict from the failing agent/tool.

        Returns:
            Standardised error result dict.
        """
        return {
            "status": "error",
            "error_message": message,
            "error_details": error_details,
            "final_result": None,
            "confidence_score": 0.0,
            "retry_count": 0,
            "phases_completed": [],
            "hitl_triggered": False,
        }
