"""
Memory Manager for the Agentic OCR Pipeline.

Manages short-term (session) and long-term (persistent) memory.

Short-term memory (STM): in-memory dict keyed by session_id, cleared at session end.
Long-term memory (LTM): JSON file persisted at memory/long_term_memory.json.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Manages short-term (session) and long-term (persistent) memory.

    Internal state:
        _stm: dict — keyed by session_id, holds per-session data
        _ltm: dict — {"strategy_performance": {image_type: {strategy_name: {...}}}}

    Short-term memory is cleared at session end to prevent data leakage.
    Long-term memory is persisted to memory/long_term_memory.json.
    """

    LTM_PATH = "memory/long_term_memory.json"
    MAX_ROLLING_WINDOW = 50  # Max recent_confidences per strategy/image_type

    def __init__(self) -> None:
        """
        Initialise the MemoryManager.

        Creates memory/ and logs/ directories if they do not exist,
        then loads long-term memory from disk.
        """
        self._stm: Dict[str, Any] = {}
        self._ltm: Dict[str, Any] = {}

        os.makedirs("memory", exist_ok=True)
        os.makedirs("logs", exist_ok=True)

        self._load_long_term_memory()

    # ------------------------------------------------------------------
    # Session Management
    # ------------------------------------------------------------------

    def start_session(self, session_id: str, image_path: str) -> None:
        """
        Initialise short-term memory for a new session.

        Args:
            session_id: Unique identifier for the session.
            image_path: Full path to the image being processed.
        """
        self._stm[session_id] = {
            "session_id": session_id,
            "image_path": image_path,
            "actions": [],
            "decisions": [],
            "tried_strategies": [],
            "start_time": datetime.now().isoformat(),
        }

    def end_session(self, session_id: str) -> None:
        """
        Persist the session log to disk and clear the STM entry.

        Writes logs/session_{session_id}.json with the full STM entry,
        then deletes the entry to prevent data leakage between sessions.

        Args:
            session_id: The session to end.
        """
        if session_id not in self._stm:
            logger.warning(
                "end_session called for unknown session_id '%s'", session_id
            )
            return

        session_data = self._stm[session_id]

        log_path = f"logs/session_{session_id}.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2)

        del self._stm[session_id]

    # ------------------------------------------------------------------
    # Short-Term Memory Operations
    # ------------------------------------------------------------------

    def record_action(self, action: Dict[str, Any]) -> None:
        """
        Append an action (with timestamp) to the active session's STM actions list.

        Determines the active session by finding the most recently started session
        in STM. If no active session exists, logs a warning and skips.

        Args:
            action: Arbitrary action dict to record.
        """
        active_session = self._get_active_session()
        if active_session is None:
            logger.warning(
                "record_action called but no active session found; skipping."
            )
            return

        action_entry = {**action, "timestamp": datetime.now().isoformat()}
        active_session["actions"].append(action_entry)

    def record_decision(self, decision: Dict[str, Any]) -> None:
        """
        Append a decision (with timestamp) to the active session's STM decisions list.

        Determines the active session by finding the most recently started session
        in STM. If no active session exists, logs a warning and skips.

        Args:
            decision: Arbitrary decision dict to record.
        """
        active_session = self._get_active_session()
        if active_session is None:
            logger.warning(
                "record_decision called but no active session found; skipping."
            )
            return

        decision_entry = {**decision, "timestamp": datetime.now().isoformat()}
        active_session["decisions"].append(decision_entry)

    def record_tried_strategy(self, session_id: str, strategy_name: str) -> None:
        """
        Add strategy_name to tried_strategies for the session (no duplicates).

        Args:
            session_id: The session to update.
            strategy_name: Name of the strategy that was tried.
        """
        session = self._stm.get(session_id)
        if session is None:
            logger.warning(
                "record_tried_strategy called for unknown session_id '%s'", session_id
            )
            return

        if strategy_name not in session["tried_strategies"]:
            session["tried_strategies"].append(strategy_name)

    def was_strategy_tried(self, session_id: str, strategy_name: str) -> bool:
        """
        Return True if strategy_name is in tried_strategies for the session.

        Args:
            session_id: The session to check.
            strategy_name: Name of the strategy to look up.

        Returns:
            True if the strategy was already tried, False otherwise.
        """
        session = self._stm.get(session_id, {})
        return strategy_name in session.get("tried_strategies", [])

    def get_session_log(self, session_id: str) -> Dict[str, Any]:
        """
        Return the STM dict for the session, or an empty dict if not found.

        Args:
            session_id: The session to retrieve.

        Returns:
            The session's STM dict, or {} if the session does not exist.
        """
        return self._stm.get(session_id, {})

    def get_session_log_json(self, session_id: str) -> str:
        """
        Return a JSON string of the session log.

        Args:
            session_id: The session to serialise.

        Returns:
            JSON-encoded string of the session log.
        """
        return json.dumps(self.get_session_log(session_id), indent=2)

    # ------------------------------------------------------------------
    # Long-Term Memory Operations
    # ------------------------------------------------------------------

    def get_long_term_memory(self) -> Dict[str, Any]:
        """
        Return the full LTM dict.

        Returns:
            The long-term memory dict.
        """
        return self._ltm

    def update_long_term_memory(
        self,
        image_type: str,
        strategy: Dict[str, Any],
        confidence_score: float,
        success: bool,
    ) -> None:
        """
        Update the rolling performance record for an image_type/strategy combination.

        Rolling window is capped at MAX_ROLLING_WINDOW (50) recent_confidences.
        Recalculates avg_confidence. Increments attempts and successes.
        Persists to disk via _save_long_term_memory().

        Args:
            image_type: The classified image type (e.g. "dense_handwriting").
            strategy: Strategy dict containing at least a "name" key.
            confidence_score: Confidence score in [0.0, 1.0] for this attempt.
            success: Whether the attempt was considered successful.
        """
        strategy_name = strategy.get("name", "unknown")

        perf = (
            self._ltm
            .setdefault("strategy_performance", {})
            .setdefault(image_type, {})
            .setdefault(
                strategy_name,
                {
                    "attempts": 0,
                    "successes": 0,
                    "avg_confidence": 0.0,
                    "recent_confidences": [],
                },
            )
        )

        # Update rolling window
        recent: list = perf.setdefault("recent_confidences", [])
        recent.append(confidence_score)
        if len(recent) > self.MAX_ROLLING_WINDOW:
            recent.pop(0)

        # Update aggregates
        perf["attempts"] = perf.get("attempts", 0) + 1
        if success:
            perf["successes"] = perf.get("successes", 0) + 1
        perf["avg_confidence"] = sum(recent) / len(recent)

        self._save_long_term_memory()

    def get_ltm_stats(self) -> Dict[str, Any]:
        """
        Return per image_type stats.

        Returns:
            {
                image_type: {
                    strategy_name: {
                        "attempts": int,
                        "avg_confidence": float,
                        "successes": int,
                    }
                }
            }
        """
        perf = self._ltm.get("strategy_performance", {})
        stats: Dict[str, Any] = {}

        for image_type, strategies in perf.items():
            stats[image_type] = {
                strategy_name: {
                    "attempts": data.get("attempts", 0),
                    "avg_confidence": data.get("avg_confidence", 0.0),
                    "successes": data.get("successes", 0),
                }
                for strategy_name, data in strategies.items()
            }

        return stats

    def reset_long_term_memory(self) -> None:
        """
        Delete memory/long_term_memory.json (if it exists) and reinitialise self._ltm.

        After this call, self._ltm == {"strategy_performance": {}}.
        """
        if os.path.exists(self.LTM_PATH):
            os.remove(self.LTM_PATH)
        self._ltm = {"strategy_performance": {}}

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _load_long_term_memory(self) -> None:
        """
        Load LTM from memory/long_term_memory.json.

        Handles FileNotFoundError (initialises fresh) and json.JSONDecodeError
        (logs warning, initialises fresh). Sets self._ltm.
        """
        try:
            with open(self.LTM_PATH, "r", encoding="utf-8") as f:
                self._ltm = json.load(f)
        except FileNotFoundError:
            self._ltm = {"strategy_performance": {}}
        except json.JSONDecodeError as exc:
            logger.warning(
                "Long-term memory file '%s' is corrupted (%s). "
                "Initialising fresh memory store.",
                self.LTM_PATH,
                exc,
            )
            self._ltm = {"strategy_performance": {}}

    def _save_long_term_memory(self) -> None:
        """
        Write self._ltm to memory/long_term_memory.json with indent=2.
        """
        with open(self.LTM_PATH, "w", encoding="utf-8") as f:
            json.dump(self._ltm, f, indent=2)

    # ------------------------------------------------------------------
    # Internal Utility
    # ------------------------------------------------------------------

    def _get_active_session(self) -> Optional[Dict[str, Any]]:
        """
        Return the STM dict for the most recently started active session.

        Uses start_time to determine recency. Returns None if STM is empty.
        """
        if not self._stm:
            return None

        # Return the session with the latest start_time
        latest_id = max(
            self._stm,
            key=lambda sid: self._stm[sid].get("start_time", ""),
        )
        return self._stm[latest_id]
