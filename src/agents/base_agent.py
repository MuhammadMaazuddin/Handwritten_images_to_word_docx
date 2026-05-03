from abc import ABC, abstractmethod
from typing import Dict, Any
from datetime import datetime
import logging


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the agentic system.

    Provides common functionality:
    - Logging
    - Error handling
    - Memory access
    - Decision recording
    """

    def __init__(self, name: str, memory_manager: 'MemoryManager'):
        self.name = name
        self.memory_manager = memory_manager
        self.logger = logging.getLogger(f"agent.{name}")

    @abstractmethod
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main execution method that each agent must implement.

        Args:
            inputs: Agent-specific input dictionary

        Returns:
            Agent-specific output dictionary with at minimum:
            - status: "success" or "error"
            - output: The agent's result
            - execution_time_seconds: float
        """
        pass

    def log_decision(self, decision_type: str, inputs: Dict[str, Any],
                     output: Dict[str, Any], rationale: str) -> None:
        """Record a decision to the decision log via memory manager."""
        decision_entry = {
            "agent": self.name,
            "decision_type": decision_type,
            "timestamp": datetime.now().isoformat(),
            "inputs": inputs,
            "output": output,
            "rationale": rationale
        }
        self.memory_manager.record_decision(decision_entry)

    def handle_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Standard error handling for all agents.

        Returns error result dict instead of raising.
        """
        self.logger.error(f"{self.name} error: {error}", exc_info=True)
        return {
            "status": "error",
            "error_message": str(error),
            "error_type": type(error).__name__,
            "context": context,
            "execution_time_seconds": 0.0
        }
