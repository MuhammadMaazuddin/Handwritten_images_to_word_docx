from abc import ABC, abstractmethod
from typing import Dict, Any
import time
import logging


class AgentTool(ABC):
    """
    Abstract base class for all agent tools.

    All tools expose a uniform invoke(inputs: dict) -> dict interface.
    Tools wrap Phase 1 components and handle errors gracefully.
    """

    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"tool.{name}")

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invoke the tool with the given inputs.

        Catches ALL exceptions and returns a structured dict.

        Returns:
            {
                "status": "success" | "error",
                "output": Any,
                "execution_time_seconds": float,
                "error_message": str  (only on error)
            }
        """
        start_time = time.time()

        try:
            result = self._execute(inputs)
            execution_time = time.time() - start_time

            return {
                "status": "success",
                "output": result,
                "execution_time_seconds": execution_time
            }

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Tool {self.name} error: {e}", exc_info=True)

            return {
                "status": "error",
                "output": None,
                "execution_time_seconds": execution_time,
                "error_message": str(e),
                "error_type": type(e).__name__
            }

    @abstractmethod
    def _execute(self, inputs: Dict[str, Any]) -> Any:
        """
        Internal execution method. Subclasses implement this.
        May raise exceptions — they will be caught by invoke().
        """
        pass
