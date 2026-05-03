from typing import Any, Dict

from src.agents.tools.base_tool import AgentTool
from src.postprocessing.llm_corrector import LLMCorrector


class CorrectorTool(AgentTool):
    """
    Agent tool wrapper for the LLMCorrector Phase 1 component.

    Calls correct_text() followed by structure_content() on the same
    LLMCorrector instance to avoid redundant initialisation.

    Inputs:
        text (str): Raw OCR text to be corrected and structured.

    Outputs (on success):
        corrected_text (str): Spelling/OCR-error-corrected text.
        structured_text (str): Markdown-structured version of the corrected text.
    """

    def __init__(self):
        super().__init__("corrector_tool")

    def _execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        corrector = LLMCorrector()
        corrected_text: str = corrector.correct_text(inputs["text"])
        structured_text: str = corrector.structure_content(corrected_text)
        return {
            "corrected_text": corrected_text,
            "structured_text": structured_text,
        }
