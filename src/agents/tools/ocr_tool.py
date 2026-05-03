from typing import Any, Dict

from src.agents.tools.base_tool import AgentTool
from src.ocr.hybrid_ocr import HybridOCR


class OCRTool(AgentTool):
    """
    Agent tool wrapper for the HybridOCR Phase 1 component.

    Lazily initialises one HybridOCR instance per engine name so that
    expensive model loads are deferred until first use and then reused
    across calls within the same session.

    Inputs:
        image_path (str): Absolute or relative path to the image to OCR.
        strategy_params (dict, optional): May contain:
            engine (str): One of "auto", "florence", "got", "easyocr", "api".
                          Defaults to "auto".

    Outputs (on success):
        text (str): Extracted text.
        confidence (float): Confidence score in [0, 1].
        method (str): Name of the OCR backend that produced the result.
    """

    def __init__(self):
        super().__init__("ocr_tool")
        self._ocr_instances: Dict[str, HybridOCR] = {}

    def _execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        engine: str = inputs.get("strategy_params", {}).get("engine", "auto")
        image_path: str = inputs["image_path"]

        if engine not in self._ocr_instances:
            self._ocr_instances[engine] = HybridOCR(
                prefer_local=(engine != "api"),
                local_model=engine,
            )

        return self._ocr_instances[engine].extract_text_from_image(image_path)

    def cleanup(self) -> None:
        """Release all loaded OCR model instances and clear the cache."""
        for instance in self._ocr_instances.values():
            instance.cleanup()
        self._ocr_instances.clear()
