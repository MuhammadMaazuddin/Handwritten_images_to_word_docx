from typing import Any, Dict

from src.agents.tools.base_tool import AgentTool
from src.preprocessing.diagram_detector import DiagramDetector


class DiagramTool(AgentTool):
    """
    Agent tool wrapper for the DiagramDetector Phase 1 component.

    Inputs:
        image_path (str): Absolute or relative path to the source image.

    Outputs (on success):
        has_diagrams (bool): Whether any diagram regions were detected.
        diagram_regions (list): List of detected diagram region dicts.
        text_only_image (str): Path to the image with diagrams masked out.
        original_image (str): Echo of the input image_path.
    """

    def __init__(self):
        super().__init__("diagram_tool")

    def _execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        image_path: str = inputs["image_path"]
        return DiagramDetector().detect_and_extract(image_path)
