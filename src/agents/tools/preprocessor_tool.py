from typing import Any, Dict

from src.agents.tools.base_tool import AgentTool
from src.preprocessing.image_processor import ImagePreprocessor


class PreprocessorTool(AgentTool):
    """
    Agent tool wrapper for the ImagePreprocessor Phase 1 component.

    Inputs:
        image_path (str): Absolute or relative path to the source image.

    Outputs (on success):
        preprocessed_path (str): Path to the saved preprocessed PNG.
        original_path (str): Echo of the input image_path.
    """

    def __init__(self):
        super().__init__("preprocessor_tool")

    def _execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        image_path: str = inputs["image_path"]
        _binary_array, output_path = ImagePreprocessor().preprocess(image_path)
        return {
            "preprocessed_path": output_path,
            "original_path": image_path,
        }
