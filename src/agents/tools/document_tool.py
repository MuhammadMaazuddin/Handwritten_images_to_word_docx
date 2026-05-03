from typing import Any, Dict, List, Optional

from src.agents.tools.base_tool import AgentTool
from src.document_generation.word_generator import WordGenerator


class DocumentTool(AgentTool):
    """
    Agent tool wrapper for the WordGenerator Phase 1 component.

    Inputs:
        content (str): Text content to write into the document.
        title (str, optional): Document title. Defaults to "Converted Notes".
        diagrams (list, optional): List of diagram dicts (path, bbox, center_y).
                                   Passed through to WordGenerator unchanged.

    Outputs (on success):
        output_path (str): Filesystem path of the generated .docx file.
    """

    def __init__(self):
        super().__init__("document_tool")

    def _execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        content: str = inputs["content"]
        title: str = inputs.get("title", "Converted Notes")
        diagrams: Optional[List[Dict]] = inputs.get("diagrams", None)

        output_path: str = WordGenerator().create_document(
            content, title, diagrams=diagrams
        )
        return {"output_path": output_path}
