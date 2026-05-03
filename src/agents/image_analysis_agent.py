"""
Image Analysis Agent for the Agentic OCR Pipeline.

Analyses an input image using OpenCV to extract characteristics such as
resolution, noise level, contrast, and diagram presence, then classifies
the image type and recommends preprocessing enhancements.

Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 17.2
"""

import time
from typing import Any, Dict

import cv2
import numpy as np

from src.agents.base_agent import BaseAgent


class ImageAnalysisAgent(BaseAgent):
    """
    Agent that analyses an image and produces an ImageAnalysisReport.

    Perceive phase: loads the image, extracts characteristics, classifies
    the image type, and recommends enhancements.  All characteristic
    extraction is guarded so that a single failure records "undetermined"
    rather than aborting the whole analysis (Req 2.5).
    """

    IMAGE_TYPES = [
        "dense_handwriting",
        "mixed_text_diagram",
        "formula_heavy",
        "low_quality",
        "printed_text",
    ]
    LOW_QUALITY_DPI_THRESHOLD = 150

    def __init__(self, memory_manager: Any) -> None:
        super().__init__("image_analysis", memory_manager)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyse the image at inputs["image_path"] and return an analysis report.

        Args:
            inputs: {"image_path": str}

        Returns:
            {
                "status": "success",
                "output": ImageAnalysisReport dict,
                "execution_time_seconds": float,
            }
            or the standard error dict from handle_error() on failure.
        """
        start_time = time.time()
        try:
            image_path: str = inputs["image_path"]

            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")

            # --- Extract characteristics, catching individual failures ---
            try:
                resolution = self._estimate_resolution(img)
            except Exception as exc:
                self.logger.warning("_estimate_resolution failed: %s", exc)
                resolution = "undetermined"

            try:
                noise_level = self._estimate_noise_level(img)
            except Exception as exc:
                self.logger.warning("_estimate_noise_level failed: %s", exc)
                noise_level = "undetermined"

            try:
                contrast_ratio = self._estimate_contrast(img)
            except Exception as exc:
                self.logger.warning("_estimate_contrast failed: %s", exc)
                contrast_ratio = "undetermined"

            try:
                has_diagrams = self._detect_diagrams_presence(img)
            except Exception as exc:
                self.logger.warning("_detect_diagrams_presence failed: %s", exc)
                has_diagrams = "undetermined"

            # --- Classify image type ---
            try:
                image_type = self._classify_image_type(
                    img, resolution, noise_level, has_diagrams
                )
            except Exception as exc:
                self.logger.warning("_classify_image_type failed: %s", exc)
                image_type = "undetermined"

            # --- Recommend enhancements ---
            try:
                recommended_enhancements = self._recommend_enhancements(
                    resolution, noise_level, contrast_ratio
                )
            except Exception as exc:
                self.logger.warning("_recommend_enhancements failed: %s", exc)
                recommended_enhancements = []

            # --- Build report ---
            height, width = img.shape[:2]
            analysis_report: Dict[str, Any] = {
                "image_type": image_type,
                "resolution_dpi": resolution,
                "noise_level": noise_level,
                "contrast_ratio": contrast_ratio,
                "has_diagrams": has_diagrams,
                "image_dimensions": {"width": width, "height": height},
                "recommended_enhancements": recommended_enhancements,
            }

            # --- Record to memory ---
            self.memory_manager.record_action(
                {
                    "agent": self.name,
                    "action": "image_analysis",
                    "result": analysis_report,
                }
            )

            execution_time = time.time() - start_time
            return {
                "status": "success",
                "output": analysis_report,
                "execution_time_seconds": float(execution_time),
            }

        except Exception as e:
            return self.handle_error(e, inputs)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _estimate_resolution(self, img: np.ndarray) -> float:
        """
        Estimate DPI using an A4-page heuristic.

        Assumes the image represents an A4 page (8.27 × 11.69 inches) and
        derives DPI from the pixel dimensions.

        Args:
            img: BGR image array.

        Returns:
            Estimated DPI as a float.
        """
        height, width = img.shape[:2]
        dpi_width = width / 8.27
        dpi_height = height / 11.69
        return (dpi_width + dpi_height) / 2

    def _estimate_noise_level(self, img: np.ndarray) -> float:
        """
        Estimate noise level using Laplacian variance, normalised to [0, 1].

        A higher value indicates more high-frequency content (noise or fine
        detail); a lower value indicates a smoother (potentially blurry) image.

        Args:
            img: BGR or grayscale image array.

        Returns:
            Normalised noise level in [0.0, 1.0].
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return min(1.0, laplacian_var / 1000.0)

    def _estimate_contrast(self, img: np.ndarray) -> float:
        """
        Estimate contrast as the max/min pixel ratio.

        Returns float('inf') when the minimum pixel value is 0 (pure black
        present), indicating very high contrast.

        Args:
            img: BGR or grayscale image array.

        Returns:
            Contrast ratio (max_val / min_val), or float('inf') if min == 0.
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        min_val = np.min(gray)
        max_val = np.max(gray)

        if min_val == 0:
            return float("inf")
        return float(max_val) / float(min_val)

    def _detect_diagrams_presence(self, img: np.ndarray) -> bool:
        """
        Detect whether the image likely contains diagrams using Hough line detection.

        Uses Canny edge detection followed by probabilistic Hough transform.
        More than 10 detected line segments is treated as evidence of diagrams.

        Args:
            img: BGR or grayscale image array.

        Returns:
            True if more than 10 line segments are detected, False otherwise.
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=50,
            minLineLength=30,
            maxLineGap=10,
        )
        return lines is not None and len(lines) > 10

    def _classify_image_type(
        self,
        img: np.ndarray,
        resolution: Any,
        noise_level: Any,
        has_diagrams: Any,
    ) -> str:
        """
        Classify the image into one of IMAGE_TYPES using a priority-ordered ruleset.

        Priority order:
          1. "undetermined" characteristics → "low_quality"
          2. Low DPI or high noise → "low_quality"
          3. Low noise and high DPI → "printed_text"
          4. Diagrams detected → "mixed_text_diagram"
          5. Default → "dense_handwriting"

        Args:
            img: BGR image array (unused directly; kept for interface consistency).
            resolution: Estimated DPI or "undetermined".
            noise_level: Normalised noise level or "undetermined".
            has_diagrams: Boolean or "undetermined".

        Returns:
            One of the IMAGE_TYPES strings.
        """
        # 1. Undetermined characteristics → low quality
        if resolution == "undetermined" or noise_level == "undetermined":
            return "low_quality"

        # 2. Low DPI or high noise → low quality
        if resolution < self.LOW_QUALITY_DPI_THRESHOLD or noise_level > 0.7:
            return "low_quality"

        # 3. Low noise and high DPI → printed text
        if noise_level < 0.2 and resolution > 200:
            return "printed_text"

        # 4. Diagrams detected → mixed text/diagram
        if has_diagrams is True:
            return "mixed_text_diagram"

        # 5. Default
        return "dense_handwriting"

    def _recommend_enhancements(
        self,
        resolution: Any,
        noise_level: Any,
        contrast_ratio: Any,
    ) -> list:
        """
        Build a list of actionable enhancement recommendations.

        Args:
            resolution: Estimated DPI or "undetermined".
            noise_level: Normalised noise level or "undetermined".
            contrast_ratio: Contrast ratio, float('inf'), or "undetermined".

        Returns:
            List of recommendation strings (may be empty).
        """
        recommendations = []

        if resolution != "undetermined" and resolution < self.LOW_QUALITY_DPI_THRESHOLD:
            recommendations.append(
                "Scan at higher resolution (≥300 DPI recommended)"
            )

        if noise_level != "undetermined" and noise_level > 0.6:
            recommendations.append("Apply denoising preprocessing")

        if (
            contrast_ratio != "undetermined"
            and contrast_ratio != float("inf")
            and contrast_ratio < 2.0
        ):
            recommendations.append("Increase brightness and contrast")

        return recommendations
