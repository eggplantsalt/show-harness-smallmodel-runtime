"""The single image and pixel-coordinate convention used by Runtime V3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class CanonicalImageAdapter:
    """Convert raw LIBERO OpenGL frames into canonical top-left-origin images.

    LIBERO's current robosuite configuration emits OpenGL-convention frames. The
    vertical row flip here is the one spatial transform applied before Runtime V3
    image consumers; points and half-open boxes use this same coordinate mapping.
    """

    orientation: str = "vertical_flip"

    def __post_init__(self) -> None:
        if self.orientation not in {"identity", "vertical_flip", "horizontal_flip", "rotate_180"}:
            raise ValueError(f"unsupported canonical orientation: {self.orientation!r}")

    def transform_image(self, image: np.ndarray) -> np.ndarray:
        array = np.asarray(image)
        if array.ndim < 2:
            raise ValueError("image must have at least two dimensions")
        return np.ascontiguousarray(self._transform_raster(array))

    def transform_mask(self, mask: np.ndarray) -> np.ndarray:
        array = np.asarray(mask)
        if array.ndim != 2:
            raise ValueError("mask must be a two-dimensional raster")
        return np.ascontiguousarray(self._transform_raster(array))

    def transform_point(
        self, point_xy: Sequence[float], *, width: int, height: int
    ) -> tuple[float, float]:
        x, y = (float(value) for value in point_xy)
        width, height = int(width), int(height)
        if self.orientation == "vertical_flip":
            y = height - 1 - y
        elif self.orientation == "horizontal_flip":
            x = width - 1 - x
        elif self.orientation == "rotate_180":
            x, y = width - 1 - x, height - 1 - y
        return float(x), float(y)

    def transform_projected_point(
        self, point_xy: Sequence[float], *, width: int, height: int
    ) -> tuple[float, float]:
        """Map camera_geometry's row-down projection through raw OpenGL pixels.

        ``project_point`` computes image rows in the conventional top-left / OpenCV
        frame. LIBERO's OpenGL observation retains bottom-left row order, so first
        convert that projected value to the raw raster, then apply this adapter's
        canonical transform. This avoids applying a vertical flip twice.
        """
        x, y = (float(value) for value in point_xy)
        raw_opengl_y = int(height) - 1 - y
        return self.transform_point((x, raw_opengl_y), width=width, height=height)

    def transform_bbox(
        self, bbox_xyxy: Sequence[float], *, width: int, height: int
    ) -> tuple[float, float, float, float]:
        """Transform an ``[x0, y0, x1, y1)`` half-open pixel box."""
        x0, y0, x1, y1 = (float(value) for value in bbox_xyxy)
        width, height = int(width), int(height)
        if self.orientation == "vertical_flip":
            y0, y1 = height - y1, height - y0
        elif self.orientation == "horizontal_flip":
            x0, x1 = width - x1, width - x0
        elif self.orientation == "rotate_180":
            x0, x1 = width - x1, width - x0
            y0, y1 = height - y1, height - y0
        return float(x0), float(y0), float(x1), float(y1)

    def _transform_raster(self, array: np.ndarray) -> np.ndarray:
        if self.orientation == "vertical_flip":
            return np.flipud(array)
        if self.orientation == "horizontal_flip":
            return np.fliplr(array)
        if self.orientation == "rotate_180":
            return np.rot90(array, 2)
        return array
