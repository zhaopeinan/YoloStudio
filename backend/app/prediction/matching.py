from __future__ import annotations

from dataclasses import dataclass

from app.db.models import Annotation


@dataclass(frozen=True)
class Box:
    x_center: float
    y_center: float
    width: float
    height: float


def _corners(box: Box) -> tuple[float, float, float, float]:
    return (
        box.x_center - box.width / 2,
        box.y_center - box.height / 2,
        box.x_center + box.width / 2,
        box.y_center + box.height / 2,
    )


def iou(left: Box, right: Box) -> float:
    left_x1, left_y1, left_x2, left_y2 = _corners(left)
    right_x1, right_y1, right_x2, right_y2 = _corners(right)
    x1 = max(left_x1, right_x1)
    y1 = max(left_y1, right_y1)
    x2 = min(left_x2, right_x2)
    y2 = min(left_y2, right_y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = left.width * left.height
    right_area = right.width * right.height
    union = left_area + right_area - intersection
    return 0.0 if union <= 0 else intersection / union


def annotation_box(annotation: Annotation) -> Box:
    return Box(
        x_center=annotation.x_center,
        y_center=annotation.y_center,
        width=annotation.width,
        height=annotation.height,
    )
