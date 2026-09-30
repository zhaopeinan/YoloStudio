from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# Tolerate float / frontend rounding overshoot when boxes sit on the image edge.
_BBOX_EDGE_SLACK = 1e-3
_MIN_BBOX_SIZE = 1e-6


def _clamp_unit(value: float) -> float:
    return min(1.0, max(0.0, value))


class AnnotationWrite(BaseModel):
    class_id: int
    x_center: float = Field(..., ge=0, le=1)
    y_center: float = Field(..., ge=0, le=1)
    width: float = Field(..., gt=0, le=1)
    height: float = Field(..., gt=0, le=1)
    track_id: str | None = Field(default=None, max_length=120)
    edge_tags: list[str] = Field(default_factory=list)

    @field_validator("edge_tags")
    @classmethod
    def strip_edge_tags(cls, value: list[str]) -> list[str]:
        return [tag.strip() for tag in value if tag.strip()]

    @model_validator(mode="after")
    def bbox_must_stay_in_frame(self) -> "AnnotationWrite":
        left = self.x_center - self.width / 2
        right = self.x_center + self.width / 2
        top = self.y_center - self.height / 2
        bottom = self.y_center + self.height / 2

        if (
            left < -_BBOX_EDGE_SLACK
            or right > 1 + _BBOX_EDGE_SLACK
            or top < -_BBOX_EDGE_SLACK
            or bottom > 1 + _BBOX_EDGE_SLACK
        ):
            if left < -_BBOX_EDGE_SLACK or right > 1 + _BBOX_EDGE_SLACK:
                raise ValueError("Bounding box exceeds image width")
            raise ValueError("Bounding box exceeds image height")

        left = _clamp_unit(left)
        right = _clamp_unit(right)
        top = _clamp_unit(top)
        bottom = _clamp_unit(bottom)
        width = max(right - left, _MIN_BBOX_SIZE)
        height = max(bottom - top, _MIN_BBOX_SIZE)
        # Keep right/bottom inside the frame if min-size push would overshoot.
        if left + width > 1:
            left = max(0.0, 1.0 - width)
        if top + height > 1:
            top = max(0.0, 1.0 - height)

        object.__setattr__(self, "x_center", left + width / 2)
        object.__setattr__(self, "y_center", top + height / 2)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)
        return self


class AnnotationReplace(BaseModel):
    annotations: list[AnnotationWrite]
    annotation_status: Literal["unreviewed", "annotated", "negative"] | None = None


class AnnotationRead(BaseModel):
    id: int
    image_id: int
    class_id: int
    class_name: str
    class_color: str
    x_center: float
    y_center: float
    width: float
    height: float
    track_id: str | None
    edge_tags: list[str]


class AnnotationList(BaseModel):
    items: list[AnnotationRead]
    annotation_status: Literal["unreviewed", "annotated", "negative"]
