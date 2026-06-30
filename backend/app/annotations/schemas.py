from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator


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
        half_width = self.width / 2
        half_height = self.height / 2
        if self.x_center - half_width < 0 or self.x_center + half_width > 1:
            raise ValueError("Bounding box exceeds image width")
        if self.y_center - half_height < 0 or self.y_center + half_height > 1:
            raise ValueError("Bounding box exceeds image height")
        return self


class AnnotationReplace(BaseModel):
    annotations: list[AnnotationWrite]


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
