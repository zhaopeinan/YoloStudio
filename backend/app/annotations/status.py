from __future__ import annotations

from typing import Literal

from app.db.models import Image

AnnotationStatus = Literal["unreviewed", "annotated", "negative"]
ANNOTATION_STATUSES = {"unreviewed", "annotated", "negative"}


def get_annotation_status(image: Image) -> AnnotationStatus:
    if image.annotations:
        return "annotated"
    raw_status = (image.metadata_ or {}).get("annotation_status")
    if raw_status in ANNOTATION_STATUSES:
        return raw_status
    return "unreviewed"


def set_annotation_status(image: Image, status: AnnotationStatus) -> None:
    metadata = dict(image.metadata_ or {})
    metadata["annotation_status"] = status
    image.metadata_ = metadata
