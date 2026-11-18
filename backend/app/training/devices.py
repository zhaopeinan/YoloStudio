from __future__ import annotations


def parse_ultralytics_device(value: str | int | list | None):
    """Normalize UI/API device values for Ultralytics YOLO.train(device=...)."""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, list):
        parsed = [int(item) for item in value]
        if len(parsed) == 1:
            return parsed[0]
        return parsed

    text = str(value).strip()
    if not text:
        return None
    if "," in text:
        parsed = [int(part.strip()) for part in text.split(",") if part.strip() != ""]
        if not parsed:
            return text
        if len(parsed) == 1:
            return parsed[0]
        return parsed
    if text.isdigit():
        return int(text)
    return text
