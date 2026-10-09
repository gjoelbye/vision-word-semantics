"""How much writing the CRAFT text detector finds in a photograph.

Only detection runs, no recognition. A scan records the box count and the share
of the frame the boxes cover, and :func:`is_excluded` decides from that record
whether a photograph is excluded. The class photographs and the SPEECH-COCO
photographs are scanned this way.
"""
from __future__ import annotations

import numpy as np

#: Detection thresholds, set low so that faint writing is found.
TEXT_THRESHOLD = 0.25
LOW_TEXT = 0.15
LINK_THRESHOLD = 0.15


def reader(gpu: bool = True):
    """The CRAFT detector. Loading is slow, so build it once per scan."""
    import easyocr

    return easyocr.Reader(["en"], gpu=gpu, verbose=False)


def text_density(detector, image) -> dict:
    """``n_boxes``, ``text_area_ratio``, ``width`` and ``height`` for one image,
    from one detection pass without rotations."""
    array = np.asarray(image.convert("RGB")) if hasattr(image, "convert") else image
    height, width = array.shape[:2]
    boxes = detector.detect(array, text_threshold=TEXT_THRESHOLD,
                            low_text=LOW_TEXT, link_threshold=LINK_THRESHOLD)[0]
    # easyocr returns a list holding one list of boxes per image.
    if boxes and isinstance(boxes[0], list):
        boxes = boxes[0]
    covered = 0.0
    for box in boxes or []:
        x0, x1, y0, y1 = box[0], box[1], box[2], box[3]
        covered += max(0, x1 - x0) * max(0, y1 - y0)
    return {"n_boxes": len(boxes or []),
            "text_area_ratio": float(covered / max(1, width * height)),
            "width": int(width), "height": int(height)}


def is_excluded(record: dict) -> bool:
    """Whether a scan record fails the filter: any detection fails it."""
    return record["n_boxes"] > 0 or record["text_area_ratio"] > 0.0
