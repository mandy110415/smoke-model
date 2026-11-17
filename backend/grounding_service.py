import os
from functools import lru_cache

import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

GROUNDING_MODEL = os.getenv(
    "GROUNDING_MODEL",
    "IDEA-Research/grounding-dino-tiny",
)

GROUNDING_PROMPT = os.getenv(
    "GROUNDING_PROMPT",
    "smoke. smoke plume. white smoke. gray smoke. industrial smoke.",
)

BOX_THRESHOLD = float(
    os.getenv("GROUNDING_BOX_THRESHOLD", "0.25")
)

TEXT_THRESHOLD = float(
    os.getenv("GROUNDING_TEXT_THRESHOLD", "0.20")
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def load_grounding_model():
    processor = AutoProcessor.from_pretrained(
        GROUNDING_MODEL,
    )
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        GROUNDING_MODEL,
    ).to(DEVICE).eval()

    return processor, model


@torch.inference_mode()
def detect_smoke_candidates(image):
    processor, model = load_grounding_model()

    inputs = processor(
        images=image,
        text=GROUNDING_PROMPT,
        return_tensors="pt",
    )

    inputs = {
        k: v.to(DEVICE)
        if hasattr(v, "to")
        else v
        for k, v in inputs.items()
    }

    outputs = model(**inputs)

    target_sizes = torch.tensor(
        [[image.height, image.width]],
        device=DEVICE,
    )

    # Transformers API compatibility
    if hasattr(processor, "post_process_grounded_object_detection"):
        processed = processor.post_process_grounded_object_detection(
            outputs,
            inputs.get("input_ids"),
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            target_sizes=target_sizes,
        )[0]
    else:
        processed = processor.post_process_object_detection(
            outputs,
            threshold=BOX_THRESHOLD,
            target_sizes=target_sizes,
        )[0]

    candidates = []

    boxes = processed.get("boxes", [])
    scores = processed.get("scores", [])
    text_labels = (
        processed.get("text_labels")
        or processed.get("labels")
        or ["smoke"] * len(boxes)
    )

    for box, score, label in zip(
        boxes,
        scores,
        text_labels,
    ):
        if hasattr(box, "detach"):
            box = box.detach().cpu().tolist()
        if hasattr(score, "detach"):
            score = float(score.detach().cpu().item())

        candidates.append({
            "class_name": str(label),
            "confidence": float(score),
            "bbox_xyxy": [float(v) for v in box],
        })

    return {
        "model": GROUNDING_MODEL,
        "prompt": GROUNDING_PROMPT,
        "box_threshold": BOX_THRESHOLD,
        "text_threshold": TEXT_THRESHOLD,
        "candidates": candidates,
    }
