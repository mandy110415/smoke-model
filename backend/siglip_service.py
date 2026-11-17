import os
from functools import lru_cache
from typing import Dict, List

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

SIGLIP_MODEL = os.getenv(
    "SIGLIP_MODEL",
    "google/siglip2-base-patch16-224",
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Multiple text prototypes are used for each semantic class instead of a
# single short label. This makes the evidence more robust for smoke/cloud
# hard negatives in industrial scenes.
PROTOTYPES: Dict[str, List[str]] = {
    "smoke": [
        "This is a photo of smoke emitted from an industrial chimney.",
        "This is a photo of a plume of industrial smoke rising from an exhaust stack.",
        "This is a photo of gray smoke connected to an emission source.",
        "This is a photo of white smoke originating from a factory pipe.",
        "This is a photo of a diffuse smoke plume spreading from industrial equipment.",
    ],
    "cloud": [
        "This is a photo of a natural cloud floating in the sky.",
        "This is a photo of a white cloud not connected to an industrial source.",
        "This is a photo of a large cloud formation in the background sky.",
        "This is a photo of thin clouds spread horizontally across the sky.",
        "This is a photo of a bright cloud above an industrial area.",
    ],
    "steam": [
        "This is a photo of steam emitted from industrial equipment.",
        "This is a photo of white water vapor near a pipe or vent.",
        "This is a photo of condensed steam in an industrial scene.",
    ],
    "fog": [
        "This is a photo of fog or atmospheric haze.",
        "This is a photo of diffuse mist covering the background.",
        "This is a photo of hazy air without a clear emission source.",
    ],
    "background": [
        "This is a photo of sky, building, roof, equipment, or background without smoke.",
        "This is a photo of industrial structures without visible smoke.",
        "This is a photo of a roof or building edge against the sky.",
    ],
}


@lru_cache(maxsize=1)
def load_siglip():
    processor = AutoProcessor.from_pretrained(SIGLIP_MODEL)

    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    model = AutoModel.from_pretrained(
        SIGLIP_MODEL,
        torch_dtype=dtype,
    ).to(DEVICE).eval()

    return processor, model


def siglip_status():
    return {
        "model": SIGLIP_MODEL,
        "device": DEVICE,
        "classes": list(PROTOTYPES.keys()),
    }


def _normalize(features: torch.Tensor) -> torch.Tensor:
    return features / features.norm(dim=-1, keepdim=True).clamp(min=1e-12)


def _encode_text_prototypes(processor, model):
    labels = []
    texts = []

    for label, prompts in PROTOTYPES.items():
        for prompt in prompts:
            labels.append(label)
            texts.append(prompt)

    text_inputs = processor(
        text=texts,
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )
    text_inputs = {
        k: v.to(DEVICE) if hasattr(v, "to") else v
        for k, v in text_inputs.items()
    }

    with torch.inference_mode():
        text_features = model.get_text_features(**text_inputs)

    return labels, _normalize(text_features.float())


@lru_cache(maxsize=1)
def _cached_text_features():
    processor, model = load_siglip()
    return _encode_text_prototypes(processor, model)


def _score_image(image: Image.Image):
    processor, model = load_siglip()
    labels, text_features = _cached_text_features()

    image_inputs = processor(
        images=image,
        return_tensors="pt",
    )
    image_inputs = {
        k: v.to(DEVICE) if hasattr(v, "to") else v
        for k, v in image_inputs.items()
    }

    with torch.inference_mode():
        image_features = model.get_image_features(**image_inputs)

    image_features = _normalize(image_features.float())
    similarities = (image_features @ text_features.T)[0]

    per_class = {}
    for label in PROTOTYPES:
        idxs = [i for i, x in enumerate(labels) if x == label]
        vals = similarities[idxs]
        per_class[label] = float(vals.mean().item())

    return per_class


def analyze_region(
    crop_image: Image.Image,
    context_image: Image.Image,
):
    """
    Produce contrastive semantic evidence from two visual scales:
    - region: the exact candidate / annotation crop
    - context: an expanded crop around the region

    SigLIP 2 is evidence only. It does not make the final audit decision.
    """
    try:
        region_scores = _score_image(crop_image)
        context_scores = _score_image(context_image)

        # Context is slightly more important because smoke/cloud distinction
        # often depends on whether a plume is connected to an emission source.
        fused_scores = {
            label: 0.4 * region_scores[label] + 0.6 * context_scores[label]
            for label in PROTOTYPES
        }

        smoke = fused_scores["smoke"]
        cloud = fused_scores["cloud"]
        strongest_negative_label = max(
            (k for k in fused_scores if k != "smoke"),
            key=lambda k: fused_scores[k],
        )
        strongest_negative_score = fused_scores[strongest_negative_label]

        return {
            "available": True,
            "model": SIGLIP_MODEL,
            "region_scores": region_scores,
            "context_scores": context_scores,
            "fused_scores": fused_scores,
            "smoke_cloud_margin": smoke - cloud,
            "smoke_vs_best_negative_margin": smoke - strongest_negative_score,
            "strongest_negative": {
                "label": strongest_negative_label,
                "score": strongest_negative_score,
            },
        }

    except Exception as exc:
        return {
            "available": False,
            "model": SIGLIP_MODEL,
            "error": f"siglip_error: {exc}",
        }
