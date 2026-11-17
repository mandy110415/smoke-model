import json
import os
import re
from functools import lru_cache

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

VLM_MODEL = os.getenv(
    "VLM_MODEL",
    "HuggingFaceTB/SmolVLM-500M-Instruct",
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_NEW_TOKENS = int(
    os.getenv("VLM_MAX_NEW_TOKENS", "120")
)


@lru_cache(maxsize=1)
def load_vlm():
    processor = AutoProcessor.from_pretrained(
        VLM_MODEL,
        trust_remote_code=True,
    )

    dtype = (
        torch.float16
        if DEVICE == "cuda"
        else torch.float32
    )

    model = AutoModelForVision2Seq.from_pretrained(
        VLM_MODEL,
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    model = model.to(DEVICE).eval()
    return processor, model


def vlm_status():
    return {
        "model": VLM_MODEL,
        "device": DEVICE,
    }


def _parse_label(text: str):
    upper = text.upper()

    match = re.search(
        r"LABEL\s*:\s*(SMOKE|NON_SMOKE|UNCERTAIN)",
        upper,
    )
    if match:
        return match.group(1)

    if "NON_SMOKE" in upper or "NON-SMOKE" in upper:
        return "NON_SMOKE"
    if re.search(r"\bSMOKE\b", upper):
        return "SMOKE"

    return "UNCERTAIN"


def _run_vlm(images, prompt):
    processor, model = load_vlm()

    messages = [{
        "role": "user",
        "content": (
            [{"type": "image"} for _ in images]
            + [{"type": "text", "text": prompt}]
        ),
    }]

    text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=text,
        images=images,
        return_tensors="pt",
    )

    inputs = {
        k: v.to(DEVICE)
        if hasattr(v, "to")
        else v
        for k, v in inputs.items()
    }

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    generated = output[:, inputs["input_ids"].shape[1]:]
    answer = processor.batch_decode(
        generated,
        skip_special_tokens=True,
    )[0].strip()

    return answer


def _siglip_prompt_block(siglip_evidence):
    if not siglip_evidence or not siglip_evidence.get("available"):
        return (
            "SigLIP 2 semantic evidence is unavailable for this region. "
            "Rely on visual evidence and be conservative."
        )

    fused = siglip_evidence.get("fused_scores", {})
    region = siglip_evidence.get("region_scores", {})
    context = siglip_evidence.get("context_scores", {})

    # Compact JSON makes the evidence easy for the VLM to read while keeping
    # it explicitly non-authoritative.
    compact = {
        "region": {k: round(v, 4) for k, v in region.items()},
        "context": {k: round(v, 4) for k, v in context.items()},
        "fused": {k: round(v, 4) for k, v in fused.items()},
        "smoke_cloud_margin": round(
            siglip_evidence.get("smoke_cloud_margin", 0.0), 4
        ),
        "smoke_vs_best_negative_margin": round(
            siglip_evidence.get("smoke_vs_best_negative_margin", 0.0), 4
        ),
        "strongest_negative": siglip_evidence.get("strongest_negative"),
    }

    return (
        "SigLIP 2 contrastive semantic evidence (supporting evidence only; "
        "do NOT blindly follow the scores):\n"
        + json.dumps(compact, ensure_ascii=False)
    )


def verify_region(
    crop_image: Image.Image,
    context_image: Image.Image,
    full_image: Image.Image,
    source: str,
    siglip_evidence=None,
):
    """
    Final semantic verification.

    Inputs:
    - exact candidate crop
    - expanded context crop
    - full image
    - SigLIP 2 contrastive evidence

    SigLIP 2 assists the VLM but never directly determines the audit label.
    """

    if source == "existing_annotation":
        context = (
            "This bounding box currently has the pseudo-label 'smoke'. "
            "Do NOT assume the pseudo-label is correct. Your task is to verify "
            "whether THIS BOX ITSELF is a valid smoke annotation."
        )
    else:
        context = (
            "This bounding box was proposed by Grounding DINO as a possible "
            "smoke candidate. Do NOT assume the proposal is correct."
        )

    siglip_block = _siglip_prompt_block(siglip_evidence)

    prompt = f"""
You are verifying smoke pseudo-labels in an industrial outdoor vision dataset.

{context}

You are given three images:
Image 1 = the exact candidate/annotation crop.
Image 2 = an expanded context crop around the candidate (approximately 3x region scale when possible).
Image 3 = the original full image.

{siglip_block}

Use BOTH visual reasoning and the SigLIP 2 evidence.
The SigLIP 2 evidence is auxiliary; if it conflicts with clear visual context, trust the visual context.

STRICT smoke-vs-cloud verification rules:
1. SMOKE requires visible smoke-like material inside the proposed box.
2. Prefer SMOKE when the material has a plausible spatial connection to a chimney, exhaust pipe, vent, fire, or industrial emission source and forms a plume/dispersion pattern.
3. Prefer NON_SMOKE when the region is mainly natural cloud, sky, roof, building, equipment, reflection, fog, haze, or background.
4. A cloud floating in the sky without a plausible connection to an emission source is NON_SMOKE even if it is white/gray and visually similar to smoke.
5. For an existing annotation, do NOT answer SMOKE merely because there is smoke somewhere nearby in the full image. The proposed bounding box itself must meaningfully cover smoke.
6. Use UNCERTAIN only when the evidence is genuinely insufficient after considering the exact crop, expanded context, full image, and SigLIP 2 evidence.

Return exactly one label:
LABEL: SMOKE
or
LABEL: NON_SMOKE
or
LABEL: UNCERTAIN

Then one short line:
REASON: <brief visual reason, mention source connection or cloud/background evidence when relevant>
""".strip()

    try:
        answer = _run_vlm(
            [crop_image, context_image, full_image],
            prompt,
        )

        label = _parse_label(answer)

        return {
            "label": label,
            "raw_answer": answer,
            "model": VLM_MODEL,
            "source": source,
            "siglip_evidence": siglip_evidence,
        }

    except Exception as exc:
        return {
            "label": "UNCERTAIN",
            "raw_answer": None,
            "reason": f"vlm_error: {exc}",
            "model": VLM_MODEL,
            "source": source,
            "siglip_evidence": siglip_evidence,
        }
