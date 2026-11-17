from PIL import Image

from siglip_service import analyze_region
from vlm_service import verify_region


def crop_region(image: Image.Image, box, padding=0.08):
    x1, y1, x2, y2 = [float(v) for v in box]

    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    px = w * padding
    py = h * padding

    x1 = max(0, int(round(x1 - px)))
    y1 = max(0, int(round(y1 - py)))
    x2 = min(image.width, int(round(x2 + px)))
    y2 = min(image.height, int(round(y2 + py)))

    if x2 <= x1 or y2 <= y1:
        return None

    return image.crop((x1, y1, x2, y2))


def verify(image, box):
    crop = crop_region(image, box, padding=0.08)
    context_crop = crop_region(image, box, padding=1.00)

    if crop is None or context_crop is None:
        return {
            "label": "UNCERTAIN",
            "reason": "invalid_crop",
            "source": "grounding_candidate",
            "siglip_evidence": None,
        }

    siglip_evidence = analyze_region(
        crop_image=crop,
        context_image=context_crop,
    )

    return verify_region(
        crop_image=crop,
        context_image=context_crop,
        full_image=image,
        source="grounding_candidate",
        siglip_evidence=siglip_evidence,
    )


def analyze_candidates(image, candidates):
    smoke_candidates = []
    rejected_candidates = []
    uncertain = []

    for cand in candidates:
        verification = verify(
            image=image,
            box=cand["bbox_xyxy"],
        )

        item = {
            "candidate": cand,
            "siglip_evidence": verification.get("siglip_evidence"),
            "vlm_verification": verification,
        }

        label = verification.get("label", "UNCERTAIN").upper()

        if label == "SMOKE":
            item["status"] = "smoke_candidate"
            item["reason"] = (
                "Grounding DINO 找到候選區域，且 SigLIP 2 語意證據與 VLM 驗證支持為煙霧。"
            )
            smoke_candidates.append(item)

        elif label == "NON_SMOKE":
            item["status"] = "non_smoke_rejected"
            item["reason"] = (
                "Grounding DINO 找到候選區域，但 SigLIP 2 + VLM 判定為 non-smoke，故淘汰。"
            )
            rejected_candidates.append(item)

        else:
            item["status"] = "uncertain"
            item["reason"] = (
                "Grounding DINO 找到候選區域，但 SigLIP 2 + VLM 仍無法可靠區分 smoke / non-smoke。"
            )
            uncertain.append(item)

    return {
        "summary": {
            "grounding_candidate_count": len(candidates),
            "smoke_candidate_count": len(smoke_candidates),
            "non_smoke_rejected_count": len(rejected_candidates),
            "uncertain_count": len(uncertain),
        },
        "smoke_candidate": smoke_candidates,
        "non_smoke_rejected": rejected_candidates,
        "uncertain": uncertain,
    }
