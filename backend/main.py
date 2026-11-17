import io

from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image

from grounding_service import detect_smoke_candidates
from audit_service import analyze_candidates
from vlm_service import vlm_status
from siglip_service import siglip_status

app = FastAPI(
    title="Smoke Candidate Analysis: Grounding DINO + SigLIP 2 + VLM",
    version="4.0.0",
)


def read_image(data: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image") from exc


@app.get("/")
def root_info():
    return {
        "service": "Smoke Candidate Analysis",
        "pipeline": [
            "Grounding DINO candidate search",
            "SigLIP 2 smoke/cloud hard-negative representation",
            "VLM semantic verification using SigLIP 2 evidence",
        ],
        "annotation_file_required": False,
        "result_classes": [
            "smoke_candidate",
            "non_smoke_rejected",
            "uncertain",
        ],
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "annotation_file_required": False,
        "siglip": siglip_status(),
        "vlm": vlm_status(),
    }


async def _analyze(file: UploadFile):
    image_bytes = await file.read()
    image = read_image(image_bytes)

    grounding = detect_smoke_candidates(image)
    analysis_result = analyze_candidates(
        image=image,
        candidates=grounding["candidates"],
    )

    return {
        "image": {
            "filename": file.filename,
            "width": image.width,
            "height": image.height,
        },
        "grounding_dino": grounding,
        "analysis": analysis_result,
    }


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    return await _analyze(file)


# Backward-compatible alias for the previous frontend/API.
@app.post("/api/audit")
async def audit_alias(file: UploadFile = File(...)):
    return await _analyze(file)
