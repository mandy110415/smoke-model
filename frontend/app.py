import io
import os

import requests
import streamlit as st
from PIL import Image, ImageDraw

BACKEND = os.getenv("BACKEND_URL", "http://backend:8000")

st.set_page_config(
    page_title="Smoke Candidate Analysis",
    layout="wide",
)

st.title("煙霧候選區域分析")
st.caption(
    "Grounding DINO 候選搜尋 + SigLIP 2 語意表徵 + VLM 驗證"
)

with st.sidebar:
    st.subheader("系統狀態")
    try:
        health = requests.get(f"{BACKEND}/health", timeout=30).json()
        st.success("Backend 已連線")
        st.json(health)
    except Exception as exc:
        st.error("Backend 尚未連線")
        st.caption(str(exc))

image_file = st.file_uploader(
    "上傳原始圖片",
    type=["jpg", "jpeg", "png"],
)

if image_file is not None:
    st.image(
        image_file,
        caption="待分析影像",
        use_container_width=True,
    )

if st.button(
    "執行分析",
    type="primary",
    disabled=image_file is None,
):
    with st.spinner(
        "Grounding DINO 搜尋候選、SigLIP 2 分析 smoke/cloud，再進行 VLM 驗證..."
    ):
        response = requests.post(
            f"{BACKEND}/api/analyze",
            files={
                "file": (
                    image_file.name,
                    image_file.getvalue(),
                    image_file.type,
                ),
            },
            timeout=1800,
        )

    if not response.ok:
        st.error(f"Backend {response.status_code}: {response.text}")
        st.stop()

    result = response.json()
    analysis = result["analysis"]
    summary = analysis["summary"]

    cols = st.columns(4)
    cols[0].metric("Grounding 候選", summary["grounding_candidate_count"])
    cols[1].metric("煙霧候選", summary["smoke_candidate_count"])
    cols[2].metric("非煙霧淘汰", summary["non_smoke_rejected_count"])
    cols[3].metric("待確認", summary["uncertain_count"])

    image = Image.open(io.BytesIO(image_file.getvalue())).convert("RGB")
    draw = ImageDraw.Draw(image)

    def draw_box(box, text, color):
        box = [int(round(v)) for v in box]
        draw.rectangle(box, outline=color, width=5)
        draw.text((box[0], max(0, box[1] - 20)), text, fill=color)

    for item in analysis["smoke_candidate"]:
        draw_box(
            item["candidate"]["bbox_xyxy"],
            "Smoke candidate",
            "red",
        )

    for item in analysis["uncertain"]:
        draw_box(
            item["candidate"]["bbox_xyxy"],
            "Uncertain",
            "yellow",
        )

    st.subheader("分析結果")
    st.image(image, use_container_width=True)

    with st.expander(f"煙霧候選 ({summary['smoke_candidate_count']})"):
        st.json(analysis["smoke_candidate"])

    with st.expander(f"非煙霧淘汰 ({summary['non_smoke_rejected_count']})"):
        st.json(analysis["non_smoke_rejected"])

    with st.expander(f"待確認 ({summary['uncertain_count']})"):
        st.json(analysis["uncertain"])

    with st.expander("SigLIP 2 語意證據說明"):
        st.markdown(
            "每個 Grounding DINO 候選框會計算 smoke / cloud / steam / fog / background "
            "的語意相似度，再交由 VLM 結合局部與全圖上下文做最終判斷。"
        )

    with st.expander("完整 API response"):
        st.json(result)
