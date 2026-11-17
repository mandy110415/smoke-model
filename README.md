# Smoke Candidate Analysis — Grounding DINO + SigLIP 2 + VLM

此版本已完全移除 YOLO `.txt` 預標註檔依賴。

## 新流程

1. 上傳一張原始圖片。
2. Grounding DINO 搜尋可能的煙霧候選框。
3. SigLIP 2 對候選框與 context crop 計算 smoke / cloud / steam / fog / background 語意證據。
4. VLM 結合局部、context、完整影像與 SigLIP 2 證據做驗證。
5. 輸出：煙霧候選、非煙霧淘汰、待確認。

## 不再需要

- `data/labels/*.txt`
- `LABEL_DIR`
- `REQUIRE_LABEL_FILE`
- Annotation IoU matching
- YOLO 預標註檔

## 啟動

Windows CMD：

```cmd
copy .env.example .env
docker compose down --remove-orphans
docker compose build --no-cache
docker compose up -d
docker compose ps
```

Frontend：`http://localhost:8001`

Backend health：`http://localhost:8000/health`

## 原始圖片資料夾

`data/raw_images/` 保留給下一階段建立 SigLIP 2 訓練資料流程使用；目前前端仍可直接上傳單張圖片分析。
