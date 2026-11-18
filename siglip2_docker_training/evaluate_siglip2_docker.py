from pathlib import Path
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoModel, AutoProcessor
from tqdm import tqdm

MODEL_DIR = Path("/app/output/best_model")
TEST_DIR = Path("/app/data/test")

CLASS_NAMES = [
    "black_smoke",
    "steam",
    "cloud",
    "normal_background",
]

PROMPTS = {
    "black_smoke": [
        "this is a photo of black industrial smoke",
        "this is a photo of dark smoke from an industrial source",
        "this is a photo of a black smoke plume from a factory",
        "this is a photo of dense dark smoke in an industrial scene",
    ],
    "steam": [
        "this is a photo of white industrial smoke",
        "this is a photo of a white smoke plume from an industrial source",
        "this is a photo of light-colored smoke from a factory exhaust",
        "this is a photo of white smoke spreading in an industrial scene",
    ],
    "cloud": [
        "this is a photo of a natural cloud in the sky",
        "this is a photo of white clouds in the background sky",
        "this is a photo of a cloud formation not connected to an emission source",
        "this is a photo of clouds spreading across the sky",
    ],
    "normal_background": [
        "this is a photo of an industrial background without smoke",
        "this is a photo of factory buildings without smoke",
        "this is a photo of a normal industrial scene",
        "this is a photo of an industrial background with no smoke plume",
    ],
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class FolderDataset(Dataset):
    def __init__(self, root):
        self.samples = []
        self.class_to_idx = {n: i for i, n in enumerate(CLASS_NAMES)}
        for name in CLASS_NAMES:
            folder = root / name
            if not folder.exists():
                raise FileNotFoundError(f"Missing folder: {folder}")
            for p in sorted(folder.iterdir()):
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                    self.samples.append((p, self.class_to_idx[name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return Image.open(path).convert("RGB"), label, str(path)


def collate_fn(batch, processor):
    images, labels, paths = zip(*batch)
    inputs = processor(images=list(images), return_tensors="pt")
    return inputs, torch.tensor(labels), paths


@torch.no_grad()
def build_text_prototypes(model, processor, device):
    protos = []
    for name in CLASS_NAMES:
        inputs = processor(
            text=[x.lower() for x in PROMPTS[name]],
            padding="max_length",
            max_length=64,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        f = model.get_text_features(**inputs)
        f = F.normalize(f, dim=-1)
        protos.append(F.normalize(f.mean(dim=0), dim=-1))
    return torch.stack(protos)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    model = AutoModel.from_pretrained(MODEL_DIR).to(device).eval()

    ds = FolderDataset(TEST_DIR)
    loader = DataLoader(
        ds,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: collate_fn(b, processor),
    )

    prototypes = build_text_prototypes(model, processor, device)
    confusion = torch.zeros(len(CLASS_NAMES), len(CLASS_NAMES), dtype=torch.int64)

    rows = []
    correct = 0
    total = 0

    for image_inputs, labels, paths in tqdm(loader):
        image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
        labels = labels.to(device)

        image_features = model.get_image_features(**image_inputs)
        image_features = F.normalize(image_features, dim=-1)

        logits = image_features @ prototypes.T

        if hasattr(model, "logit_scale"):
            logits = logits * model.logit_scale.exp().clamp(max=100)
        if hasattr(model, "logit_bias"):
            logits = logits + model.logit_bias

        probs = torch.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)

        correct += (preds == labels).sum().item()
        total += labels.numel()

        for y, pred, path, prob in zip(labels.cpu(), preds.cpu(), paths, probs.cpu()):
            confusion[y, pred] += 1
            rows.append({
                "path": path,
                "true": CLASS_NAMES[y.item()],
                "pred": CLASS_NAMES[pred.item()],
                "scores": {
                    CLASS_NAMES[i]: float(prob[i])
                    for i in range(len(CLASS_NAMES))
                },
                "white_smoke_cloud_margin": float(
                    prob[CLASS_NAMES.index("steam")] -
                    prob[CLASS_NAMES.index("cloud")]
                ),
            })

    accuracy = correct / max(total, 1)

    print("\nTest accuracy:", f"{accuracy:.4f}")
    print("\nConfusion matrix (rows=true, cols=pred):")
    print("Classes:", CLASS_NAMES)
    print(confusion)

    out = Path("/app/output/test_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "accuracy": accuracy,
            "classes": CLASS_NAMES,
            "confusion_matrix": confusion.tolist(),
            "predictions": rows,
        }, f, ensure_ascii=False, indent=2)

    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
