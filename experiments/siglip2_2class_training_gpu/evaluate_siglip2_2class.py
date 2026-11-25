from pathlib import Path
import csv
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from transformers import AutoModel, AutoProcessor

DATA_ROOT = Path("/app/data")
MODEL_DIR = Path("/app/output/best_model")
OUTPUT_ROOT = Path("/app/output")

BATCH_SIZE = 16
CLASS_NAMES = ["cloud", "smoke"]

TEXT_PROMPTS = {
    "cloud": [
        "a photo of natural clouds in the sky",
        "a photo of a cloud formation",
        "a photo of clouds not connected to an emission source",
        "a photo of white or gray clouds in the sky",
    ],
    "smoke": [
        "a photo of smoke",
        "a photo of a smoke plume",
        "a photo of industrial smoke from an emission source",
        "a photo of white, gray, or black smoke",
    ],
}

class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, folder, processor):
        self.ds = ImageFolder(folder)
        self.processor = processor

        expected = {name: i for i, name in enumerate(CLASS_NAMES)}
        if self.ds.class_to_idx != expected:
            raise RuntimeError(
                f"Class mapping mismatch. Expected {expected}, got {self.ds.class_to_idx}"
            )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        path, label = self.ds.samples[idx]
        image, _ = self.ds[idx]
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)
        return pixel_values, label, path

def build_text_prototypes(model, processor, device):
    model.eval()
    prototypes = []

    with torch.no_grad():
        for class_name in CLASS_NAMES:
            prompts = TEXT_PROMPTS[class_name]
            text_inputs = processor(
                text=prompts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            feats = model.get_text_features(**text_inputs)
            feats = F.normalize(feats, dim=-1)
            proto = feats.mean(dim=0)
            proto = F.normalize(proto, dim=-1)
            prototypes.append(proto)

    return torch.stack(prototypes, dim=0)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Model :", MODEL_DIR)

    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    model = AutoModel.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    test_ds = EvalDataset(DATA_ROOT / "test", processor)
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    prototypes = build_text_prototypes(model, processor, device)

    # rows=true class, cols=pred class
    confusion = [[0, 0], [0, 0]]
    results = []
    correct = 0
    total = 0

    with torch.no_grad():
        for pixel_values, labels, paths in test_loader:
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)

            image_features = model.get_image_features(pixel_values=pixel_values)
            image_features = F.normalize(image_features, dim=-1)

            logits = image_features @ prototypes.T
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=1)

            for i in range(labels.size(0)):
                y = labels[i].item()
                p = preds[i].item()

                confusion[y][p] += 1
                correct += int(y == p)
                total += 1

                cloud_prob = probs[i, 0].item()
                smoke_prob = probs[i, 1].item()

                results.append({
                    "path": paths[i],
                    "true_label": CLASS_NAMES[y],
                    "pred_label": CLASS_NAMES[p],
                    "cloud_score": cloud_prob,
                    "smoke_score": smoke_prob,
                    "smoke_minus_cloud": smoke_prob - cloud_prob,
                    "correct": int(y == p),
                })

    accuracy = correct / total

    cloud_total = sum(confusion[0])
    smoke_total = sum(confusion[1])

    cloud_to_smoke = confusion[0][1] / cloud_total if cloud_total else 0.0
    smoke_to_cloud = confusion[1][0] / smoke_total if smoke_total else 0.0

    metrics = {
        "test_accuracy": accuracy,
        "cloud_to_smoke_rate": cloud_to_smoke,
        "smoke_to_cloud_rate": smoke_to_cloud,
        "confusion_matrix": {
            "labels": CLASS_NAMES,
            "matrix": confusion,
        },
        "test_count": total,
    }

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_ROOT / "test_results.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    with open(OUTPUT_ROOT / "test_predictions.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "path",
                "true_label",
                "pred_label",
                "cloud_score",
                "smoke_score",
                "smoke_minus_cloud",
                "correct",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print("\n==============================")
    print("TEST RESULTS")
    print("==============================")
    print(f"Test accuracy       : {accuracy:.4f}")
    print(f"Cloud -> Smoke rate : {cloud_to_smoke:.4f}")
    print(f"Smoke -> Cloud rate : {smoke_to_cloud:.4f}")

    print("\nConfusion Matrix")
    print("                 Pred Cloud   Pred Smoke")
    print(f"True Cloud       {confusion[0][0]:10d}   {confusion[0][1]:10d}")
    print(f"True Smoke       {confusion[1][0]:10d}   {confusion[1][1]:10d}")

    print("\nSaved:")
    print(OUTPUT_ROOT / "test_results.json")
    print(OUTPUT_ROOT / "test_predictions.csv")

if __name__ == "__main__":
    main()
