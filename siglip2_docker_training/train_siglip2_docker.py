from pathlib import Path
import json
import random
import math

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoModel, AutoProcessor
from tqdm import tqdm

MODEL_ID = "google/siglip2-base-patch16-224"

DATA_ROOT = Path("/app/data")
OUTPUT_DIR = Path("/app/output")

SEED = 42
EPOCHS = 20
BATCH_SIZE = 4
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0
UNFREEZE_LAST_N_VISION_LAYERS = 2
EARLY_STOPPING_PATIENCE = 5

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


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FolderDataset(Dataset):
    def __init__(self, root, class_names):
        self.samples = []
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}
        for class_name in class_names:
            class_dir = root / class_name
            if not class_dir.exists():
                raise FileNotFoundError(f"Missing folder: {class_dir}")
            for p in sorted(class_dir.iterdir()):
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                    self.samples.append((p, self.class_to_idx[class_name]))
        if not self.samples:
            raise RuntimeError(f"Dataset is empty: {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return Image.open(path).convert("RGB"), label, str(path)


def collate_fn(batch, processor):
    images, labels, paths = zip(*batch)
    inputs = processor(images=list(images), return_tensors="pt")
    return inputs, torch.tensor(labels, dtype=torch.long), list(paths)


def freeze_and_unfreeze(model):
    for p in model.parameters():
        p.requires_grad = False

    layers = model.vision_model.encoder.layers
    n = min(UNFREEZE_LAST_N_VISION_LAYERS, len(layers))

    for layer in layers[-n:]:
        for p in layer.parameters():
            p.requires_grad = True

    if hasattr(model.vision_model, "post_layernorm"):
        for p in model.vision_model.post_layernorm.parameters():
            p.requires_grad = True

    if hasattr(model, "visual_projection"):
        for p in model.visual_projection.parameters():
            p.requires_grad = True

    if hasattr(model, "logit_scale") and isinstance(model.logit_scale, torch.nn.Parameter):
        model.logit_scale.requires_grad = True
    if hasattr(model, "logit_bias") and isinstance(model.logit_bias, torch.nn.Parameter):
        model.logit_bias.requires_grad = True


@torch.no_grad()
def build_class_text_embeddings(model, processor, device):
    class_embeddings = []
    model.eval()

    for class_name in CLASS_NAMES:
        text_inputs = processor(
            text=[p.lower() for p in PROMPTS[class_name]],
            padding="max_length",
            max_length=64,
            truncation=True,
            return_tensors="pt",
        )
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

        feats = model.get_text_features(**text_inputs)
        feats = F.normalize(feats, dim=-1)

        prototype = F.normalize(feats.mean(dim=0), dim=-1)
        class_embeddings.append(prototype)

    return torch.stack(class_embeddings, dim=0)


def compute_logits(model, image_inputs, class_text_embeddings):
    image_features = model.get_image_features(**image_inputs)
    image_features = F.normalize(image_features, dim=-1)

    logits = image_features @ class_text_embeddings.T

    if hasattr(model, "logit_scale"):
        logits = logits * model.logit_scale.exp().clamp(max=100)

    if hasattr(model, "logit_bias"):
        logits = logits + model.logit_bias

    return logits


def run_epoch(model, loader, class_text_embeddings, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    iterator = tqdm(loader, leave=False)

    for image_inputs, labels, _ in iterator:
        image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        logits = compute_logits(model, image_inputs, class_text_embeddings)
        loss = F.cross_entropy(logits, labels)

        if is_train:
            loss.backward()
            optimizer.step()

        preds = logits.argmax(dim=1)

        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_count += labels.size(0)

        iterator.set_postfix(loss=f"{loss.item():.4f}")

    return {
        "loss": total_loss / max(total_count, 1),
        "accuracy": total_correct / max(total_count, 1),
    }


def save_checkpoint(model, processor, output_dir, metadata):
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    with open(output_dir / "training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("SigLIP 2 smoke/cloud partial fine-tuning")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Model : {MODEL_ID}")
    print(f"Data  : {DATA_ROOT}")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(device)

    freeze_and_unfreeze(model)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    print(f"Trainable parameters: {trainable:,} / {total:,}")
    print(f"Trainable ratio: {trainable / total:.2%}")

    train_ds = FolderDataset(DATA_ROOT / "train", CLASS_NAMES)
    val_ds = FolderDataset(DATA_ROOT / "val", CLASS_NAMES)

    print(f"Train images: {len(train_ds)}")
    print(f"Val images  : {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=lambda b: collate_fn(b, processor),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=lambda b: collate_fn(b, processor),
    )

    class_text_embeddings = build_class_text_embeddings(
        model, processor, device
    ).detach()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_loss = math.inf
    no_improve = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")

        train_metrics = run_epoch(
            model, train_loader, class_text_embeddings, device, optimizer
        )

        with torch.no_grad():
            val_metrics = run_epoch(
                model, val_loader, class_text_embeddings, device
            )

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(row)

        print(
            f"Train loss={row['train_loss']:.4f}, "
            f"acc={row['train_accuracy']:.4f}"
        )
        print(
            f"Val   loss={row['val_loss']:.4f}, "
            f"acc={row['val_accuracy']:.4f}"
        )

        if row["val_loss"] < best_val_loss:
            best_val_loss = row["val_loss"]
            no_improve = 0

            metadata = {
                "base_model": MODEL_ID,
                "classes": CLASS_NAMES,
                "class_semantics": {
                    "black_smoke": "black smoke",
                    "steam": "white smoke",
                    "cloud": "cloud",
                    "normal_background": "background",
                },
                "prompts": PROMPTS,
                "best_epoch": epoch,
                "best_val_loss": best_val_loss,
                "history": history,
            }

            save_checkpoint(
                model,
                processor,
                OUTPUT_DIR / "best_model",
                metadata,
            )

            print("Saved new best model.")
        else:
            no_improve += 1

        if no_improve >= EARLY_STOPPING_PATIENCE:
            print(
                f"Early stopping after {EARLY_STOPPING_PATIENCE} "
                f"epochs without validation improvement."
            )
            break

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print("\nTraining completed.")
    print(f"Best model: {OUTPUT_DIR / 'best_model'}")


if __name__ == "__main__":
    main()
