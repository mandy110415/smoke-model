from pathlib import Path
import json
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from transformers import AutoModel, AutoProcessor
from tqdm import tqdm

MODEL_NAME = "google/siglip2-base-patch16-224"
DATA_ROOT = Path("/app/data")
OUTPUT_ROOT = Path("/app/output")
BEST_MODEL_DIR = OUTPUT_ROOT / "best_model"

SEED = 42
BATCH_SIZE = 16
MAX_EPOCHS = 20
PATIENCE = 5
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4

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

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class SiglipDataset(torch.utils.data.Dataset):
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
        image, label = self.ds[idx]
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)
        return pixel_values, label

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

def enable_partial_finetuning(model):
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze last 2 vision encoder layers
    if hasattr(model, "vision_model") and hasattr(model.vision_model, "encoder"):
        layers = model.vision_model.encoder.layers
        for layer in layers[-2:]:
            for p in layer.parameters():
                p.requires_grad = True

    # Vision post layer norm
    if hasattr(model, "vision_model") and hasattr(model.vision_model, "post_layernorm"):
        for p in model.vision_model.post_layernorm.parameters():
            p.requires_grad = True

    # Visual projection if present
    if hasattr(model, "visual_projection"):
        for p in model.visual_projection.parameters():
            p.requires_grad = True

    # SigLIP scale/bias if trainable params
    for attr in ["logit_scale", "logit_bias"]:
        if hasattr(model, attr):
            obj = getattr(model, attr)
            if isinstance(obj, torch.nn.Parameter):
                obj.requires_grad = True

def evaluate(model, loader, prototypes, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for pixel_values, labels in loader:
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)

            image_features = model.get_image_features(pixel_values=pixel_values)
            image_features = F.normalize(image_features, dim=-1)

            logits = image_features @ prototypes.T
            loss = F.cross_entropy(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total

def main():
    set_seed(SEED)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("SigLIP 2 two-class fine-tuning")
    print("Device:", device)
    print("Model :", MODEL_NAME)

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)

    enable_partial_finetuning(model)

    train_ds = SiglipDataset(DATA_ROOT / "train", processor)
    val_ds = SiglipDataset(DATA_ROOT / "val", processor)

    print("Train images:", len(train_ds))
    print("Val images  :", len(val_ds))
    print("Class mapping:", train_ds.ds.class_to_idx)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    prototypes = build_text_prototypes(model, processor, device).detach()

    trainable = [p for p in model.parameters() if p.requires_grad]
    print("Trainable parameters:", sum(p.numel() for p in trainable))

    optimizer = torch.optim.AdamW(
        trainable,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    best_val_loss = float("inf")
    no_improve = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{MAX_EPOCHS}")

        for pixel_values, labels in pbar:
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            image_features = model.get_image_features(pixel_values=pixel_values)
            image_features = F.normalize(image_features, dim=-1)

            logits = image_features @ prototypes.T
            loss = F.cross_entropy(logits, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / total
        train_acc = correct / total

        # Recompute prototypes after each epoch because vision changes but text stays fixed;
        # text prototype itself remains fixed, so reuse is valid.
        val_loss, val_acc = evaluate(model, val_loader, prototypes, device)

        print(f"Train loss={train_loss:.4f}, acc={train_acc:.4f}")
        print(f"Val   loss={val_loss:.4f}, acc={val_acc:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0

            BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(BEST_MODEL_DIR)
            processor.save_pretrained(BEST_MODEL_DIR)

            with open(OUTPUT_ROOT / "training_history.json", "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)

            print("Saved best model.")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"Early stopping after {PATIENCE} epochs without validation improvement.")
                break

    print("\nTraining completed.")
    print("Best model:", BEST_MODEL_DIR)

if __name__ == "__main__":
    main()
