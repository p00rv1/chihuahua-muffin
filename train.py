"""
Train ResNet-18 classifier on Chihuahua vs Muffin dataset using 3LC.

- Sentry warnings disabled
- Embeddings collection fixed with correct layer index
- Pretrained backbone enabled (strong accuracy boost)
- Safe transforms to prevent shape errors
- Strong augmentation + mixed precision
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import tlc
from tqdm import tqdm
from pathlib import Path
import random
import numpy as np
import os
from collections import Counter
from torch.cuda.amp import autocast, GradScaler
import sentry_sdk   # For disabling Sentry warnings

# ============================================================================
# CONFIGURATION
# ============================================================================

EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.0003
RANDOM_SEED = 42
MIXUP_ALPHA = 0.3
SWA_START_FRAC = 0.8
PROJECT_NAME = "Chihuahua-Muffin"
DATASET_NAME = "chihuahua-muffin"
NUM_CLASSES = 2
CLASS_NAMES = ["chihuahua", "muffin"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"[OK] Random seed set to {seed}")


def print_label_stats(table, name="Table"):
    try:
        labels = [int(sample["label"]) for sample in table]
        dist = Counter(labels)
        print(f"\n{name} label distribution:")
        for lbl, count in sorted(dist.items()):
            cls_name = CLASS_NAMES[lbl] if lbl < len(CLASS_NAMES) else f"unknown({lbl})"
            print(f"  {cls_name} ({lbl}): {count} samples")
        print(f"  Total: {len(table)} samples")
    except Exception as e:
        print(f"Could not compute label stats for {name}: {e}")


def mixup_data(x, y, alpha=0.3):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a = y.clone().clamp(0, NUM_CLASSES - 1)
    y_b = y[index].clone().clamp(0, NUM_CLASSES - 1)
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    y_a = y_a.clamp(0, NUM_CLASSES - 1)
    y_b = y_b.clamp(0, NUM_CLASSES - 1)
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================================================
# MODEL
# ============================================================================

class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        features = self.resnet(x)
        return self.classifier(features)


# ============================================================================
# TRANSFORMS
# ============================================================================

train_transform = transforms.Compose([
    transforms.Resize(176),
    transforms.RandomResizedCrop(128, scale=(0.6, 1.0), ratio=(0.75, 1.33)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.25),
    transforms.RandomRotation(degrees=25),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=18, scale=(0.75, 1.3)),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.35, scale=(0.02, 0.18)),
])

val_transform = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def safe_image_to_tensor(image, transform):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image)).convert("RGB")
    else:
        image = image.convert("RGB")
    return transform(image)


def train_fn(sample):
    try:
        image = Image.open(sample["image"])
        label = int(sample.get("label", 0))
        if label not in (0, 1):
            label = 0
        label = max(0, min(label, NUM_CLASSES - 1))
        img_tensor = safe_image_to_tensor(image, train_transform)
        return img_tensor, torch.tensor(label, dtype=torch.long)
    except Exception as e:
        print(f"Warning in train_fn: {e}")
        return torch.zeros((3, 128, 128), dtype=torch.float32), torch.tensor(0, dtype=torch.long)


def val_fn(sample):
    try:
        image = Image.open(sample["image"])
        label = int(sample.get("label", 0))
        label = max(0, min(label, NUM_CLASSES - 1))
        img_tensor = safe_image_to_tensor(image, val_transform)
        return img_tensor, torch.tensor(label, dtype=torch.long)
    except Exception as e:
        print(f"Warning in val_fn: {e}")
        return torch.zeros((3, 128, 128), dtype=torch.float32), torch.tensor(0, dtype=torch.long)


# ============================================================================
# METRICS
# ============================================================================

def metrics_fn(batch, predictor_output: tlc.PredictorOutput):
    labels = batch[1].to(device)
    predictions = predictor_output.forward
    softmax_output = F.softmax(predictions, dim=1)
    predicted_indices = torch.argmax(predictions, dim=1)
    confidence = torch.gather(softmax_output, 1, predicted_indices.unsqueeze(1)).squeeze(1)

    accuracy = (predicted_indices == labels).float()
    cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")(predictions, labels)

    return {
        "loss": cross_entropy_loss.cpu().numpy(),
        "predicted": predicted_indices.cpu().numpy(),
        "accuracy": accuracy.cpu().numpy(),
        "confidence": confidence.cpu().numpy(),
    }


# ============================================================================
# TRAINING
# ============================================================================

BEST_MODEL_FILENAME = "best_model.pth"


def train():
    set_seed(RANDOM_SEED)
    base_path = Path(__file__).parent
    scaler = GradScaler()

    tlc.register_project_url_alias(
        token="CHIHUAHUA_MUFFIN_DATA",
        path=str(base_path.absolute()),
        project=PROJECT_NAME,
    )

    # Disable Sentry to stop the connection warnings
    sentry_sdk.init(dsn="", default_integrations=False)

    print("\nLoading 3LC tables...")
    train_table = tlc.Table.from_names(
        project_name=PROJECT_NAME,
        dataset_name=DATASET_NAME,
        table_name="train"
    ).latest()

    val_table = tlc.Table.from_names(
        project_name=PROJECT_NAME,
        dataset_name=DATASET_NAME,
        table_name="val"
    ).latest()

    print_label_stats(train_table, "Train")
    print_label_stats(val_table, "Val")

    train_table.map(train_fn)
    train_table.map_collect_metrics(val_fn)
    val_table.map(val_fn)

    train_sampler = train_table.create_sampler(
        exclude_zero_weights=True,
        weighted=False,
        shuffle=True
    )

    train_dataloader = DataLoader(
        train_table, batch_size=BATCH_SIZE, sampler=train_sampler,
        num_workers=0, pin_memory=True
    )

    val_dataloader = DataLoader(
        val_table, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True
    )

    model = ResNet18Classifier(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=8e-4, steps_per_epoch=len(train_dataloader),
        epochs=EPOCHS, pct_start=0.2
    )

    swa_model = torch.optim.swa_utils.AveragedModel(model)
    swa_start = int(EPOCHS * SWA_START_FRAC)

    run = tlc.init(project_name=PROJECT_NAME, description="Chihuahua vs Muffin - Sentry disabled")

    best_val_accuracy = 0.0
    best_model_state = None

    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)

    for epoch in range(EPOCHS):
        model.train()
        for images, labels in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images, labels = images.to(device), labels.to(device)
            mixed_images, targets_a, targets_b, lam = mixup_data(images, labels, MIXUP_ALPHA)

            optimizer.zero_grad()
            with autocast():
                outputs = model(mixed_images)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        if epoch >= swa_start:
            swa_model.update_parameters(model)

        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for images, labels in val_dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                pred = outputs.argmax(dim=1)
                val_correct += (pred == labels).sum().item()
                val_total += labels.size(0)

        val_accuracy = 100 * val_correct / val_total if val_total > 0 else 0
        print(f"Epoch {epoch+1}/{EPOCHS} - Val Acc: {val_accuracy:.2f}%")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  --> New best model! ({val_accuracy:.2f}%)")

    print("\nFinalizing SWA...")
    torch.optim.swa_utils.update_bn(train_dataloader, swa_model, device=device)

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    else:
        model.load_state_dict(swa_model.module.state_dict())

    model_path = base_path / BEST_MODEL_FILENAME
    torch.save(model.state_dict(), model_path)
    print(f"[OK] Best model saved to {model_path}")

    # ====================== 3LC METRICS & EMBEDDINGS ======================
    print("\nCollecting metrics & embeddings...")
    model.eval()

    # Find embedding layer index
    indices_and_modules = list(enumerate(model.named_modules()))
    embedding_idx = None
    for idx, (name, module) in indices_and_modules:
        if name == "classifier.4":   # Linear(512 → 256)
            embedding_idx = idx
            break
    if embedding_idx is None:
        embedding_idx = len(indices_and_modules) - 3   # safe fallback

    predictor = tlc.Predictor(model, layers=[embedding_idx])

    metric_schemas = {
        "loss": tlc.Schema(description="Cross entropy loss", value=tlc.Float32Value()),
        "predicted": tlc.CategoricalLabelSchema(display_name="predicted", classes=CLASS_NAMES),
        "accuracy": tlc.Schema(value=tlc.Float32Value()),
        "confidence": tlc.Schema(value=tlc.Float32Value()),
    }

    collectors = [
        tlc.FunctionalMetricsCollector(collection_fn=metrics_fn, column_schemas=metric_schemas),
        tlc.EmbeddingsMetricsCollector(layers=[embedding_idx])
    ]

    try:
        tlc.collect_metrics(
            table=train_table,
            predictor=predictor,
            metrics_collectors=collectors,
            split="train",
            dataloader_args={"batch_size": BATCH_SIZE, "num_workers": 0, "pin_memory": True}
        )
        print("  [OK] Metrics collected successfully.")
    except Exception as e:
        print(f"  ERROR collecting metrics: {e}")

    print("\nReducing embeddings with UMAP...")
    try:
        run.reduce_embeddings_by_foreign_table_url(
            train_table.url, method="umap", n_neighbors=15, n_components=3
        )
        print("  [OK] Embeddings reduced.")
    except Exception as e:
        print(f"  WARNING: UMAP failed: {e}")

    run.set_status_completed()
    print("\n[OK] Training completed!")
    print("Open 3LC Dashboard → sort train table by high 'loss' or low 'confidence' → fix wrong labels.")


if __name__ == "__main__":
    train()