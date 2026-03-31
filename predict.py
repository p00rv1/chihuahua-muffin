"""
Predict on test images and create submission.csv for Kaggle.

- Loads best_model.pth (saved by train.py). Exits with a clear message if missing.
- Runs inference on data/test/ (flat folder; no labels).
- Writes submission.csv with columns image_id, prediction, confidence.
  If sample_submission.csv exists, output is aligned to its image_ids (same order;
  missing test images get prediction 0, confidence 0.5 so the file is valid).
- submission.csv is overwritten each run.

Usage:
    python predict.py

Outputs:
    submission.csv  - Kaggle submission (overwritten each run).
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import csv

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH = Path("best_model.pth")
TEST_DIR = Path("data/test")
OUTPUT_PATH = Path("submission.csv")
SAMPLE_SUBMISSION_PATH = Path("sample_submission.csv")
NUM_CLASSES = 2
CLASS_NAMES = ["chihuahua", "muffin"]
BATCH_SIZE = 32
IMAGE_SIZE = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ============================================================================
# MODEL — must be an exact copy of ResNet18Classifier in train.py
# ============================================================================

class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # weights=None because we load our own trained weights below
        self.resnet = models.resnet18(weights=None)
        features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

        # Matches train.py exactly: 512 → 256 → num_classes, Dropout 0.4
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
        return self.classifier(self.resnet(x))


# ============================================================================
# DATASET
# ============================================================================

class TestDataset(Dataset):
    """Flat test folder: all images, no labels. Returns (image_tensor, image_id)."""
    def __init__(self, image_dir: Path, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.images = []
        if self.image_dir.exists():
            seen = set()
            for ext in ["*.jpg", "*.jpeg", "*.png"]:
                for img in self.image_dir.glob(ext):
                    key = img.name.lower()
                    if key not in seen:
                        seen.add(key)
                        self.images.append(img)
        self.images.sort(key=lambda x: x.name)
        print(f"  Found {len(self.images)} images in {image_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Warning: Could not load {img_path}: {e}")
            image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (128, 128, 128))
        if self.transform:
            image = self.transform(image)
        # Use stem (no extension) as image_id to match typical Kaggle format
        return image, img_path.stem


# ============================================================================
# TRANSFORMS — must match val_transform in train.py exactly
# ============================================================================

test_transform = transforms.Compose([
    transforms.Resize(128),           # matches val_transform: Resize(128)
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ============================================================================
# INFERENCE
# ============================================================================

def predict_on_dataset(model, dataloader, device):
    predictions = []
    model.eval()
    with torch.no_grad():
        for images, filenames in tqdm(dataloader, desc="Predicting"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidences, predicted = probs.max(1)
            for fid, pred, conf in zip(
                filenames,
                predicted.cpu().numpy(),
                confidences.cpu().numpy()
            ):
                predictions.append({
                    "image_id": fid,
                    "prediction": int(pred),
                    "confidence": float(conf),
                })
    return predictions


def load_expected_image_ids():
    """Load image_id order from sample_submission.csv if present; else None."""
    if not SAMPLE_SUBMISSION_PATH.exists():
        return None
    with open(SAMPLE_SUBMISSION_PATH, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "image_id" not in (reader.fieldnames or []):
            return None
        return [row["image_id"] for row in reader]


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("  Chihuahua vs Muffin - Prediction")
    print("=" * 60)

    # ── 1. Load model ────────────────────────────────────────────────────────
    if not MODEL_PATH.exists():
        print(f"\n[ERROR] Model not found: {MODEL_PATH}")
        print("  Run train.py first to train and save best_model.pth.")
        return 1

    print("\n[1/4] Loading model...")
    try:
        state = torch.load(MODEL_PATH, map_location=device)
    except Exception as e:
        print(f"[ERROR] Could not load model file: {e}")
        return 1

    model = ResNet18Classifier(num_classes=NUM_CLASSES)
    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        print(f"[ERROR] Model state_dict incompatible with architecture: {e}")
        print("  Make sure ResNet18Classifier here matches train.py exactly.")
        return 1

    model = model.to(device)
    model.eval()
    print(f"  [OK] Loaded {MODEL_PATH}")

    # ── 2. Run inference ─────────────────────────────────────────────────────
    print("\n[2/4] Predicting on test images...")
    if not TEST_DIR.exists():
        print(f"  [ERROR] Test directory not found: {TEST_DIR}")
        return 1

    test_dataset = TestDataset(TEST_DIR, transform=test_transform)
    if len(test_dataset) == 0:
        print(f"  [ERROR] No images found in {TEST_DIR}.")
        return 1

    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    predictions = predict_on_dataset(model, test_loader, device)
    pred_by_id = {p["image_id"]: p for p in predictions}
    print(f"  [OK] Predicted {len(predictions)} images")

    # ── 3. Align to sample_submission order ──────────────────────────────────
    print("\n[3/4] Aligning to submission format...")
    expected_ids = load_expected_image_ids()
    if expected_ids is not None:
        rows = []
        for image_id in expected_ids:
            if image_id in pred_by_id:
                rows.append(pred_by_id[image_id])
            else:
                # Image listed in sample but not found in test folder
                rows.append({"image_id": image_id, "prediction": 0, "confidence": 0.5})
        predictions = rows
        print(f"  [OK] Aligned to {SAMPLE_SUBMISSION_PATH} ({len(predictions)} rows)")
    else:
        print(f"  [INFO] No {SAMPLE_SUBMISSION_PATH} found; order = test folder order")

    # ── 4. Write CSV ──────────────────────────────────────────────────────────
    print("\n[4/4] Writing submission.csv...")
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "prediction", "confidence"])
        writer.writeheader()
        writer.writerows(predictions)
    print(f"  [OK] Written to {OUTPUT_PATH} (overwrites previous file)")

    print("\n" + "=" * 60)
    print("  Submission ready for Kaggle upload")
    print("  Columns: image_id | prediction (0=chihuahua, 1=muffin) | confidence")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    exit(main())