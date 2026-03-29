# Chihuahua vs Muffin – 3LC Kaggle Starter Kit

**The 3LC x MNNIT Hackathon: Data-Centric AI with 3LC**

> Can you tell a chihuahua from a muffin? Build an image classifier using data-centric AI with 3LC and compete on the Kaggle leaderboard.

**Setup and workflow:** See the competition **Overview**, **Description**, and **Iterative Workflow** tabs on Kaggle for the full guide. You must also complete the **evaluation & feedback form** by the deadline (see Overview on Kaggle).

---

## Quick start

### 1. Environment

```bash
python -m venv 3lc-env
# Windows: 3lc-env\Scripts\activate
# Mac/Linux: source 3lc-env/bin/activate

# CPU: install 3LC and PyTorch. For GPU: install PyTorch with CUDA first, then 3LC (see Kaggle "Iterative Workflow - Environment Setup").
pip install 3lc joblib pytz umap-learn torch torchvision
```

The starter uses **UMAP** for embedding reduction (3D embeddings in the Dashboard). This works on both CPU and GPU and avoids environment-specific issues that PaCMAP can have (e.g. on some Windows/PyTorch setups). 3LC also supports PaCMAP if you want to switch `method="pacmap"` in `train.py` and install `pacmap`.

### 2. 3LC login

- Create account: [https://account.3lc.ai](https://account.3lc.ai)
- API key: [https://account.3lc.ai/api-key](https://account.3lc.ai/api-key)

```bash
3lc login <your_api_key>
3lc service   # Required to use the 3LC Dashboard (label undefined, view embeddings)
```

### 3. Data

The kit includes a pre-split `data/` folder:

```
data/
├── train/
│   ├── chihuahua/   (labeled)
│   ├── muffin/     (labeled)
│   └── undefined/  (unlabeled — use 3LC to label and add to training)
├── val/
│   ├── chihuahua/
│   └── muffin/
└── test/           (flat — only images, no subfolders; for submission)
```

**No test data is used during training.** Only `train` and `val` are registered in 3LC; `predict.py` reads `data/test/` only for inference.

### 4. Run order (train and submit)

```bash
python register_tables.py   # once — create 3LC tables from data/train, data/val
python train.py             # train model; writes best_model.pth
python predict.py           # run inference; writes submission.csv
```

Upload `submission.csv` to the Kaggle competition.

---

## Competition at a glance

| Item | Details |
|------|--------|
| **Task** | Binary classification: chihuahua (0) vs muffin (1) |
| **Model** | ResNet-18 (fixed); no pretrained weights — train from scratch |
| **Metric** | Accuracy on hidden test set |
| **Tool** | 3LC (required) |

---

## Data-centric loop

1. **Train** – `train.py` → 3LC run with embeddings.
2. **Analyze** – Dashboard: embeddings, per-sample metrics.
3. **Fix data** – Label `undefined`, correct labels, adjust weights.
4. **Retrain** – `train.py` uses `.latest()` tables.
5. **Submit** – `predict.py` → `submission.csv` → Kaggle.

---

## Data at a glance (no ambiguity)

| Set | Labeled | Unlabeled | Total |
|-----|---------|-----------|-------|
| **Train** | 100 (50 chihuahua, 50 muffin) | 3,579 undefined | 3,679 |
| **Val** | 1,000 (500 per class, balanced) | 0 | 1,000 |
| **Test** | Hidden | — | 1,184 |

You start with **100 labeled** and **3,579 unlabeled**. Val is 1,000 (500 per class) for stable feedback. Use 3LC to label undefined samples and retrain to improve accuracy.

## Outputs (what each script produces)

| Script | Output | Behavior when run again |
|--------|--------|--------------------------|
| `register_tables.py` | 3LC tables (train, val) | **Idempotent:** if tables already exist, skips and does not overwrite; safe to run multiple times |
| `train.py` | `best_model.pth` | **Overwrites** previous best model |
| `predict.py` | `submission.csv` | **Overwrites** previous submission file |

So: each training run replaces `best_model.pth`; each prediction run replaces `submission.csv`. There is no versioning; the latest run is the one that counts. For a different checkpoint, rename or copy `best_model.pth` before running `train.py` again.

---

## Loading tables: .latest() vs URL

- **Default:** `train.py` loads tables by **name** with **`.latest()`**, so it always uses the newest revision (including any edits you make in the 3LC Dashboard).
- **Optional:** To train on a **specific** table revision, you can load by URL. In `train.py`, comment out the "OPTION 1" block and uncomment the "OPTION 2" block, then paste your train and val table URLs (from the 3LC Dashboard → Tables tab → copy URL).

---

## Files in this kit

| File | Purpose |
|------|--------|
| `config.yaml` | Competition and training config |
| `register_tables.py` | Register train/val in 3LC (run once) |
| `train.py` | Training with 3LC; saves `best_model.pth` |
| `predict.py` | Inference on test; writes `submission.csv` |
| `sample_submission.csv` | Required submission format and image_ids |
| `data/` | Pre-split train, val, and test images |
| `README.md` | This file |

---

## Submission format

`submission.csv` must have:

- `image_id` – same IDs as in `sample_submission.csv`
- `prediction` – 0 (chihuahua) or 1 (muffin)
- `confidence` – float in [0, 1]

---

## Dataset and pipeline verification

- **Splits:** Only `data/train` and `data/val` are registered in 3LC. **Test data is never used for training.** `predict.py` reads only `data/test/` for inference.
- **Train/val:** Train has labeled (chihuahua, muffin) and undefined (weight=0 until you label in Dashboard). Val has only labeled classes. Class distributions can be checked in the 3LC Dashboard after running `register_tables.py`.
- **Reproducibility:** `train.py` sets a fixed random seed (see `RANDOM_SEED` in the script) so training is deterministic for the same data and code.
- **Submission alignment:** If `sample_submission.csv` is present, `predict.py` writes a submission with the same `image_id`s in the same order; missing test images get a default prediction so the file is valid for Kaggle.

---

## For reviewers and live demo

- **Run order:** `register_tables.py` (once) → `train.py` → `predict.py`. No silent dependencies on uncommitted or external state.
- **Failures are explicit:** Missing model, missing test dir, empty test folder, or invalid model file all print a clear error and exit with non-zero status.
- **Overwrite behavior:** Each run of `train.py` overwrites `best_model.pth`; each run of `predict.py` overwrites `submission.csv`. No versioning; documented in README and in script docstrings.
- **Table loading:** Participants see which tables are used: `train.py` prints train and val table URLs after loading. Optional URL-based loading is available (commented out) for pinning to a specific revision.

---

## Resources

- [3LC Documentation](https://docs.3lc.ai)

Good luck, and may the best data win.
