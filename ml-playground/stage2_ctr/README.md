Below is a **clean, ready-to-commit `README.md`**.
You can copy-paste this **as-is** into:

```
D:\repo\lixia_homejob\ml-playground\stage2_ctr\README.md
```

---

```markdown
# Stage 2 — CTR Modeling (DNN → DCN)

This stage builds a **CTR modeling pipeline** on the **Criteo DAC dataset**, starting from a **simple DNN baseline** and then extending to **DCN (Deep & Cross Network)**.

Key principles:
- **Large data stays outside git**
- **Code + configs stay inside git**
- **Windows + CUDA friendly**
- Step-by-step, debuggable, production-style setup

---

## 0. Repo & Data Layout

### Code (tracked by git)
```

stage2_ctr/
├── configs/
├── datasets/
├── models/
├── scripts/
├── README.md

```

### Large dataset (NOT tracked by git)
```

D:\repo\large data\criteo\dac

````

The repo reads data via absolute path defined in config.

---

## 1. Environment Setup

### Conda environment
```powershell
conda activate mtml
python -c "import torch; print(torch.__version__)"
````

### GPU check

```powershell
python - <<EOF
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
EOF
```

---

## 2. Dataset

* Dataset: **Kaggle Criteo DAC** (`mrkmakr/criteo-dataset`)
* Downloaded **manually**
* Extracted to:

```
D:\repo\large data\criteo\dac
```

### Schema (no header in file)

| Field           | Count         |
| --------------- | ------------- |
| Label           | 1             |
| Dense features  | 13 (`I1–I13`) |
| Sparse features | 26 (`C1–C26`) |

---

## 3. Git Safety

Create `.gitignore`:

```gitignore
/data/
/**/data/

*.zip
*.tar
*.gz
*.7z

*.pt
*.pth
*.ckpt

logs/
*.log

__pycache__/
*.pyc
.ipynb_checkpoints/
```

---

## 4. Config

`configs/local.yaml`

```yaml
data:
  dac_dir: "D:/repo/large data/criteo/dac"
  train_file: "train.txt"

preprocess:
  dense_transform: "log1p"
  hash_bins: 2000000

split:
  val_ratio: 0.05
  seed: 42

loader:
  batch_size: 4096
  num_workers: 0      # Windows safe default
  pin_memory: true

model:
  embed_dim: 16
  hidden_dims: [256, 128, 64]

train:
  lr: 0.001
  epochs: 2
  steps_per_epoch: 2000
  log_every: 100
```

---

## 5. Data Pipeline

### Design

* **IterableDataset** (streaming, no full memory load)
* Sparse features → **hashing**
* Per-field offset to form global embedding id:

```
global_id = field_id * hash_bins + hashed_id
total_embeddings = 26 * hash_bins
```

### Code

* `datasets/criteo_dac.py`

---

## 6. Sanity Check (Critical Step)

### Purpose

Verify **data → tensor → model input** correctness before training:

* parsing
* shapes
* value ranges
* hashing bounds
* label distribution
* train / val split

### Run

```powershell
cd stage2_ctr
python -m scripts.sanity_check_dataloader
```

Expected:

* dense shape `[B, 13]`
* sparse shape `[B, 26]`
* no NaN / out-of-range ids

> Always run scripts with `python -m`, never as raw `.py`.

---

## 7. Baseline Model — DNN CTR

### Model

* One large embedding table for sparse features
* Dense + flattened embeddings concatenated
* MLP → logit → sigmoid (via BCEWithLogitsLoss)

### Train

```powershell
python -m scripts.train_dnn
```

### Required device checks (printed during training)

* `torch.cuda.is_available: True`
* model on `cuda:0`
* dense / sparse / y on `cuda:0`

⚠️ Important:

```python
y = y.to(device)
```

`.to()` is **not in-place**.

---

## 8. GPU Verification (Windows)

### Two GPUs visible

* **GPU 0**: AMD Radeon (integrated, display)
* **GPU 1**: NVIDIA RTX 5090 (training)

### Correct way to verify GPU usage

**Task Manager → Performance → GPU 1**

* Watch **CUDA / Compute**
* Watch **Dedicated GPU memory**

### Why `nvidia-smi` shows `N/A`

* Windows runs NVIDIA in **WDDM mode**
* Per-process GPU memory often not reported
* This is normal and expected

---

## 9. Known Windows Issues & Fixes

### Import error

```
ModuleNotFoundError: No module named 'datasets'
```

Fix:

```powershell
python -m scripts.sanity_check_dataloader
```

Ensure:

```
datasets/__init__.py
scripts/__init__.py
```

---

### DataLoader multiprocessing error

```
Can't pickle local object ...
```

Fix:

* Use `num_workers: 0`, OR
* Move `collate_fn` to top-level function

---

## 10. Current Status

✅ Data pipeline verified
✅ DNN baseline training on GPU
✅ GPU usage confirmed (RTX 5090)
✅ Loss decreasing as expected

---

## 11. Next Steps

### Step 11.1 — Metrics

* Add **AUC**
* Add logging for:

  * train loss
  * val logloss
  * val AUC

### Step 11.2 — DCN (CrossNet v1)

* Add cross layers before MLP
* Compare vs DNN baseline

### Step 11.3 — DCNv2 (optional)

* Low-rank / mixture cross layers
* Memory vs gain analysis

### Step 11.4 — Performance tuning

* hash_bins vs embed_dim
* optimizer memory
* AMP / mixed precision
* DataLoader parallelism

---

## 12. Runbook (Quick Commands)

```powershell
# sanity check
python -m scripts.sanity_check_dataloader

# train DNN
python -m scripts.train_dnn

# GPU monitor
nvidia-smi -l 1
```

---

**Stage 2 philosophy**

> Build the smallest correct system first,
> then add modeling power (DCN) with confidence.

```

---

If you want, next I can:
- add a **Results table** section (date / config / metrics),
- or continue straight into **DCN (CrossNet v1) implementation** step-by-step.
```
