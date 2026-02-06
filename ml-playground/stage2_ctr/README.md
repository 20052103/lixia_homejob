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
Stage 2 — CTR Modeling Results & Conclusions
Experimental Setup

Dataset: Criteo DAC (Kaggle version), fixed gold eval dataset (gold_eval.txt)

Features:

13 dense features (log / log1p transformed)

26 sparse categorical features (hashing trick)

Artifacts reused across all models:

Same feature hashing (hash_bins=2,000,000)

Same dense preprocessing

Same gold eval split

Metric:

Primary: Logloss

Secondary: AUC

Baseline probability:
Positive rate ≈ 0.251

All models were evaluated on the same gold dataset to ensure apples-to-apples comparison.

Model Comparison (Gold Eval)
Model	Logloss ↓	AUC ↑	Calibration (avg_pred vs pos_rate)
DNN baseline	0.5285	0.7570	0.2620 vs 0.2513
DCN v1 (CrossNet v1)	0.5319	0.7535	0.2513 vs 0.2513
DCN v2 (Low-rank + MoE)	0.5257	0.7594	0.2688 vs 0.2513
Key Observations
1. DNN baseline is already strong

The DNN significantly outperforms a naive baseline (always predicting global positive rate).

AUC ≈ 0.757 indicates strong ranking ability from embeddings + MLP alone.

This confirms that most first-order and some interaction signals are already captured.

2. DCN v1 underperforms the DNN

DCN v1 shows worse logloss and AUC than the DNN baseline.

However, it achieves near-perfect calibration (avg_pred ≈ pos_rate).

This behavior aligns with known limitations of rank-1 CrossNet v1:

Single global interaction pattern

Limited expressive power

Tendency to smooth predictions toward the mean

Conclusion: DCN v1 is too rigid for large-scale, noisy CTR data when the DNN baseline is strong.

3. DCN v2 outperforms both DNN and DCN v1

DCN v2 achieves the best logloss and AUC among all models.

Improvements over DNN:

Logloss ↓ ~0.0028

AUC ↑ ~0.0024

This gain is meaningful and typical for production CTR systems.

The improvement comes from DCN v2’s architectural advantages:

Low-rank cross layers (higher expressive power than rank-1)

Mixture-of-experts (MoE) with gating, enabling:

Sample-specific interaction patterns

Adaptive feature crossing across different user/content contexts

4. Calibration trade-off

DCN v2 slightly over-predicts probabilities (avg_pred > pos_rate).

This is a common trade-off when optimizing ranking performance.

Calibration can be corrected independently (e.g., temperature scaling) without affecting AUC.

Final Conclusion

When using a strong embedding-based DNN baseline, DCN v1 may fail to add value due to its limited interaction capacity.
DCN v2, with low-rank and gated mixture-of-experts cross layers, successfully captures higher-order, sample-dependent feature interactions and delivers consistent gains in both logloss and AUC on a fixed gold evaluation set.

This validates DCN v2 as the preferred cross-based architecture for CTR modeling in this setup.

Next Steps

Optional calibration (temperature scaling) on gold/validation data

Hyperparameter sweeps on DCN v2 (rank, experts, learning rate)

Extend comparison to DCNv2 vs wider/deeper DNN baselines