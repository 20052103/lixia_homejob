# Stage 2 — CTR Modeling with Criteo DAC

This stage focuses on building, evaluating, and comparing multiple CTR models on the Criteo DAC dataset.
All models share the same feature pipeline, hashing strategy, and gold evaluation dataset to ensure
fair, apples-to-apples comparisons.

---

## Dataset

- **Source**: Criteo DAC (Kaggle version)
- **Task**: Binary classification (CTR)
- **Features**:
  - 13 dense numerical features (log / log1p transformed)
  - 26 sparse categorical features (hashing trick)
- **Positive rate** (gold eval): approximately **0.25**

---

## Feature Pipeline (Shared Across All Models)

To ensure consistency and reproducibility, all models reuse the same feature artifacts:

- Dense preprocessing:
  - Log / log1p transform
- Sparse preprocessing:
  - Hashing trick
  - `hash_bins = 2,000,000`
- Feature artifacts are saved as:
  - `feature_artifacts.yaml`
- The same artifacts are reused for:
  - Training
  - Evaluation
  - Inference

---

## Evaluation Protocol

- **Evaluation set**:
  - Fixed gold dataset: `gold_eval.txt`
- **Metrics**:
  - Logloss (primary)
  - AUC (secondary)
- **Baseline reference**:
  - Always predicting the global positive rate

All metrics reported below are computed on the same gold evaluation set.

---

## Models Implemented

### 1. DNN Baseline

- Embedding lookup for sparse features
- Concatenation with dense features
- Multi-layer perceptron (MLP)
- Output bias initialized with dataset prior

This serves as a strong baseline model.

---

### 2. DCN v1 (CrossNet v1)

- Explicit cross layers (rank-1)
- Deep tower (MLP)
- Shared cross parameters across all samples

DCN v1 improves calibration but is limited in expressive power.

---

### 3. DCN v2 (Low-rank CrossNet with MoE)

- Low-rank cross layers
- Mixture-of-experts (MoE) with gating
- Sample-dependent feature interaction patterns
- Deep tower combined with cross outputs

DCN v2 significantly increases modeling capacity for higher-order interactions.

---

## Gold Evaluation Results

## Evaluation Results (Preliminary, with Known Leakage)

> ⚠️ **Important note**  
> The current evaluation split (`gold_eval.txt`) was sampled from `train.txt` and used for
> training-time monitoring / early stopping (GBDT).  
> Therefore, the metrics below are **optimistic** and should be used for **relative comparison only**,
> not as final test results.

**Evaluation set**: `gold_eval.txt`  
**Positive rate**: ~0.251  
**Samples**: 500,000

| Model | Category | Logloss ↓ | AUC ↑ | avg_pred vs pos_rate | Notes |
|------|----------|-----------|-------|----------------------|-------|
| Always predict p | Naive baseline | ~0.555 | 0.500 | ≈ p | Reference |
| GBDT (LightGBM) | Tree-based | **0.4586** | **0.7859** | N/A | Strong tabular baseline (early-stopped on gold) |
| DCN v2 | Deep (CrossNet v2 + MoE) | 0.5257 | 0.7594 | 0.2688 vs 0.2513 | Best deep model |
| DNN baseline | Deep (Embedding + MLP) | 0.5285 | 0.7570 | 0.2620 vs 0.2513 | Strong baseline |
| DCN v1 | Deep (CrossNet v1) | 0.5319 | 0.7535 | **0.2513 vs 0.2513** | Well-calibrated, weak ranking |

---

## Intended Final Evaluation Protocol (No Leakage)

After creating clean, non-overlapping splits, all models will be evaluated using the following setup:

| Split | Usage | Notes |
|------|------|------|
| `train_minus_val_test.txt` | Model training | Parameter updates only |
| `val.txt` | Early stopping / tuning | Used by GBDT and deep models |
| `test_gold.txt` | Final evaluation | **Never used for training or model selection** |

The final comparison table will be regenerated **only on `test_gold.txt`**.

---

## Expected Relative Ordering (Based on Preliminary Results)

Although absolute values will change after removing leakage, the relative ordering is expected to remain:



- Gold eval size: **500,000 samples**
- Positive rate: **0.251**

---

## Key Observations

### Strong DNN Baseline
- The DNN baseline already captures a large portion of the signal.
- AUC around 0.757 indicates strong ranking performance from embeddings + MLP.

---

### DCN v1 Limitations
- DCN v1 underperforms the DNN in both logloss and AUC.
- It achieves excellent calibration but lacks expressive interaction capacity.
- This aligns with known limitations of rank-1 cross layers.

---

### DCN v2 Improvements
- DCN v2 outperforms both DNN and DCN v1.
- Gains over DNN baseline:
  - Logloss improvement of approximately **0.0028**
  - AUC improvement of approximately **0.0024**
- The improvement comes from:
  - Low-rank cross layers
  - Mixture-of-experts with gating
  - Sample-adaptive feature interactions

---

## Calibration Notes

- DCN v2 slightly over-predicts probabilities compared to the true positive rate.
- This is a common trade-off when optimizing ranking metrics.
- Calibration can be addressed independently using techniques such as temperature scaling.

---

## Final Conclusion

DCN v2 provides consistent improvements over a strong DNN baseline by capturing higher-order,
sample-dependent feature interactions.  
DCN v1, while well-calibrated, is too restrictive for large-scale CTR modeling.

For this setup, **DCN v2 is the preferred architecture**.

---

## Artifacts and Checkpoints

- All trained models are saved locally at:

