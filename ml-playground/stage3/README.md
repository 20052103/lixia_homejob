# Stage 3 — Attention / Transformer Models for CTR (Criteo DAC)

## 目标

Stage 3 的目标是在 **完全复用 Stage2 的数据集、feature artifact 与 gold eval 口径** 的前提下，
系统性探索 **Attention / Transformer 类模型** 在 CTR 场景下的：

- 表达能力
- 训练稳定性
- 显存与工程成本

本阶段以 **学习导向** 为主，而非面试导向。

---

## 1. 实验设置（与 Stage2 完全对齐）

| Item | Value |
|---|---|
| Dataset | Criteo DAC |
| Dense features | 13 |
| Categorical features | 26 |
| Hash bins | 2,000,000 |
| Token dimension (`d_token`) | 64 |
| Batch size | 4096 |
| Train steps / epoch | 2000 |
| Eval | validation + **gold eval (full)** |
| Metrics | logloss, AUC |

### 数据与模型路径

- Dataset  
  `D:/repo/large data/criteo/dac`
- Model checkpoints & runs  
  `D:/repo/large data/model`

---

## 2. Feature 处理与口径说明（复用 Stage2）

- Categorical features  
  - Hash 到 `[0, hash_bins)`  
  - Stage2 生成的是 **global id（field_offset + local_id）**
  - Stage3 中统一通过  
    ```python
    sparse = sparse % hash_bins
    ```
    转换为 **per-field local id**

- Dense features  
  - 使用 `log1p` 变换
  - 与 Stage2 完全一致

- Feature artifacts  
  - 直接复用 Stage2：
    ```python
    from stage2_ctr.utils.feature_artifacts import build_feature_artifacts
    ```

---

## 3. 模型对照（结构层面）

| Model | Core Idea | Interaction Modeling | Normalization | FFN | Pooling |
|---|---|---|---|---|---|
| **DCNv2** (Stage2) | Explicit feature crossing | CrossNet (low-rank) + MLP | BN / None | MLP | N/A |
| **AutoInt** | Attention over fields | Self-Attention (field-wise) | Post-LN | 2×d | Mean |
| **FT-Transformer** | Tabular Transformer | Self-Attention + Deep FFN | **Pre-LN** | **4×d (GELU)** | CLS / Mean |

---

## 4. 维度与计算复杂度（真实配置）

| Item | DCNv2 | AutoInt | FT-Transformer |
|---|---|---|---|
| Tokenization | ❌ | ✔ | ✔ |
| Number of tokens | ❌ | 27 | 27 |
| Attention map | ❌ | `[B,4,27,27]` | `[B,4,27,27]` |
| Transformer layers | ❌ | 2 | 2 |
| Token dimension | ❌ | 64 | 64 |
| FFN hidden size | ❌ | 128 | **256** |

---

## 5. 显存与工程成本分析

| Aspect | DCNv2 | AutoInt | FT-Transformer |
|---|---|---|---|
| Embedding size | Large | Large | Large |
| Attention memory | ❌ | High | High |
| FFN activation | ❌ | Medium | **High** |
| Optimizer state (Adam) | Medium | High | **Very High (naive)** |
| OOM risk | Low | Medium | **High without optimizer split** |
| Training stability | High | Medium | **High** |

### Optimizer 设计（关键工程决策）

Stage3 统一采用 **分组优化器**：

- **Embedding parameters** → `SGD`
- **Other parameters** → `AdamW`

该策略显著降低了 embedding 的 optimizer state 显存占用，
并解决了 FT-Transformer 在大 batch 下的 OOM 问题。

---

## 6. 实验结果（Gold Eval）

> ⚠️ 所有模型均使用 **同一份 gold eval 文件**，结果可直接横向比较。

| Model | Gold Logloss ↓ | Notes |
|---|---|---|
| **DCNv2** (Stage2 baseline) | ~0.52 | Strong explicit-cross baseline |
| **AutoInt** | ~0.47–0.48 | Attention over fields |
| **FT-Transformer** | **0.4677** | Best so far, stable |

示例（FT-Transformer）：
