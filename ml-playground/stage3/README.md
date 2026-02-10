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

## X. 显存定量剖析（Attention / FFN / Optimizer）

本节用当前真实配置对训练显存的主要来源做数量级估算，帮助理解为什么：
- Transformer 类模型比 DCNv2 更吃显存
- FT-Transformer 在 naive AdamW 下容易 OOM
- “Embedding SGD + Other AdamW” 是关键工程策略

### X.1 实验配置（真实运行口径）

- Batch size `B = 4096`
- Token count `F = 27`（26 sparse + 1 dense；若 pooling=cls 则 F=28）
- Token dim `d = 64`
- Heads `H = 4`（`d_head = d/H = 16`）
- Layers `L = 2`
- FT-Transformer FFN: `d → 4d → d`（即 `64 → 256 → 64`）
- Hash bins `= 2,000,000`
- dtype: fp32（默认）

以下估算用 fp32（4 bytes / float）。

---

### X.2 Attention 的显存热点（Activation）

#### (1) Attention map `A = softmax(QK^T)`
Attention weights 形状：
- `A`: `[B, H, F, F]`

代入数值：
- 元素数：`4096 * 4 * 27 * 27 = 11,943,936`
- 显存：`11,943,936 * 4 bytes ≈ 45.6 MB`

> 训练时不仅要存 A，还要存 softmax 的中间结果用于 backward，
> 实际通常是 **~2× 到 3×** 这个量级（实现相关）。

#### (2) Q/K/V 投影激活
每个是 `[B, F, d]`：
- 元素数：`4096 * 27 * 64 = 7,077,888`
- 单个显存：`~27.0 MB`
- Q+K+V 合计：`~81.0 MB`

#### Attention 小结（单层数量级）
- `A`（45.6MB） + `QKV`（81MB） + 其它 buffer（O, dropout, residual, ln 等）
- 单层 attention 训练激活通常落在 **~150–250 MB**（fp32，数量级）

---

### X.3 FFN 的显存热点（Activation）

FT-Transformer FFN 中间层（hidden）形状：
- `[B, F, 4d]` = `[4096, 27, 256]`

代入数值：
- 元素数：`4096 * 27 * 256 = 28,311,552`
- 显存：`28,311,552 * 4 bytes ≈ 108.0 MB`

训练时 backward 需要保存激活（以及 GELU 等中间量），
因此 FFN 实际也会接近 **~2×** 的量级（实现相关）。

#### FFN 小结（单层数量级）
- 仅 FFN hidden 就 **~108 MB**
- 加上 activation saved tensors，单层 FFN 训练激活通常是 **~150–250 MB**（fp32，数量级）

---

### X.4 Transformer 总 activation（数量级）

以 **每层 ~300–500MB** 的数量级（Attention + FFN）粗估：

- `L=2` → **~600MB – 1GB** 的训练激活量级（fp32）

> 实际显存还会叠加：
> - embedding lookup 输出 tokens
> - head MLP
> - dataloader 的 pinned buffer（若启用）
> - CUDA caching allocator 的 reserved segments / fragmentation

---

### X.5 Optimizer（最容易被忽略、但最致命）

#### (1) Embedding 参数量
使用 shared embedding table：
- 参数：`hash_bins * d = 2,000,000 * 64 = 128,000,000` floats
- fp32 参数显存：`128,000,000 * 4 bytes ≈ 512 MB`

#### (2) AdamW 的状态量（naive 情况）
Adam/AdamW 为每个参数维护：
- 参数本体 `param`
- 梯度 `grad`
- 一阶动量 `exp_avg`
- 二阶动量 `exp_avg_sq`

对 embedding 而言，单 embedding table 的理论显存（fp32）：
- `param` ~512MB
- `grad`  ~512MB
- `m`     ~512MB
- `v`     ~512MB
- 合计：**~2.0 GB**

再叠加 foreach/multi-tensor kernel 的临时 buffer + allocator reserve，
很容易在 `optimizer.ste

