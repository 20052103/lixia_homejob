# 🎯 ML-Playground: From Zero to CTR/MTML

从零开始用 PyTorch 手写训练代码，逐步体验和掌握深度学习在表格数据和推荐系统中的应用

## 🎯 总体目标

从 零开始用 PyTorch 手写训练代码，逐步体验和掌握：

- Tabular / CTR / 推荐系统的 **完整训练流程**
- Embedding + Feature Cross（MLP / FM / DeepFM / DCN）
- MTML（多任务学习）：shared-bottom / 多 loss
- GPU 训练 & 吞吐调优
- 工业级 CTR / Ranking 的核心建模思想
- **不依赖高层训练框架**（如 Lightning、DeepCTR），所有关键逻辑自己写。

## 🧱 技术栈（固定）

- Python 3.10 (conda)
- PyTorch 2.9.1 + cu128
- GPU: NVIDIA RTX 5090 (SM_120)
- numpy / pandas / scikit-learn
- tqdm

## 📁 项目结构（推荐）

```
ml-playground/
├── README.md
├── stage1_adult/
│   ├── data/
│   │   ├── adult.data
│   │   └── adult.test
│   ├── 01_load_and_peek.py
│   ├── 02_preprocess.py
│   ├── 03_train_mtml.py
│   └── adult_stage1.npz
│
├── stage2_ctr/
│   ├── data/
│   ├── 01_make_subset.py
│   ├── 02_dataloader.py
│   ├── models/
│   │   ├── dnn.py
│   │   ├── deepfm.py
│   │   └── dcn.py
│   └── train_ctr.py
│
└── envs/
    └── mtml.yaml
```

---

## 🟢 Stage 0：环境与 GPU 验证（已完成）

### 目标

搭建长期可复用的 ML/GPU 开发环境

确认 Blackwell (SM_120) 被 PyTorch 原生识别

### 关键检查

```python
torch.cuda.is_available()        # True
torch.cuda.get_device_name(0)    # RTX 5090
torch.cuda.get_device_capability(0)  # (12, 0)
```

### 产出

- 稳定的 conda env: `mtml`
- PyTorch 2.9.1 + cu128 正常工作

---

## 🟢 Stage 1：Adult Census Income（MTML + Embedding）

### 🎯 目标

- 从 **原始 CSV 到 GPU 训练**
- 体验 **多任务学习（MTML）**
- 练习 **categorical embedding + dense feature**

### 📊 数据

- **UCI Adult Dataset**
- ~32k samples
- 6 numeric + 8 categorical features
- label 不平衡（>50K ≈ 24%）

### 🧠 任务定义

- **Task A（主任务）**：收入是否 >50K（二分类）
- **Task B（辅助任务）**：是否已婚（从特征派生）

### 🏗️ 模型结构

```
[numeric features]
[categorical features → embedding]
            ↓
        concat
            ↓
     shared MLP backbone
        /            \
   head A           head B
 (income)         (married)
```

**Shared-bottom MTML**
- 两个 BCEWithLogits loss
- 加权 loss：L = wA * LA + wB * LB

### 📂 脚本说明

#### `01_load_and_peek.py`
- 读取 CSV
- 基础 EDA / 缺失检查

#### `02_preprocess.py`
- train/val/test split
- 标准化 numeric
- categorical → index + OOV
- 生成 `adult_stage1.npz`

#### `03_train_mtml.py`
- PyTorch Dataset / DataLoader
- GPU 训练
- AUC 评估（sklearn）

### ✅ 你学到什么

- 从 0 写 tabular DNN
- embedding 的真实用法
- MTML 的 trade-off（主任务 vs 辅助任务）
- GPU batch / DataLoader 基础调优

---

## 🟡 Stage 2：CTR 预测（Criteo / DeepFM / DCN）

### 🎯 目标

- 进入工业级 CTR 场景
- 学会 **显式特征交叉**
- 对比不同结构的建模能力

### 📊 数据

- **Criteo Display Ads**（或等价 CTR 数据）
- 特征：
  - 13 dense
  - 26 sparse（高基数）

### 🔧 特征处理

- **dense**：log1p / 标准化
- **sparse**：
  - hash trick（如 2^20）
  - embedding lookup
  - 流式 DataLoader（避免一次性读大文件）

### 🧠 模型逐步升级

#### Baseline DNN
```
concat → MLP → CTR
```

#### DeepFM
```
FM（二阶交叉）
     +
   Deep MLP
```

#### DCN v1
```
Cross Network（显式高阶）
     +
    MLP
```

### 📏 评估指标

- LogLoss
- AUC

### ✅ 你学到什么

- 为什么 FM / DCN 在 CTR 有优势
- 显式 vs 隐式 feature cross
- embedding + hash 的工业做法
- CTR 训练的真实工程形态

---

## 🔵 Stage 3（可选进阶）

- MMoE / PLE（多任务 CTR）
- Focal Loss / class reweight
- 更大数据量 + 吞吐对比
- 推理 latency / batch size 影响

---

## 🧠 核心原则

✅ 不依赖黑盒框架

✅ 所有关键逻辑可读、可改、可 debug

✅ 优先"理解结构"，再追 SOTA
