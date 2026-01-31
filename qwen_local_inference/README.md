# Qwen2.5-7B 本地推理助手

一个基于PyTorch和CUDA GPU的Qwen2.5-7B本地推理应用，提供温柔女性化的Tkinter GUI界面。

## ✨ 特性

- **GPU加速推理**: 使用NVIDIA RTX 5090进行高速推理
- **本地运行**: 无需云服务，完全离线使用
- **温柔UI**: 采用柔和紫粉色配色的女性化界面
- **多轮对话**: 支持上下文感知的对话历史
- **快速响应**: 2-3秒生成一次回复

## 🔧 系统要求

### 硬件
- **GPU**: NVIDIA RTX 5090（或其他支持CUDA的GPU）
- **VRAM**: 最少16GB（推荐32GB+）
- **RAM**: 最少8GB系统内存
- **存储**: 模型占用约15GB空间

### 软件
- **OS**: Windows 10/11
- **Python**: 3.11+
- **CUDA**: 12.8+（已集成在PyTorch中）

## 📦 环境配置

### 1. 虚拟环境设置

```bash
# 进入项目目录
cd d:\repo\lixia_homejob\qwen_local_inference

# 虚拟环境已创建：venv_311
# 之后运行Python时使用虚拟环境中的python：
.\venv_311\Scripts\python.exe
```

### 2. PyTorch配置

**已安装版本**: PyTorch 2.9.1+cu128

如果需要重新安装：
```bash
.\venv_311\Scripts\pip.exe install torch==2.9.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

**验证CUDA支持**:
```bash
.\venv_311\Scripts\python.exe -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'PyTorch: {torch.__version__}')"
```

### 3. 必要依赖

已安装的核心包：
- torch 2.9.1+cu128
- transformers 5.0.0+
- accelerate
- safetensors
- numpy
- tkinter (Python内置)

## 📂 项目结构

```
qwen_local_inference/
├── main.py                 # 应用入口
├── config.py              # 配置文件
├── model_manager.py       # 模型加载管理
├── inference_engine.py    # GPU推理引擎
├── ui.py                  # Tkinter GUI界面
├── venv_311/              # Python虚拟环境
└── README.md              # 本文件

模型存储位置：
D:\repo\model\
└── models--Qwen--Qwen2.5-7B-Instruct/
```

## 🚀 使用方法

### 启动应用

```bash
cd d:\repo\lixia_homejob\qwen_local_inference
.\venv_311\Scripts\python.exe main.py
```

应用会：
1. 检查CUDA和GPU信息
2. 加载Qwen2.5-7B-Instruct模型（首次加载较慢）
3. 打开Tkinter GUI窗口

### 使用GUI

1. **输入问题**: 在下方文本框输入您的问题
2. **发送消息**: 
   - 点击"💕 发送"按钮
   - 或按 `Ctrl+Enter` 快捷键
3. **清空对话**: 点击"🗑️ 清空"按钮清除历史记录
4. **退出应用**: 点击"👋 再见"或关闭窗口

### 配置调整

编辑 `config.py` 修改以下参数：

```python
# 模型配置
MODEL_CACHE_DIR = "../../model"  # 模型存储位置

# 推理参数
MAX_TOKENS = 1024              # 最大生成长度
TEMPERATURE = 0.7              # 生成温度（0-1）
TOP_P = 0.95                   # 核采样参数
TOP_K = 40                      # Top-K采样

# 模型加载配置
USE_FLASH_ATTENTION = False     # Flash Attention开关
```

## ⚙️ 配置说明

### PyTorch配置
- **Version**: 2.9.1+cu128（支持RTX 5090的SM_120架构）
- **Device**: cuda（自动使用GPU）
- **Dtype**: float16（节省显存）

### 推理参数
- **Temperature**: 越高越随机（0.7较平衡）
- **Top P**: 核采样，控制回复多样性
- **Top K**: 限制候选词汇数量
- **Max Tokens**: 单次生成的最大长度

## 🔍 故障排除

### CUDA不可用
```bash
# 检查CUDA状态
.\venv_311\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"

# 结果应该是：True
```

### 模型加载失败
```bash
# 检查模型路径
# 确保 D:\repo\model\models--Qwen--Qwen2.5-7B-Instruct\ 存在

# 如果路径不对，修改 config.py 中的 MODEL_CACHE_DIR
```

### 内存不足
- 减少 `MAX_TOKENS`
- 关闭其他应用释放内存
- 使用 `TOP_K` 和 `TOP_P` 减少计算量

### 响应很慢
- 首次推理需要编译CUDA kernel，后续会快速
- 确保 CUDA 可用
- 检查GPU是否被其他程序占用

## 📊 性能指标

在RTX 5090上的测试结果：
- **模型加载**: 1-2秒
- **单次推理**: 2-3秒（生成128个token）
- **显存使用**: 约14.19GB（float16精度）
- **精度**: float16（fp16）

## 🛠️ 开发信息

### 依赖版本
- PyTorch: 2.9.1+cu128
- Transformers: 5.0.0+
- CUDA: 12.8+
- Python: 3.11+

### 关键代码说明

**model_manager.py**: 负责模型加载和设备信息
```python
# 使用device_map="auto"自动分配GPU/CPU内存
# 使用dtype=torch.float16节省显存
```

**inference_engine.py**: GPU推理引擎
```python
# 维持对话历史
# 使用HuggingFace chat template格式化消息
# torch.no_grad()优化推理
```

**ui.py**: Tkinter GUI界面
```python
# 柔和配色：#f5f0f8（背景）、#d984d9（按钮）
# 后台线程处理推理，避免UI卡顿
# 支持Ctrl+Enter快捷键
```

## 📝 更新日志

### 2026-01-30
- ✅ 恢复Qwen2.5-7B PyTorch GPU推理
- ✅ 安装PyTorch 2.9.1+cu128
- ✅ 禁用Flash Attention（flash_attn未安装）
- ✅ 修复torch_dtype废弃警告
- ✅ 模型移至D:\repo\model（外部存储）
- ✅ 清理项目文件（删除GGUF相关）
- ✅ 推送代码到GitHub

## 📧 支持

如有问题，请检查：
1. CUDA和GPU是否可用
2. 模型路径是否正确
3. 虚拟环境是否激活
4. Python版本是否为3.11+

---

**作者**: Lixia  
**最后更新**: 2026-01-30  
**Python**: 3.11  
**PyTorch**: 2.9.1+cu128
