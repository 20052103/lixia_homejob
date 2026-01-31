"""
配置文件 - Qwen2.5-7B PyTorch + CUDA GPU推理
"""

# 模型配置
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MODEL_CACHE_DIR = "../../model"  # 外部模型目录

# 推理设备配置
DEVICE = "cuda"  # 使用CUDA GPU
DTYPE = "auto"   # 自动精度（float16）

# 推理参数
MAX_TOKENS = 1024
TEMPERATURE = 0.7
TOP_P = 0.95
TOP_K = 40

# 模型加载配置
USE_FLASH_ATTENTION = True  # 使用Flash Attention 2
LOAD_IN_8BIT = False        # 不使用8-bit量化
