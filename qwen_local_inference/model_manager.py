"""
模型管理器 - 加载和管理Qwen2.5-7B模型
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import MODEL_NAME, MODEL_CACHE_DIR, DEVICE, DTYPE, USE_FLASH_ATTENTION


class ModelManager:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._ensure_cache_dir()
    
    def _ensure_cache_dir(self):
        """确保模型缓存目录存在"""
        os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    
    def download_model(self):
        """下载并加载模型"""
        try:
            print(f"📥 从HuggingFace加载 {MODEL_NAME}...")
            
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME,
                cache_dir=MODEL_CACHE_DIR,
                trust_remote_code=True
            )
            
            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                cache_dir=MODEL_CACHE_DIR,
                device_map="auto",
                dtype=torch.float16 if DTYPE == "auto" else DTYPE,
                trust_remote_code=True,
                attn_implementation="flash_attention_2" if USE_FLASH_ATTENTION else None
            )
            
            print("✅ 模型加载完成")
            return True
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return False
    
    def get_model(self):
        return self.model
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def get_device_info(self):
        """获取设备信息"""
        info = {
            "device": DEVICE,
            "cuda_available": torch.cuda.is_available(),
            "torch_version": torch.__version__,
            "model_dtype": str(self.model.dtype) if self.model else "N/A"
        }
        
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_total"] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
            info["sm_capability"] = f"SM_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}"
        
        return info
