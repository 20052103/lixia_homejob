"""
主程序入口 - 初始化和启动应用（PyTorch + CUDA GPU）
"""

import os
import tkinter as tk
from tkinter import messagebox
import sys
from pathlib import Path

# 禁用Hugging Face symlink警告（Windows不支持）
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

from model_manager import ModelManager
from inference_engine import InferenceEngine
from ui import QwenGUI
from config import MODEL_NAME, DEVICE


def main():
    """主程序入口"""
    
    print("=" * 60)
    print("[启动] Qwen2.5-7B 本地推理系统 (PyTorch + CUDA GPU)")
    print("=" * 60)
    
    # 初始化模型管理器
    model_manager = ModelManager()
    
    # 显示设备信息
    device_info = model_manager.get_device_info()
    print(f"\n[系统信息]")
    print(f"  - 推理设备: {device_info['device'].upper()}")
    print(f"  - CUDA可用: {device_info['cuda_available']}")
    print(f"  - 模型精度: {device_info['model_dtype']}")
    
    if device_info['cuda_available']:
        print(f"  - GPU型号: {device_info.get('gpu_name', 'N/A')}")
        print(f"  - GPU显存: {device_info.get('gpu_memory_total', 'N/A')}")
        print(f"  - SM能力: {device_info.get('sm_capability', 'N/A')}")
    
    print(f"  - PyTorch版本: {device_info['torch_version']}")
    print(f"  - 模型: {MODEL_NAME}\n")
    
    # 创建Tkinter根窗口
    root = tk.Tk()
    root.withdraw()  # 先隐藏窗口
    
    try:
        # 显示加载提示
        messagebox.showinfo(
            "加载模型",
            "正在加载Qwen2.5-7B...\n\n"
            "首次运行会自动下载模型\n"
            "这可能需要10-20分钟，请耐心等待..."
        )
        
        # 加载模型
        print("🔄 正在加载模型...")
        if not model_manager.download_model():
            messagebox.showerror("错误", "模型加载失败，请检查网络和内存")
            root.destroy()
            return
        
        # 初始化推理引擎
        print("🔄 正在初始化推理引擎...")
        model = model_manager.get_model()
        tokenizer = model_manager.get_tokenizer()
        inference_engine = InferenceEngine(model, tokenizer)
        
        print("✅ 模型和推理引擎初始化完成\n")
        
        # 显示窗口并创建GUI
        root.deiconify()
        gui = QwenGUI(root, inference_engine)
        
        # 欢迎信息
        gui.display_message(
            "系统",
            "欢迎使用Qwen2.5-7B本地推理助手！\n"
            "在下方输入您的问题，我会为您提供帮助。\n"
            "按 Ctrl+Enter 发送消息。",
            "system"
        )
        
        print("=" * 60)
        print("✅ 应用启动完成，窗口已打开")
        print("=" * 60)
        
        # 启动GUI主循环
        root.mainloop()
    
    except Exception as e:
        messagebox.showerror("错误", f"应用启动失败:\n{str(e)}")
        root.destroy()
        sys.exit(1)


if __name__ == "__main__":
    main()
