"""
用户界面 - Tkinter GUI with 温柔的女性化风格
"""

import tkinter as tk
from tkinter import scrolledtext, messagebox
import threading
from config import DEVICE


class QwenGUI:
    def __init__(self, root, inference_engine):
        self.root = root
        self.engine = inference_engine
        self.root.title("✨ Qwen2.5 温柔助手 ✨")
        self.root.geometry("900x700")
        
        # 配置颜色和字体
        self.bg_color = "#f5f0f8"      # 柔和紫粉色背景
        self.btn_color = "#d984d9"     # 柔和紫色按钮
        self.accent_color = "#e8a8e8"  # 柔和粉紫色
        self.text_color = "#5a4a6a"    # 暖色深色文字
        self.root.config(bg=self.bg_color)
        
        self._create_widgets()
    
    def _create_widgets(self):
        """创建界面组件"""
        # 标题
        title_frame = tk.Frame(self.root, bg=self.bg_color)
        title_frame.pack(pady=10)
        
        title_label = tk.Label(
            title_frame,
            text="✨ Qwen2.5 温柔助手 ✨",
            font=("微软雅黑", 18, "bold"),
            fg=self.text_color,
            bg=self.bg_color
        )
        title_label.pack()
        
        subtitle_label = tk.Label(
            title_frame,
            text="让我陪你聊天～",
            font=("微软雅黑", 10),
            fg=self.accent_color,
            bg=self.bg_color
        )
        subtitle_label.pack()
        
        # 消息显示区域
        self.message_display = scrolledtext.ScrolledText(
            self.root,
            width=100,
            height=20,
            font=("微软雅黑", 10),
            bg="#fff5f7",
            fg=self.text_color,
            state="disabled",
            relief="flat"
        )
        self.message_display.pack(padx=15, pady=10, fill="both", expand=True)
        
        # 输入框
        input_frame = tk.Frame(self.root, bg=self.bg_color)
        input_frame.pack(padx=15, pady=5, fill="x")
        
        tk.Label(
            input_frame,
            text="你的问题:",
            font=("微软雅黑", 10),
            fg=self.text_color,
            bg=self.bg_color
        ).pack(side="left")
        
        self.input_text = tk.Text(
            self.root,
            height=3,
            font=("微软雅黑", 10),
            bg="#fdf3ff",
            fg=self.text_color,
            relief="flat"
        )
        self.input_text.pack(padx=15, fill="x")
        self.input_text.bind("<Control-Return>", self._on_send)
        
        # 按钮框
        button_frame = tk.Frame(self.root, bg=self.bg_color)
        button_frame.pack(pady=10)
        
        send_btn = tk.Button(
            button_frame,
            text="💕 发送",
            command=self._on_send,
            font=("微软雅黑", 10, "bold"),
            bg=self.btn_color,
            fg="white",
            relief="flat",
            padx=20,
            pady=8
        )
        send_btn.pack(side="left", padx=5)
        
        clear_btn = tk.Button(
            button_frame,
            text="🗑️ 清空",
            command=self._on_clear,
            font=("微软雅黑", 10, "bold"),
            bg=self.accent_color,
            fg="white",
            relief="flat",
            padx=20,
            pady=8
        )
        clear_btn.pack(side="left", padx=5)
        
        exit_btn = tk.Button(
            button_frame,
            text="👋 再见",
            command=self.root.quit,
            font=("微软雅黑", 10, "bold"),
            bg="#c97fc9",
            fg="white",
            relief="flat",
            padx=20,
            pady=8
        )
        exit_btn.pack(side="left", padx=5)
        
        # 状态栏
        self.status_label = tk.Label(
            self.root,
            text="✅ 已准备好",
            font=("微软雅黑", 9),
            fg=self.accent_color,
            bg=self.bg_color
        )
        self.status_label.pack(pady=5)
    
    def display_message(self, sender, message, msg_type="chat"):
        """显示消息"""
        self.message_display.config(state="normal")
        
        if msg_type == "system":
            self.message_display.insert("end", f"🎀 {message}\n\n")
        elif sender == "user":
            self.message_display.insert("end", f"👤 你: {message}\n")
        else:
            self.message_display.insert("end", f"✨ 助手: {message}\n")
        
        self.message_display.insert("end", "━" * 50 + "\n")
        self.message_display.see("end")
        self.message_display.config(state="disabled")
    
    def _on_send(self, event=None):
        """发送消息处理"""
        user_input = self.input_text.get("1.0", "end-1c").strip()
        if not user_input:
            return
        
        self.display_message("user", user_input)
        self.input_text.delete("1.0", "end")
        self.status_label.config(text="⏳ 思考中...")
        
        # 在后台线程中运行推理
        thread = threading.Thread(target=self._generate_response, args=(user_input,))
        thread.daemon = True
        thread.start()
    
    def _generate_response(self, user_input):
        """生成响应（后台线程）"""
        try:
            response = self.engine.generate(user_input)
            self.root.after(0, self.display_message, "assistant", response)
        except Exception as e:
            error_msg = f"错误: {str(e)}"
            self.root.after(0, self.display_message, "system", error_msg)
        finally:
            self.root.after(0, lambda: self.status_label.config(text="✅ 已准备好"))
    
    def _on_clear(self):
        """清空对话"""
        self.engine.clear_history()
        self.message_display.config(state="normal")
        self.message_display.delete("1.0", "end")
        self.message_display.config(state="disabled")
        self.display_message("system", "对话已清空，我们重新开始吧～")
