"""
推理引擎 - 处理模型推理和对话历史
"""

import torch
from config import MAX_TOKENS, TEMPERATURE, TOP_P, TOP_K


class InferenceEngine:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.conversation_history = []
    
    def generate(self, user_message):
        """生成响应"""
        # 添加用户消息到对话历史
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # 使用聊天模板格式化对话
        text = self.tokenizer.apply_chat_template(
            self.conversation_history,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 分词
        model_inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        # 生成响应
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
                do_sample=True
            )
        
        # 解码响应
        response = self.tokenizer.decode(
            generated_ids[0][model_inputs.input_ids.shape[-1]:],
            skip_special_tokens=True
        )
        
        # 添加助手响应到对话历史
        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })
        
        return response.strip()
    
    def clear_history(self):
        """清除对话历史"""
        self.conversation_history = []
