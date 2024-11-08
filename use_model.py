# -*- coding: utf-8 -*-
"""
@Time ： 10/31/24 1:42 PM
@Auth ： yangmingxing
@File ：use_model.py
@IDE ：PyCharm
@Motto:no bug
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.utils._pytree")

# 指定合并后的模型路径
merged_model_path = "./output/DeepSeek_full"

# 加载模型
# model = AutoModelForCausalLM.from_pretrained(merged_model_path, torch_dtype=torch.float16, device_map="auto")
# INT8 量化是减少模型显存占用的常见方法，将模型权重从 FP32 降到 INT8。可显著减少 GPU 内存需求，性能下降通常也在可接受范围内。
model = AutoModelForCausalLM.from_pretrained(merged_model_path, load_in_8bit=True, device_map="auto")

if hasattr(model, 'tie_weights'):
    model.tie_weights()

tokenizer = AutoTokenizer.from_pretrained(merged_model_path)

# 使用模型生成文本示例
input_text = '''
###重要信息
 -你是一个幽默且知书达理的女性，请直接以女性的角度回复，注意角色不要混乱，你是女性，你是女性，你是女性，用户和你是男朋友关系，你会积极对用户调侃，长度20字。

 User:今天好累啊'''



inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

# 生成
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,  # 可调整生成长度
        do_sample=True,
        top_p=0.95,
        temperature=0.7,
        num_return_sequences=1
    )

# 解码生成的文本
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("生成的文本:", generated_text)

