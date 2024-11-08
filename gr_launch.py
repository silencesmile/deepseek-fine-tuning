# -*- coding: utf-8 -*-
"""
@Time ： 10/29/24 5:14 PM
@Auth ： 公众号：阿三先生
@File ：gr.py
@IDE ：PyCharm
@Motto:no bug
"""
import gradio as gr
import torch
import json
import time

from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
#from log_info import set_no_infos

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.utils._pytree")

# 加载合并后的模型和分词器
#model_path = "./output/merged_model"  # 替换为合并后的模型路径

model_path = "./output/DeepSeek_full2"
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
# 节省资源
#model = AutoModelForCausalLM.from_pretrained(model_path, load_in_8bit=True, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)


def set_no_infos(log, log_file):
  with open(log_file, "a") as file:
    date = datetime.now()
    text_info = f'\n{date} $ {log}'
    file.write(text_info)

default_persona = '''
###重要信息
 -你是一个幽默且知书达理的女性，请直接以女性的角度回复，注意角色不要混乱，你是女性，你是女性，你是女性，用户和你是男朋友关系，你会积极对用户调侃，长度20字。'''

default_input = 'User:今天好累啊'

# 定义生成文本的函数
def generate_text(input1, input2):
    s = time.time()

    prompt = f"User: {input1}\n {input2} \n\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    #inputs = tokenizer(prompt, return_tensors="pt")


    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            #do_sample=True,
            top_p=0.95,
            top_k=1,
            temperature=0.95,
            max_time=5.0
            #early_stopping=True,
            #num_beams=3,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    #print(f"{datetime.now()} -- diff time:{time.time() - s}")

    event_time = time.time() - s

    prompt_respones = {
           'prompt': prompt,
           'response': generated_text,
           'diff_time':f"{event_time:.2f}"
            }

    # set_no_infos(json.dumps(prompt_respones, ensure_ascii=False), "./logs/chat.log")

    content_text = generated_text[len(prompt): ]
    return content_text

# 创建 Gradio 接口
with gr.Blocks() as demo:
    gr.Markdown("## 文本生成应用")
    
    input1 = gr.Textbox(label="人设信息(可自行修改)", value=default_persona)
    input2 = gr.Textbox(label="聊天内容(可自行修改)", value=default_input)
    output = gr.Textbox(label="模型输出信息")

    generate_button = gr.Button("生成对话")
    generate_button.click(fn=generate_text, inputs=[input1, input2], outputs=output)

# 启动 Gradio 应用
#demo.launch()
demo.launch(server_name="0.0.0.0", server_port=6006,share=True)

