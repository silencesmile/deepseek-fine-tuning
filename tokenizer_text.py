# -*- coding: utf-8 -*-
"""
@Time ： 10/29/24 5:14 PM
@Auth ： 公众号：阿三先生
@File ：111.py
@IDE ：PyCharm
@Motto:no bug
"""
import tokenizer
import torch
import pandas as pd
from datasets import Dataset

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained('./deepseek-ai/deepseek-llm-7b-chat/', use_fast=False, trust_remote_code=True)
tokenizer.padding_side = 'right' # padding在右边

'''
Lora训练的数据是需要经过格式化、编码之后再输入给模型进行训练的，如果是熟悉Pytorch模型训练流程的同学会知道，
我们一般需要将输入文本编码为input_ids，将输出文本编码为labels，编码之后的结果都是多维的向量。
'''

def process_func(example):
  MAX_LENGTH = 384  # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
  input_ids, attention_mask, labels = [], [], []
  instruction = tokenizer(f"User: {example['instruction'] + example['input']}\n\n",
                          add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
  response = tokenizer(f"Assistant: {example['output']}<｜end▁of▁sentence｜>", add_special_tokens=False)
  input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
  attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
  labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
  if len(input_ids) > MAX_LENGTH:  # 做一个截断
    input_ids = input_ids[:MAX_LENGTH]
    attention_mask = attention_mask[:MAX_LENGTH]
    labels = labels[:MAX_LENGTH]
  return {
    "input_ids": input_ids,
    "attention_mask": attention_mask,
    "labels": labels
  }

def get_tokenized_id(json_file):
  df = pd.read_json(json_file)
  ds = Dataset.from_pandas(df)

  # 处理数据集
  tokenized_id = ds.map(process_func, remove_columns=ds.column_names)
  # print(tokenized_id)

  return tokenized_id


if __name__ == '__main__':
  # 处理数据集
  json_file = './data.json'
  tokenized_id = get_tokenized_id(json_file)
  print(tokenized_id)

  # 数据检查
  result = tokenizer.decode(tokenized_id[0]['input_ids'])

  result2 = tokenizer.decode(list(filter(lambda x: x != -100, tokenized_id[0]["labels"])))
  print(result, result2)

  # text = {
  #     "instruction": "现在你要扮演皇帝身边的女人--甄嬛",
  #     "input": "你是谁？",
  #     "output": "家父是大理寺少卿甄远道。"
  # }
  #
  #
  # result = process_func(text)
  # print(result)

