import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm
import json
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import *

from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    apply_rotary_pos_emb,
    repeat_kv
)

# from transformers.models.qwen2.modeling_qwen2 import (
#     Qwen2Attention,
#     apply_rotary_pos_emb,
#     repeat_kv
# )


def modify_model_attn(model, save_dirpath):
    """
    修改模型的前向传播以捕获注意力模式
    """
    def custom_attn_forward(self, 
                            hidden_states: torch.Tensor,
                            position_embeddings: tuple[torch.Tensor, torch.Tensor],
                            *args, **kwargs):
        # print("Enter Attn custom forward")

        # 获取层索引
        layer_idx = self.layer_idx
        layer_save_dirpath = os.path.join(save_dirpath, f"layer_{layer_idx}")
        os.makedirs(layer_save_dirpath, exist_ok=True)
        
        # 准备注意力计算
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        torch.save(hidden_states, os.path.join(layer_save_dirpath, "h.pt"))
        torch.save(query_states, os.path.join(layer_save_dirpath, "q_unrope.pt"))
        torch.save(key_states, os.path.join(layer_save_dirpath, "k_unrope.pt"))
        torch.save(value_states, os.path.join(layer_save_dirpath, "v.pt"))

        # 应用位置编码
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        torch.save(query_states, os.path.join(layer_save_dirpath, "q_rope.pt"))
        torch.save(key_states, os.path.join(layer_save_dirpath, "k_rope.pt"))

        print(f"Layer {layer_idx} saved qkvhv for shape {query_states.shape=} {key_states.shape=}")

        # 处理键值重复
        # rep_nums = query_states.shape[1] // key_states.shape[1]
        # key_states = repeat_kv(key_states, rep_nums)

        return self._original_forward(hidden_states=hidden_states,
                                    position_embeddings=position_embeddings,
                                    *args, **kwargs)

    # 修改所有层的注意力前向传播
    for layer in model.model.layers:
        self_attention = layer.self_attn
        self_attention._original_forward = self_attention.forward
        self_attention.forward = custom_attn_forward.__get__(self_attention, type(self_attention))

    return model

if __name__ == "__main__":
    model_path = '/inspire/hdd/project/heziweiproject/liuxiaoran-240108120089/projects_zgliu/models/Llama-3_2-3B'
    save_dirpath = '/inspire/hdd/project/heziweiproject/liuxiaoran-240108120089/projects_zgliu/projects/huffkv/attn_analysis/result'


    # from utils import load_from_longbench_jsonl

    line_idx = 46
    opencompass_doot_dir = '/inspire/hdd/project/heziweiproject/liuxiaoran-240108120089/projects_zgliu/projects/huffkv/huffkv-opencompass'
    
    # dataset_path = os.path.join(opencompass_doot_dir, 'data/Longbench/data/narrativeqa.jsonl')
    dataset_path = os.path.join(opencompass_doot_dir, 'data/Longbench/data/gov_report.jsonl')
    raw_text, dataset_name = load_from_longbench_jsonl(dataset_path, line_idx)

    # dataset_path = os.path.join(opencompass_doot_dir, 'data/babilong/data/qa1/16k.json')
    # raw_text, dataset_name = load_from_babilong_json(dataset_path, line_idx)
    
    save_dirpath = os.path.join(save_dirpath, os.path.basename(model_path), dataset_name)
    os.makedirs(save_dirpath, exist_ok=True)

    raw_text_savefp = os.path.join(save_dirpath, "raw_text.txt")
    with open(raw_text_savefp, 'w') as f:
        f.write(raw_text)

    save_layerdata_dirpath = os.path.join(save_dirpath, "layer_data")
    os.makedirs(save_layerdata_dirpath, exist_ok=True)
    
    # 加载模型和分词器
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        device_map='auto',
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 修改模型以捕获注意力模式
    model = modify_model_attn(model, save_layerdata_dirpath)

    

    input_ids = tokenizer(raw_text, truncation=False, padding=False, return_tensors="pt").input_ids.to(model.device)

    with torch.no_grad():
        model(input_ids)