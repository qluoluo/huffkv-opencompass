import os
from pathlib import Path
import torch
import numpy as np
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
        
        # import ipdb; ipdb.set_trace()

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

        print(f"Layer {layer_idx} saved qkvh for shape {query_states.shape=} {key_states.shape=} dtype={query_states.dtype}")

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
    # root_dir = '/inspire/hdd/project/embodied-multimodality/liuzhigeng-253108120105'
    # root_dir = '/inspire/hdd/project/exploration-topic/liuzhigeng-253108120105'
    root_dir = Path("/inspire/qb-ilm/project/exploration-topic/liuzhigeng-253108120105")
    
    # model_path = os.path.join(root_dir, "models/Llama-3_2-3B")
    model_path = Path("/inspire/hdd/global_user/liuzhigeng-253108120105/models/Llama-3.1-8B")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    save_dirpath = os.path.join(root_dir, "projects/ffa/huffkv-opencompass/opencompass/models/myModel/ffa/attn_analysis/result")
    
    # opencompass_root_dir = '/inspire/hdd/project/heziweiproject/liuxiaoran-240108120089/projects_zgliu/projects/huffkv/huffkv-opencompass'
    # opencompass_root_dir = '/inspire/hdd/project/embodied-multimodality/liuxiaoran-240108120089/projects_zgliu/projects/huffKV/huffkv-opencompass'
    opencompass_root_dir = os.path.join(root_dir, "projects/ffa/huffkv-opencompass")

    # dataset_path = os.path.join(opencompass_root_dir, 'data/Longbench/data/narrativeqa.jsonl')
    dataset_path = os.path.join(opencompass_root_dir, 'data/Longbench/data/gov_report.jsonl')
    
    # line_idx = 50
    # raw_text, dataset_name = load_from_longbench_jsonl(dataset_path, line_idx)
    line_start, line_end = 48, 68
    raw_text, dataset_name = load_from_longbench_jsonl(dataset_path, line_start, line_end)

    repeat_text_num = 1
    if repeat_text_num > 1:
        raw_text = "\n".join([raw_text] * repeat_text_num)
        dataset_name = f"{dataset_name}_repeat{repeat_text_num}"
    
    # dataset_path = os.path.join(opencompass_root_dir, 'data/babilong/data/qa1/16k.json')
    # raw_text, dataset_name = load_from_babilong_json(dataset_path, line_idx)

    input_ids = tokenizer(raw_text, truncation=False, padding=False, return_tensors="pt").input_ids
    
    sample_len_k = 256
    # sample_len_k = -1
    sample_len = sample_len_k * 1024
    if sample_len_k > 0 and input_ids.shape[-1] >= sample_len:
        print(f"cut {input_ids.shape[-1]//1024}k to {sample_len_k}k length..")
        input_ids = input_ids[..., :sample_len]
        dataset_name = f"{dataset_name}_{sample_len_k}k"

    print(f"{input_ids.shape=}")
    
    # c = input("Enter c to continue...\n")
    # if c.lower().strip() != 'c':
    #     exit()

    save_dirpath = os.path.join(save_dirpath, os.path.basename(model_path), dataset_name)
    os.makedirs(save_dirpath, exist_ok=True)
    raw_text_savefp = os.path.join(save_dirpath, "raw_text.txt")
    with open(raw_text_savefp, 'w') as f:
        f.write(raw_text)
    save_layerdata_dirpath = os.path.join(save_dirpath, "layer_data")
    os.makedirs(save_layerdata_dirpath, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        # torch_dtype=torch.bfloat16, 
        # dtype=torch.bfloat16, 
        dtype=torch.float16, 
        device_map='auto',
        trust_remote_code=True,
    )
    model = modify_model_attn(model, save_layerdata_dirpath)

    with torch.no_grad():
        input_ids = input_ids.to(model.device)
        model(input_ids)