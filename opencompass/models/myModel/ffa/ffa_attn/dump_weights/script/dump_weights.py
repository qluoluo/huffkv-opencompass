import os
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
    repeat_kv,
    Optional, Cache, TransformersKwargs, Unpack
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
                            attention_mask: Optional[torch.Tensor],
                            past_key_values: Optional[Cache] = None,
                            cache_position: Optional[torch.LongTensor] = None,
                            **kwargs: Unpack[TransformersKwargs],
                        ):
        # print("Enter Attn custom forward")
        # 获取层索引
        layer_idx = self.layer_idx
        print(f"Enter {layer_idx=}")
        # print(f"{hidden_states.shape=}")
        
        if hidden_states.shape[-2] == 1:
            # decode stage
            
            layer_save_dirpath = os.path.join(save_dirpath, f"layer_{layer_idx}")
            os.makedirs(layer_save_dirpath, exist_ok=True)
            
            save_fp = os.path.join(layer_save_dirpath, "attn_input.pt")
            
            torch.save(dict(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                
                q_proj_state=self.q_proj.state_dict(),
                k_proj_state=self.k_proj.state_dict(),
                v_proj_state=self.v_proj.state_dict(),
                o_proj_state=self.o_proj.state_dict(),
            ), save_fp)

            print(f"Layer {layer_idx} saved Attn input to {save_fp}")
            
            if layer_idx == self.config.num_hidden_layers - 1:
                exit()


        return self._original_forward(
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values,
                cache_position,
                **kwargs,
            )

    # 修改所有层的注意力前向传播
    for layer in model.model.layers:
        self_attention = layer.self_attn
        self_attention._original_forward = self_attention.forward
        self_attention.forward = custom_attn_forward.__get__(self_attention, type(self_attention))

    return model

if __name__ == "__main__":
    root_dir = '/inspire/hdd/project/embodied-multimodality/liuzhigeng-253108120105'
    # model_path = '/inspire/hdd/project/heziweiproject/liuxiaoran-240108120089/projects_zgliu/models/Llama-3_2-3B'
    # model_path = '/inspire/hdd/project/embodied-multimodality/liuxiaoran-240108120089/projects_zgliu/models/Llama-3_2-3B'
    model_path = os.path.join(root_dir, "models/Llama-3_2-3B")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # save_dirpath = '/inspire/hdd/project/heziweiproject/liuxiaoran-240108120089/projects_zgliu/projects/huffkv/attn_analysis/result'
    # save_dirpath = '/inspire/hdd/project/embodied-multimodality/liuxiaoran-240108120089/projects_zgliu/projects/huffKV/huffkv-opencompass/opencompass/models/myModel/bucket_attn/attn_analysis/result'

    save_dirpath = os.path.join(root_dir, "projects/ffa/huffkv-opencompass/opencompass/models/myModel/ffa/ffa_Attention/dump_weights/result")

    
    # opencompass_root_dir = '/inspire/hdd/project/heziweiproject/liuxiaoran-240108120089/projects_zgliu/projects/huffkv/huffkv-opencompass'
    # opencompass_root_dir = '/inspire/hdd/project/embodied-multimodality/liuxiaoran-240108120089/projects_zgliu/projects/huffKV/huffkv-opencompass'
    opencompass_root_dir = os.path.join(root_dir, "projects/ffa/huffkv-opencompass")

    # dataset_path = os.path.join(opencompass_root_dir, 'data/Longbench/data/narrativeqa.jsonl')
    dataset_path = os.path.join(opencompass_root_dir, 'data/Longbench/data/gov_report.jsonl')
    
    # line_idx = 50
    # raw_text, dataset_name = load_from_longbench_jsonl(dataset_path, line_idx)
    line_start, line_end = 48, 57
    raw_text, dataset_name = load_from_longbench_jsonl(dataset_path, line_start, line_end)

    repeat_text_num = 1
    if repeat_text_num > 1:
        raw_text = "\n".join([raw_text] * repeat_text_num)
        dataset_name = f"{dataset_name}_repeat{repeat_text_num}"
    
    # dataset_path = os.path.join(opencompass_root_dir, 'data/babilong/data/qa1/16k.json')
    # raw_text, dataset_name = load_from_babilong_json(dataset_path, line_idx)

    input_ids = tokenizer(raw_text, truncation=False, padding=False, return_tensors="pt").input_ids

    print(f"{input_ids.shape=}")
    # c = input("Enter c to continue...")
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
        dtype=torch.bfloat16, 
        # dtype=torch.float16, 
        device_map='auto',
        trust_remote_code=True,
    )
    model = modify_model_attn(model, save_layerdata_dirpath)

    with torch.no_grad():
        input_ids = input_ids.to(model.device)
        # model(input_ids)
        model.generate(input_ids)