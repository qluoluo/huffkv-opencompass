import os
import torch
import gc
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import json
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
import glob

from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    apply_rotary_pos_emb,
    repeat_kv
)

def clear_kv_cache(model):
    """Clear KV cache from all layers"""
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        for layer in model.model.layers:
            if hasattr(layer, 'self_attn'):
                # Clear past_key_value if it exists
                if hasattr(layer.self_attn, 'past_key_values'):
                    layer.self_attn.past_key_values = None
                
                # Clear any cached states
                if hasattr(layer.self_attn, '_cache'):
                    layer.self_attn._cache = None
    
    # Also clear model-level cache if it exists
    if hasattr(model, 'past_key_values'):
        model.past_key_values = None
    
    # Force garbage collection
    gc.collect()
    torch.cuda.empty_cache()

def compute_attention_stats(attn_weights):
    """
    Compute statistics for attention weights
    attn_weights: tensor of shape [batch, num_heads, seq_len, seq_len]
    Returns dict with stats for each head
    """
    stats = {}
    batch_size, num_heads, seq_len, _ = attn_weights.shape
    
    for head_idx in range(num_heads):
        head_attn = attn_weights[0, head_idx].float()  # [seq_len, seq_len]
        head_attn_flat = head_attn.flatten()
        
        # Sort for percentile calculations
        sorted_attn, _ = torch.sort(head_attn_flat, descending=True)
        
        head_stats = {
            'max': head_attn_flat.max().item(),
            'mean': head_attn_flat.mean().item(),
            'min': head_attn_flat.min().item(),
            'top1%': sorted_attn[:max(1, len(sorted_attn) // 100)].mean().item(),
            'top5%': sorted_attn[:max(1, len(sorted_attn) // 20)].mean().item(),
            'top10%': sorted_attn[:max(1, len(sorted_attn) // 10)].mean().item(),
            'top20%': sorted_attn[:max(1, len(sorted_attn) // 5)].mean().item(),
            'top50%': sorted_attn[:max(1, len(sorted_attn) // 2)].mean().item(),
        }
        
        stats[f'head_{head_idx}'] = head_stats
    
    return stats

def modify_model_attn(model, attention_stats):
    """
    修改模型的前向传播以捕获注意力模式和统计信息
    """
    def custom_attn_forward(self, 
                            hidden_states: torch.Tensor,
                            position_embeddings: tuple[torch.Tensor, torch.Tensor],
                            attention_mask=None,
                            *args, **kwargs):
        
        # 获取层索引
        layer_idx = self.layer_idx
        # layer_save_dirpath = os.path.join(save_dirpath, f"layer_{layer_idx}")
        # os.makedirs(layer_save_dirpath, exist_ok=True)
        
        # 准备注意力计算
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # 应用位置编码
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # 计算注意力权重
        key_states_repeated = repeat_kv(key_states, self.num_key_value_groups)
        
        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states_repeated.transpose(2, 3)) / np.sqrt(self.head_dim)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # # don not Apply softmax !
        # attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # 计算并保存统计信息
        stats = compute_attention_stats(attn_weights)
        attention_stats[f'layer_{layer_idx}'].append(stats)
        
        # print(f"Layer {layer_idx} processed attention for shape {query_states.shape}")

        # 调用原始 forward
        return self._original_forward(hidden_states=hidden_states,
                                    position_embeddings=position_embeddings,
                                    attention_mask=attention_mask,
                                    *args, **kwargs)

    # 修改所有层的注意力前向传播
    for layer in model.model.layers:
        self_attention = layer.self_attn
        self_attention._original_forward = self_attention.forward
        self_attention.forward = custom_attn_forward.__get__(self_attention, type(self_attention))

    return model

def reset_attention_stats(model):
    """Reset the hooks to clear attention stats"""
    for layer in model.model.layers:
        self_attention = layer.self_attn
        if hasattr(self_attention, '_original_forward'):
            self_attention.forward = self_attention._original_forward

def load_jsonl(file_path):
    """Load data from jsonl file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

if __name__ == "__main__":
    root_dir = '/inspire/hdd/project/heziweiproject/heziwei-25044'

    model_path = os.path.join(root_dir, "public/download_ckpts/meta-llama_Llama-3.2-3B")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    save_dirpath = os.path.join(root_dir, "projects_zyning/huffkv-opencompass/opencompass/models/myModel/ffa/attn_analysis/result")
    
    opencompass_root_dir = os.path.join(root_dir, "projects_zyning/huffkv-opencompass")

    data_dir = os.path.join(opencompass_root_dir, 'data/LongBench/data')
    
    # 获取所有 jsonl 文件
    jsonl_files = glob.glob(os.path.join(data_dir, "*.jsonl"))
    print(f"Found {len(jsonl_files)} jsonl files")
    
    sample_len_k = 64
    sample_len = sample_len_k * 1024

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        dtype=torch.bfloat16, 
        device_map='auto',
        trust_remote_code=True,
    )
    
    # 处理每个 jsonl 文件
    for jsonl_file in tqdm(jsonl_files, desc="Processing files"):
        dataset_name = os.path.splitext(os.path.basename(jsonl_file))[0]
        print(f"\n{'='*50}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*50}")
        
        # 加载数据
        data = load_jsonl(jsonl_file)
        print(f"Loaded {len(data)} samples from {dataset_name}")
        
        # 处理每个样本
        for idx, sample in enumerate(tqdm(data, desc=f"Processing {dataset_name}")):
            # 从样本中提取文本（根据 LongBench 的格式调整）
            # 通常 LongBench 的格式包含 'context' 或 'input' 字段
            if 'context' in sample:
                raw_text = sample['context']
            elif 'input' in sample:
                raw_text = sample['input']
            elif 'text' in sample:
                raw_text = sample['text']
            else:
                # 如果没有明确的文本字段，尝试组合可能的字段
                raw_text = " ".join([str(v) for k, v in sample.items() if isinstance(v, str)])
            
            # Tokenize
            input_ids = tokenizer(raw_text, truncation=False, padding=False, return_tensors="pt").input_ids
            
            # 限制长度
            original_len_k = input_ids.shape[-1] // 1024
            if sample_len_k > 0 and input_ids.shape[-1] >= sample_len:
                input_ids = input_ids[..., :sample_len]
                len_suffix = f"{sample_len_k}k"
            else:
                len_suffix = f"{original_len_k}k"
            
            print(f"\nSample {idx}: {input_ids.shape[-1]} tokens ({len_suffix})")
            
            # 创建保存目录
            sample_save_dirpath = os.path.join(
                save_dirpath, 
                os.path.basename(model_path), 
                dataset_name,
                f"sample_{idx}_{len_suffix}"
            )
            os.makedirs(sample_save_dirpath, exist_ok=True)
            
            # 保存原始文本
            with open(os.path.join(sample_save_dirpath, "raw_text.txt"), 'w', encoding='utf-8') as f:
                f.write(raw_text)
            
            # 保存样本元数据
            with open(os.path.join(sample_save_dirpath, "metadata.json"), 'w') as f:
                json.dump({
                    'sample_idx': idx,
                    'dataset': dataset_name,
                    'original_length': input_ids.shape[-1],
                    'length_k': len_suffix,
                }, f, indent=2)
            
            # save_layerdata_dirpath = os.path.join(sample_save_dirpath, "layer_data")
            # os.makedirs(save_layerdata_dirpath, exist_ok=True)
            
            # 初始化统计信息存储
            attention_stats = defaultdict(list)
            
            # 修改模型
            model = modify_model_attn(model, attention_stats)
            
            # 运行模型
            with torch.no_grad():
                input_ids = input_ids.to(model.device)
                try:
                    model(input_ids, use_cache=False)
                except Exception as e:
                    print(f"Error processing sample {idx}: {e}")
                    continue
            
            # 保存统计信息
            summary_stats = {}
            for layer_name, stats_list in attention_stats.items():
                summary_stats[layer_name] = stats_list
            
            with open(os.path.join(sample_save_dirpath, "all_layers_attn_stats.json"), 'w') as f:
                json.dump(summary_stats, f, indent=2)
            
            # 重置模型（移除 hooks）
            reset_attention_stats(model)
            
            # 清理显存
            torch.cuda.empty_cache()
            clear_kv_cache(model)
            
            print(f"Saved attention statistics to {sample_save_dirpath}")
    
    print("\n" + "="*50)
    print("All datasets processed!")
    print("="*50)
