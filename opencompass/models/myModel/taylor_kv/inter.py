from sympy import jacobi
import os, torch
import torch.nn as nn
import math

def precompute_coefficients(k, v, q0):
    """
    预处理系数函数，支持任意前置维度
    
    参数:
    k: 键向量, 形状为 [..., l, d]
    v: 值向量, 形状为 [..., l, d]
    q0: 展开点, 形状为 [..., 1, d]
    
    返回:
    base: 基准值, 形状为 [..., d]
    jacobian: 雅可比矩阵, 形状为 [..., d, d]
    """
    # 计算权重
    weights = torch.exp(torch.matmul(q0, k.transpose(-1, -2))).squeeze(-2)  # 形状: [..., l]
    
    # 计算基准值
    base = torch.sum(weights.unsqueeze(-1) * v, dim=-2)  # 形状: [..., d]
    
    # 计算雅可比矩阵
    # 使用 einsum 进行高效计算，支持任意前置维度
    # jacobian = torch.einsum('...l, ...ld, ...ld -> ...d d', weights, v, k)  # 形状: [..., d, d]
    jacobian = torch.einsum('...l, ...l i, ...l j -> ...i j', weights, v, k)  # 形状: [..., d, d]
    
    return base, jacobian

def approximate(q, base, jacobian, q0):
    """
    使用一阶展开近似计算函数值，支持任意前置维度
    
    参数:
    q: 查询向量, 形状为 [..., 1, d]
    base: 基准值, 形状为 [..., d]
    jacobian: 雅可比矩阵, 形状为 [..., d, d]
    q0: 展开点, 形状为 [..., 1, d]
    
    返回:
    近似值, 形状为 [..., d]
    """
    delta = q - q0  # 形状: [..., 1, d]
    # 使用 einsum 处理任意维度
    return base + torch.einsum('...i d, ...d d -> ...i d', delta, jacobian).squeeze(-2)  # 形状: [..., d]

def exact_computation(q, k, v):
    """
    精确计算函数值，支持任意前置维度
    
    参数:
    q: 查询向量, 形状为 [..., 1, d]
    k: 键向量, 形状为 [..., l, d]
    v: 值向量, 形状为 [..., l, d]
    
    返回:
    精确值, 形状为 [..., d]
    """
    weights = torch.exp(torch.matmul(q, k.transpose(-1, -2))).squeeze(-2)  # 形状: [..., l]
    return torch.sum(weights.unsqueeze(-1) * v, dim=-2)  # 形状: [..., d]

# if __name__ == "__main__":
def rand_test():
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    
    # 参数设置
    n = 100  # 键值对数量
    d = 128   # 向量维度
    batch_size = 1  # 批次大小
    num_heads = 8     # 序列长度
    
    # 测试序列处理情况
    print("\n测试序列处理情况:")
    k_seq = torch.randn(batch_size, num_heads, n, d)  # 键向量 [batch_size, seq_len, n, d]
    v_seq = torch.randn(batch_size, num_heads, n, d)  # 值向量 [batch_size, seq_len, n, d]
    q0_seq = torch.randn(batch_size, num_heads, 1, d) # 展开点 [batch_size, seq_len, 1, d]

    q0_seq = q0_seq / math.sqrt(d)
    
    # 预处理系数
    base_seq, jacobian_seq = precompute_coefficients(k_seq, v_seq, q0_seq)
    
    # 生成测试查询点
    test_points_seq = q0_seq + 0.001 * torch.randn(batch_size, num_heads, 1, d)
    
    # 精确计算
    exact_val_seq = exact_computation(test_points_seq, k_seq, v_seq)
    
    # 近似计算
    approx_val_seq = approximate(test_points_seq, base_seq, jacobian_seq, q0_seq)
    
    print(f'{exact_val_seq.shape=}, {approx_val_seq.shape=}')
    print(f'{exact_val_seq=}, {approx_val_seq=}')

    print(f'{(exact_val_seq - approx_val_seq).abs().mean()=}')
    print(f'{(exact_val_seq - approx_val_seq).abs().max()=}')

    # # 计算误差
    # abs_error_seq = torch.norm(exact_val_seq - approx_val_seq, dim=-1)
    # rel_error_seq = abs_error_seq / torch.norm(exact_val_seq, dim=-1)
    
    # print(f"序列处理平均绝对误差: {abs_error_seq.mean().item():.6f}")
    # print(f"序列处理平均相对误差: {rel_error_seq.mean().item():.6f}")

def real_test():
    exp_root = '/inspire/hdd/project/heziweiproject/liuxiaoran-240108120089/projects_zgliu/projects/huffkv/attn_analysis/result/Llama-3_2-3B/longbench_narrativeqa_42'
    layer_data_root = os.path.join(exp_root, 'layer_data')
    from transformers.models.llama.modeling_llama import repeat_kv
    
    # Iterate through the layers and plot the attention weights and their distribution
    for layer_idx, layer_qkvh_data in enumerate(load_qkvh(layer_data_root)):
        q_rope = layer_qkvh_data["q_rope"].to('cuda')
        k_rope = layer_qkvh_data["k_rope"].to('cuda')
        v = layer_qkvh_data["v"].to('cuda')
        
        # sample_seq_len = -1
        sample_seq_len = 8 * 1024
        if sample_seq_len > 0:
            q_rope = q_rope[..., :sample_seq_len, :]
            k_rope = k_rope[..., :sample_seq_len, :]

        val_len = 16
        q_val = q_rope[..., -val_len:, :]
        q_rope = q_rope[..., :-val_len, :]

        inter_len = 1024
        inter_q_rope = q_rope[..., -inter_len:, :]
        base_list, jacobi_list = [], []
        for i in range(inter_len):
            now_inter_q = inter_q_rope[..., i:i+1, :]
            base, jacobian = precompute_coefficients(k_rope, v, now_inter_q)
            base_list.append(base)
            jacobi_list.append(jacobian)

        q_rope = q_rope[..., -1:, :]  # Focusing on the last position (query part)

        bsz, num_heads, seq_len, head_dim = q_rope.shape
        _, num_kv_heads, _, _ = k_rope.shape
        head_group = num_heads // num_kv_heads
        k_rope = repeat_kv(k_rope, head_group)
        v = repeat_kv(v, head_group)

        assert bsz == 1, f"Batch size must be 1, but got {bsz}"




if __name__ == "__main__":
    rand_test()