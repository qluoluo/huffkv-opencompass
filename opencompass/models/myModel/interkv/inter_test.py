import os
import torch
import math

def precompute_coefficients_inter(q0, k, v):
    """
    预处理以支持插值函数
    参数:
    k: 键向量, 形状为 [..., l, d]
    v: 值向量, 形状为 [..., l, d]
    q0: 展开点, 形状为 [..., 1, d]
    """

    # 计算权重
    weights = torch.exp(torch.matmul(q0, k.transpose(-1, -2))).squeeze(-2)  # 形状: [..., l]
    qkv_base = torch.sum(weights.unsqueeze(-1) * v, dim=-2)  # 形状: [..., d]
    qkv_jacobian = torch.einsum('...l, ...l i, ...l j -> ...i j', weights, v, k)  # 形状: [..., d, d]

    qk_base = torch.sum(weights, dim=-1)  # 形状: [...]
    qk_gradient = torch.sum(weights.unsqueeze(-1) * k, dim=-2)  # 形状: [..., d]

    return {
        "q0": q0,
        "qkv_base": qkv_base,
        "qkv_jacobian": qkv_jacobian,
        "qk_base": qk_base,
        "qk_gradient": qk_gradient,
    }

def approximate_qkv(q, co_dict):
    """
    使用一阶展开近似计算 ∑_i e^{q^T k_i} v_i
    
    参数:
    q: 查询向量, 形状为 [..., 1, d]
    co_dict: 系数字典, 包含 q0, qkv_base, qkv_jacobian
    
    返回:
    近似值, 形状为 [..., d]
    """

    q0, qkv_base, qkv_jacobian = co_dict['q0'], co_dict['qkv_base'], co_dict['qkv_jacobian']
    delta = q - q0  # 形状: [..., 1, d]
    # 使用 einsum 处理任意维度
    return qkv_base + torch.einsum('...i d, ...d d -> ...i d', delta, qkv_jacobian).squeeze(-2)  # 形状: [..., d]

def approximate_qk(q, co_dict):
    """
    使用一阶展开近似计算 ∑_i e^{q^T k_i}
    
    参数:
    q: 查询向量, 形状为 [..., 1, d]
    co_dict: 系数字典, 包含 q0, qk_base, qk_gradient
    
    返回:
    近似值, 形状为 [...]
    """
    q0, qk_base, qk_gradient = co_dict['q0'], co_dict['qk_base'], co_dict['qk_gradient']
    delta = (q - q0).squeeze(-2)  # 形状: [..., d]
    return qk_base + torch.sum(qk_gradient * delta, dim=-1)  # 形状: [...]

def exact_computation_qkv(q, k, v):
    """
    精确计算向量函数值，支持任意前置维度
    用于 ∑_i e^{q^T k_i} v_i
    
    参数:
    q: 查询向量, 形状为 [..., 1, d]
    k: 键向量, 形状为 [..., l, d]
    v: 值向量, 形状为 [..., l, d]
    
    返回:
    精确值, 形状为 [..., d]
    """
    weights = torch.exp(torch.matmul(q, k.transpose(-1, -2))).squeeze(-2)  # 形状: [..., l]
    return torch.sum(weights.unsqueeze(-1) * v, dim=-2)  # 形状: [..., d]

def exact_computation_qk(q, k):
    """
    精确计算标量函数值，支持任意前置维度
    用于 ∑_i e^{q^T k_i}
    
    参数:
    q: 查询向量, 形状为 [..., 1, d]
    k: 键向量, 形状为 [..., l, d]
    
    返回:
    精确值, 形状为 [...]
    """
    weights = torch.exp(torch.matmul(q, k.transpose(-1, -2))).squeeze(-2)  # 形状: [..., l]
    return torch.sum(weights, dim=-1)  # 形状: [...]

def rand_test():
    # 设置随机种子以确保可重复性
    torch.manual_seed(44)
    
    # 参数设置
    n = 100  # 键值对数量
    d = 128   # 向量维度
    batch_size = 1  # 批次大小
    num_heads = 8     # 序列长度
    
    # 测试序列处理情况
    print("\n测试序列处理情况:")
    k = torch.randn(batch_size, num_heads, n, d)  # 键向量 [batch_size, seq_len, n, d]
    v = torch.randn(batch_size, num_heads, n, d)  # 值向量 [batch_size, seq_len, n, d]
    q = torch.randn(batch_size, num_heads, 1, d) # 展开点 [batch_size, seq_len, 1, d]

    q = q / math.sqrt(d)
    
    # 预处理系数
    co_dict = precompute_coefficients_inter(q, k, v)
    
    # 生成测试查询点
    test_q = q + 0.01 * torch.randn(batch_size, num_heads, 1, d)
    print(f'{q.flatten()[:10]=}\n{test_q.flatten()[:10]=}')
    
    exact_qkv = exact_computation_qkv(q, k, v)
    exact_qk = exact_computation_qk(q, k)
    exact_attn = exact_qkv / exact_qk.unsqueeze(-1)  # 形状: [batch_size, num_heads, d]
    
    # 近似计算
    approx_qkv = approximate_qkv(test_q, co_dict)
    approx_qk = approximate_qk(test_q, co_dict)
    approx_attn = approx_qkv / approx_qk.unsqueeze(-1)  # 形状: [batch_size, num_heads, d]

    print(f'{(exact_qk - approx_qk).flatten()[:10]=}')
    print(f'{(exact_qkv - approx_qkv).flatten()[:10]=}')
    
    # 计算误差
    qkv_error = torch.norm(exact_qkv - approx_qkv, p=2) / torch.norm(exact_qkv, p=2)
    qk_error = torch.norm(exact_qk - approx_qk, p=2) / torch.norm(exact_qk, p=2)
    attn_error = torch.norm(exact_attn - approx_attn, p=2) / torch.norm(exact_attn, p=2)
    
    print(f"QKV 相对误差: {qkv_error.item():.6f}")
    print(f"QK 相对误差: {qk_error.item():.6f}")
    print(f"注意力输出相对误差: {attn_error.item():.6f}")
    
    return {
        "exact_attn": exact_attn,
        "approx_attn": approx_attn,
        "errors": (qkv_error, qk_error, attn_error)
    }

def real_test():
    from utils import load_qkvh
    exp_root = '/inspire/hdd/project/heziweiproject/liuxiaoran-240108120089/projects_zgliu/projects/huffkv/attn_analysis/result/Llama-3_2-3B/longbench_narrativeqa_42'
    layer_data_root = os.path.join(exp_root, 'layer_data')
    from transformers.models.llama.modeling_llama import repeat_kv
    
    # Iterate through the layers and plot the attention weights and their distribution
    for layer_idx, layer_qkvh_data in enumerate(load_qkvh(layer_data_root)):

        print(f"Layer {layer_idx}")

        q_rope = layer_qkvh_data["q_rope"].to('cuda')
        k_rope = layer_qkvh_data["k_rope"].to('cuda')
        v = layer_qkvh_data["v"].to('cuda')

        bsz, num_heads, seq_len, head_dim = q_rope.shape
        _, num_kv_heads, _, _ = k_rope.shape
        head_group = num_heads // num_kv_heads
        k_rope = repeat_kv(k_rope, head_group)
        v = repeat_kv(v, head_group)

        assert bsz == 1, f"Batch size must be 1, but got {bsz}"
        
        # 总体的采样长度
        # sample_seq_len = -1
        sample_seq_len = 8 * 1024
        if sample_seq_len > 0:
            q_rope = q_rope[..., :sample_seq_len, :]
            k_rope = k_rope[..., :sample_seq_len, :]
            v = v[..., :sample_seq_len, :]

        q_rope = q_rope / math.sqrt(head_dim)

        # 截取部分作为计算验证
        val_len = 16
        q_val = q_rope[..., -val_len:, :]
        q_rope = q_rope[..., :-val_len, :]
        k_rope = k_rope[..., :-val_len, :]
        v = v[..., :-val_len, :]

        inter_len = 8
        inter_q_rope = q_rope[..., -inter_len:, :]
        
        co_dict_list = []
        for i in range(inter_len):
            now_inter_q = inter_q_rope[..., i:i+1, :]
            # print(f'{i=}, {now_inter_q.shape=}, {k_rope.shape=}, {v.shape=}')
            # import ipdb; ipdb.set_trace()
            co_dict = precompute_coefficients_inter(now_inter_q, k_rope, v)
            co_dict_list.append(co_dict)

        for i in range(val_len):
            now_q = q_val[..., i:i+1, :]
            exact_qkv = exact_computation_qkv(now_q, k_rope, v)
            exact_qk = exact_computation_qk(now_q, k_rope)
            exact_attn = exact_qkv / exact_qk.unsqueeze(-1)

            approx_qk_list = []
            approx_qkv_list = []
            for i in range(inter_len):
                approx_qkv = approximate_qkv(now_q, co_dict_list[i])
                approx_qk = approximate_qk(now_q, co_dict_list[i])
                approx_attn = approx_qkv / approx_qk.unsqueeze(-1)
                # print(f'{i=}, mean={(exact_attn - approx_attn).abs().mean()} max={(exact_attn - approx_attn).abs().max()}')

                approx_qk_list.append(approx_qk)
                approx_qkv_list.append(approx_qkv)
            
            approx_qk_mean = torch.stack(approx_qk_list).mean(dim=0)
            approx_qkv_mean = torch.stack(approx_qkv_list).mean(dim=0)
            approx_attn = approx_qkv_mean / approx_qk_mean.unsqueeze(-1)
            print(f'avg mean={(exact_attn - approx_attn).abs().mean()} max={(exact_attn - approx_attn).abs().max()}')
            print()

            # exit()
        exit()


if __name__ == "__main__":
    # rand_test()
    real_test()