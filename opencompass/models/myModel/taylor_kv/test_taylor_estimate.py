import torch

def precompute_k_related(k, order=2):
    """
    预计算与 k 和 k0 相关的常量部分，支持选择任意阶数的展开。
    
    参数：
    k: 含有多个向量 k_i 的矩阵，形状为 (n, d)
    order: 阶数，支持0、1、2阶
    
    返回：
    一个字典，包含与阶数相关的预计算结果
    """
    # 计算 k 的平均值作为 k_0
    k0 = torch.mean(k, dim=0)
    
    # 获取 k 的数量 n
    n = k.shape[0]
    d = k.shape[1]  # 维度
    
    # 计算差值
    k_diff = k - k0
    
    # 创建一个字典用于存储不同阶数的结果
    result_dict = {'k0': k0, 'n': n, 'order': order}
    
    # 零阶项: sum_i 1 = n
    result_dict['S0'] = n
    
    # 一阶项: sum_i (k_i - k_0)
    if order >= 1:
        result_dict['S1'] = torch.sum(k_diff, dim=0)
    
    # 二阶项: sum_i (k_i - k_0)(k_i - k_0)^T
    if order >= 2:
        # 使用更高效的方法计算外积的和
        # 注意: 这里使用矩阵乘法而不是循环，以提高效率
        S2 = torch.matmul(k_diff.t(), k_diff)
        result_dict['S2'] = S2
    
    return result_dict


def estimate_sum(q, precomputed):
    """
    给定 q 和包含预计算结果的字典，计算最终的估算结果。
    
    参数：
    q: 目标向量 q, 形状为 (d,)
    precomputed: 包含预计算结果的字典
    
    返回：
    估算的总和
    """
    k0 = precomputed['k0']
    order = precomputed['order']
    
    # 计算 q^T k_0
    q_k0 = torch.dot(q, k0)
    
    # 计算 e^(q^T k_0)
    exp_q_k0 = torch.exp(q_k0)
    
    # 初始化结果项，首先加上零阶项：n * exp(q^T k_0)
    result = precomputed['S0']
    
    # 一阶项: exp(q^T k_0) * q^T * sum_i (k_i - k_0)
    if order >= 1:
        q_S1 = torch.dot(q, precomputed['S1'])
        result += q_S1
    
    # 二阶项: (1/2) * exp(q^T k_0) * q^T * [sum_i (k_i - k_0)(k_i - k_0)^T] * q
    if order >= 2:
        # 计算二次型: q^T S2 q
        q_S2_q = torch.matmul(torch.matmul(q, precomputed['S2']), q)
        result += 0.5 * q_S2_q
    
    return result * exp_q_k0


def precompute_kv_related(k, v, order=2):
    """
    预计算与 k, v 和 k0 相关的常量部分，支持选择任意阶数的展开。
    
    参数：
    k: 含有多个向量 k_i 的矩阵，形状为 (n, d_k)
    v: 含有多个向量 v_i 的矩阵，形状为 (n, d_v)
    order: 阶数，支持0、1、2阶
    
    返回：
    一个字典，包含与阶数相关的预计算结果
    """
    # 计算 k 的平均值作为 k_0
    k0 = torch.mean(k, dim=0)
    
    # 获取 k 的数量 n
    n = k.shape[0]
    d_k = k.shape[1]  # k 的维度
    d_v = v.shape[1]  # v 的维度
    
    # 计算差值
    k_diff = k - k0
    
    # 创建一个字典用于存储不同阶数的结果
    result_dict = {'k0': k0, 'n': n, 'order': order, 'd_v': d_v}
    
    # 零阶项: sum_i v_i
    result_dict['S0_v'] = torch.sum(v, dim=0)
    
    # 一阶项: sum_i v_i (k_i - k_0)^T
    if order >= 1:
        # 使用高效矩阵乘法计算
        S1_v = torch.matmul(v.t(), k_diff)  # 形状: (d_v, d_k)
        result_dict['S1_v'] = S1_v
    
    # 二阶项: 对于每个 v 的分量，计算 sum_i v_i,j (k_i - k_0)(k_i - k_0)^T
    if order >= 2:
        # 初始化 S2_v 张量，形状为 (d_v, d_k, d_k)
        S2_v = torch.zeros(d_v, d_k, d_k, device=k.device, dtype=k.dtype)
        
        # 对于每个 v 的分量
        for j in range(d_v):
            # 对于每个样本 i，计算 v_i[j] * (k_diff[i] 的外积)
            # 使用向量化方法提高效率
            S2_v[j] = torch.matmul((v[:, j].unsqueeze(1) * k_diff).t(), k_diff)
        
        result_dict['S2_v'] = S2_v
    
    return result_dict


def estimate_sum_with_v(q, precomputed):
    """
    给定 q 和包含预计算结果的字典，计算 ∑_i e^{q^T k_i} v_i 的估算结果。
    
    参数：
    q: 目标向量 q, 形状为 (d_k,)
    precomputed: 包含预计算结果的字典
    
    返回：
    估算的总和，形状为 (d_v,)
    """
    k0 = precomputed['k0']
    order = precomputed['order']
    d_v = precomputed['d_v']
    
    # 计算 q^T k_0
    q_k0 = torch.dot(q, k0)
    
    # 计算 e^(q^T k_0)
    exp_q_k0 = torch.exp(q_k0)
    
    # 初始化结果向量，首先加上零阶项：sum_i v_i * exp(q^T k_0)
    result = precomputed['S0_v'].clone()
    
    # 一阶项: exp(q^T k_0) * [sum_i v_i (k_i - k_0)^T] q
    if order >= 1:
        # S1_v 的形状是 (d_v, d_k)，q 是 (d_k,)
        # 矩阵乘法得到形状为 (d_v,) 的向量
        q_S1_v = torch.matmul(precomputed['S1_v'], q)
        result += q_S1_v
    
    # 二阶项: (1/2) * exp(q^T k_0) * sum_i v_i (q^T (k_i - k_0))^2
    if order >= 2:
        # 对于每个 v 的分量 j，计算 q^T S2_v[j] q
        q_S2_v_q = torch.zeros(d_v, dtype=q.dtype, device=q.device)
        for j in range(d_v):
            # 计算二次型: q^T S2_v[j] q
            q_S2_v_q[j] = torch.matmul(torch.matmul(q, precomputed['S2_v'][j]), q)
        
        result += 0.5 * q_S2_v_q
    
    return result * exp_q_k0


def rand_test():
    # 示例用法：
    torch.manual_seed(42)  # 设置随机种子以确保结果可重现
    
    # 创建示例数据
    n, d_k, d_v = 5, 3, 2
    k = torch.randn(n, d_k)  # k_i 的矩阵
    v = torch.randn(n, d_v)  # v_i 的矩阵
    
    # 给定 q，目标向量 q
    q = torch.randn(1, d_k)
    
    # 计算真实的总和，∑_i e^{q^T k_i} v_i
    exponents = torch.matmul(k, q)
    exp_values = torch.exp(exponents)
    # 扩展维度以进行广播乘法
    exp_values_expanded = exp_values.unsqueeze(1).expand(-1, d_v)
    true_result = torch.sum(exp_values_expanded * v, dim=0)
    print("True result:", true_result)
    
    # 比较不同 order 的结果：
    for order in range(0, 3):  # 从 0 到 2 阶展开
            
        print(f"\nComparing for order {order}:")
        precomputed = precompute_kv_related(k, v, order)
        result = estimate_sum_with_v(q, precomputed)
        print(f"Estimated result (order {order}): {result}")
        print(f"Difference between estimated and true result: {result - true_result}")
        print(f"Relative error: {(result - true_result).abs() / true_result.abs() * 100}%")
    
    # 也测试没有 v 的情况
    print("\n" + "="*50)
    print("Testing without v (original function):")
    
    # 计算真实的总和，exp(q^T k_i) 的和
    true_result_no_v = torch.exp(torch.matmul(k, q)).sum()
    print("True result (no v):", true_result_no_v.item())
    
    for order in range(0, 3):  # 从 0 到 2 阶展开
        print(f"\nComparing for order {order}:")
        precomputed = precompute_k_related(k, order)
        result = estimate_sum(q, precomputed)
        print(f"Estimated result (order {order}): {result.item()}")
        print(f"Difference between estimated and true result: {result.item() - true_result_no_v.item()}")
        print(f"Relative error: {abs(result.item() - true_result_no_v.item()) / abs(true_result_no_v.item()) * 100}%")


def real_kv_test():
    from utils import load_qkvh
    import os
    from tqdm import tqdm
    import torch
    from transformers.models.llama.modeling_llama import repeat_kv

    exp_root = '/inspire/hdd/project/heziweiproject/liuxiaoran-240108120089/projects_zgliu/projects/huffkv/attn_analysis/result/Llama-3_2-3B/longbench_narrativeqa_42'
    layer_data_root = os.path.join(exp_root, 'layer_data')

    # 迭代处理每一层的数据
    for layer_idx, layer_qkvh_data in enumerate(load_qkvh(layer_data_root)):
        if layer_idx < 5:
            continue
        q_rope = layer_qkvh_data["q_rope"].to('cuda')
        k_rope = layer_qkvh_data["k_rope"].to('cuda')
        v = layer_qkvh_data["v"].to('cuda')

        # k_rope -= k_rope.mean(dim=-2, keepdim=True)
        
        # 可选：限制序列长度以加快测试速度
        sample_seq_len = 20
        if sample_seq_len > 0:
            q_rope = q_rope[:, :, :sample_seq_len, :]
            k_rope = k_rope[:, :, :sample_seq_len, :]
            v = v[:, :, :sample_seq_len, :]

        # 只取最后一个位置的查询
        q_rope = q_rope[..., -1:, :]

        bsz, num_heads, _, head_dim = q_rope.shape
        _, num_kv_heads, seq_len, _ = k_rope.shape
        head_group = num_heads // num_kv_heads
        
        # 重复 k 和 v 以匹配头的数量
        k_rope = repeat_kv(k_rope, head_group)
        v = repeat_kv(v, head_group)

        assert bsz == 1, f"Batch size must be 1, but got {bsz}"

        # import ipdb; ipdb.set_trace()

        # 重塑张量以便逐头处理
        q_rope, k_rope, v = q_rope.squeeze(0), k_rope.squeeze(0), v.unsqueeze(0)

        # 初始化误差统计
        errors_with_v = {order: [] for order in range(3)}
        errors_without_v = {order: [] for order in range(3)}

        # 逐头处理
        for head in range(num_heads):
            q_head = q_rope[head]  # [1, head_dim]
            k_head = k_rope[head]  # [seq_len, head_dim]
            v_head = v[head]       # [seq_len, d_v]

            import ipdb; ipdb.set_trace()

            # 计算真实注意力权重（有 v）
            # exponents = torch.matmul(k_head, q_head)
            exponents = torch.sum(k_head * q_head, dim=-1)
            exp_values = torch.exp(exponents)
            true_weights_with_v = (exp_values * v_head).sum(dim=0)
            true_weights_with_v *= torch.exp(max_exp)

            # 计算真实注意力权重（无 v）
            true_weights_without_v = exp_values.sum() * torch.exp(max_exp)

            # 测试不同阶数的近似
            for order in range(3):
                # 有 v 的近似
                precomputed_with_v = precompute_kv_related(k_head, v_head, order)
                approx_with_v = estimate_sum_with_v(q_head, precomputed_with_v)
                error_with_v = (approx_with_v - true_weights_with_v).abs().mean().item()
                errors_with_v[order].append(error_with_v)

                # 无 v 的近似
                precomputed_without_v = precompute_k_related(k_head, order)
                approx_without_v = estimate_sum(q_head, precomputed_without_v)
                error_without_v = (approx_without_v - true_weights_without_v).abs().item()
                errors_without_v[order].append(error_without_v)

        # 打印该层的平均误差
        print(f"\nLayer {layer_idx} errors (with v):")
        for order in range(3):
            avg_error = sum(errors_with_v[order]) / num_heads
            print(f"Order {order}: {avg_error:.6e}")

        print(f"Layer {layer_idx} errors (without v):")
        for order in range(3):
            avg_error = sum(errors_without_v[order]) / num_heads
            print(f"Order {order}: {avg_error:.6e}")

if __name__ == "__main__":
    rand_test()
    # real_kv_test()
    pass