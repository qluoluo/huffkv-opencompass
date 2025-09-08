import torch

def precompute_taylor_approx(keys, values, order=2, k0=None):
    """
    预计算泰勒展开近似所需的统计量（支持0阶、1阶和2阶）
    
    参数:
        keys: 形状为 (N, d_k) 的张量，表示键向量
        values: 形状为 (N, d_v) 的张量，表示值向量
        order: 泰勒展开的阶数 (0, 1, 或 2)
        k0: 可选，形状为 (d_k,) 的张量，表示展开点。如果为None，则使用键的均值
    
    返回:
        包含预计算结果的字典，包括阶数信息和相应的统计量
    """
    if order not in [0, 1, 2]:
        raise ValueError("order must be 0, 1, or 2")
    
    N, d_k = keys.shape
    d_v = values.shape[1]
    
    # 确定展开点
    if k0 is None:
        k0 = torch.mean(keys, dim=0)
    
    # 初始化结果字典
    precomputed = {
        'order': order,
        'k0': k0,
        'd_k': d_k,
        'd_v': d_v
    }
    
    # 计算偏移量
    offsets = keys - k0.unsqueeze(0)  # 形状: (N, d_k)
    
    # 0阶项 (常数项)
    if order >= 0:
        S_v = torch.sum(values, dim=0)  # 形状: (d_v,)
        precomputed['S_v'] = S_v
    
    # 1阶项 (线性项)
    if order >= 1:
        # M_v = Σ_i v_i (k_i - k0)^T
        M_v = torch.sum(values.unsqueeze(2) * offsets.unsqueeze(1), dim=0)  # 形状: (d_v, d_k)
        precomputed['M_v'] = M_v
    
    # 2阶项 (二次项)
    if order >= 2:
        # 为每个值维度创建一个矩阵
        M_list = []
        for j in range(d_v):
            # 对于每个值维度j，计算 M_j = Σ_i v_{i,j} * (k_i - k0) * (k_i - k0)^T
            v_j = values[:, j]  # 形状: (N,)
            M_j = torch.sum(v_j.unsqueeze(1).unsqueeze(2) * 
                           offsets.unsqueeze(2) * offsets.unsqueeze(1), dim=0)
            M_list.append(M_j)
        
        # 将列表转换为张量 (d_v, d_k, d_k)
        M_tensor = torch.stack(M_list, dim=0)
        precomputed['M_tensor'] = M_tensor
    
    return precomputed

def compute_taylor_approx(q, precomputed):
    """
    使用预计算的统计量计算泰勒展开近似
    
    参数:
        q: 形状为 (d_k,) 的张量，表示查询向量
        precomputed: 预计算结果的字典，包含阶数信息和相应的统计量
    
    返回:
        近似结果，形状为 (d_v,) 的张量
    """
    order = precomputed['order']
    k0 = precomputed['k0']
    
    # 计算指数项
    a = torch.dot(q, k0)
    exp_a = torch.exp(a)
    
    # 初始化结果
    result = precomputed['S_v']  # 0阶项
    
    # 添加1阶项（如果适用）
    if order >= 1:
        b = torch.matmul(precomputed['M_v'], q)
        result += b
    
    # 添加2阶项（如果适用）
    if order >= 2:
        d_v = precomputed['d_v']
        M_tensor = precomputed['M_tensor']
        
        # 计算二阶项
        c = torch.zeros(d_v, device=q.device)
        for j in range(d_v):
            M_j = M_tensor[j]
            c[j] = torch.matmul(q, torch.matmul(M_j, q))
        
        result += 0.5 * c
    
    # 最后一起乘以指数项
    result *= exp_a
    
    return result

# 示例使用
def rand_test():
    # 设置随机种子以确保可重复性
    torch.manual_seed(43)
    
    # 创建示例数据
    N, d_k, d_v = 1000, 64, 64
    keys = torch.randn(N, d_k)
    values = torch.randn(N, d_v)
    q = torch.randn(d_k) / 8
    
    # 测试不同阶数的近似
    orders = [0, 1, 2]
    results = {}
    
    # 计算精确值（用于比较）
    print("开始计算精确值...")
    exact_result = torch.zeros(d_v)
    for i in range(N):
        exact_result += torch.exp(torch.dot(q, keys[i])) * values[i]
    print("精确计算完成")
    
    for order in orders:
        print(f"\n=== 测试 {order} 阶近似 ===")
        
        # 预计算
        print("开始预计算...")
        precomputed = precompute_taylor_approx(keys, values, order=order)
        print("预计算完成")
        
        # 计算近似值
        print("开始计算近似值...")
        approx_result = compute_taylor_approx(q, precomputed)
        print("近似计算完成")
        
        # 计算相对误差
        error = torch.norm(approx_result - exact_result) / torch.norm(exact_result)
        print(f"相对误差: {error.item():.6f}")
        
        # 存储结果
        results[order] = {
            'approx': approx_result,
            'error': error
        }
    
    # 打印比较结果
    print("\n=== 不同阶数近似结果比较 ===")
    print(f"精确值的范数: {torch.norm(exact_result).item():.4f}")
    
    for order in orders:
        error = results[order]['error'].item()
        print(f"{order}阶近似相对误差: {error:.6f}")
    
    # 打印前5个维度的值比较
    print("\n前5个维度的值比较:")
    print("维度\t精确值\t\t0阶\t\t1阶\t\t2阶")
    for i in range(5):
        exact_val = exact_result[i].item()
        order0_val = results[0]['approx'][i].item()
        order1_val = results[1]['approx'][i].item()
        order2_val = results[2]['approx'][i].item()
        
        print(f"{i}\t{exact_val:.4f}\t\t{order0_val:.4f}\t\t{order1_val:.4f}\t\t{order2_val:.4f}")

if __name__ == "__main__":
    rand_test()