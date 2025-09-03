import torch

# 统计量预处理函数
def process_statistics(k):
    """
    计算 k_i 的均值（假设为零）和协方差矩阵。
    
    :param k: Tensor, shape (n, d), k_i 的数据矩阵，n 是样本数量，d 是特征维度
    :return: 协方差矩阵
    """
    # 假设 k_i 的均值为零，无需计算均值
    k_centered = k  # 直接使用零均值，即 k - 0
    cov_k = torch.matmul(k_centered.T, k_centered) / k.size(0)
    
    return cov_k

# 泰勒展开估计函数
def taylor_expansion_estimate(k, q):
    """
    基于统计量计算泰勒展开的估计。
    
    :param k: Tensor, shape (n, d), k_i 的数据矩阵
    :param q: Tensor, shape (d,), q 向量
    :return: 泰勒展开估计的结果
    """
    # 计算统计量（仅协方差矩阵）
    cov_k = process_statistics(k)
    
    # 初始化 e^{q^T k_i^0} 这一项，假设 k0 为零时的指数部分
    exp_term = torch.exp(torch.matmul(torch.zeros_like(k[0]), q))  # e^{q^T k0}，k0为零
    
    # 第一项是 e^{q^T k0}
    first_term = exp_term * k.size(0)  # 对每个 k_i 求和，乘以样本数量
    
    # 第二项，由于均值为零，第一阶项消失
    second_term = torch.tensor(0.0)  # 第一阶项消失
    
    # 第三项：二阶项，基于协方差矩阵和 q 的二阶影响
    # 这里将 q 转换为列向量形式，避免使用过时的 T 操作
    second_order_term = 0.5 * torch.matmul(torch.matmul(q, cov_k), q.T) * exp_term
    
    # 组合所有项
    total_estimate = first_term + second_term + second_order_term
    
    return total_estimate

# 计算精确的结果
def exact_sum(k, q):
    """
    计算精确的总和：sum(e^{q^T k_i})。
    
    :param k: Tensor, shape (n, d), k_i 的数据矩阵
    :param q: Tensor, shape (d,), q 向量
    :return: 精确的总和
    """
    return torch.sum(torch.exp(torch.matmul(k, q)))

# 示例使用
if __name__ == "__main__":
    # 假设 k 是一个 100x10 的张量，表示有 100 个 10 维的样本
    n, d = 100, 10
    k = torch.randn(n, d)  # 随机生成 k_i 样本数据，确保 k 是 n x d 的形状
    k = k - k.mean(dim=0, keepdim=True)
    
    # 假设 q 是一个 10 维的向量
    q = torch.randn(d)  # 随机生成 q 向量，确保 q 是 d 维的
    
    # 计算泰勒展开估计
    estimate = taylor_expansion_estimate(k, q)
    
    # 计算精确的总和
    exact = exact_sum(k, q)
    
    # 计算误差
    error = torch.abs(exact - estimate)
    
    print("Exact Sum:", exact)
    print("Taylor Expansion Estimate:", estimate)
    print("Error:", error)
