#!/usr/bin/env python3
"""
GPU压力测试工具 - 可指定显存利用率

使用方法:
  python gpu_stress.py                    # 默认80%显存利用率，所有GPU
  python gpu_stress.py -m 60              # 60%显存利用率，所有GPU  
  python gpu_stress.py -m 90 -g 0,1       # 90%显存利用率，仅GPU 0和1
  python gpu_stress.py --memory 50 --gpu 2 # 50%显存利用率，仅GPU 2

监控命令:
  nvidia-smi -l 1                         # 实时监控
  kill $(cat /tmp/gpu_stress_main.pid)    # 停止程序
"""

import torch
import time
from datetime import datetime
import os
import signal
import psutil
import numpy as np

def get_gpu_memory():
    """获取每个GPU的显存使用情况"""
    torch.cuda.synchronize()
    return [torch.cuda.memory_allocated(i)/1024**2 for i in range(torch.cuda.device_count())]

def get_gpu_utilization():
    """获取GPU使用率"""
    try:
        import pynvml
        pynvml.nvmlInit()
        utilization = []
        for i in range(torch.cuda.device_count()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            utilization.append(util.gpu)
        return utilization
    except:
        return [0] * torch.cuda.device_count()

def stress_gpu(gpu_id, matrix_size=8192, sleep_time=0, memory_usage_percentage=80):
    """在指定GPU上运行矩阵乘法运算，可指定显存利用率"""
    torch.cuda.set_device(gpu_id)
    
    # 获取GPU显存信息
    gpu_memory_total = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**2  # MB
    gpu_memory_reserved = torch.cuda.memory_reserved(gpu_id) / 1024**2  # MB
    available_memory = gpu_memory_total - gpu_memory_reserved - 512  # 预留512MB作为缓冲
    
    # 根据指定百分比计算目标显存使用量
    target_memory_usage = available_memory * (memory_usage_percentage / 100.0)
    
    print(f'GPU {gpu_id} 总显存: {gpu_memory_total:.2f} MB, 可用显存: {available_memory:.2f} MB')
    print(f'目标显存利用率: {memory_usage_percentage}% ({target_memory_usage:.2f} MB)')
    
    # 动态计算最优矩阵大小
    bytes_per_element = 4  # float32
    
    # 计算矩阵配置
    num_main_matrices = max(6, int(memory_usage_percentage / 10))  # 根据百分比调整矩阵数量
    num_temp_matrices = max(3, int(memory_usage_percentage / 20))
    total_matrices = num_main_matrices + num_temp_matrices
    
    memory_per_matrix = (target_memory_usage * 1024**2) / total_matrices  # 字节
    elements_per_matrix = memory_per_matrix / bytes_per_element
    optimal_size = int(np.sqrt(elements_per_matrix))
    
    # 确保矩阵大小合理
    optimal_size = max(min(optimal_size, 16384), 1024)  # 限制在1024-16384之间
    
    print(f'GPU {gpu_id} 使用矩阵大小: {optimal_size}x{optimal_size}')
    print(f'矩阵配置: {num_main_matrices}个主矩阵 + {num_temp_matrices}个临时矩阵')
    
    # 创建矩阵
    matrices = []
    for i in range(num_main_matrices):
        matrix = torch.randn(optimal_size, optimal_size, dtype=torch.float32, device=f'cuda:{gpu_id}')
        matrices.append(matrix)
    
    # 创建额外的大矩阵用于持续计算
    large_matrix_a = torch.randn(optimal_size, optimal_size, dtype=torch.float32, device=f'cuda:{gpu_id}')
    large_matrix_b = torch.randn(optimal_size, optimal_size, dtype=torch.float32, device=f'cuda:{gpu_id}')
    
    # 创建临时计算空间
    temp_results = [torch.empty(optimal_size, optimal_size, dtype=torch.float32, device=f'cuda:{gpu_id}') 
                   for _ in range(num_temp_matrices)]
    
    print(f'GPU {gpu_id} 初始显存占用: {get_gpu_memory()[gpu_id]:.2f} MB')
    
    # 创建状态文件
    pid = os.getpid()
    status_file = f'/tmp/gpu_stress_{gpu_id}_{pid}.status'
    with open(status_file, 'w') as f:
        f.write('running')
    
    iteration_count = 0
    start_time = time.time()
    
    try:
        while True:
            # 检查状态文件是否存在
            if not os.path.exists(status_file):
                print(f'GPU {gpu_id} 状态文件被删除，程序退出')
                break
                
            # 执行密集的矩阵运算
            for i in range(len(matrices)-1):
                temp_idx = i % len(temp_results)
                
                # 矩阵乘法运算链
                torch.matmul(matrices[i], matrices[i+1], out=temp_results[temp_idx])
                temp_results[temp_idx] = torch.relu(temp_results[temp_idx])
                temp_results[temp_idx] = torch.sigmoid(temp_results[temp_idx])
                temp_results[temp_idx] = torch.tanh(temp_results[temp_idx])
                
                # 与大型矩阵运算
                temp_result2 = torch.matmul(large_matrix_a, temp_results[temp_idx])
                matrices[i] = torch.add(temp_results[temp_idx], temp_result2)
            
            # 额外的持续计算负载
            for _ in range(3):
                temp = torch.matmul(large_matrix_a, large_matrix_b)
                large_matrix_a = torch.relu(temp)
                large_matrix_b = torch.sigmoid(temp)
            
            iteration_count += 1
            
            # 每1000次迭代记录一次状态
            if iteration_count % 1000 == 0:
                current_time = time.time()
                elapsed = current_time - start_time
                if elapsed >= 60:  # 每分钟打印一次状态
                    mem_used = get_gpu_memory()[gpu_id]
                    mem_percentage = (mem_used / gpu_memory_total) * 100
                    util = get_gpu_utilization()[gpu_id]
                    print(f'[{datetime.now()}] GPU {gpu_id} - 显存: {mem_used:.2f}MB ({mem_percentage:.1f}%), 使用率: {util}%, 迭代: {iteration_count}')
                    start_time = current_time
            
            # 只在必要时同步
            if iteration_count % 5000 == 0:
                torch.cuda.synchronize(gpu_id)
                
    finally:
        # 清理状态文件
        if os.path.exists(status_file):
            os.remove(status_file)

def main():
    import argparse
    
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='GPU压力测试工具 - 可指定显存利用率')
    parser.add_argument('--memory', '-m', type=int, default=80, 
                       help='显存利用率百分比 (10-95), 默认80%%')
    parser.add_argument('--gpu', '-g', type=str, default='all',
                       help='指定GPU ID (0,1,2...) 或 "all" 使用所有GPU, 默认all')
    
    args = parser.parse_args()
    
    # 验证参数
    if args.memory < 10 or args.memory > 95:
        print('❌ 显存利用率必须在10-95%之间')
        return
    
    # 获取可用的GPU数量
    num_gpus = torch.cuda.device_count()
    print(f'发现 {num_gpus} 个GPU设备')
    
    # 确定要使用的GPU
    if args.gpu == 'all':
        target_gpus = list(range(num_gpus))
    else:
        try:
            target_gpus = [int(x.strip()) for x in args.gpu.split(',')]
            # 验证GPU ID
            for gpu_id in target_gpus:
                if gpu_id >= num_gpus or gpu_id < 0:
                    print(f'❌ GPU ID {gpu_id} 无效，可用GPU: 0-{num_gpus-1}')
                    return
        except ValueError:
            print(f'❌ GPU参数格式错误，使用格式: 0,1,2 或 all')
            return
    
    print(f'📊 配置信息:')
    print(f'   - 目标显存利用率: {args.memory}%')
    print(f'   - 使用GPU: {target_gpus}')
    
    # 保存主进程PID
    with open('/tmp/gpu_stress_main.pid', 'w') as f:
        f.write(str(os.getpid()))
    
    # 在指定GPU上启动进程
    child_pids = []
    for gpu_id in target_gpus:
        pid = os.fork()
        if pid == 0:  # 子进程
            try:
                print(f'在 GPU {gpu_id} 上启动压力测试 (显存目标: {args.memory}%)')
                stress_gpu(gpu_id, memory_usage_percentage=args.memory)
            except KeyboardInterrupt:
                print(f'GPU {gpu_id} 测试终止')
            finally:
                os._exit(0)
        else:
            child_pids.append(pid)
    
    print(f'所有GPU进程已启动，主进程PID: {os.getpid()}')
    print('要停止程序，请运行: kill $(cat /tmp/gpu_stress_main.pid)')
    
    try:
        # 主进程等待所有子进程
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print('收到终止信号，正在清理...')
    finally:
        # 清理所有子进程
        for pid in child_pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except:
                pass
        # 清理PID文件
        if os.path.exists('/tmp/gpu_stress_main.pid'):
            os.remove('/tmp/gpu_stress_main.pid')
        print('程序已终止')

if __name__ == '__main__':
    main()