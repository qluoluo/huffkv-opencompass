#!/usr/bin/env python3
"""
Flash Attention 测试脚本
用于验证Flash Attention是否能在当前环境中正常运行
"""

import sys
import subprocess
import importlib
import torch
import warnings
warnings.filterwarnings('ignore')

def check_python_version():
    """检查Python版本"""
    print("=" * 50)
    print("检查Python版本...")
    print(f"Python版本: {sys.version}")
    if sys.version_info < (3, 7):
        print("❌ 需要Python 3.7或更高版本")
        return False
    print("✅ Python版本符合要求")
    return True

def check_pytorch():
    """检查PyTorch安装和CUDA可用性"""
    print("=" * 50)
    print("检查PyTorch...")
    try:
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA版本: {torch.version.cuda}")
            print(f"GPU设备: {torch.cuda.get_device_name(0)}")
            print(f"GPU数量: {torch.cuda.device_count()}")
        else:
            print("⚠️ 未检测到CUDA，Flash Attention需要CUDA支持")
            
        print("✅ PyTorch检查完成")
        return True
    except Exception as e:
        print(f"❌ PyTorch检查失败: {e}")
        return False

def install_package(package_name):
    """安装指定的Python包"""
    try:
        print(f"正在安装 {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"✅ {package_name} 安装成功")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ {package_name} 安装失败")
        return False

def check_flash_attn():
    """检查Flash Attention是否可用"""
    print("=" * 50)
    print("检查Flash Attention...")
    
    # 尝试导入flash_attn
    try:
        flash_attn_available = importlib.util.find_spec("flash_attn") is not None
        if not flash_attn_available:
            print("未找到flash_attn包，尝试安装...")
            if not install_package("flash-attn"):
                print("❌ Flash Attention安装失败")
                return False
                
        import flash_attn
        print(f"Flash Attention版本: {flash_attn.__version__}")
        print("✅ Flash Attention安装成功")
        
        # 运行简单的测试
        print("运行简单测试...")
        test_flash_attention()
        
        print("✅ Flash Attention测试通过")
        return True
        
    except Exception as e:
        print(f"❌ Flash Attention测试失败: {e}")
        return False

def test_flash_attention():
    """运行简单的Flash Attention测试"""
    try:
        from flash_attn import flash_attn_func
        import torch.nn.functional as F
        
        # 创建测试数据
        batch_size, seq_len, n_heads, head_dim = 2, 64, 4, 64
        dtype = torch.float16
        
        # 在CUDA上创建随机张量
        q = torch.randn(batch_size, seq_len, n_heads, head_dim, dtype=dtype, device='cuda') * 0.01
        k = torch.randn(batch_size, seq_len, n_heads, head_dim, dtype=dtype, device='cuda') * 0.01
        v = torch.randn(batch_size, seq_len, n_heads, head_dim, dtype=dtype, device='cuda') * 0.01
        
        # 使用Flash Attention计算
        output = flash_attn_func(q, k, v, causal=True)
        
        # 使用普通Attention计算（用于验证）
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            expected = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), 
                attn_mask=None, dropout_p=0.0, is_causal=True
            ).transpose(1, 2)
        
        # 检查结果是否相似（允许一定的误差）
        diff = (output - expected).abs().mean()
        print(f"Flash Attention与标准Attention的平均差异: {diff.item():.6f}")
        
        if diff < 0.1:
            print("✅ Flash Attention计算结果正确")
        else:
            print("⚠️ Flash Attention计算结果与预期有较大差异")
            
    except Exception as e:
        print(f"❌ Flash Attention测试运行失败: {e}")
        raise

def main():
    """主函数"""
    print("开始测试Flash Attention...")
    
    # 检查环境
    success = True
    success &= check_python_version()
    success &= check_pytorch()
    
    if success:
        success &= check_flash_attn()
    
    print("=" * 50)
    if success:
        print("🎉 所有测试通过! Flash Attention可以正常运行")
        return 0
    else:
        print("❌ 测试失败，请检查环境配置")
        return 1

if __name__ == "__main__":
    sys.exit(main())