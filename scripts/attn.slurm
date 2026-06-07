#!/bin/bash
#SBATCH --job-name=flash_compile
#SBATCH --time=4:00:00            # 2小时足够编译
#SBATCH --nodes=1                  # 单节点编译足够
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32         # 32核心用于并行编译
#SBATCH --gpus-per-node=1          # 只需要1个GPU测试编译
#SBATCH --mem=128G                 # 128GB内存足够
#SBATCH --partition=ghx4           # GH200分区
#SBATCH --account=bffh-dtai-gh
#SBATCH --output=/u/yzou1/logs/flash_compile_%j.out
#SBATCH --error=/u/yzou1/logs/flash_compile_%j.err

# 排除可能有问题的节点（可选）
#SBATCH --exclude=gh077

echo "=== Flash Attention 编译作业开始于: $(date) ==="
echo "作业ID: $SLURM_JOB_ID"
echo "节点: $(hostname)"
echo "GPU分配: $SLURM_GPUS_ON_NODE"

# 加载必要模块
module load gcc/11.4.0
module load cuda/12.6.1

# 激活conda环境
conda activate smolvlm_env

echo "=== 环境信息 ==="
echo "GCC版本: $(gcc --version | head -1)"
echo "CUDA版本: $(nvcc --version | grep release)"
echo "Python版本: $(python --version)"
echo "PyTorch版本: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA可用: $(python -c 'import torch; print(torch.cuda.is_available())')"

# 设置环境变量
export TORCH_EXTENSIONS_DISABLE_VERSION_CHECK=1
export TORCH_CUDA_ARCH_LIST="9.0"          # GH200是Hopper 9.0
export MAX_JOBS=8                         # 与cpus-per-task匹配
export CUDA_HOME=$CUDA_PATH
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo "=== 编译设置 ==="
echo "TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"
echo "MAX_JOBS: $MAX_JOBS"
echo "CUDA_HOME: $CUDA_HOME"

# 进入目录
cd /u/yzou1/flash-attention/hopper

echo "=== 清理之前的构建 ==="
rm -rf build/

echo "=== 开始编译 flash-attention hopper 版本 ==="
echo "编译开始时间: $(date)"

# 编译（添加详细输出）
time python setup.py install --verbose

echo "=== 编译完成 ==="
echo "完成时间: $(date)"

# 验证安装
echo "=== 验证安装 ==="
python -c "
import torch
try:
    import flash_attn
    print(f'✅ Flash Attention 安装成功!')
    print(f'版本: {flash_attn.__version__}')
    print(f'PyTorch CUDA: {torch.version.cuda}')
    print(f'设备: {torch.cuda.get_device_name(0)}')

    # 快速测试
    from flash_attn import flash_attn_func
    q = torch.randn(1, 4, 512, 64, device='cuda', dtype=torch.float16)
    k = torch.randn(1, 4, 512, 64, device='cuda', dtype=torch.float16)
    v = torch.randn(1, 4, 512, 64, device='cuda', dtype=torch.float16)

    output = flash_attn_func(q, k, v)
    print(f'✅ 测试通过! 输出形状: {output.shape}')
except Exception as e:
    print(f'❌ 安装失败: {e}')
    import traceback
    traceback.print_exc()
"

echo "=== 作业结束 ==="