#!/usr/bin/env bash
# =============================================================
# torch 修复脚本 — 解决 torch 被 xformers 篡改的问题
# 使用方法: bash fix_torch.sh
# =============================================================

echo ">>> 当前 torch 环境:"
python -c "
import torch, torchvision, torchaudio
print(f'  torch:       {torch.__version__}')
print(f'  torchvision: {torchvision.__version__}')
print(f'  torchaudio:  {torchaudio.__version__}')
print(f'  CUDA:        {torch.version.cuda}')
" 2>&1

echo ""
echo ">>> 检测 CUDA 版本..."
CUDA_VER=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null)
echo "    CUDA: $CUDA_VER"

# 根据 CUDA 版本判断应该用什么 torch
# AutoDL 5090 通常是 cu128
if [[ "$CUDA_VER" == "12.8" ]] || [[ "$CUDA_VER" == "12.8."* ]]; then
    echo "    检测到 CUDA 12.8 → 尝试恢复 torch 2.8.0+cu128"
    echo ""
    echo ">>> 卸载被篡改的 torch..."
    pip uninstall -y torch torchvision torchaudio xformers 2>/dev/null

    echo ""
    echo ">>> 重新安装 torch 2.8.0+cu128..."
    pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --no-cache-dir

    echo ""
    echo ">>> 重新安装 xformers (--no-deps 防止再次篡改)..."
    pip install xformers --no-deps --no-cache-dir 2>/dev/null || echo "    xformers 安装失败(可选)"

elif [[ "$CUDA_VER" == "12.4" ]] || [[ "$CUDA_VER" == "12.4."* ]]; then
    echo "    检测到 CUDA 12.4 → 尝试恢复 torch 2.5.1+cu124 (RPF原始环境)"
    echo ""
    pip uninstall -y torch torchvision torchaudio xformers 2>/dev/null
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124 --no-cache-dir
    pip install xformers==0.0.29 --no-deps --no-cache-dir 2>/dev/null || echo "    xformers 安装失败(可选)"

else
    echo "    未知 CUDA 版本: $CUDA_VER"
    echo "    请手动确认 AutoDL 镜像自带的 torch 版本，然后运行:"
    echo "    pip install torch==<VERSION> torchvision torchaudio --index-url https://download.pytorch.org/whl/<CUDA_TAG>"
fi

echo ""
echo ">>> 修复后验证:"
python -c "
import torch, torchvision, torchaudio
print(f'  torch:       {torch.__version__}')
print(f'  torchvision: {torchvision.__version__}')
print(f'  torchaudio:  {torchaudio.__version__}')
print(f'  CUDA:        {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'  GPU:         {torch.cuda.get_device_name(0)}')
    print('  ✅ CUDA 可用!')
else:
    print('  ❌ CUDA 不可用!')
" 2>&1

echo ""
echo ">>> 如果 torch 恢复成功，重新运行: bash install_autodl.sh"
