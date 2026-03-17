#!/usr/bin/env bash
# =============================================================
# H200 八卡环境安装脚本
#
# 前提: 系统已预装 torch 2.8.0 + CUDA 12.x，不要动 torch！
#
# ⚠ 不装 torch（已有 2.8.0）
# ⚠ 不装 xformers（代码不需要，且它会偷偷拉新 torch）
# ⚠ layer.py 已注释掉 @torch.compile（torch 2.8 有 CuteDSL bug）
#
# 用法:
#   bash install_h200.sh
# =============================================================
set -e

echo "=============================================="
echo "  RSBM-PCR H200 安装脚本"
echo "=============================================="

# ──────────────────────────────────────────────────
# 0. 读取当前 torch 版本（不安装！）
# ──────────────────────────────────────────────────
echo ""
echo ">>> [0/6] 检查已有环境..."

TORCH_FULL=$(python -c "import torch; print(torch.__version__)")
TORCH_SHORT=$(python -c "import torch; v=torch.__version__.split('+')[0]; print(v)")
TORCH_MAJOR_MINOR=$(python -c "import torch; v=torch.__version__.split('+')[0].rsplit('.',1)[0]; print(v)")
CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)")
CUDA_TAG=$(python -c "import torch; print('cu'+torch.version.cuda.replace('.',''))")
PY_VERSION=$(python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "N/A")
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")

echo "    PyTorch:     $TORCH_FULL  ← 不会动这个!"
echo "    CUDA:        $CUDA_VERSION"
echo "    Python:      $PY_VERSION"
echo "    GPU:         $GPU_NAME × $GPU_COUNT"

# ──────────────────────────────────────────────────
# 1. 纯 Python 依赖（不碰 torch）
#    ⚠ 不装 xformers! 它会拉 torch>=2.10
# ──────────────────────────────────────────────────
echo ""
echo ">>> [1/6] 安装纯 Python 依赖（不装 xformers）..."
pip install --no-cache-dir \
    ninja \
    "diffusers>=0.33.0" \
    "lightning>=2.5.0" \
    "torchmetrics>=1.6.0" \
    "trimesh>=4.6.0" \
    addict \
    scipy \
    h5py \
    tqdm \
    "hydra-core>=1.3.0" \
    wandb \
    mitsuba \
    matplotlib \
    rich \
    "huggingface-hub>=0.26.0" \
    einops \
    fvcore \
    iopath

# 检查 torch 有没有被动
TORCH_CHECK=$(python -c "import torch; print(torch.__version__)")
if [ "$TORCH_CHECK" != "$TORCH_FULL" ]; then
    echo "    ⚠ torch 被篡改: $TORCH_FULL → $TORCH_CHECK，正在恢复..."
    pip install "torch==${TORCH_FULL}" --no-deps --no-cache-dir
else
    echo "    ✅ torch 版本安全: $TORCH_CHECK"
fi

# ──────────────────────────────────────────────────
# 2. PyG 散列算子 (PointTransformerV3 需要)
# ──────────────────────────────────────────────────
echo ""
echo ">>> [2/6] 安装 PyG 散列算子..."
PYG_URL="https://data.pyg.org/whl/torch-${TORCH_SHORT}+${CUDA_TAG}.html"
echo "    PyG wheel URL: $PYG_URL"

pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f "$PYG_URL" --no-cache-dir 2>&1 \
|| {
    echo "    ⚠ 预编译 wheel 不可用，尝试源码编译..."
    pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
        --no-cache-dir 2>&1 \
    || echo "    ❌ PyG 安装失败（可能需要手动安装）"
}

# ──────────────────────────────────────────────────
# 3. FlashAttention (layer.py 硬依赖)
# ──────────────────────────────────────────────────
echo ""
echo ">>> [3/6] 安装 FlashAttention..."
export MAX_JOBS=2   # 防止编译时 OOM

FLASH_OK=0

# 方案A: pip 直接装（可能有预编译 wheel）
echo "    方案A: pip install flash-attn..."
pip install flash-attn --no-build-isolation --no-cache-dir 2>&1 \
&& { echo "    ✅ flash-attn 安装成功"; FLASH_OK=1; } \
|| echo "    方案A 失败"

# 方案B: 遍历 GitHub 预编译 wheel
if [ "$FLASH_OK" -eq 0 ]; then
    echo "    方案B: 尝试 GitHub 预编译 wheel..."
    for FA_VER in "2.8.3" "2.7.4.post1" "2.7.3"; do
        for CU in "cu12" "$CUDA_TAG"; do
            for ABI in "FALSE" "TRUE"; do
                WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v${FA_VER}/flash_attn-${FA_VER}+${CU}torch${TORCH_MAJOR_MINOR}cxx11abi${ABI}-${PY_VERSION}-${PY_VERSION}-linux_x86_64.whl"
                echo "      尝试: flash_attn-${FA_VER}+${CU}torch${TORCH_MAJOR_MINOR}cxx11abi${ABI}"
                pip install "$WHEEL_URL" --no-cache-dir 2>/dev/null \
                && { echo "    ✅ flash-attn wheel 安装成功"; FLASH_OK=1; break 3; }
            done
        done
    done
fi

if [ "$FLASH_OK" -eq 0 ]; then
    echo "    ❌ flash-attn 自动安装失败"
    echo "    手动方案: 访问 https://github.com/Dao-AILab/flash-attention/releases"
    echo "    搜索匹配 torch${TORCH_MAJOR_MINOR} + ${PY_VERSION} 的 .whl"
fi

# ──────────────────────────────────────────────────
# 4. PyTorch3D (eval 指标计算需要)
# ──────────────────────────────────────────────────
echo ""
echo ">>> [4/6] 安装 PyTorch3D..."
export MAX_JOBS=2
PT3D_OK=0

# 方案A: pip
echo "    方案A: pip install pytorch3d..."
pip install pytorch3d --no-cache-dir 2>&1 \
&& { echo "    ✅ pytorch3d (pip)"; PT3D_OK=1; } \
|| echo "    方案A 失败"

# 方案B: Facebook 预编译 wheel
if [ "$PT3D_OK" -eq 0 ]; then
    echo "    方案B: 预编译 wheel..."
    for CUPT in "cu121_pyt${TORCH_SHORT//./}" "cu128_pyt${TORCH_SHORT//./}" "cu121_pyt251" "cu124_pyt251"; do
        WHEEL_URL="https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/${PY_VERSION}_${CUPT}/pytorch3d-0.7.8-${PY_VERSION}-${PY_VERSION}-linux_x86_64.whl"
        echo "      尝试: ${PY_VERSION}_${CUPT}"
        pip install "$WHEEL_URL" --no-deps --no-cache-dir 2>/dev/null \
        && { echo "    ✅ pytorch3d (wheel)"; PT3D_OK=1; break; }
    done
fi

# 方案C: 源码编译（最后手段）
if [ "$PT3D_OK" -eq 0 ]; then
    echo "    方案C: 源码编译..."
    pip install "git+https://github.com/facebookresearch/pytorch3d.git" \
        --no-build-isolation --no-cache-dir 2>&1 \
    && { echo "    ✅ pytorch3d (源码)"; PT3D_OK=1; } \
    || echo "    ❌ pytorch3d 安装失败"
fi

# ──────────────────────────────────────────────────
# 5. Spconv
# ──────────────────────────────────────────────────
echo ""
echo ">>> [5/6] 安装 Spconv..."
pip install "spconv-${CUDA_TAG}" --no-cache-dir 2>&1 \
|| pip install spconv-cu120 --no-cache-dir 2>&1 \
|| pip install spconv-cu121 --no-cache-dir 2>&1 \
|| echo "    ❌ spconv 安装失败"

# ──────────────────────────────────────────────────
# 6. 最终验证
# ──────────────────────────────────────────────────
echo ""
echo ">>> [6/6] 最终验证..."

# 再检查一次 torch
TORCH_NOW=$(python -c "import torch; print(torch.__version__)")
if [ "$TORCH_NOW" != "$TORCH_FULL" ]; then
    echo "    ⚠ torch 被篡改: $TORCH_FULL → $TORCH_NOW"
    echo "    正在强制恢复..."
    pip install "torch==${TORCH_FULL}" --no-deps --no-cache-dir
fi

# 清理残留
SITE_PKG=$(python -c "import site; print(site.getsitepackages()[0])")
rm -rf "${SITE_PKG}/~orch"* 2>/dev/null

python << 'PYEOF'
import sys
print(f'\n  Python:          {sys.version.split()[0]}')

import torch
print(f'  PyTorch:         {torch.__version__}')
print(f'  CUDA:            {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'  GPU:             {torch.cuda.get_device_name(0)}')
    print(f'  GPU Count:       {torch.cuda.device_count()}')
    props = torch.cuda.get_device_properties(0)
    print(f'  GPU Memory:      {props.total_mem / 1024**3:.1f} GB')

checks = [
    ('flash_attn',     'flash_attn',     True),
    ('pytorch3d',      'pytorch3d',      True),
    ('torch_scatter',  'torch_scatter',  True),
    ('spconv',         'spconv',         True),
    ('lightning',      'lightning',       True),
    ('torchmetrics',   'torchmetrics',   True),
    ('trimesh',        'trimesh',        True),
    ('h5py',           'h5py',           True),
    ('hydra',          'hydra',          True),
    ('diffusers',      'diffusers',      False),
    ('wandb',          'wandb',          False),
    ('mitsuba',        'mitsuba',        False),
    ('einops',         'einops',         False),
]

all_ok = True
for name, mod, critical in checks:
    try:
        m = __import__(mod)
        ver = getattr(m, '__version__', '✅')
        print(f'  {name:18s} {ver}')
    except ImportError:
        marker = '❌ 必需!' if critical else '⚠ 可选'
        print(f'  {name:18s} {marker}')
        if critical:
            all_ok = False

# 验证 RSBM 模块可以导入
print()
try:
    from rectified_point_flow.flow_model import PointCloudDiT
    print('  RSBM PointCloudDiT   ✅ 可导入')
except Exception as e:
    print(f'  RSBM PointCloudDiT   ❌ {e}')
    all_ok = False

try:
    from rectified_point_flow.rsbm_modeling import RSBMRectifiedPointFlow
    print('  RSBMRectifiedPointFlow ✅ 可导入')
except Exception as e:
    print(f'  RSBMRectifiedPointFlow ❌ {e}')
    all_ok = False

print()
if all_ok:
    print('  🎉 所有依赖就绪！可以开始训练！')
else:
    print('  ⚠ 有依赖未就绪，请根据上方日志修复')
PYEOF

echo ""
echo "=============================================="
echo "  安装完毕"
echo "=============================================="
echo ""
echo "下一步:"
echo "  1. bash run_full_experiments.sh verify   # 验证 DDP"
echo "  2. bash run_full_experiments.sh phase1   # 下载数据集"
echo "  3. bash run_full_experiments.sh phase2   # 开始训练"
