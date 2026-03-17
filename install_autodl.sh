#!/usr/bin/env bash
# =============================================================
# AutoDL 安装脚本 v4 — 彻底解决 torch 被篡改问题
#
# 根因: xformers 0.0.35 依赖 torch>=2.10，fallback 分支把 torch 拉升了
# 修复: ① 删掉 xformers（代码里根本没用）
#       ② 清理 ~orch 残留
#       ③ flash-attn MAX_JOBS=2 防 OOM
#       ④ 所有 pip install 都不允许动 torch
# =============================================================

echo "=============================================="
echo "  RSBM-PCR AutoDL 安装脚本 v4"
echo "=============================================="

# ──────────────────────────────────────────────────
# 0. 国内镜像 + 清理残留
# ──────────────────────────────────────────────────
echo ""
echo ">>> [0/7] 环境准备..."
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ 2>/dev/null
pip config set global.trusted-host mirrors.aliyun.com 2>/dev/null

# 清理 ~orch 这种半残留包 (之前 uninstall 不干净留下的)
echo "    清理 site-packages 中的残留文件..."
SITE_PKG=$(python -c "import site; print(site.getsitepackages()[0])")
rm -rf "${SITE_PKG}/~orch"* 2>/dev/null && echo "    已清理 ~orch 残留" || echo "    无残留需清理"

# ──────────────────────────────────────────────────
# 1. 锁定 torch 版本
# ──────────────────────────────────────────────────
echo ""
echo ">>> [1/7] 锁定 PyTorch 版本..."
TORCH_FULL=$(python -c "import torch; print(torch.__version__)")
TORCH_SHORT=$(python -c "import torch; v=torch.__version__.split('+')[0]; print(v)")
TORCH_MAJOR_MINOR=$(python -c "import torch; v=torch.__version__.split('+')[0].rsplit('.',1)[0]; print(v)")
CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)")
CUDA_TAG=$(python -c "import torch; print('cu'+torch.version.cuda.replace('.',''))")
PY_VERSION=$(python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "N/A")

echo "    PyTorch:  $TORCH_FULL"
echo "    CUDA:     $CUDA_VERSION"
echo "    CUDA tag: $CUDA_TAG"
echo "    Python:   $PY_VERSION"
echo "    Torch MM: $TORCH_MAJOR_MINOR"
echo "    GPU:      $GPU_NAME"

# ──────────────────────────────────────────────────
# 2. 核心 Python 包 (纯 Python，不碰 torch)
# ──────────────────────────────────────────────────
echo ""
echo ">>> [2/7] 安装核心 Python 依赖..."
pip install --no-cache-dir \
    ninja \
    diffusers \
    lightning \
    torchmetrics \
    trimesh \
    addict \
    scipy \
    h5py \
    tqdm \
    hydra-core \
    wandb \
    mitsuba \
    matplotlib \
    rich \
    huggingface-hub \
    einops \
    fvcore \
    iopath

echo ""
echo "    检查 torch 是否被动了..."
TORCH_CHECK=$(python -c "import torch; print(torch.__version__)")
if [ "$TORCH_CHECK" != "$TORCH_FULL" ]; then
    echo "    ⚠ torch 被修改了: $TORCH_FULL → $TORCH_CHECK，正在恢复..."
    pip install "torch==${TORCH_FULL}" --no-deps --no-cache-dir
fi

# ──────────────────────────────────────────────────
# 3. PyG 散列算子
# ──────────────────────────────────────────────────
echo ""
echo ">>> [3/7] 安装 PyG 散列算子..."
PYG_URL="https://data.pyg.org/whl/torch-${TORCH_SHORT}+${CUDA_TAG}.html"
echo "    PyG wheel URL: $PYG_URL"

pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f "$PYG_URL" --no-cache-dir 2>&1 \
|| {
    echo "    ⚠ PyG 预编译不可用，尝试源码编译..."
    pip install torch-scatter torch-sparse torch-cluster torch-spline-conv --no-cache-dir 2>&1 \
    || echo "    ❌ PyG 安装失败"
}

# ──────────────────────────────────────────────────
# 4. FlashAttention (必需! layer.py 硬依赖)
# ──────────────────────────────────────────────────
echo ""
echo ">>> [4/7] 安装 FlashAttention..."
export MAX_JOBS=2

FLASH_OK=0

# 方案A: pip install (可能自带预编译 wheel)
echo "    方案A: pip install flash-attn --no-build-isolation..."
pip install flash-attn --no-build-isolation --no-cache-dir 2>&1 \
&& { echo "    ✅ flash-attn 安装成功"; FLASH_OK=1; } \
|| echo "    方案A 失败"

# 方案B: 遍历预编译 wheel
if [ "$FLASH_OK" -eq 0 ]; then
    echo "    方案B: 尝试预编译 wheel..."
    for FA_VER in "2.7.4.post1" "2.7.3" "2.6.3"; do
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
    echo ""
    echo "    ❌ flash-attn 自动安装失败"
    echo "    手动方案:"
    echo "    1. 访问 https://github.com/Dao-AILab/flash-attention/releases"
    echo "    2. 搜索匹配 torch${TORCH_MAJOR_MINOR} + ${PY_VERSION} 的 .whl"
    echo "    3. wget <URL> && pip install ./flash_attn-xxx.whl"
fi

# ──────────────────────────────────────────────────
# 5. PyTorch3D (eval/metrics.py 需要)
# ──────────────────────────────────────────────────
echo ""
echo ">>> [5/7] 安装 PyTorch3D..."
export MAX_JOBS=2
PT3D_OK=0

# 方案A: pip
echo "    方案A: pip install pytorch3d..."
pip install pytorch3d --no-cache-dir 2>&1 \
&& { echo "    ✅ pytorch3d (pip)"; PT3D_OK=1; } \
|| echo "    方案A 失败"

# 方案B: Facebook 预编译 wheel (多种组合)
if [ "$PT3D_OK" -eq 0 ]; then
    echo "    方案B: Facebook 预编译 wheel..."
    for CUPT in "cu121_pyt${TORCH_SHORT//./}" "cu128_pyt${TORCH_SHORT//./}" "cu121_pyt251" "cu124_pyt251"; do
        WHEEL_URL="https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/${PY_VERSION}_${CUPT}/pytorch3d-0.7.8-${PY_VERSION}-${PY_VERSION}-linux_x86_64.whl"
        echo "      尝试: ${PY_VERSION}_${CUPT}"
        pip install "$WHEEL_URL" --no-deps --no-cache-dir 2>/dev/null \
        && { echo "    ✅ pytorch3d (wheel)"; PT3D_OK=1; break; }
    done
fi

# 方案C: conda
if [ "$PT3D_OK" -eq 0 ]; then
    echo "    方案C: conda..."
    conda install -y pytorch3d -c pytorch3d 2>&1 \
    && { echo "    ✅ pytorch3d (conda)"; PT3D_OK=1; } \
    || echo "    方案C 失败"
fi

# 方案D: 从源码编译 (最后手段)
if [ "$PT3D_OK" -eq 0 ]; then
    echo "    方案D: 源码编译..."
    # 先试 gitee 镜像
    pip install "git+https://gitee.com/mirrors/pytorch3d.git" --no-build-isolation --no-cache-dir 2>&1 \
    && { echo "    ✅ pytorch3d (gitee源码)"; PT3D_OK=1; } \
    || {
        # 最后试 GitHub
        pip install "git+https://github.com/facebookresearch/pytorch3d.git" --no-build-isolation --no-cache-dir 2>&1 \
        && { echo "    ✅ pytorch3d (github源码)"; PT3D_OK=1; } \
        || echo "    ❌ pytorch3d 安装失败"
    }
fi

# ──────────────────────────────────────────────────
# 6. Spconv
# ──────────────────────────────────────────────────
echo ""
echo ">>> [6/7] 安装 Spconv..."
pip install "spconv-${CUDA_TAG}" --no-cache-dir 2>&1 \
|| pip install spconv-cu120 --no-cache-dir 2>&1 \
|| pip install spconv-cu121 --no-cache-dir 2>&1 \
|| echo "    ❌ spconv 安装失败"

# ──────────────────────────────────────────────────
# 7. 最终 torch 版本校验
# ──────────────────────────────────────────────────
echo ""
echo ">>> [7/7] 最终 torch 版本校验..."
TORCH_NOW=$(python -c "import torch; print(torch.__version__)")
if [ "$TORCH_NOW" != "$TORCH_FULL" ]; then
    echo "    ⚠⚠⚠ torch 被篡改了: $TORCH_FULL → $TORCH_NOW"
    echo "    正在强制恢复..."
    pip install "torch==${TORCH_FULL}" --no-deps --no-cache-dir 2>&1 \
    || echo "    ❌ 无法恢复 torch，请运行 bash fix_torch.sh"
else
    echo "    ✅ torch 版本安全: $TORCH_NOW"
fi

# 再次清理 ~orch 残留
rm -rf "${SITE_PKG}/~orch"* 2>/dev/null

# ──────────────────────────────────────────────────
# 验证
# ──────────────────────────────────────────────────
echo ""
echo "=============================================="
echo "  安装验证"
echo "=============================================="
python << 'PYEOF'
import sys
print(f'  Python:          {sys.version.split()[0]}')

import torch
print(f'  torch:           {torch.__version__}')
print(f'  CUDA:            {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'  GPU:             {torch.cuda.get_device_name(0)}')
    props = torch.cuda.get_device_properties(0)
    print(f'  GPU Memory:      {props.total_mem / 1024**3:.1f} GB')

checks = [
    ('torchvision',    'torchvision'),
    ('lightning',      'lightning'),
    ('flash_attn',     'flash_attn'),
    ('pytorch3d',      'pytorch3d'),
    ('torch_scatter',  'torch_scatter'),
    ('spconv',         'spconv'),
    ('trimesh',        'trimesh'),
    ('h5py',           'h5py'),
    ('hydra',          'hydra'),
    ('diffusers',      'diffusers'),
]

all_ok = True
for name, mod in checks:
    try:
        m = __import__(mod)
        ver = getattr(m, '__version__', '✅')
        critical = name in ('flash_attn', 'pytorch3d', 'torch_scatter', 'spconv', 'lightning')
        print(f'  {name:18s} {ver}')
    except ImportError:
        critical = name in ('flash_attn', 'pytorch3d', 'torch_scatter', 'spconv', 'lightning')
        marker = '❌ 必需!' if critical else '❌ 可选'
        print(f'  {name:18s} {marker}')
        if critical:
            all_ok = False

print()
if all_ok:
    print('  🎉 所有必需依赖已就绪！可以开始训练了！')
else:
    print('  ⚠ 有必需依赖未安装，请参考上方日志手动安装')
PYEOF

echo ""
echo "=============================================="
echo "  脚本执行完毕"
echo "=============================================="
