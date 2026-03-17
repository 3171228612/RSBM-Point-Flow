#!/bin/bash
# ============================================================================
#  RSBM Full Experiment Suite for 8x H200
#
#  8 卡全给 RSBM 训练，RPF 对比直接用预训练权重。
#
#  Usage:
#    bash run_full_experiments.sh phase1   # 下载数据
#    bash run_full_experiments.sh verify   # 验证 DDP (2卡, 2epoch, ~5min)
#    bash run_full_experiments.sh phase2   # 训练 RSBM (8卡, 2000epoch)
#    bash run_full_experiments.sh phase3   # 消融实验 (单卡)
#    bash run_full_experiments.sh phase4   # 速度测试 (单卡)
#    bash run_full_experiments.sh all      # 全部
#
#  断点续训: 自动! 只要 output 目录下有 last.ckpt 就会自动恢复。
#            如果要从头训，先删掉对应的 output 目录。
#
#  Estimated time (8x H200):
#    Phase 1: ~2 hours (download ~86 GB)
#    Phase 2: ~1.5-2 days (2000 epochs, 8 GPUs, bf16-mixed)
#    Phase 3: ~30 minutes
#    Phase 4: ~10 minutes
# ============================================================================

set -e

# ======================== Configuration ========================
DATA_ROOT="/mlx_devbox/users/zhaoliangjie/dataset"
CODE_ROOT="/mlx_devbox/users/zhaoliangjie/RSBM-Point-Flow-main"
ENCODER_CKPT="./weights/RPF_base_full_anchorfree_ep2000.ckpt"

# Training
NUM_GPUS=8          # ← 全部 8 卡给 RSBM
BATCH_SIZE=40       # per GPU, H200 (141GB) 完全够用
NUM_WORKERS=16      # per GPU
MAX_EPOCHS=2000

# Ablation
RESULT_DIR="./ablation_results"
N_GENS=3            # number of generation samples per test case

# RPF baseline: 直接用预训练权重对比，不需要重新训
RPF_PRETRAINED_CKPT="${ENCODER_CKPT}"  # 同一个 checkpoint 就是 RPF 完整模型

cd ${CODE_ROOT}

# ======================== DDP 防挂措施 ========================
# NCCL 超时: 默认 30 分钟可能在 validation 时不够（6 个数据集 val 很慢）
export NCCL_TIMEOUT=3600              # 1 小时
export NCCL_DEBUG=WARN                # 报错时打印 NCCL 信息（不要用 INFO，太吵）
export NCCL_IB_DISABLE=0              # H200 有 InfiniBand，别禁
export NCCL_P2P_LEVEL=NVL             # NVLink 点对点传输
export NCCL_SOCKET_IFNAME=eth0        # 指定网络接口（按实际改）

# PyTorch DDP 设置
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1   # 异步错误处理，避免 hang 后无日志
export TORCH_NCCL_BLOCKING_WAIT=0          # 非阻塞等待
export TORCH_DISTRIBUTED_DEBUG=OFF         # 关闭 debug（生产模式）

# OMP 线程: 防止 CPU 线程过多互相抢
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# CUDA
export CUDA_LAUNCH_BLOCKING=0         # 异步 launch (更快)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # 减少显存碎片

echo "=============================================="
echo "  RSBM Experiment Suite (8x H200)"
echo "=============================================="
echo "  DATA_ROOT:  ${DATA_ROOT}"
echo "  CODE_ROOT:  ${CODE_ROOT}"
echo "  GPUs:       ${NUM_GPUS}"
echo "  Precision:  bf16-mixed"
echo "  Batch/GPU:  ${BATCH_SIZE}"
echo "  Epochs:     ${MAX_EPOCHS}"
echo "=============================================="

# ======================== Phase 1: Download ========================
phase1_download() {
    echo ""
    echo "=============================================="
    echo "  Phase 1: Download Datasets & Checkpoints"
    echo "=============================================="

    mkdir -p ${DATA_ROOT}
    mkdir -p ./weights

    # ---- Datasets (6 for flow model training, total ~86 GB) ----
    echo ""
    echo "Downloading datasets to ${DATA_ROOT}/ ..."
    echo "  [1/6] IKEA (293 MB)..."
    wget -c -O ${DATA_ROOT}/ikea.hdf5 \
        "https://storage.googleapis.com/flow-asm/ikea.hdf5"

    echo "  [2/6] PartNet (52 GB) ..."
    wget -c -O ${DATA_ROOT}/partnet.hdf5 \
        "https://storage.googleapis.com/flow-asm/partnet.hdf5"

    echo "  [3/6] BreakingBad-Everyday (27 GB) ..."
    # IMPORTANT: Config expects filename 'everyday.hdf5', not 'breaking_bad_vol.hdf5'
    wget -c -O ${DATA_ROOT}/everyday.hdf5 \
        "https://storage.googleapis.com/flow-asm/breaking_bad_vol.hdf5"

    echo "  [4/6] Two-by-Two (259 MB) ..."
    # IMPORTANT: Config expects filename 'twobytwo.hdf5', not '2by2.hdf5'
    wget -c -O ${DATA_ROOT}/twobytwo.hdf5 \
        "https://storage.googleapis.com/flow-asm/2by2.hdf5"

    echo "  [5/6] ModelNet-40 (2 GB) ..."
    wget -c -O ${DATA_ROOT}/modelnet.hdf5 \
        "https://storage.googleapis.com/flow-asm/modelnet.hdf5"

    echo "  [6/6] TUD-L (4 GB) ..."
    wget -c -O ${DATA_ROOT}/tudl.hdf5 \
        "https://storage.googleapis.com/flow-asm/tudl.hdf5"

    echo ""
    echo "All datasets downloaded. Verifying..."
    ls -lh ${DATA_ROOT}/*.hdf5

    # ---- Checkpoint (pretrained encoder) ----
    echo ""
    echo "Downloading pretrained RPF checkpoint..."
    python -c "
from rectified_point_flow.utils import download_rfp_checkpoint
path = download_rfp_checkpoint('RPF_base_full_anchorfree_ep2000.ckpt', './weights')
print(f'Checkpoint saved to: {path}')
"

    echo ""
    echo "Phase 1 complete!"
    echo "  Datasets: ${DATA_ROOT}/"
    echo "  Checkpoint: ./weights/"
}

# ======================== Phase 2: Training ========================
phase2_train() {
    echo ""
    echo "=============================================="
    echo "  Phase 2: Train RSBM (${NUM_GPUS} GPUs)"
    echo "=============================================="

    # ---- 检查断点续训 ----
    CKPT_FILE="./output/RSBM_6ds_ep2000/last.ckpt"
    if [ -f "${CKPT_FILE}" ]; then
        echo "  📌 发现 last.ckpt，自动断点续训!"
        echo "  📌 如需从头训练，请先: rm -rf ./output/RSBM_6ds_ep2000"
    else
        echo "  🆕 首次训练，从头开始"
    fi
    echo ""

    # ---- 8 卡 RSBM ----
    python train.py \
        --config-name "RSBM_main" \
        data_root="${DATA_ROOT}" \
        trainer.devices=${NUM_GPUS} \
        trainer.strategy="ddp" \
        trainer.num_nodes=1 \
        data.batch_size=${BATCH_SIZE} \
        data.num_workers=${NUM_WORKERS} \
        trainer.max_epochs=${MAX_EPOCHS} \
        model.encoder_ckpt="${ENCODER_CKPT}" \
        +model.frozen_encoder=true \
        experiment_name="RSBM_6ds_ep2000"

    echo ""
    echo "Phase 2 complete! RSBM training finished."
}

# ---- Quick DDP verification (2 GPU, 2 epochs) ----
phase2_verify() {
    echo ""
    echo "=============================================="
    echo "  DDP Verify: Quick Test (2 GPU, 2 epochs)"
    echo "=============================================="
    echo ""
    echo "  测试: 2 卡 DDP + BF16 + IKEA 数据集"
    echo "  预期: ~3 分钟完成，不报错即通过"
    echo ""

    # 先清掉旧测试输出（防止断点续训干扰）
    rm -rf ./output/RSBM_ddp_test

    python train.py \
        --config-name "RSBM_main" \
        data_root="${DATA_ROOT}" \
        data=ikea \
        trainer.devices=2 \
        trainer.strategy="ddp" \
        trainer.max_epochs=2 \
        data.batch_size=4 \
        data.num_workers=4 \
        model.encoder_ckpt="${ENCODER_CKPT}" \
        +model.frozen_encoder=true \
        experiment_name="RSBM_ddp_test"

    VERIFY_EXIT=$?

    echo ""
    if [ ${VERIFY_EXIT} -eq 0 ]; then
        echo "✅ DDP 验证通过! 2 卡 BF16 训练正常。"
        echo "   可以放心跑 phase2 了。"
    else
        echo "❌ DDP 验证失败! 请检查上方错误日志。"
        echo ""
        echo "常见问题:"
        echo "  1. NCCL 超时 → 检查 GPU 互联 (nvidia-smi topo -m)"
        echo "  2. OOM → 降低 batch_size"
        echo "  3. flash_attn 报错 → 重装 flash-attn"
        echo "  4. torch.compile 报错 → 确认 layer.py 里 @torch.compile 已注释"
    fi

    # Cleanup test output
    rm -rf ./output/RSBM_ddp_test
}

# ======================== Phase 3: Ablation ========================
phase3_ablation() {
    echo ""
    echo "=============================================="
    echo "  Phase 3: Full Ablation Experiments"
    echo "=============================================="

    RSBM_CKPT="./output/RSBM_6ds_ep2000/last.ckpt"
    # RPF 对比直接用预训练权重
    RPF_CKPT="${RPF_PRETRAINED_CKPT}"

    # 检查 checkpoint
    if [ ! -f "${RSBM_CKPT}" ]; then
        echo "❌ RSBM checkpoint 不存在: ${RSBM_CKPT}"
        echo "   请先完成 Phase 2 训练。"
        exit 1
    fi
    if [ ! -f "${RPF_CKPT}" ]; then
        echo "❌ RPF checkpoint 不存在: ${RPF_CKPT}"
        echo "   请先运行 Phase 1 下载预训练权重。"
        exit 1
    fi

    mkdir -p ${RESULT_DIR}

    # ---- 3a: RSBM Schedule × Steps ablation ----
    echo ""
    echo "=== 3a: RSBM Schedule × Steps Ablation ==="
    for DATASET in ikea partnet everyday twobytwo modelnet tudl; do
        for SCHED in linear uniform karras; do
            for STEPS in 3 5 10 20 30 50; do
                echo ">>> RSBM | ${DATASET} | ${SCHED} | ${STEPS} steps"
                rm -rf ./demo/results/

                python sample.py --config-name RSBM_demo \
                    ckpt_path="${RSBM_CKPT}" \
                    data=${DATASET} \
                    data_root="${DATA_ROOT}" \
                    data.batch_size=1 \
                    data.num_workers=4 \
                    model.inference_sampling_steps=${STEPS} \
                    model.sigma_schedule=${SCHED} \
                    model.n_generations=${N_GENS} \
                    visualizer.renderer=none \
                2>&1 | tail -5 || true

                python -c "
import json, glob, numpy as np, os
files = sorted(glob.glob('./demo/results/${DATASET}_*.json'))
if files:
    metrics = [json.load(open(f)) for f in files]
    result = {
        'method': 'RSBM', 'dataset': '${DATASET}',
        'schedule': '${SCHED}', 'steps': ${STEPS},
        'part_accuracy': float(np.mean([m.get('part_accuracy', 0) for m in metrics])),
        'object_chamfer': float(np.mean([m.get('object_chamfer', 0) for m in metrics])),
        'rotation_error': float(np.mean([m.get('rotation_rmse', m.get('rotation_error', 0)) for m in metrics])),
        'translation_error': float(np.mean([m.get('translation_rmse', m.get('translation_error', 0)) for m in metrics])),
    }
    os.makedirs('${RESULT_DIR}/schedule', exist_ok=True)
    with open(f'${RESULT_DIR}/schedule/rsbm_${DATASET}_${SCHED}_s${STEPS}.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f'  PA={result[\"part_accuracy\"]:.4f} CD={result[\"object_chamfer\"]:.6f}')
else:
    print('  No results')
"
            done
        done
    done

    # ---- 3b: RPF Steps ablation (预训练权重, Euler baseline) ----
    echo ""
    echo "=== 3b: RPF Steps Ablation (pretrained) ==="
    for DATASET in ikea partnet everyday twobytwo modelnet tudl; do
        for STEPS in 3 5 10 20 30 50; do
            echo ">>> RPF | ${DATASET} | euler | ${STEPS} steps"
            rm -rf ./demo/results/

            python sample.py --config-name RPF_base_demo \
                ckpt_path="${RPF_CKPT}" \
                data=${DATASET} \
                data_root="${DATA_ROOT}" \
                data.batch_size=1 \
                data.num_workers=4 \
                model.inference_sampling_steps=${STEPS} \
                model.n_generations=${N_GENS} \
                visualizer.renderer=none \
            2>&1 | tail -5 || true

            python -c "
import json, glob, numpy as np, os
files = sorted(glob.glob('./demo/results/${DATASET}_*.json'))
if files:
    metrics = [json.load(open(f)) for f in files]
    result = {
        'method': 'RPF', 'dataset': '${DATASET}',
        'schedule': 'euler', 'steps': ${STEPS},
        'part_accuracy': float(np.mean([m.get('part_accuracy', 0) for m in metrics])),
        'object_chamfer': float(np.mean([m.get('object_chamfer', 0) for m in metrics])),
        'rotation_error': float(np.mean([m.get('rotation_rmse', m.get('rotation_error', 0)) for m in metrics])),
        'translation_error': float(np.mean([m.get('translation_rmse', m.get('translation_error', 0)) for m in metrics])),
    }
    os.makedirs('${RESULT_DIR}/schedule', exist_ok=True)
    with open(f'${RESULT_DIR}/schedule/rpf_${DATASET}_euler_s${STEPS}.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f'  PA={result[\"part_accuracy\"]:.4f} CD={result[\"object_chamfer\"]:.6f}')
else:
    print('  No results')
"
        done
    done

    # ---- 3c: Print Summary ----
    echo ""
    echo "=============================================="
    echo "  ABLATION RESULTS SUMMARY"
    echo "=============================================="
    python -c "
import json, glob, os
import numpy as np

files = sorted(glob.glob('${RESULT_DIR}/schedule/*.json'))
results = [json.load(open(f)) for f in files]

# ---- Per-dataset table ----
datasets = sorted(set(r['dataset'] for r in results))
methods_schedules = sorted(set((r['method'], r['schedule']) for r in results))
steps_list = sorted(set(r['steps'] for r in results))

for ds in datasets:
    print(f'\n===== Dataset: {ds} =====')
    print(f'{\"Method\":>6} {\"Sched\":>8} | ' + ' | '.join(f'{s:>5}s' for s in steps_list))
    print('-' * (18 + len(steps_list) * 9))
    for method, sched in methods_schedules:
        row = []
        for s in steps_list:
            match = [r for r in results if r['dataset']==ds and r['method']==method
                     and r['schedule']==sched and r['steps']==s]
            if match:
                row.append(f'{match[0][\"part_accuracy\"]:.3f}')
            else:
                row.append('  N/A')
        print(f'{method:>6} {sched:>8} | ' + ' | '.join(f'{v:>5}' for v in row))

# ---- Overall average table ----
print(f'\n===== OVERALL AVERAGE (all datasets) =====')
print(f'{\"Method\":>6} {\"Sched\":>8} | ' + ' | '.join(f'{s:>7}s' for s in steps_list))
print('-' * (18 + len(steps_list) * 11))
for method, sched in methods_schedules:
    row = []
    for s in steps_list:
        matches = [r for r in results if r['method']==method
                   and r['schedule']==sched and r['steps']==s]
        if matches:
            avg_pa = np.mean([m['part_accuracy'] for m in matches])
            avg_cd = np.mean([m['object_chamfer'] for m in matches])
            row.append(f'{avg_pa:.3f}')
        else:
            row.append('    N/A')
    print(f'{method:>6} {sched:>8} | ' + ' | '.join(f'{v:>7}' for v in row))
"

    echo ""
    echo "Phase 3 complete! Results saved to ${RESULT_DIR}/schedule/"
}

# ======================== Phase 4: Speed Benchmark ========================
phase4_speed() {
    echo ""
    echo "=============================================="
    echo "  Phase 4: Inference Speed Benchmark"
    echo "=============================================="

    RSBM_CKPT="./output/RSBM_6ds_ep2000/last.ckpt"
    RPF_CKPT="${RPF_PRETRAINED_CKPT}"

    mkdir -p ${RESULT_DIR}/speed

    # Use the existing benchmark script if available
    if [ -f "benchmark_inference.py" ]; then
        echo "Using benchmark_inference.py for precise CUDA-event timing..."

        # RSBM at different step counts
        for STEPS in 3 5 10 50; do
            echo ">>> RSBM | Heun | ${STEPS} steps"
            python benchmark_inference.py \
                --config-name RSBM_demo \
                ckpt_path="${RSBM_CKPT}" \
                data=ikea \
                data_root="${DATA_ROOT}" \
                data.batch_size=1 \
                model.inference_sampling_steps=${STEPS} \
                +benchmark.num_warmup=10 \
                +benchmark.num_runs=50 \
            || true
        done

        # RPF baseline
        for STEPS in 10 20 50; do
            echo ">>> RPF | Euler | ${STEPS} steps"
            python benchmark_inference.py \
                --config-name RPF_base_demo \
                ckpt_path="${RPF_CKPT}" \
                data=ikea \
                data_root="${DATA_ROOT}" \
                data.batch_size=1 \
                model.inference_sampling_steps=${STEPS} \
                +benchmark.num_warmup=10 \
                +benchmark.num_runs=50 \
            || true
        done
    else
        echo "benchmark_inference.py not found. Running wall-clock speed test..."

        for MODEL_TYPE in RSBM RPF; do
            if [ "${MODEL_TYPE}" = "RSBM" ]; then
                CKPT="${RSBM_CKPT}"
                CONFIG="RSBM_demo"
                STEPS_LIST="3 5 10 50"
            else
                CKPT="${RPF_CKPT}"
                CONFIG="RPF_base_demo"
                STEPS_LIST="10 20 50"
            fi

            for STEPS in ${STEPS_LIST}; do
                echo ">>> Speed: ${MODEL_TYPE} | ${STEPS} steps"
                START_TIME=$(date +%s%N)

                python sample.py --config-name ${CONFIG} \
                    ckpt_path="${CKPT}" \
                    data=ikea \
                    data_root="${DATA_ROOT}" \
                    data.batch_size=1 \
                    data.num_workers=4 \
                    model.inference_sampling_steps=${STEPS} \
                    model.n_generations=1 \
                    visualizer.renderer=none \
                2>&1 | tail -3 || true

                END_TIME=$(date +%s%N)
                ELAPSED_MS=$(( (END_TIME - START_TIME) / 1000000 ))
                echo "  Wall time: ${ELAPSED_MS} ms for 18 samples"
                echo "  Per sample: $(( ELAPSED_MS / 18 )) ms"
            done
        done
    fi

    echo ""
    echo "Phase 4 complete!"
}

# ======================== Main ========================
case "${1:-all}" in
    phase1|download)
        phase1_download
        ;;
    phase2|train)
        phase2_train
        ;;
    verify|test)
        phase2_verify
        ;;
    phase3|ablation)
        phase3_ablation
        ;;
    phase4|speed)
        phase4_speed
        ;;
    all)
        phase1_download
        phase2_train
        phase3_ablation
        phase4_speed
        ;;
    *)
        echo "Usage: bash $0 {phase1|phase2|verify|phase3|phase4|all}"
        echo ""
        echo "Phases:"
        echo "  phase1   - 下载数据集 (~86 GB) + 预训练权重"
        echo "  verify   - DDP 快速验证 (2 卡, 2 epoch, ~3 min)"
        echo "  phase2   - 训练 RSBM (${NUM_GPUS} 卡, ${MAX_EPOCHS} epoch)"
        echo "  phase3   - 消融实验 (单卡, ~30 min)"
        echo "  phase4   - 速度测试 (单卡, ~10 min)"
        echo "  all      - 全部跑完"
        echo ""
        echo "断点续训: 自动! output 目录下有 last.ckpt 就恢复。"
        echo "从头训练: 先 rm -rf ./output/RSBM_6ds_ep2000"
        ;;
esac
