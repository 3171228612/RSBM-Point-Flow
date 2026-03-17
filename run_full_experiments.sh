#!/bin/bash
# ============================================================================
#  RSBM Full Experiment Suite for 8x H200
#
#  This script runs ALL experiments needed for the paper:
#    Phase 1: Download datasets + checkpoints
#    Phase 2: Train RSBM + RPF baseline (parallel on 4+4 GPUs)
#    Phase 3: Ablation experiments (single GPU, fast)
#    Phase 4: Inference speed benchmark
#
#  Usage:
#    # Run specific phase:
#    bash run_full_experiments.sh phase1   # Download data
#    bash run_full_experiments.sh phase2   # Train models
#    bash run_full_experiments.sh phase3   # Ablation
#    bash run_full_experiments.sh phase4   # Speed benchmark
#    bash run_full_experiments.sh all      # Everything
#
#  Estimated time (8x H200):
#    Phase 1: ~2 hours (download ~86 GB)
#    Phase 2: ~3-4 days (2000 epochs, 4 GPUs each)
#    Phase 3: ~30 minutes
#    Phase 4: ~10 minutes
# ============================================================================

set -e

# ======================== Configuration ========================
DATA_ROOT="/mlx_devbox/users/zhaoliangjie/dataset"
CODE_ROOT="/mlx_devbox/users/zhaoliangjie/Rectified-Point-Flow-main"
ENCODER_CKPT="./weights/RPF_base_full_anchorfree_ep2000.ckpt"

# Training
BATCH_SIZE=40       # per GPU, 40 for 80GB+ GPU
NUM_WORKERS=16      # per GPU
MAX_EPOCHS=2000

# Ablation
RESULT_DIR="./ablation_results"
N_GENS=3            # number of generation samples per test case

cd ${CODE_ROOT}

# ======================== Phase 1: Download ========================
phase1_download() {
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
    # Auto-download via HuggingFace (built into sample.py)
    # Or manually:
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
    echo "=============================================="
    echo "  Phase 2: Train RSBM + RPF (parallel)"
    echo "=============================================="
    
    # ---- Option A: 8 GPUs split into 2 groups of 4 ----
    echo ""
    echo "Training RSBM on GPU 0-3 and RPF on GPU 4-7 in parallel..."
    echo "Expected: ~3-4 days for 2000 epochs"
    
    # RSBM (GPU 0-3)
    CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
        --config-name "RSBM_main" \
        data_root="${DATA_ROOT}" \
        trainer.devices=4 \
        trainer.strategy="ddp" \
        trainer.num_nodes=1 \
        data.batch_size=${BATCH_SIZE} \
        data.num_workers=${NUM_WORKERS} \
        trainer.max_epochs=${MAX_EPOCHS} \
        model.encoder_ckpt="${ENCODER_CKPT}" \
        +model.frozen_encoder=true \
        experiment_name="RSBM_6ds_ep2000" \
    &
    RSBM_PID=$!
    
    # RPF baseline (GPU 4-7, same conditions)
    CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py \
        --config-name "RPF_base_main" \
        data_root="${DATA_ROOT}" \
        trainer.devices=4 \
        trainer.strategy="ddp" \
        trainer.num_nodes=1 \
        data.batch_size=${BATCH_SIZE} \
        data.num_workers=${NUM_WORKERS} \
        trainer.max_epochs=${MAX_EPOCHS} \
        model.encoder_ckpt="${ENCODER_CKPT}" \
        +model.frozen_encoder=true \
        experiment_name="RPF_6ds_ep2000" \
    &
    RPF_PID=$!
    
    echo "RSBM PID: ${RSBM_PID}"
    echo "RPF PID: ${RPF_PID}"
    echo ""
    echo "Both training jobs launched. Monitor with:"
    echo "  tail -f ./output/RSBM_6ds_ep2000/wandb/latest-run/files/output.log"
    echo "  tail -f ./output/RPF_6ds_ep2000/wandb/latest-run/files/output.log"
    
    # Wait for both
    wait ${RSBM_PID}
    echo "RSBM training finished!"
    wait ${RPF_PID}
    echo "RPF training finished!"
    
    echo "Phase 2 complete!"
}

# ---- Quick DDP verification (2 GPU, 2 epochs) ----
phase2_verify() {
    echo "=============================================="
    echo "  Phase 2 Verify: Quick DDP Test (2 GPU, 2 epochs)"
    echo "=============================================="
    
    # Quick test on 2 GPUs, only ikea dataset, 2 epochs
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
        experiment_name="RSBM_ddp_test" \
    
    echo ""
    if [ $? -eq 0 ]; then
        echo "✅ DDP test PASSED! Multi-GPU training works."
    else
        echo "❌ DDP test FAILED. Check error messages above."
    fi
    
    # Cleanup test output
    rm -rf ./output/RSBM_ddp_test
}

# ======================== Phase 3: Ablation ========================
phase3_ablation() {
    echo "=============================================="
    echo "  Phase 3: Full Ablation Experiments"
    echo "=============================================="
    
    RSBM_CKPT="./output/RSBM_6ds_ep2000/last.ckpt"
    RPF_CKPT="./output/RPF_6ds_ep2000/last.ckpt"
    
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
    
    # ---- 3b: RPF Steps ablation (Euler baseline) ----
    echo ""
    echo "=== 3b: RPF Steps Ablation ==="
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
    echo "=============================================="
    echo "  Phase 4: Inference Speed Benchmark"
    echo "=============================================="
    
    RSBM_CKPT="./output/RSBM_6ds_ep2000/last.ckpt"
    RPF_CKPT="./output/RPF_6ds_ep2000/last.ckpt"
    
    mkdir -p ${RESULT_DIR}/speed
    
    python -c "
import torch
import time
import json
import os

# Benchmark function: measure sampling-only time
def benchmark_sampling(sample_fn, model, data_dict, latent, num_runs=50, warmup=10):
    \"\"\"Benchmark sampling speed using CUDA events.\"\"\"
    times = []
    for i in range(warmup + num_runs):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        
        _ = sample_fn(data_dict, latent)
        
        end.record()
        torch.cuda.synchronize()
        
        if i >= warmup:
            times.append(start.elapsed_time(end))
    
    return {
        'mean_ms': float(sum(times) / len(times)),
        'std_ms': float((sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5),
        'min_ms': float(min(times)),
        'max_ms': float(max(times)),
    }

print('Speed benchmark requires trained checkpoints.')
print('Run after Phase 2 completes.')
print('Use: python benchmark_inference.py for detailed speed comparison.')
"
    
    # Use the existing benchmark script if available
    if [ -f "benchmark_inference.py" ]; then
        python benchmark_inference.py \
            --rsbm_ckpt "${RSBM_CKPT}" \
            --rpf_ckpt "${RPF_CKPT}" \
            --data_root "${DATA_ROOT}" \
            --output "${RESULT_DIR}/speed/benchmark.json"
    else
        echo "benchmark_inference.py not found. Running manual speed test..."
        
        # Manual speed comparison via inference time measurement
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
        echo "  phase1   - Download datasets (~86 GB) and checkpoints"
        echo "  phase2   - Train RSBM + RPF (4+4 GPUs, ~3-4 days)"
        echo "  verify   - Quick DDP test (2 GPUs, 2 epochs, ~5 min)"
        echo "  phase3   - Schedule ablation (single GPU, ~30 min)"
        echo "  phase4   - Inference speed benchmark (~10 min)"
        echo "  all      - Run everything"
        ;;
esac
