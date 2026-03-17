#!/bin/bash
# ============================================================================
#  Bridge-Optimal Sampling Schedule Ablation
#  
#  Tests RSBM with different sigma schedules (linear, uniform, karras)
#  at different step counts, and compares with RPF baseline.
#
#  Usage:
#    chmod +x run_schedule_ablation.sh
#    bash run_schedule_ablation.sh
#
#  Results saved to: ./ablation_results/schedule_ablation/
# ============================================================================

set -e

# ---- Configuration ----
DATA_ROOT="/mlx_devbox/users/zhaoliangjie/dataset"
RSBM_CKPT="./output/RSBM_eps05_val50/last.ckpt"
RPF_CKPT="./output/RPF_base/last.ckpt"
# If using pretrained RPF, uncomment:
# RPF_CKPT="./weights/RPF_base_full_anchorfree_ep2000.ckpt"

RESULT_DIR="./ablation_results/schedule_ablation"
STEPS_LIST="3 5 10 20 30 50"
SCHEDULE_LIST="linear uniform karras"
N_GENS=3

mkdir -p ${RESULT_DIR}

# ============================================================================
#  Part 1: RSBM — Schedule x Steps Ablation
# ============================================================================
echo "=============================================="
echo "  RSBM Schedule Ablation"
echo "=============================================="

for SCHED in ${SCHEDULE_LIST}; do
  for STEPS in ${STEPS_LIST}; do
    echo ""
    echo ">>> RSBM | schedule=${SCHED} | steps=${STEPS}"
    rm -rf ./demo/results/

    python sample.py --config-name RSBM_demo \
        ckpt_path="${RSBM_CKPT}" \
        data=ikea \
        data_root="${DATA_ROOT}" \
        data.batch_size=1 \
        data.num_workers=4 \
        model.inference_sampling_steps=${STEPS} \
        model.sigma_schedule=${SCHED} \
        model.n_generations=${N_GENS} \
        visualizer.renderer=none \
    || echo "  [WARN] sample.py exited with error (print_eval_table bug, results still saved)"

    # Parse results from JSON
    python -c "
import json, glob, numpy as np, os
files = sorted(glob.glob('./demo/results/ikea_*.json'))
if files:
    metrics = [json.load(open(f)) for f in files]
    result = {
        'method': 'RSBM',
        'schedule': '${SCHED}',
        'steps': ${STEPS},
        'part_accuracy': float(np.mean([m.get('part_accuracy', 0) for m in metrics])),
        'object_chamfer': float(np.mean([m.get('object_chamfer', 0) for m in metrics])),
        'rotation_error': float(np.mean([m.get('rotation_rmse', m.get('rotation_error', 0)) for m in metrics])),
        'translation_error': float(np.mean([m.get('translation_rmse', m.get('translation_error', 0)) for m in metrics])),
    }
    outfile = '${RESULT_DIR}/rsbm_${SCHED}_steps${STEPS}.json'
    with open(outfile, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'  PA: {result[\"part_accuracy\"]:.4f} | CD: {result[\"object_chamfer\"]:.6f} | RE: {result[\"rotation_error\"]:.4f} | TE: {result[\"translation_error\"]:.6f}')
else:
    print('  [ERROR] No result files found')
"
  done
done

# ============================================================================
#  Part 2: RPF Baseline — Steps Ablation (for comparison)
# ============================================================================
echo ""
echo "=============================================="
echo "  RPF Baseline Steps Ablation"
echo "=============================================="

for STEPS in ${STEPS_LIST}; do
    echo ""
    echo ">>> RPF | steps=${STEPS}"
    rm -rf ./demo/results/

    python sample.py --config-name RPF_base_demo \
        ckpt_path="${RPF_CKPT}" \
        data=ikea \
        data_root="${DATA_ROOT}" \
        data.batch_size=1 \
        data.num_workers=4 \
        model.inference_sampling_steps=${STEPS} \
        model.n_generations=${N_GENS} \
        visualizer.renderer=none \
    || echo "  [WARN] sample.py exited with error (print_eval_table bug, results still saved)"

    python -c "
import json, glob, numpy as np, os
files = sorted(glob.glob('./demo/results/ikea_*.json'))
if files:
    metrics = [json.load(open(f)) for f in files]
    result = {
        'method': 'RPF',
        'schedule': 'euler',
        'steps': ${STEPS},
        'part_accuracy': float(np.mean([m.get('part_accuracy', 0) for m in metrics])),
        'object_chamfer': float(np.mean([m.get('object_chamfer', 0) for m in metrics])),
        'rotation_error': float(np.mean([m.get('rotation_rmse', m.get('rotation_error', 0)) for m in metrics])),
        'translation_error': float(np.mean([m.get('translation_rmse', m.get('translation_error', 0)) for m in metrics])),
    }
    outfile = '${RESULT_DIR}/rpf_euler_steps${STEPS}.json'
    with open(outfile, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'  PA: {result[\"part_accuracy\"]:.4f} | CD: {result[\"object_chamfer\"]:.6f} | RE: {result[\"rotation_error\"]:.4f} | TE: {result[\"translation_error\"]:.6f}')
else:
    print('  [ERROR] No result files found')
"
done

# ============================================================================
#  Part 3: Print Summary Table
# ============================================================================
echo ""
echo "=============================================="
echo "  FULL ABLATION RESULTS"
echo "=============================================="

python -c "
import json, glob, os

files = sorted(glob.glob('${RESULT_DIR}/*.json'))
results = [json.load(open(f)) for f in files]

# Sort: method, schedule, steps
results.sort(key=lambda x: (x['method'], x['schedule'], x['steps']))

header = f'{\"Method\":>6} | {\"Schedule\":>8} | {\"Steps\":>5} | {\"PA\":>8} | {\"CD\":>10} | {\"RE\":>8} | {\"TE\":>10}'
print(header)
print('-' * len(header))

prev_method = ''
for r in results:
    if r['method'] != prev_method and prev_method:
        print('-' * len(header))
    prev_method = r['method']
    print(f'{r[\"method\"]:>6} | {r[\"schedule\"]:>8} | {r[\"steps\"]:>5} | {r[\"part_accuracy\"]:>8.4f} | {r[\"object_chamfer\"]:>10.6f} | {r[\"rotation_error\"]:>8.4f} | {r[\"translation_error\"]:>10.6f}')

# Highlight best RSBM config
rsbm_results = [r for r in results if r['method'] == 'RSBM']
if rsbm_results:
    best_cd = min(rsbm_results, key=lambda x: x['object_chamfer'])
    best_pa = max(rsbm_results, key=lambda x: x['part_accuracy'])
    print()
    print(f'Best CD:  RSBM {best_cd[\"schedule\"]} {best_cd[\"steps\"]}steps → CD={best_cd[\"object_chamfer\"]:.6f}')
    print(f'Best PA:  RSBM {best_pa[\"schedule\"]} {best_pa[\"steps\"]}steps → PA={best_pa[\"part_accuracy\"]:.4f}')

# Compare: best RSBM few-step vs RPF 50-step
rpf_50 = [r for r in results if r['method'] == 'RPF' and r['steps'] == 50]
rsbm_5 = [r for r in results if r['method'] == 'RSBM' and r['steps'] == 5]
if rpf_50 and rsbm_5:
    rpf = rpf_50[0]
    # Find best 5-step schedule
    rsbm_5_best = min(rsbm_5, key=lambda x: x['object_chamfer'])
    cd_improve = (rpf['object_chamfer'] - rsbm_5_best['object_chamfer']) / rpf['object_chamfer'] * 100
    print()
    print(f'Key comparison: RSBM-{rsbm_5_best[\"schedule\"]}-5steps vs RPF-50steps')
    print(f'  CD: {rsbm_5_best[\"object_chamfer\"]:.6f} vs {rpf[\"object_chamfer\"]:.6f} ({cd_improve:+.1f}%)')
    print(f'  PA: {rsbm_5_best[\"part_accuracy\"]:.4f} vs {rpf[\"part_accuracy\"]:.4f}')
    print(f'  NFE: 9 vs 50 (5.6x faster)')
"

echo ""
echo "Done! Results saved to ${RESULT_DIR}/"
