#!/bin/bash
# =============================================================================
# MedZFS Full Training Pipeline
# =============================================================================
# Runs the complete three-stage training for all datasets.
# Usage: bash scripts/run_training.sh
# =============================================================================

set -e

SEED=42
GPU=0

echo "=============================================="
echo "  MedZFS Training Pipeline"
echo "=============================================="

for DATASET in abd_mri abd_ct cmr; do
    CONFIG="configs/train_${DATASET}.yaml"

    echo ""
    echo "=============================="
    echo "  Training on: $DATASET"
    echo "=============================="

    # Stage 1: Zero-shot pre-training
    echo "  → Stage 1: Zero-shot pre-training..."
    python -m training.train \
        --config "$CONFIG" \
        --stage 1 \
        --seed $SEED \
        --gpu $GPU

    # Stage 2: Episodic meta-learning
    echo "  → Stage 2: Episodic meta-learning..."
    python -m training.train \
        --config "$CONFIG" \
        --stage 2 \
        --seed $SEED \
        --gpu $GPU \
        --resume "checkpoints/stage1_best.pth"

    # Stage 3: Joint bilevel optimization
    echo "  → Stage 3: Joint bilevel optimization..."
    python -m training.train \
        --config "$CONFIG" \
        --stage 3 \
        --seed $SEED \
        --gpu $GPU \
        --resume "checkpoints/stage2_best.pth"

    echo "  ✓ $DATASET training complete!"
done

echo ""
echo "=============================================="
echo "  All training complete!"
echo "  Running evaluation..."
echo "=============================================="

python -m evaluation.evaluate \
    --config configs/eval.yaml \
    --checkpoint checkpoints/stage3_best.pth \
    --shots 0 1 5 10 \
    --datasets abd_mri abd_ct cmr \
    --output_dir results/

echo "  ✓ Evaluation complete! Results in results/"
