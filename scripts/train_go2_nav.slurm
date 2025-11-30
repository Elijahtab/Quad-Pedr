#!/bin/bash
#SBATCH --job-name=go2_nav_train
#SBATCH --output=/auto/ugrad_space/leonarlg/IsaacLab/logs/go2_nav_train_%j.out
#SBATCH --error=/auto/ugrad_space/leonarlg/IsaacLab/logs/go2_nav_train_%j.err

#SBATCH --partition=opengpu.p
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=07:00:00
#SBATCH --exclude=poison

# Usage:
#   sbatch train_go2_nav_all.sh 1        # Milestone 1 (flat, fresh)
#   sbatch train_go2_nav_all.sh 2        # Milestone 2 (flat, maybe resume)
#   sbatch train_go2_nav_all.sh 25       # Milestone 2.5 (ROUGH, resume from 2)
#   sbatch train_go2_nav_all.sh 3        # Milestone 3 (ROUGH, resume from 2.5 if exists)

MILESTONE="${1:-1}"

echo "Starting Milestone ${MILESTONE} on node: $(hostname)"

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

# IsaacLab root
cd /auto/ugrad_space/leonarlg/IsaacLab

mkdir -p /auto/ugrad_space/leonarlg/IsaacLab/outputs/go2_nav/

TASK_ID="Isaac-Nav-Go2-Flat-v0"
NUM_ENVS=2048
MAX_ITERS=2000
EXTRA_ARGS=""

###############################################
# Helper: Auto-detect past run directories
###############################################

LOG_ROOT="/auto/ugrad_space/leonarlg/IsaacLab/logs/rsl_rl"

# Look for runs:
# - flat / milestone-2
# - rough / milestone-2.5
LATEST_FLAT_RUN=$(ls -td ${LOG_ROOT}/unitree_go2_flat/* 2>/dev/null | head -1)
LATEST_ROUGH25_RUN=$(ls -td ${LOG_ROOT}/unitree_go2_rough25/* 2>/dev/null | head -1)

###############################################
# Milestone switching logic
###############################################
case "$MILESTONE" in

  1)
    echo ">>> Milestone 1: FLAT, fresh training."
    TASK_ID="Isaac-Nav-Go2-Flat-v0"
    MAX_ITERS=500
    ;;

  2)
    echo ">>> Milestone 2: FLAT, optional resume."

    RUN_NAME="${RUN_NAME:-}"
    CKPT="${CKPT:-}"

    if [[ -n "$RUN_NAME" && -n "$CKPT" ]]; then
        EXTRA_ARGS="--resume --load_run ${RUN_NAME} --checkpoint ${CKPT}"
        echo "Resuming manually from $RUN_NAME ($CKPT)"
    elif [[ -n "$LATEST_FLAT_RUN" ]]; then
        echo "Auto-detected latest flat run: $LATEST_FLAT_RUN"
        EXTRA_ARGS="--resume --load_run $(basename $LATEST_FLAT_RUN)"
    else
        echo "No flat checkpoints found → starting Milestone 2 from scratch."
    fi

    TASK_ID="Isaac-Nav-Go2-Flat-v0"
    MAX_ITERS=1000
    ;;

  25)
    echo ">>> Milestone 2.5: ROUGH, resume from Milestone 2."

    if [[ -n "$LATEST_FLAT_RUN" ]]; then
        echo "Using FLAT checkpoint: $LATEST_FLAT_RUN"
        EXTRA_ARGS="--resume --load_run $(basename $LATEST_FLAT_RUN)"
    else
        echo "No flat checkpoint found → Milestone 2.5 starts from scratch."
    fi

    TASK_ID="Isaac-Nav-Go2-Rough-v0"
    MAX_ITERS=1000
    ;;

  3)
    echo ">>> Milestone 3: ROUGH terrain."

    RUN_NAME="${RUN_NAME:-}"
    CKPT="${CKPT:-}"

    # -------------------------------
    # NEW: Resume if 2.5 exists
    # -------------------------------
    if [[ -n "$RUN_NAME" && -n "$CKPT" ]]; then
        EXTRA_ARGS="--resume --load_run ${RUN_NAME} --checkpoint ${CKPT}"
        echo "Manual resume for M3: $RUN_NAME ($CKPT)"

    elif [[ -n "$LATEST_ROUGH25_RUN" ]]; then
        echo "Auto-resuming from Milestone 2.5 checkpoint: $LATEST_ROUGH25_RUN"
        EXTRA_ARGS="--resume --load_run $(basename $LATEST_ROUGH25_RUN)"

    else
        echo "No Milestone 2.5 checkpoints found → M3 starts fresh."
    fi

    TASK_ID="Isaac-Nav-Go2-Rough-v0"
    MAX_ITERS=2000
    ;;

  *)
    echo "!! Unknown milestone ${MILESTONE}. Use: 1, 2, 25, or 3."
    exit 1
    ;;
esac

echo ""
echo "------------------------------------------"
echo "Running milestone ${MILESTONE} with:"
echo "Task:        ${TASK_ID}"
echo "Iterations:  ${MAX_ITERS}"
echo "Extra args:  ${EXTRA_ARGS}"
echo "------------------------------------------"
echo ""

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task ${TASK_ID} \
    --num_envs ${NUM_ENVS} \
    --max_iterations ${MAX_ITERS} \
    --headless \
    ${EXTRA_ARGS}

echo "Training for Milestone ${MILESTONE} finished!"