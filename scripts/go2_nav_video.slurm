#!/bin/bash
#SBATCH --job-name=go2_nav_play
#SBATCH --partition=opengpu.p
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/auto/ugrad_space/leonarlg/IsaacLab/logs/go2_nav_play_%j.out
#SBATCH --error=/auto/ugrad_space/leonarlg/IsaacLab/logs/go2_nav_play_%j.err
#SBATCH --exclude=poison   # keep this!

echo "Node: $(hostname)"
echo "Starting GO2 video rollout with play.py..."

source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

cd /auto/ugrad_space/leonarlg/IsaacLab

########################
# CONFIG YOU CHANGE
########################

# Which training run + checkpoint to visualize:
RUN_DIR="2025-11-29_20-00-41"
MODEL_PT="model_499.pt"

# Video length in env steps
VIDEO_STEPS=500

# This is the policy log root for the ROUGH task
LOG_ROOT="/auto/ugrad_space/leonarlg/IsaacLab/logs/rsl_rl"
ENV_SRC="unitree_go2_flat"
TASK_ID="Isaac-Nav-Go2-Flat-v0"

# Where play.py will write the video by default:
VIDEO_SRC_DIR="${LOG_ROOT}/${ENV_SRC}/${RUN_DIR}/videos/play"

# Where YOU want to store “final” videos (shared across milestones)
FINAL_VIDEO_ROOT="/auto/ugrad_space/leonarlg/IsaacLab/outputs/go2_nav/videos"
mkdir -p "${FINAL_VIDEO_ROOT}"

# FULL checkpoint path
CKPT_PATH="${LOG_ROOT}/${ENV_SRC}/${RUN_DIR}/${MODEL_PT}"

echo "Using run dir: ${RUN_DIR}"
echo "Checkpoint:    ${CKPT_PATH}"
echo "Video src dir: ${VIDEO_SRC_DIR}"
echo "Final videos:  ${FINAL_VIDEO_ROOT}"

########################
# RUN PLAY.PY
########################

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task ${TASK_ID} \
    --num_envs 1 \
    --resume \
    --load_run "${RUN_DIR}" \
    --checkpoint "${CKPT_PATH}" \
    --video \
    --video_length ${VIDEO_STEPS} \
    --headless \
    --device cuda \
    --info

echo "Play/video rollout finished. Collecting videos..."

########################
# COPY VIDEO(S) OUT
########################

# IsaacLab usually writes .mp4s under: ${VIDEO_SRC_DIR}/*.mp4
if compgen -G "${VIDEO_SRC_DIR}/*.mp4" > /dev/null; then
    TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"

    for f in "${VIDEO_SRC_DIR}"/*.mp4; do
        base="$(basename "$f" .mp4)"
        # Example name: m3_unitree_go2_rough_2025-11-28_21-06-33_policy_0_2025-11-29_17-00-00.mp4
        DEST="${FINAL_VIDEO_ROOT}/${RUN_DIR}_${base}_${TIMESTAMP}.mp4"
        cp "$f" "$DEST"
        echo "Saved final video: ${DEST}"
    done
else
    echo "WARNING: No .mp4 files found in ${VIDEO_SRC_DIR}"
fi

echo "Play/video job finished."