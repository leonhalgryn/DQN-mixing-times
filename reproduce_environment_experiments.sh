#!/usr/bin/env bash
set -euo pipefail

# Generic full-reproduction script for one environment.
#
# Usage:
#   bash scripts/reproduce_environment.sh cartpole
#   bash scripts/reproduce_environment.sh frozenlake
#   bash scripts/reproduce_environment.sh lunarlander
#
# By default, this script uses the defaults from the Hydra configs.
#
# Optional overrides:
#   SEED=123
#   SAMPLER=yu_contiguous_block
#   MAX_LAG=50
#   KNN=40
#   N_ROLLOUTS=500
#   NUM_CORES=-1
#   DATA_ROOT=data/yu_sampling
#   FIGURES_ROOT=figures/yu_sampling

if [ "$#" -ne 1 ]; then
    echo "Usage: bash scripts/reproduce_environment.sh <environment>"
    echo ""
    echo "Available environments:"
    echo "  cartpole"
    echo "  frozenlake"
    echo "  lunarlander"
    exit 1
fi

ENV_KEY="$1"

case "${ENV_KEY}" in
    cartpole)
        DQN_CONFIG="cartpole"
        MODEL_NAME="cartpole"
        ;;
    frozenlake)
        DQN_CONFIG="frozenlake"
        MODEL_NAME="frozenlake"
        ;;
    lunarlander)
        DQN_CONFIG="lunarlander"
        MODEL_NAME="lunarlander"
        ;;
    *)
        echo "Unknown environment: ${ENV_KEY}"
        echo ""
        echo "Available environments:"
        echo "  cartpole"
        echo "  frozenlake"
        echo "  lunarlander"
        exit 1
        ;;
esac

DATA_ROOT="${DATA_ROOT:-data/yu_sampling}"
FIGURES_ROOT="${FIGURES_ROOT:-figures/yu_sampling}"

BUFFER_DIR="${DATA_ROOT}/replay_buffers"
MODEL_DIR="${DATA_ROOT}/models"
MINIBATCH_DIR="${DATA_ROOT}/minibatches"
TAU_DIR="${DATA_ROOT}/tau_estimates"
FIGURE_DIR="${FIGURES_ROOT}"

MODEL_PATH="${MODEL_DIR}/${MODEL_NAME}.pt"
MINIBATCH_PATH="${MINIBATCH_DIR}/${MODEL_NAME}_minibatches.npy"
MINIBATCH_TAU_PATH="${TAU_DIR}/${MODEL_NAME}_minibatches.npy"
TRAJECTORY_TAU_PATH="${TAU_DIR}/${MODEL_NAME}_trajectory.npy"

MINIBATCH_FIGURE_PATH="${FIGURE_DIR}/${MODEL_NAME}_minibatches.pdf"
TRAJECTORY_FIGURE_PATH="${FIGURE_DIR}/${MODEL_NAME}_trajectory.pdf"

mkdir -p \
    "${BUFFER_DIR}" \
    "${MODEL_DIR}" \
    "${MINIBATCH_DIR}" \
    "${TAU_DIR}" \
    "${FIGURE_DIR}"

export HYDRA_FULL_ERROR=1

TRAIN_OVERRIDES=(
    dqn="${DQN_CONFIG}"
    buffer_path="${BUFFER_DIR}"
    save_model_path="${MODEL_PATH}"
    minibatch_log_path="${MINIBATCH_PATH}"
    render=false
)

MINIBATCH_OVERRIDES=(
    minibatch_file="${MINIBATCH_PATH}"
    output="${MINIBATCH_TAU_PATH}"
)

TRAJECTORY_OVERRIDES=(
    train/dqn="${DQN_CONFIG}"
    load_model_path="${MODEL_PATH}"
    output="${TRAJECTORY_TAU_PATH}"
)

if [[ -n "${SEED:-}" ]]; then
    TRAIN_OVERRIDES+=(seed="${SEED}")
    TRAJECTORY_OVERRIDES+=(seed="${SEED}")
fi

if [[ -n "${SAMPLER:-}" ]]; then
    TRAIN_OVERRIDES+=(sampler="${SAMPLER}")
fi

if [[ -n "${MAX_LAG:-}" ]]; then
    MINIBATCH_OVERRIDES+=(max_lag="${MAX_LAG}")
    TRAJECTORY_OVERRIDES+=(max_lag="${MAX_LAG}")
fi

if [[ -n "${TRAJECTORY_MAX_LAG:-}" ]]; then
    TRAJECTORY_OVERRIDES+=(max_lag="${TRAJECTORY_MAX_LAG}")
fi

if [[ -n "${KNN:-}" ]]; then
    MINIBATCH_OVERRIDES+=(knn="${KNN}")
    TRAJECTORY_OVERRIDES+=(knn="${KNN}")
fi


if [[ -n "${NUM_CORES:-}" ]]; then
    MINIBATCH_OVERRIDES+=(num_cores="${NUM_CORES}")
    TRAJECTORY_OVERRIDES+=(num_cores="${NUM_CORES}")
fi

if [[ -n "${N_ROLLOUTS:-}" ]]; then
    TRAJECTORY_OVERRIDES+=(n_rollouts="${N_ROLLOUTS}")
fi

echo "============================================================"
echo "Reproducing experiment for environment: ${ENV_KEY}"
echo "DQN config: ${DQN_CONFIG}"
echo ""
echo "Using Hydra defaults for all parameters except:"
echo "  - environment config"
echo "  - output paths"
echo ""
echo "Optional shell overrides currently set:"
echo "  SEED=${SEED:-<config default>}"
echo "  SAMPLER=${SAMPLER:-<config default>}"
echo "  MAX_LAG=${MAX_LAG:-<config default>}"
echo "  KNN=${KNN:-<config default>}"
echo "  N_ROLLOUTS=${N_ROLLOUTS:-<config default>}"
echo "  NUM_CORES=${NUM_CORES:-<config default>}"
echo "============================================================"
echo ""

echo "Step 1/5: Training DQN and logging minibatches..."
python -m experiments.train_agent "${TRAIN_OVERRIDES[@]}"

echo ""
echo "Step 2/5: Estimating minibatch tau-mixing coefficients..."
python -m experiments.estimate_minibatch_mixing "${MINIBATCH_OVERRIDES[@]}"

echo ""
echo "Step 3/5: Estimating trajectory tau-mixing coefficients..."
python -m experiments.estimate_trajectory_dependence "${TRAJECTORY_OVERRIDES[@]}"

echo ""
echo "Step 4/5: Plotting minibatch tau-mixing estimates..."
python -m experiments.plot_tau_estimates \
    --tau_estimates_path="${MINIBATCH_TAU_PATH}" \
    --save_path="${MINIBATCH_FIGURE_PATH}"

echo ""
echo "Step 5/5: Plotting trajectory tau-mixing estimates..."
python -m experiments.plot_tau_estimates \
    --tau_estimates_path="${TRAJECTORY_TAU_PATH}" \
    --save_path="${TRAJECTORY_FIGURE_PATH}"

echo ""
echo "============================================================"
echo "Completed reproduction for environment: ${ENV_KEY}"
echo ""
echo "Saved model:"
echo "  ${MODEL_PATH}"
echo ""
echo "Saved minibatches:"
echo "  ${MINIBATCH_PATH}"
echo ""
echo "Saved tau estimates:"
echo "  ${MINIBATCH_TAU_PATH}"
echo "  ${TRAJECTORY_TAU_PATH}"
echo ""
echo "Saved figures:"
echo "  ${MINIBATCH_FIGURE_PATH}"
echo "  ${TRAJECTORY_FIGURE_PATH}"
echo "============================================================"