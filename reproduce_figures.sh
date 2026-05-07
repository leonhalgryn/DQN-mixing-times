#!/usr/bin/env bash
set -euo pipefail

mkdir -p figures

for env in cartpole frozenlake lunarlander; do
  python -m experiments.plot_tau_estimates \
    --tau_estimates_path "./data/yu_sampling/tau_estimates/${env}_trajectory.npy" \
    --save_path "./figures/${env}_trajectory_tau.pdf"

  python -m experiments.plot_tau_estimates \
    --tau_estimates_path "./data/yu_sampling/tau_estimates/${env}_minibatches.npy" \
    --save_path "./figures/${env}_minibatch_tau.pdf"
done

echo "Figures saved to ./figures"