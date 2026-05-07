import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import argparse

mpl.rcParams["text.usetex"] = True


def plot_tau_estimates(tau_estimates, save_path=None):
    _, ax = plt.subplots(figsize=(6, 4))
    if tau_estimates.ndim == 1:
        tau_estimates = tau_estimates[np.newaxis, :]
    _, max_lag = tau_estimates.shape
    tau_estimates = np.where(np.isnan(tau_estimates), 0, tau_estimates)
    avg_tau = np.nanmean(tau_estimates, axis=0)
    std_tau = np.nanstd(tau_estimates, axis=0)

    n = np.sum(~np.isnan(tau_estimates), axis=0)
    se_tau = std_tau / np.sqrt(n)
    err = 2 * se_tau

    x = np.arange(1, max_lag + 1)
    ax.plot(x, avg_tau, linewidth=2, label="Mean")
    ax.fill_between(x, avg_tau - err, avg_tau + err, alpha=0.3, label=r"$\pm 2$ SE")

    ax.set_xlabel(r"$k$ (lag)", fontsize=18)
    ax.set_ylabel(r"$\hat{\tau}^m_X(k)$", fontsize=18)
    ax.legend()

    if save_path:
        plt.savefig(save_path, dpi=1200, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau_estimates_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, default=None)

    args = parser.parse_args()

    tau_estimates = np.load(args.tau_estimates_path)

    plot_tau_estimates(tau_estimates, save_path=args.save_path)
