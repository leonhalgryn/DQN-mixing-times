import numpy as np
import ot
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


def estimate_tau_mixing(
    buffer,
    max_lag=10,
    m=1,
    n_neighbours=20,
    show_progress=True,
    use_sinkhorn=False,
    sinkhorn_reg=1e-2,
    leave_one_out=True,
    n_permutations=1,
):
    """
    Estimate a finite-window Wasserstein tau-mixing proxy.

    For each lag k, this estimates the truncated dependence score

        tau_X^m(k) ≈ max_i W_1( L_hat(Y | Z_i), L_hat(Y) ),

        (this corresponds to Eq. (D.10) in the paper)

    where Z_i = (X_i, ..., X_{i+m-1}) is a length-m history window and
    Y_i = X_{i+m-1+k} is the k-step-ahead target. The conditional law is
    approximated by the empirical distribution of the futures associated with
    the nearest-neighbor histories of Z_i, and the marginal law by the empirical
    distribution of all valid futures. A permutation baseline is subtracted to
    reduce finite-sample bias, giving

        max(tau_obs(k) - tau_perm(k), 0).

        (this corresponds to Eq. (D.14) in the paper)

    This is a computable proxy for the finite-memory target tau_X^m(k), not a
    direct estimate of the full population tau-mixing coefficient tau_X(k).

    Parameters
    ----------
    buffer : array_like, shape (T, d)
        Time series sample.
    max_lag : int, default=10
        Maximum lag k to estimate.
    m : int, default=1
        History/window length used for conditioning.
    n_neighbours : int, default=20
        Number of nearest neighbors used to approximate the local conditional
        law. Spelled with British English to match the function argument.
    show_progress : bool, default=True
        If True, display a tqdm progress bar over lags.
    use_sinkhorn : bool, default=False
        If True, use entropic optimal transport via ``ot.sinkhorn2``. Otherwise
        use exact Earth Mover's Distance via ``ot.emd2``.
    sinkhorn_reg : float, default=1e-2
        Entropic regularization parameter used only when ``use_sinkhorn=True``.
    leave_one_out : bool, default=True
        If True, exclude each history window from its own neighborhood.
    n_permutations : int, default=1
        Number of random permutations used to estimate the finite-sample
        baseline.

    Returns
    -------
    tau_estimates : ndarray, shape (max_lag,)
        Nonnegative permutation-centered estimates for lags 1, ..., max_lag.
        Entries are NaN when too few observations are available for a lag
        (see Eq. (D.15) and (D.16) in the paper) -- these are later set to zero
        when plotting the estimated coefficients.
    """
    buffer = np.asarray(buffer)
    T, d = buffer.shape
    tau_estimates = np.full(max_lag, np.nan)

    for lag in tqdm(range(1, max_lag + 1), disable=not show_progress):
        N = T - m - lag + 1
        if N <= 0:
            continue

        Z = np.array([buffer[t : t + m].reshape(-1) for t in range(N)])
        Y = buffer[m - 1 + lag : m - 1 + lag + N]

        k_query = n_neighbours + 1 if leave_one_out else n_neighbours
        if N <= k_query:
            continue

        nbrs = NearestNeighbors(n_neighbors=k_query)
        nbrs.fit(Z)

        a_global = np.ones(N) / N
        local_w1 = []

        for i in range(N):
            _, idx = nbrs.kneighbors(Z[i : i + 1])
            idx = idx[0]

            if leave_one_out:
                idx = idx[idx != i]

            idx = idx[:n_neighbours]
            if len(idx) < n_neighbours:
                continue

            local_sample = Y[idx]
            a_local = np.ones(len(local_sample)) / len(local_sample)

            M = ot.dist(local_sample, Y, metric="euclidean")

            if use_sinkhorn:
                w1 = ot.sinkhorn2(a_local, a_global, M, reg=sinkhorn_reg)
            else:
                w1 = ot.emd2(a_local, a_global, M)

            local_w1.append(w1)

        perm_stats = []

        for _ in range(n_permutations):

            perm = np.random.permutation(N)
            Y_perm = Y[perm]

            local_w1_perm = []

            for i in range(N):
                _, indices = nbrs.kneighbors(Z[i : i + 1])
                local_sample = Y_perm[indices[0]]

                k = len(local_sample)
                a_local = np.ones(k) / k

                M = ot.dist(local_sample, Y_perm, metric="euclidean")
                w1 = ot.emd2(a_local, a_global, M)

                local_w1_perm.append(w1)

            perm_stats.append(np.max(local_w1_perm))

        tau_perm = np.mean(perm_stats)

        if local_w1:
            tau_obs = np.max(local_w1)
            tau_estimates[lag - 1] = max(tau_obs - tau_perm, 0)

    return tau_estimates


if __name__ == "__main__":
    pass
