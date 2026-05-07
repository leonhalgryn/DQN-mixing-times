import numpy as np
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from src.utils.mixing_estimation import estimate_tau_mixing
from src.utils.buffers import load_minibatches


@hydra.main(
    version_base=None,
    config_path="conf",
    config_name="mixing_estimation/minibatches/config",
)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    minibatch_file = to_absolute_path(cfg.minibatch_file)
    output_path = Path(to_absolute_path(cfg.output))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    minibatches = load_minibatches(minibatch_file)

    if minibatches.ndim == 4:
        if minibatches.shape[0] != 1:
            raise ValueError(
                "Expected a singleton first dimension for 4D minibatches, "
                f"but got shape {minibatches.shape}."
            )
        minibatches = minibatches.squeeze(axis=0)

    if minibatches.ndim != 3:
        raise ValueError(
            "Expected minibatches to have shape (n_minibatches, batch_size, dim), "
            f"but got shape {minibatches.shape}."
        )

    n_minibatches, _, _ = minibatches.shape

    tau_estimates_list = Parallel(n_jobs=cfg.num_cores)(
        delayed(estimate_tau_mixing)(
            minibatches[i],
            max_lag=cfg.max_lag,
            n_neighbours=cfg.knn,
            show_progress=False,
        )
        for i in tqdm(range(n_minibatches), desc="Minibatches")
    )

    tau_estimates = np.stack(tau_estimates_list, axis=0)

    np.save(output_path, tau_estimates)
    print(f"Tau estimates saved to {output_path}")


if __name__ == "__main__":
    main()