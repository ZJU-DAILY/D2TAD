"""Training and evaluation"""

import os

import hydra
import numpy as np
import run_train
import torch.multiprocessing as mp
import utils
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import OmegaConf, open_dict


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg):
    ngpus = cfg.ngpus
    if "load_dir" in cfg:
        hydra_cfg_path = os.path.join(cfg.load_dir, ".hydra/hydra.yaml")
        hydra_cfg = OmegaConf.load(hydra_cfg_path).hydra

        work_dir = cfg.load_dir
        n_iter = cfg.training.n_iters
        batch_size = cfg.training.batch_size

        cfg = utils.load_hydra_config_from_run(cfg.load_dir)
        with open_dict(cfg):
            cfg.training.n_iters = n_iter
            cfg.training.batch_size = batch_size

        # work_dir = cfg.work_dir
        utils.makedirs(work_dir)
    else:
        hydra_cfg = HydraConfig.get()
        work_dir = (
            hydra_cfg.run.dir
            if hydra_cfg.mode == RunMode.RUN
            else os.path.join(hydra_cfg.sweep.dir, hydra_cfg.sweep.subdir)
        )
        utils.makedirs(work_dir)

    with open_dict(cfg):
        cfg.ngpus = ngpus
        cfg.work_dir = work_dir
        cfg.wandb_name = os.path.basename(os.path.normpath(work_dir))

    # Run the training pipeline
    # Use a per-process launch port when several single-GPU trainings run in parallel.
    port = int(os.environ.get("D2TAD_MASTER_PORT", "18468"))
    logger = utils.get_logger(os.path.join(work_dir, "logs"))

    hydra_cfg = HydraConfig.get()
    if hydra_cfg.mode != RunMode.RUN:
        logger.info(f"Run id: {hydra_cfg.job.id}")

    try:
        mp.set_start_method("forkserver")
        mp.spawn(
            run_train.run_multiprocess, args=(ngpus, cfg, port), nprocs=ngpus, join=True
        )
    except Exception as e:
        logger.critical(e, exc_info=True)
        raise


if __name__ == "__main__":
    main()
