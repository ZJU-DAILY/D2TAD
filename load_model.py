import os
from pathlib import Path
import torch
from model import SEDD
import utils
from model.ema import ExponentialMovingAverage
import graph_lib
import noise_lib

from omegaconf import OmegaConf

def load_model_hf(dir, device):
    score_model = SEDD.from_pretrained(dir).to(device)
    graph = graph_lib.get_graph(score_model.config, device)
    noise = noise_lib.get_noise(score_model.config).to(device)
    return score_model, graph, noise


def _resolve_run_root(path: Path) -> Path:
    """Find the nearest ancestor that contains a Hydra config."""
    if path.is_file():
        path = path.parent
    current = path
    while True:
        if (current / ".hydra" / "config.yaml").exists():
            return current
        if current.parent == current:
            raise FileNotFoundError(
                f"Could not find a Hydra config for {path}. Please pass a training run directory."
            )
        current = current.parent


def _resolve_checkpoint_path(original_path: Path, run_root: Path) -> Path:
    """Determine which checkpoint file to load."""
    if original_path.is_file():
        return original_path
    ckpt_path = original_path / "checkpoints-meta" / "checkpoint.pth"
    if ckpt_path.exists():
        return ckpt_path
    ckpt_path = run_root / "checkpoints-meta" / "checkpoint.pth"
    if ckpt_path.exists():
        return ckpt_path
    raise FileNotFoundError(
        f"Could not find checkpoint.pth under {original_path} or {run_root}."
    )


def load_model_local(root_dir, device):
    path = Path(root_dir).expanduser().resolve()
    run_root = _resolve_run_root(path)

    cfg = utils.load_hydra_config_from_run(run_root)
    graph = graph_lib.get_graph(cfg, device)
    noise = noise_lib.get_noise(cfg).to(device)
    score_model = SEDD(cfg).to(device)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=cfg.training.ema)

    ckpt_path = _resolve_checkpoint_path(path, run_root)
    loaded_state = torch.load(ckpt_path, map_location=device)
    # state_dict = loaded_state.get("model", loaded_state) if isinstance(loaded_state, dict) else loaded_state
    # score_model.load_state_dict(state_dict, strict=False)

    score_model.load_state_dict(loaded_state['model'])
    ema.load_state_dict(loaded_state['ema'])

    ema.store(score_model.parameters())
    ema.copy_to(score_model.parameters())

    # if isinstance(loaded_state, dict) and "ema" in loaded_state:
    #     ema.load_state_dict(loaded_state["ema"])
    #     ema.store(score_model.parameters())
    #     ema.copy_to(score_model.parameters())
    return score_model, graph, noise


def load_model(root_dir, device):
    # path = Path(root_dir).expanduser()
    # if path.exists():
    #     return load_model_local(root_dir, device)
    try:
        return load_model_hf(root_dir, device)
    except:
    # except Exception:
        return load_model_local(root_dir, device)
