"""
train_vitra_navi.py - Navigation training for VITRA.

Current scope:
- Train only (evaluation is temporarily disabled).
- Load pretrained VLM backbone from model_load_path (default: VITRA-VLA/VITRA-VLA-3B).
- Reinitialize DiT-S action head and train only action head parameters.
- Use LeRobot-format navigation data with image + language + action.
- Deterministic episode split: 95% train by seed.
"""

import argparse
import json
import math
import os
import time
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from vitra.utils.config_utils import load_config
from vitra.utils.data_utils import PaddedCollatorForHandPrediction


DEFAULT_PRETRAIN = "VITRA-VLA/VITRA-VLA-3B"
DEFAULT_TRAIN_RATIO = 0.95
DEFAULT_WAYPOINT_HORIZON = 16


def _get_load_model():
    """Lazy import to avoid heavy model deps for stats-only command."""
    try:
        import huggingface_hub as _hf_hub

        if not hasattr(_hf_hub, "is_offline_mode"):
            def _is_offline_mode():
                return str(os.environ.get("HF_HUB_OFFLINE", "0")).lower() in ("1", "true", "yes")

            _hf_hub.is_offline_mode = _is_offline_mode
    except Exception:
        pass

    from vitra.models.vla_builder import load_model

    return load_model


def _load_stats_json(data_dir: str) -> dict:
    stats_path = os.path.join(data_dir, "meta", "stats.json")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Missing LeRobot stats: {stats_path}")
    with open(stats_path) as f:
        return json.load(f)


def _resolve_existing_dir(raw_path: str, config_path: str) -> str:
    """Resolve dataset dir robustly across different launch CWDs."""
    p = Path(raw_path)
    candidates = []

    if p.is_absolute():
        candidates.append(p)
    else:
        # 1) As provided (relative to current CWD)
        candidates.append(Path.cwd() / p)
        # 2) Relative to config file dir
        cfg_dir = Path(config_path).resolve().parent
        candidates.append(cfg_dir / p)
        # 3) Relative to VITRA root
        vitra_root = Path(__file__).resolve().parents[1]
        candidates.append(vitra_root / p)
        # 4) Relative to workspace root (parent of VITRA root)
        workspace_root = Path(__file__).resolve().parents[2]
        candidates.append(workspace_root / p)

    for c in candidates:
        if c.exists() and c.is_dir():
            return str(c.resolve())

    checked = "\n  - " + "\n  - ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Could not resolve data_dir: {raw_path}\nChecked:{checked}"
    )


def _pick(stats: dict, key: str, fallbacks) -> Optional[dict]:
    if key in stats:
        return stats[key]
    for k in fallbacks:
        if k in stats:
            return stats[k]
    return None


def compute_statistics_from_lerobot(
    data_dir: str,
    action_col: str = "action",
    state_col: str = "observation.state",
) -> dict:
    stats = _load_stats_json(data_dir)
    action_stats = _pick(stats, action_col, ["action"])
    if action_stats is None:
        raise KeyError(f"Action stats key not found: {action_col}")

    state_stats = _pick(stats, state_col, ["observation.state", "state", "observation.pose"])
    out = {
        "dataset_name": os.path.basename(os.path.abspath(data_dir)),
        "action_dim": len(action_stats["mean"]),
        "action_mean": action_stats["mean"],
        "action_std": action_stats["std"],
        "action_min": action_stats.get("min"),
        "action_max": action_stats.get("max"),
    }

    if state_stats is not None:
        out.update(
            {
                "state_dim": len(state_stats["mean"]),
                "state_mean": state_stats["mean"],
                "state_std": state_stats["std"],
                "state_min": state_stats.get("min"),
                "state_max": state_stats.get("max"),
            }
        )
    else:
        out.update(
            {
                "state_dim": 0,
                "state_mean": [],
                "state_std": [],
                "state_min": [],
                "state_max": [],
            }
        )

    return out


class ZNormalizer:
    def __init__(self, mean, std):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.maximum(np.asarray(std, dtype=np.float32), 1e-6)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std


class XYOnlyNormalizer:
    """Normalize only x/y channels and keep angular channels unchanged."""

    def __init__(self, xy_mean, xy_std):
        self.xy_mean = np.asarray(xy_mean, dtype=np.float32)
        self.xy_std = np.maximum(np.asarray(xy_std, dtype=np.float32), 1e-6)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        y = x.copy()
        y[..., 0:2] = (y[..., 0:2] - self.xy_mean) / self.xy_std
        return y


class XYPerStepNormalizer:
    """Normalize x/y per waypoint index for [H, D] action chunks."""

    def __init__(self, xy_mean_h2, xy_std_h2):
        m = np.asarray(xy_mean_h2, dtype=np.float32)
        s = np.maximum(np.asarray(xy_std_h2, dtype=np.float32), 1e-6)
        if m.ndim != 2 or m.shape[1] != 2:
            raise ValueError(f"xy_mean_h2 must be [H,2], got {m.shape}")
        if s.shape != m.shape:
            raise ValueError(f"xy_std_h2 shape mismatch: {s.shape} vs {m.shape}")
        self.xy_mean = m
        self.xy_std = s

    def normalize(self, x: np.ndarray) -> np.ndarray:
        y = x.copy()
        if y.ndim != 2 or y.shape[0] != self.xy_mean.shape[0] or y.shape[1] < 2:
            raise ValueError(f"Expected [H,D] action with H={self.xy_mean.shape[0]}, got {y.shape}")
        y[:, 0:2] = (y[:, 0:2] - self.xy_mean) / self.xy_std
        return y


def _to_waypoints_xytheta(action_raw: np.ndarray, waypoint_horizon: int) -> np.ndarray:
    """Convert raw action to [H, 3] = [x, y, theta]."""
    a = np.asarray(action_raw, dtype=np.float32)
    if a.ndim == 1 and a.shape[0] == waypoint_horizon * 3:
        return a.reshape(waypoint_horizon, 3)
    if a.ndim == 2 and a.shape == (waypoint_horizon, 3):
        return a
    raise ValueError(
        f"Unsupported waypoint action shape {a.shape}; expected ({waypoint_horizon * 3},) or ({waypoint_horizon}, 3)."
    )


def _xytheta_to_xysincos(waypoints_xytheta: np.ndarray) -> np.ndarray:
    theta = waypoints_xytheta[:, 2]
    return np.stack(
        [
            waypoints_xytheta[:, 0],
            waypoints_xytheta[:, 1],
            np.sin(theta),
            np.cos(theta),
        ],
        axis=-1,
    ).astype(np.float32)


def inspect_navigation_action_format(data_dir: str, action_col: str, waypoint_horizon: int) -> Dict[str, int]:
    files = sorted((Path(data_dir) / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files under {(Path(data_dir) / 'data')}")
    row0 = pd.read_parquet(files[0]).iloc[0]
    raw = np.asarray(row0[action_col], dtype=np.float32)

    if (raw.ndim == 1 and raw.shape[0] == waypoint_horizon * 3) or (
        raw.ndim == 2 and raw.shape == (waypoint_horizon, 3)
    ):
        return {
            "mode": 1,  # waypoint mode
            "future_window": waypoint_horizon,
            "action_dim": 4,
        }

    if raw.ndim == 1:
        return {
            "mode": 0,  # legacy per-step mode
            "future_window": waypoint_horizon,
            "action_dim": int(raw.shape[0]),
        }

    raise ValueError(f"Unsupported action format in dataset: shape={raw.shape}")


class NavigationDatasetCore:
    """LeRobot navigation core dataset with episode-level split."""

    def __init__(
        self,
        data_dir: str,
        chunk_size: int,
        split: str,
        train_episode_ratio: float,
        split_seed: int,
        action_col: str = "action",
        state_col: str = "observation.state",
        image_col: str = "observation.images.main",
        instruction_col: Optional[str] = None,
        default_instruction: str = "Navigate to the target location safely.",
        override_state_dim: Optional[int] = None,
        max_parquet_files: Optional[int] = None,
        max_rows: Optional[int] = None,
        waypoint_horizon: int = DEFAULT_WAYPOINT_HORIZON,
        waypoint_xy_norm: str = "global",
    ):
        self.data_root = Path(data_dir)
        self.chunk_size = int(chunk_size)
        self.waypoint_horizon = int(waypoint_horizon)
        self.waypoint_xy_norm = str(waypoint_xy_norm)
        self.action_col = action_col
        self.state_col = state_col
        self.image_col = image_col
        self.instruction_col = instruction_col
        self.default_instruction = default_instruction

        files = sorted((self.data_root / "data").rglob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files found under {(self.data_root / 'data')}")
        if max_parquet_files and max_parquet_files > 0:
            files = files[: int(max_parquet_files)]

        dfs, loaded = [], 0
        row_budget = int(max_rows) if (max_rows and max_rows > 0) else None
        for p in files:
            dfi = pd.read_parquet(p)
            if row_budget is not None and loaded + len(dfi) > row_budget:
                keep = max(0, row_budget - loaded)
                if keep > 0:
                    dfs.append(dfi.iloc[:keep].copy())
                break
            dfs.append(dfi)
            loaded += len(dfi)
            if row_budget is not None and loaded >= row_budget:
                break

        self.df = pd.concat(dfs, ignore_index=True)
        if "episode_index" not in self.df.columns:
            raise KeyError("Missing required column: episode_index")
        if self.action_col not in self.df.columns:
            raise KeyError(f"Missing action column: {self.action_col}")
        if self.image_col not in self.df.columns:
            raise KeyError(f"Missing image column: {self.image_col}")

        sample_action = np.asarray(self.df.iloc[0][self.action_col], dtype=np.float32)
        self.waypoint_action_mode = False
        if (sample_action.ndim == 1 and sample_action.shape[0] == self.waypoint_horizon * 3) or (
            sample_action.ndim == 2 and sample_action.shape == (self.waypoint_horizon, 3)
        ):
            self.waypoint_action_mode = True

        self.stats = compute_statistics_from_lerobot(data_dir, action_col=action_col, state_col=state_col)
        self.action_dim = 4 if self.waypoint_action_mode else int(self.stats["action_dim"])
        if self.waypoint_action_mode:
            self.chunk_size = self.waypoint_horizon
        state_dim_stats = int(self.stats.get("state_dim", 0))
        self.state_dim = int(override_state_dim) if override_state_dim is not None else state_dim_stats
        if self.state_dim <= 0:
            self.state_dim = self.action_dim

        if self.waypoint_action_mode:
            xy = []
            for a in self.df[self.action_col].values:
                wp = _to_waypoints_xytheta(a, self.waypoint_horizon)
                xy.append(wp[:, :2])
            xy_nh2 = np.stack(xy, axis=0).astype(np.float32)
            if self.waypoint_xy_norm == "per_step":
                self.xy_mean = xy_nh2.mean(axis=0).astype(np.float32)  # [H,2]
                self.xy_std = np.maximum(xy_nh2.std(axis=0).astype(np.float32), 1e-6)  # [H,2]
                self.action_norm = XYPerStepNormalizer(self.xy_mean, self.xy_std)
            elif self.waypoint_xy_norm == "global":
                xy_flat = xy_nh2.reshape(-1, 2)
                self.xy_mean = xy_flat.mean(axis=0).astype(np.float32)  # [2]
                self.xy_std = np.maximum(xy_flat.std(axis=0).astype(np.float32), 1e-6)  # [2]
                self.action_norm = XYOnlyNormalizer(self.xy_mean, self.xy_std)
            else:
                raise ValueError(
                    f"Unsupported waypoint_xy_norm={self.waypoint_xy_norm}. Use 'global' or 'per_step'."
                )
        else:
            self.action_norm = ZNormalizer(self.stats["action_mean"], self.stats["action_std"])
        self.state_norm = (
            ZNormalizer(self.stats["state_mean"], self.stats["state_std"])
            if state_dim_stats > 0
            else None
        )

        self._tasks = self._load_task_map(data_dir)
        self._ep_bounds = self._build_episode_bounds()
        self._selected_episodes = self._split_episodes(split, train_episode_ratio, split_seed)
        self._valid_indices = self._build_indices(self._selected_episodes)

        print(
            f"NavigationDatasetCore[{split}]: episodes={len(self._selected_episodes)}, "
            f"frames={len(self._valid_indices)}, chunk={self.chunk_size}, "
            f"action_dim={self.action_dim}, state_dim={self.state_dim}, "
            f"waypoint_mode={self.waypoint_action_mode}"
        )
        if self.waypoint_action_mode:
            print(
                "Navigation waypoint stats: "
                f"xy_norm={self.waypoint_xy_norm} "
                f"xy_mean_shape={list(self.xy_mean.shape)} xy_std_shape={list(self.xy_std.shape)}"
            )

    @staticmethod
    def _load_task_map(data_dir: str) -> Dict[int, str]:
        tasks = {}
        tasks_json = os.path.join(data_dir, "meta", "tasks.json")
        tasks_parquet = os.path.join(data_dir, "meta", "tasks.parquet")
        if os.path.exists(tasks_json):
            with open(tasks_json) as f:
                payload = json.load(f)
            if isinstance(payload, list):
                for e in payload:
                    tasks[int(e["task_index"])] = str(e["task"])
            elif isinstance(payload, dict):
                for k, v in payload.items():
                    tasks[int(k)] = str(v)
        elif os.path.exists(tasks_parquet):
            tdf = pd.read_parquet(tasks_parquet)
            for task_text, row in tdf.iterrows():
                tasks[int(row["task_index"])] = str(task_text)
        return tasks

    def _build_episode_bounds(self):
        epi = self.df["episode_index"].values
        episodes = sorted(self.df["episode_index"].unique())
        bounds = {}
        for ep in episodes:
            idxs = np.where(epi == ep)[0]
            bounds[int(ep)] = (int(idxs[0]), int(idxs[-1]) + 1)
        return bounds

    def _split_episodes(self, split: str, train_ratio: float, seed: int):
        if split not in ("train", "val"):
            raise ValueError("split must be train or val")
        episodes = np.array(sorted(self._ep_bounds.keys()), dtype=np.int64)
        rng = np.random.default_rng(seed)
        rng.shuffle(episodes)

        train_ratio = float(train_ratio)
        train_ratio = min(max(train_ratio, 0.0), 1.0)
        n_total = len(episodes)
        n_train = int(round(n_total * train_ratio))
        n_train = min(max(n_train, 1), n_total)
        if n_total > 1 and n_train == n_total:
            n_train = n_total - 1

        train_eps = episodes[:n_train]
        val_eps = episodes[n_train:]
        selected = train_eps if split == "train" else val_eps
        if len(selected) == 0:
            selected = train_eps
        return [int(x) for x in selected]

    def _build_indices(self, episodes):
        idxs = []
        for ep in episodes:
            s, e = self._ep_bounds[ep]
            idxs.extend(range(s, e))
        return np.asarray(idxs, dtype=np.int64)

    def __len__(self):
        return len(self._valid_indices)

    def _load_image(self, row, col):
        data = row[col]
        if isinstance(data, dict) and data.get("bytes"):
            return np.array(Image.open(BytesIO(data["bytes"])).convert("RGB"))
        if isinstance(data, dict) and data.get("path"):
            return np.array(Image.open(self.data_root / data["path"]).convert("RGB"))
        return np.array(Image.open(self.data_root / str(data)).convert("RGB"))

    def _load_instruction(self, row):
        if self.instruction_col and self.instruction_col in row.index:
            txt = row[self.instruction_col]
            if isinstance(txt, str) and txt.strip():
                return txt

        if "task_index" in row.index:
            t = row["task_index"]
            if isinstance(t, (list, np.ndarray)):
                t = int(t[0]) if len(t) > 0 else 0
            if t is not None and int(t) in self._tasks:
                return self._tasks[int(t)]

        for k in ("language_instruction", "instruction", "task"):
            if k in row.index and isinstance(row[k], str) and row[k].strip():
                return row[k]

        return self.default_instruction

    def __getitem__(self, idx):
        gidx = int(self._valid_indices[idx])
        row = self.df.iloc[gidx]
        ep = int(row["episode_index"])
        _, ep_end = self._ep_bounds[ep]

        head_rgb = self._load_image(row, self.image_col)

        state = np.zeros((self.state_dim,), dtype=np.float32)
        if self.state_col in row.index:
            raw_s = np.asarray(row[self.state_col], dtype=np.float32)
            n = min(len(raw_s), self.state_dim)
            state[:n] = raw_s[:n]
            state_mask = np.zeros((self.state_dim,), dtype=np.float32)
            state_mask[:n] = 1.0
        else:
            state_mask = np.zeros((self.state_dim,), dtype=np.float32)

        if self.waypoint_action_mode:
            raw = _to_waypoints_xytheta(row[self.action_col], self.waypoint_horizon)
            actions = _xytheta_to_xysincos(raw)
            valid = self.chunk_size
            action_mask = np.ones((self.chunk_size, self.action_dim), dtype=np.float32)
        else:
            valid = min(ep_end - gidx, self.chunk_size)
            actions = np.zeros((self.chunk_size, self.action_dim), dtype=np.float32)
            for t in range(valid):
                a = np.asarray(self.df.iloc[gidx + t][self.action_col], dtype=np.float32)
                if a.shape[0] != self.action_dim:
                    raise ValueError(
                        f"Bad action shape at row {gidx + t}: {a.shape}, expected ({self.action_dim},)"
                    )
                actions[t] = a

            action_mask = np.zeros((self.chunk_size, self.action_dim), dtype=np.float32)
            action_mask[:valid, :] = 1.0

        return {
            "head_rgb": head_rgb,
            "action_list": actions,
            "action_mask": action_mask,
            "current_state": state,
            "current_state_mask": state_mask,
            "valid_frames": valid,
            "instruction": self._load_instruction(row),
        }

    def normalize(self, sample):
        valid = int(sample["valid_frames"])
        actions = sample["action_list"].copy()
        actions[:valid] = self.action_norm.normalize(actions[:valid])

        state = sample["current_state"].copy()
        if self.state_norm is not None and sample["current_state_mask"].sum() > 0:
            state = self.state_norm.normalize(state)

        sample["action_list"] = torch.tensor(actions, dtype=torch.float32)
        sample["action_mask"] = torch.tensor(sample["action_mask"], dtype=torch.float32)
        sample["current_state"] = torch.tensor(state, dtype=torch.float32)
        sample["current_state_mask"] = torch.tensor(sample["current_state_mask"], dtype=torch.float32)
        return sample


class NavigationTrainDataset(Dataset):
    def __init__(self, core: NavigationDatasetCore, processor, camera_fov_rad: float, augment: bool, head_jitter_cfg: Optional[dict]):
        self.core = core
        self.processor = processor
        self.camera_fov_rad = float(camera_fov_rad)
        self.augment = bool(augment)
        if self.augment:
            from torchvision.transforms import ColorJitter

            cfg = head_jitter_cfg or {}
            self.head_jitter = ColorJitter(
                brightness=cfg.get("brightness", 0.2),
                contrast=cfg.get("contrast", 0.2),
                saturation=cfg.get("saturation", 0.1),
                hue=cfg.get("hue", 0.03),
            )

    def __len__(self):
        return len(self.core)

    def __getitem__(self, idx):
        sample = self.core.normalize(self.core[idx])
        img = Image.fromarray(sample["head_rgb"])
        if self.augment:
            img = self.head_jitter(img)

        text = "<image>" + sample["instruction"]
        inputs = self.processor(text=text, images=[img], return_tensors="pt").to(torch.float32)

        return {
            "pixel_values": inputs["pixel_values"],
            "input_ids": inputs["input_ids"].squeeze(0),
            "labels": None,
            "dataset_name": "navigation",
            "actions": sample["action_list"],
            "action_masks": sample["action_mask"],
            "current_state": sample["current_state"],
            "current_state_mask": sample["current_state_mask"],
            "fov": torch.tensor([self.camera_fov_rad, self.camera_fov_rad], dtype=torch.float32),
        }


def make_optimizer(model, trainer_cfg):
    lr = trainer_cfg.get("lr_action_head", trainer_cfg.get("learning_rate", 1e-4))
    params = [p for p in model.act_model.parameters() if p.requires_grad]
    n = sum(p.numel() for p in params)
    print(f"optim(action_head): params={n/1e6:.2f}M lr={lr:.1e}")
    groups = [{"params": params, "lr": lr, "name": "action_head"}]

    lora_lr = trainer_cfg.get("lr_lora_pg", trainer_cfg.get("learning_rate", 1e-4))
    lora_params = []
    seen = set()

    try:
        from vitra.utils.lora import lora_params as _lora_params

        for p in _lora_params(model):
            if p.requires_grad and id(p) not in seen:
                lora_params.append(p)
                seen.add(id(p))
    except Exception:
        pass

    if hasattr(model.model, "multi_modal_projector"):
        for p in model.model.multi_modal_projector.parameters():
            if p.requires_grad and id(p) not in seen:
                lora_params.append(p)
                seen.add(id(p))

    if lora_params:
        ln = sum(p.numel() for p in lora_params)
        print(f"optim(lora_pg): params={ln/1e6:.2f}M lr={lora_lr:.1e}")
        groups.append({"params": lora_params, "lr": lora_lr, "name": "lora_pg"})

    return torch.optim.AdamW(groups, weight_decay=trainer_cfg.get("weight_decay", 0.01))


def make_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _build_train_dataset(configs, processor):
    data_cfg = configs["data"]
    aug_cfg = configs.get("augmentation", {})
    seed = int(configs.get("seed", 42))
    train_ratio = float(data_cfg.get("train_episode_ratio", 0.95))

    core = NavigationDatasetCore(
        data_dir=data_cfg["data_dir"],
        chunk_size=int(configs.get("fwd_pred_next_n", 16)),
        split="train",
        train_episode_ratio=train_ratio,
        split_seed=seed,
        action_col=data_cfg.get("action_col", "action"),
        state_col=data_cfg.get("state_col", "observation.state"),
        image_col=data_cfg.get("main_image_col", "observation.images.main"),
        instruction_col=data_cfg.get("instruction_col", "language_instruction"),
        default_instruction=data_cfg.get("default_instruction", "Navigate to the target location safely."),
        override_state_dim=configs.get("state_encoder", {}).get("state_dim"),
        max_parquet_files=data_cfg.get("max_parquet_files"),
        max_rows=data_cfg.get("max_rows"),
        waypoint_horizon=int(data_cfg.get("waypoint_horizon", DEFAULT_WAYPOINT_HORIZON)),
        waypoint_xy_norm=data_cfg.get("waypoint_xy_norm", "global"),
    )

    dataset = NavigationTrainDataset(
        core=core,
        processor=processor,
        camera_fov_rad=float(data_cfg.get("camera_fov_rad", 1.6)),
        augment=bool(aug_cfg.get("enabled", True)),
        head_jitter_cfg=aug_cfg.get("head_color_jitter"),
    )

    collator = PaddedCollatorForHandPrediction(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right",
    )

    loader = DataLoader(
        dataset,
        batch_size=int(configs["batch_size"]),
        shuffle=True,
        num_workers=int(data_cfg.get("num_workers", 4)),
        collate_fn=collator,
        drop_last=True,
        pin_memory=True,
    )
    return core, dataset, loader


def _configure_train_setup(configs, stats, inferred_action_dim: int, inferred_window: int):
    configs.setdefault("action_model", {})
    configs.setdefault("state_encoder", {})

    # Independent navigation training defaults
    configs["model_load_path"] = configs.get("model_load_path", DEFAULT_PRETRAIN)
    configs["loss_type"] = configs.get("loss_type", "navigation")
    configs["action_model"]["model_type"] = "DiT-S"
    configs["action_model"]["use_wrist_cross_attn"] = False
    configs["fwd_pred_next_n"] = int(inferred_window)

    # Explicitly bind action dim to inferred training target format.
    configs["action_model"]["action_dim"] = int(inferred_action_dim)
    if "state_dim" not in configs["state_encoder"]:
        sdim = int(stats.get("state_dim", 0))
        configs["state_encoder"]["state_dim"] = sdim if sdim > 0 else int(configs["action_model"]["action_dim"])

    print(
        "Model config: "
        f"model_load_path={configs['model_load_path']}, "
        f"DiT={configs['action_model']['model_type']}, "
        f"action_dim={configs['action_model']['action_dim']}, "
        f"use_wrist_cross_attn={configs['action_model']['use_wrist_cross_attn']}"
    )


def _set_action_head_only_trainable(model):
    for p in model.parameters():
        p.requires_grad_(False)
    model.act_model.requires_grad_(True)
    t = sum(p.numel() for p in model.parameters() if p.requires_grad)
    a = sum(p.numel() for p in model.parameters())
    print(f"Action-head-only mode: {t/1e6:.1f}M / {a/1e6:.1f}M trainable")


def _set_action_head_lora_trainable(model):
    for p in model.parameters():
        p.requires_grad_(False)

    model.act_model.requires_grad_(True)

    if hasattr(model.model, "multi_modal_projector"):
        model.model.multi_modal_projector.requires_grad_(True)

    if getattr(model, "_has_lora", False):
        from vitra.utils.lora import lora_params as _lora_params

        for p in _lora_params(model):
            p.requires_grad_(True)

    t = sum(p.numel() for p in model.parameters() if p.requires_grad)
    a = sum(p.numel() for p in model.parameters())
    print(f"Action-head+LoRA mode: {t/1e6:.1f}M / {a/1e6:.1f}M trainable")


def _reinit_action_head(model):
    model.act_model.net.initialize_weights()
    print("Reinitialized DiT action head (fresh DiT-S)")


def train(args):
    load_model = _get_load_model()
    configs = load_config(args.config)
    trainer_cfg = configs["trainer"]
    data_cfg = configs["data"]
    lora_train_cfg = configs.get("lora_training", {})
    use_lora_mode = bool(lora_train_cfg.get("enabled", False))
    lora_freeze_steps = max(0, int(lora_train_cfg.get("freeze_steps", 3000)))

    if use_lora_mode and not configs.get("lora"):
        configs["lora"] = {
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "rank": 32,
            "alpha": 64,
        }
        print("lora_training enabled without top-level 'lora'; using default LoRA config.")

    if not use_lora_mode and "lora" in configs:
        configs.pop("lora", None)

    # Make data path stable whether script is launched from workspace root
    # or from the VITRA subdirectory.
    data_cfg["data_dir"] = _resolve_existing_dir(data_cfg["data_dir"], args.config)
    print(f"Resolved data_dir: {data_cfg['data_dir']}")

    seed = int(configs.get("seed", 42))
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Seed: {seed}")

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(output_dir, "tb_logs"))
    device = torch.device("cuda")

    stats = compute_statistics_from_lerobot(
        data_cfg["data_dir"],
        action_col=data_cfg.get("action_col", "action"),
        state_col=data_cfg.get("state_col", "observation.state"),
    )

    inferred = inspect_navigation_action_format(
        data_cfg["data_dir"],
        action_col=data_cfg.get("action_col", "action"),
        waypoint_horizon=int(data_cfg.get("waypoint_horizon", DEFAULT_WAYPOINT_HORIZON)),
    )
    inferred_action_dim = int(inferred["action_dim"])
    inferred_window = int(inferred["future_window"])
    print(
        f"Inferred action format: mode={'waypoint' if inferred['mode'] == 1 else 'legacy'} "
        f"window={inferred_window} action_dim={inferred_action_dim}"
    )

    _configure_train_setup(configs, stats, inferred_action_dim=inferred_action_dim, inferred_window=inferred_window)
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(configs, f, indent=2)

    model = load_model(configs).to(device).train()
    _reinit_action_head(model)
    _set_action_head_only_trainable(model)
    model.use_bf16 = bool(configs.get("use_bf16", False))

    if use_lora_mode:
        print(
            "LoRA training mode enabled: "
            f"phase1(action-head-only) steps={lora_freeze_steps}, "
            "then phase2(action-head+LoRA+P_g)."
        )

    core, dataset, loader = _build_train_dataset(configs, model.processor)

    # Save derived training-target stats in addition to raw LeRobot stats.
    stats_out = dict(stats)
    if core.waypoint_action_mode:
        stats_out["target_format"] = "xysincos"
        stats_out["waypoint_horizon"] = int(core.waypoint_horizon)
        stats_out["waypoint_xy_norm"] = str(core.waypoint_xy_norm)
        stats_out["target_xy_mean"] = np.asarray(core.xy_mean, dtype=np.float32).tolist()
        stats_out["target_xy_std"] = np.asarray(core.xy_std, dtype=np.float32).tolist()

    with open(os.path.join(output_dir, "statistics.json"), "w") as f:
        json.dump(stats_out, f, indent=2)

    print(f"Dataset(train): {len(dataset)} samples, batch={configs['batch_size']}")

    optimizer = make_optimizer(model, trainer_cfg)
    max_steps = int(trainer_cfg.get("max_steps", 6000))
    warmup_steps = int(trainer_cfg.get("warmup_steps", 100))
    scheduler = make_scheduler(optimizer, warmup_steps, max_steps)
    lora_phase2_started = False

    grad_accum = int(trainer_cfg.get("grad_accum", 4))
    use_fp16 = bool(configs.get("use_fp16", False))
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    save_every = int(trainer_cfg.get("save_every", 1000))
    log_every = int(trainer_cfg.get("log_every", 50))

    step, micro, loss_sum = 0, 0, 0.0
    t0 = time.time()
    print(f"Training {max_steps} steps (grad_accum={grad_accum})")

    while step < max_steps:
        for batch in loader:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=bool(configs.get("use_bf16", False))):
                out = model.forward(
                    pixel_values=batch["pixel_values"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    action_labels=batch["actions"],
                    action_masks=batch["action_masks"],
                    current_state_mask=batch["current_state_mask"],
                    current_state=batch["current_state"],
                    fov=batch["fov"],
                )

            loss = out["loss"] / grad_accum
            scaler.scale(loss).backward()
            loss_sum += out["loss"].item()
            micro += 1

            if micro % grad_accum != 0:
                continue

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(trainer_cfg.get("gradient_clip_val", 1.0)))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            step += 1

            if use_lora_mode and (not lora_phase2_started) and step >= lora_freeze_steps:
                _set_action_head_lora_trainable(model)
                optimizer = make_optimizer(model, trainer_cfg)
                remaining_steps = max(max_steps - step, 1)
                phase2_warmup = int(lora_train_cfg.get("warmup_steps", warmup_steps))
                scheduler = make_scheduler(optimizer, phase2_warmup, remaining_steps)
                lora_phase2_started = True
                print(
                    "LoRA phase2 started at step "
                    f"{step}: optimizer/scheduler reset for remaining {remaining_steps} steps"
                )

            if step % log_every == 0:
                avg = loss_sum / (log_every * grad_accum)
                for gi, pg in enumerate(optimizer.param_groups):
                    writer.add_scalar(f"lr/{pg.get('name', f'g{gi}')}", pg["lr"], step)
                writer.add_scalar("train/loss", avg, step)
                print(f"step {step}/{max_steps} loss={avg:.4f} lr={optimizer.param_groups[0]['lr']:.2e} t={time.time()-t0:.0f}s")
                loss_sum = 0.0

            if step % save_every == 0:
                ckpt = os.path.join(output_dir, f"step_{step}.pt")
                torch.save(model.state_dict(), ckpt)
                print(f"Saved -> {ckpt}")

            if step >= max_steps:
                break

    # final = os.path.join(output_dir, "final.pt")
    # torch.save(model.state_dict(), final)
    # writer.close()
    # print(f"Done -> {final}")


def evaluate(_args):
    print("Evaluation is temporarily disabled for navigation training.")


def main():
    parser = argparse.ArgumentParser(description="VITRA navigation training")
    sub = parser.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--config", required=True)
    t.add_argument("--output_dir", default="./checkpoints/navigation")

    e = sub.add_parser("eval")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--config", default=None)

    s = sub.add_parser("stats")
    s.add_argument("--data_dir", required=True)
    s.add_argument("--action_col", default="action")
    s.add_argument("--state_col", default="observation.state")

    args = parser.parse_args()
    if args.cmd == "train":
        train(args)
    elif args.cmd == "eval":
        evaluate(args)
    elif args.cmd == "stats":
        print(
            json.dumps(
                compute_statistics_from_lerobot(
                    args.data_dir, action_col=args.action_col, state_col=args.state_col
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
