"""
grasp_dataset.py — Robot dataset for single-arm grasp VLA fine-tuning.

Native 7-dim actions and states (NO padding to VITRA's 192/212-dim space).
DiT first/last layers are replaced for 7-dim I/O.

State:  7-dim  [tx, ty, tz, rx, ry, rz, gripper_pos]   (camera-space EEF)
Action: 7-dim  [Δtx, Δty, Δtz, Δrx, Δry, Δrz, Δgripper]  (camera-space delta)
Wrist:  4-ch   [R, G, B, depth_metres]   (RGBD for DiT cross-attention)
Head:   3-ch   [R, G, B]                (RGB for VLM / PaliGemma)

Camera frame: OpenCV convention (x-right, y-down, z-forward).
Rotation: Euler xyz (radians) — matches VITRA's pretraining format.
Position: metres.  TCP at gripper centre (analogous to hand_mount)."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

# ──────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────

STATE_DIM = 7   # [tx, ty, tz, rx, ry, rz, gripper_pos]  (Euler xyz rotation)
ACTION_DIM = 7  # [Δtx, Δty, Δtz, Δrx, Δry, Δrz, Δgripper]  (Euler xyz delta)
GRIPPER_DIM = 6  # Index of the gripper dimension in state/action vectors


# ──────────────────────────────────────────────────────────────────────────
# Depth utilities
# ──────────────────────────────────────────────────────────────────────────

def decode_depth_from_rgb(rgb_array):
    """Decode 2-byte depth from encoded RGB image.

    The encoding from collect_data.py:
      ch0 = high byte,  ch1 = low byte,  ch2 = unused
      depth_mm = ch0 * 256 + ch1
      depth_metres = depth_mm / 1000.0

    Args:
        rgb_array: [H, W, 3] uint8 encoded depth image.

    Returns:
        depth: [H, W] float32 depth in metres.
    """
    hi = rgb_array[:, :, 0].astype(np.uint16)
    lo = rgb_array[:, :, 1].astype(np.uint16)
    depth_mm = hi * 256 + lo
    return (depth_mm.astype(np.float32) / 1000.0)


# ──────────────────────────────────────────────────────────────────────────
# Statistics
# ──────────────────────────────────────────────────────────────────────────

def compute_statistics_from_lerobot(data_dir, state_dim=STATE_DIM, action_dim=ACTION_DIM):
    """Compute per-dim Gaussian statistics from a LeRobot dataset.

    Returns dict with 'state_mean', 'state_std', 'action_mean', 'action_std'
    each of shape (dim,).  Also computes 'depth_mean' and 'depth_std' for
    wrist depth normalisation.
    """
    meta_path = os.path.join(data_dir, "meta", "episodes.jsonl")
    data_path = os.path.join(data_dir, "data")

    # Collect all parquet files (LeRobot v2 stores in data/chunk-NNN/)
    parquet_files = sorted(Path(data_path).rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files in {data_path}")

    all_states, all_actions = [], []
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        for _, row in df.iterrows():
            s = np.array(row["observation.state"], dtype=np.float32)
            a = np.array(row["action"], dtype=np.float32)
            assert s.shape == (state_dim,), f"State shape {s.shape} != ({state_dim},)"
            assert a.shape == (action_dim,), f"Action shape {a.shape} != ({action_dim},)"
            all_states.append(s)
            all_actions.append(a)

    states = np.stack(all_states)  # [N, state_dim]
    actions = np.stack(all_actions)  # [N, action_dim]

    stats = {
        "state_mean": states.mean(axis=0).tolist(),
        "state_std":  states.std(axis=0).tolist(),
        "state_min":  states.min(axis=0).tolist(),
        "state_max":  states.max(axis=0).tolist(),
        "action_mean": actions.mean(axis=0).tolist(),
        "action_std":  actions.std(axis=0).tolist(),
        "action_min":  actions.min(axis=0).tolist(),
        "action_max":  actions.max(axis=0).tolist(),
    }
    return stats


def load_or_compute_statistics(data_dir, stats_path=None):
    """Load cached statistics or compute from scratch.

    Args:
        data_dir: Path to LeRobot dataset root.
        stats_path: Path to save/load stats JSON. If None, uses data_dir/stats.json.

    Returns:
        Dictionary with 'state_mean', 'state_std', 'action_mean', 'action_std'.
    """
    if stats_path is None:
        stats_path = os.path.join(data_dir, "stats.json")

    if os.path.exists(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        print(f"Loaded statistics from {stats_path}")
        return stats

    print(f"Computing statistics from {data_dir} ...")
    stats = compute_statistics_from_lerobot(data_dir)

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics to {stats_path}")
    return stats


# ──────────────────────────────────────────────────────────────────────────
# Per-dimension Normalizer (min-max for gripper, z-score for rest)
# ──────────────────────────────────────────────────────────────────────────

class ActionNormalizer:
    """Per-dimension normalizer for 7-dim actions.

    Uses Z-score for dims 0–5 (position and rotation deltas).
    Uses **symmetric min-max → [-1, +1]** for dim 6 (gripper delta).

    Why *symmetric* min-max for the gripper?
    The raw gripper-action distribution is bimodal:
      ~80 % of frames have Δgrip ≈ 0 (approach), ~20 % have Δgrip ≈ −0.1 (close).

    *Plain* min-max uses the data range [−0.152, +0.0007], placing
    "stay still" (Δgrip = 0) at normalised +0.99 — the extreme boundary
    of [-1, +1].  The diffusion model's DDIM output easily exceeds 1.0;
    after clipping, every such sample produces a consistent +0.0007
    opening delta that accumulates over ~40 approach steps.  The opened
    gripper pushes the state out of distribution, creating a positive-
    feedback loop that prevents full closure.

    *Symmetric* min-max extends the range to [-M, +M] where
    M = max(|vmin|, |vmax|).  This places "stay still" (Δgrip = 0) at
    normalised 0.0 — the centre of the diffusion prior.  The model's
    Gaussian prior naturally biases toward 0 when uncertain, which
    correctly maps to "no movement".  Opening would require predicting
    positive values, which the training distribution (all ≤ 0) never
    encourages.
    """

    def __init__(self, mean, std, vmin=None, vmax=None,
                 gripper_dim=GRIPPER_DIM, gripper_norm="symmetric_minmax"):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std  = np.asarray(std,  dtype=np.float32)
        self.gripper_dim = gripper_dim

        if gripper_norm in ("minmax", "symmetric_minmax") and vmin is not None and vmax is not None:
            self.vmin = np.asarray(vmin, dtype=np.float32)
            self.vmax = np.asarray(vmax, dtype=np.float32)
            rng = self.vmax[gripper_dim] - self.vmin[gripper_dim]
            if rng < 1e-8:
                gripper_norm = "zscore"  # degenerate, fall back
            self.gripper_norm = gripper_norm
        else:
            self.gripper_norm = "zscore"

        # Pre-compute symmetric bounds for the gripper dim
        if self.gripper_norm == "symmetric_minmax":
            d = self.gripper_dim
            M = max(abs(float(self.vmin[d])), abs(float(self.vmax[d])))
            self._sym_lo = -M
            self._sym_hi = +M
            self._sym_rng = 2.0 * M
        elif self.gripper_norm == "minmax":
            d = self.gripper_dim
            self._sym_lo = float(self.vmin[d])
            self._sym_hi = float(self.vmax[d])
            self._sym_rng = self._sym_hi - self._sym_lo

    # ── numpy paths ───────────────────────────────────────────────────
    def normalize(self, x):
        """x: np.ndarray [..., D] → normalised copy."""
        out = (x - self.mean) / (self.std + 1e-7)
        if self.gripper_norm in ("minmax", "symmetric_minmax"):
            d = self.gripper_dim
            out[..., d] = (x[..., d] - self._sym_lo) / (self._sym_rng + 1e-7) * 2.0 - 1.0
        return out

    def denormalize(self, x):
        """x: np.ndarray [..., D] → raw values."""
        out = x * (self.std + 1e-7) + self.mean
        if self.gripper_norm in ("minmax", "symmetric_minmax"):
            d = self.gripper_dim
            g = np.clip(x[..., d], -1.0, 1.0)
            out[..., d] = (g + 1.0) / 2.0 * (self._sym_rng + 1e-7) + self._sym_lo
        return out

    # ── torch paths (for use inside training loop) ────────────────────
    def normalize_t(self, x):
        """x: Tensor [..., D] → normalised Tensor."""
        mean = torch.as_tensor(self.mean, device=x.device, dtype=x.dtype)
        std  = torch.as_tensor(self.std,  device=x.device, dtype=x.dtype)
        out = (x - mean) / (std + 1e-7)
        if self.gripper_norm in ("minmax", "symmetric_minmax"):
            d = self.gripper_dim
            lo = self._sym_lo
            hi = self._sym_hi
            out[..., d] = (x[..., d] - lo) / (hi - lo + 1e-7) * 2.0 - 1.0
        return out

    def denormalize_t(self, x):
        """x: Tensor [..., D] → raw-valued Tensor."""
        mean = torch.as_tensor(self.mean, device=x.device, dtype=x.dtype)
        std  = torch.as_tensor(self.std,  device=x.device, dtype=x.dtype)
        out = x * (std + 1e-7) + mean
        if self.gripper_norm in ("minmax", "symmetric_minmax"):
            d = self.gripper_dim
            lo = self._sym_lo
            hi = self._sym_hi
            g = x[..., d].clamp(-1.0, 1.0)
            out[..., d] = (g + 1.0) / 2.0 * (hi - lo + 1e-7) + lo
        return out


# ──────────────────────────────────────────────────────────────────────────
# Core Dataset
# ──────────────────────────────────────────────────────────────────────────

class GraspDatasetCore:
    """Load LeRobot grasp dataset for VLA fine-tuning.

    Returns native 7-dim states/actions (no padding to 192/212).
    Loads RGBD for wrist camera, RGB for head camera.
    """

    def __init__(
        self,
        data_dir,
        chunk_size=16,
        cam_head_col="observation.images.cam_head",
        cam_wrist_rgb_col="observation.images.cam_right_wrist",
        cam_wrist_depth_col="observation.depth.cam_right_wrist",
        stats_path=None,
    ):
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.cam_head_col = cam_head_col
        self.cam_wrist_rgb_col = cam_wrist_rgb_col
        self.cam_wrist_depth_col = cam_wrist_depth_col

        # Load all parquet data (LeRobot v2 stores in data/chunk-NNN/)
        data_path = os.path.join(data_dir, "data")
        parquet_files = sorted(Path(data_path).rglob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files in {data_path}")

        dfs = [pd.read_parquet(pf) for pf in parquet_files]
        self.df = pd.concat(dfs, ignore_index=True)

        # Episode boundaries
        self.episode_indices = self.df["episode_index"].values
        self.episodes = sorted(self.df["episode_index"].unique())

        # Build per-episode start/end index
        self._ep_start = {}
        self._ep_end = {}
        for ep in self.episodes:
            mask = self.episode_indices == ep
            indices = np.where(mask)[0]
            self._ep_start[ep] = int(indices[0])
            self._ep_end[ep] = int(indices[-1]) + 1

        # Normalisation statistics
        self.stats = load_or_compute_statistics(data_dir, stats_path)
        self.s_mean = np.array(self.stats["state_mean"], dtype=np.float32)
        self.s_std  = np.array(self.stats["state_std"],  dtype=np.float32)
        self.a_mean = np.array(self.stats["action_mean"], dtype=np.float32)
        self.a_std  = np.array(self.stats["action_std"],  dtype=np.float32)

        # Per-dimension normalizers (min-max for gripper, z-score for rest)
        self.action_normalizer = ActionNormalizer(
            self.a_mean, self.a_std,
            vmin=self.stats.get("action_min"),
            vmax=self.stats.get("action_max"),
        )
        self.state_normalizer = ActionNormalizer(
            self.s_mean, self.s_std,
            vmin=self.stats.get("state_min"),
            vmax=self.stats.get("state_max"),
        )

        # Image root paths
        self.data_dir_path = Path(data_dir)

        # Load task descriptions from meta/tasks.json (task_index → string)
        tasks_path = os.path.join(data_dir, "meta", "tasks.json")
        self._tasks = {}
        if os.path.exists(tasks_path):
            import json as _json
            with open(tasks_path) as f:
                tasks_list = _json.load(f)
            # tasks.json is a list of {task_index: int, task: str}
            if isinstance(tasks_list, list):
                for entry in tasks_list:
                    self._tasks[entry["task_index"]] = entry["task"]
            elif isinstance(tasks_list, dict):
                for k, v in tasks_list.items():
                    self._tasks[int(k)] = v

        # Valid sample indices: every frame that has at least 1 future frame
        self._valid_indices = []
        for ep in self.episodes:
            ep_start = self._ep_start[ep]
            ep_end = self._ep_end[ep]
            # Every frame in the episode is valid (we zero-pad the tail)
            for idx in range(ep_start, ep_end):
                self._valid_indices.append(idx)
        self._valid_indices = np.array(self._valid_indices)

        print(f"GraspDatasetCore: {len(self.episodes)} episodes, "
              f"{len(self._valid_indices)} frames, chunk={chunk_size}")

    def __len__(self):
        return len(self._valid_indices)

    def _load_image(self, row, col):
        """Load an image from the dataset (PIL → numpy [H, W, 3] uint8).

        Handles LeRobot v2 format where images are stored as inline bytes
        in parquet files (dict with 'bytes' and 'path' keys).
        Falls back to file-based loading if bytes are not available.
        """
        from io import BytesIO
        data = row[col]
        if isinstance(data, dict) and "bytes" in data and data["bytes"]:
            return np.array(Image.open(BytesIO(data["bytes"])).convert("RGB"))
        elif isinstance(data, dict) and "path" in data:
            img_path = self.data_dir_path / data["path"]
            return np.array(Image.open(img_path).convert("RGB"))
        else:
            img_path = self.data_dir_path / str(data)
            return np.array(Image.open(img_path).convert("RGB"))

    def __getitem__(self, idx):
        global_idx = self._valid_indices[idx]
        row = self.df.iloc[global_idx]
        ep = row["episode_index"]
        ep_start = self._ep_start[ep]
        ep_end = self._ep_end[ep]
        remaining = ep_end - global_idx  # frames left in episode

        # ── Head camera RGB ──
        head_rgb = self._load_image(row, self.cam_head_col)  # [H, W, 3] uint8

        # ── Wrist camera RGBD ──
        wrist_rgb = self._load_image(row, self.cam_wrist_rgb_col)  # [H, W, 3] uint8
        wrist_depth_encoded = self._load_image(row, self.cam_wrist_depth_col)  # [H, W, 3]
        wrist_depth = decode_depth_from_rgb(wrist_depth_encoded)  # [H, W] float32 metres

        # ── Current state (7-dim) ──
        state = np.array(row["observation.state"], dtype=np.float32)
        assert state.shape == (STATE_DIM,), f"Bad state shape: {state.shape}"

        # ── Action chunk (T, 7) ──
        valid_frames = min(remaining, self.chunk_size)
        actions = np.zeros((self.chunk_size, ACTION_DIM), dtype=np.float32)
        for t in range(valid_frames):
            a = np.array(self.df.iloc[global_idx + t]["action"], dtype=np.float32)
            assert a.shape == (ACTION_DIM,), f"Bad action shape: {a.shape}"
            actions[t] = a

        # ── Action mask (T, 7) — all-ones for valid, all-zeros for padding ──
        action_mask = np.zeros((self.chunk_size, ACTION_DIM), dtype=np.float32)
        action_mask[:valid_frames, :] = 1.0

        # ── State mask (7,) — all-ones (single-arm, always valid) ──
        state_mask = np.ones(STATE_DIM, dtype=np.float32)

        # ── Task instruction (from tasks.json via task_index) ──
        task_idx = row.get("task_index", 0)
        if isinstance(task_idx, (list, np.ndarray)):
            task_idx = int(task_idx[0]) if len(task_idx) > 0 else 0
        task = self._tasks.get(int(task_idx), "Left hand: None. Right hand: Grasp the object and lift it.")

        return {
            "head_rgb": head_rgb,               # [H, W, 3] uint8
            "wrist_rgb": wrist_rgb,             # [H, W, 3] uint8
            "wrist_depth": wrist_depth,         # [H, W] float32 metres
            "action_list": actions,             # [T, 7] float32
            "action_mask": action_mask,         # [T, 7] float32 binary
            "current_state": state,             # [7] float32
            "current_state_mask": state_mask,   # [7] float32 binary
            "valid_frames": valid_frames,
            "instruction": task,
        }

    def normalize(self, sample):
        """Normalize state and actions.

        Uses per-dimension normalization: Z-score for position/rotation dims,
        symmetric min-max → [-1, +1] for the gripper dim (see ActionNormalizer).
        Converts to torch tensors.  Actions beyond valid_frames stay zero.
        """
        action = sample["action_list"].copy()
        state = sample["current_state"].copy()
        valid = sample["valid_frames"]

        action[:valid] = self.action_normalizer.normalize(action[:valid])
        state = self.state_normalizer.normalize(state)

        sample["action_list"] = torch.tensor(action, dtype=torch.float32)
        sample["action_mask"] = torch.tensor(sample["action_mask"], dtype=torch.float32)
        sample["current_state"] = torch.tensor(state, dtype=torch.float32)
        sample["current_state_mask"] = torch.tensor(
            sample["current_state_mask"], dtype=torch.float32
        )
        return sample

    def denormalize_action(self, action_norm):
        """Denormalize a single action or action chunk.

        Uses per-dim denormalization (min-max for gripper, z-score for rest).

        Args:
            action_norm: numpy or torch array, shape [..., 7].

        Returns:
            action: same type, denormalized.
        """
        if isinstance(action_norm, torch.Tensor):
            return self.action_normalizer.denormalize_t(action_norm)
        return self.action_normalizer.denormalize(action_norm)

    def denormalize_state(self, state_norm):
        """Denormalize a state vector.

        Uses per-dim denormalization (min-max for gripper, z-score for rest).

        Args:
            state_norm: numpy or torch array, shape [..., 7].

        Returns:
            state: same type, denormalized.
        """
        if isinstance(state_norm, torch.Tensor):
            return self.state_normalizer.denormalize_t(state_norm)
        return self.state_normalizer.denormalize(state_norm)
