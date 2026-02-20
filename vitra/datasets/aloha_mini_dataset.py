"""
aloha_mini_dataset.py

Dataset for AlohaMini LeRobot data, compatible with VITRA training.

Data format (from collect_data.py — joint-space):
  state:  6-dim = [5 arm joint qpos + 1 gripper qpos]
  action: 6-dim = [5 arm PD delta commands + 1 gripper PD delta command]

Mapping to VITRA's unified 192-dim action / 212-dim state space:
  [51:56]  right-hand dims 0-4   ← 5 arm joint values
  [57:102] right-hand finger joints ← gripper value spread to 15 MANO curl axes

The MANO finger space is 15 joints × 3 Euler angles (XYZ).  The curl
(flexion) is the X axis.  We replicate the 1-DOF gripper value to the
curl axis of ALL 15 joints so the pretrained model sees a coordinated
grasp signal.
"""

import glob, io, json, os
import numpy as np
import pandas as pd
import torch
from PIL import Image

from vitra.datasets.dataset_utils import ActionFeature, StateFeature
from vitra.utils.data_utils import read_dataset_statistics

# Dimensions
ACTION_DIM = 6  # 5 arm joints + 1 gripper
ARM_DIM = 5
GRIPPER_DIM = 1

# Unified space constants
UNIFIED_ACTION_DIM = ActionFeature.ALL_FEATURES[1]   # 192
UNIFIED_STATE_DIM  = StateFeature.ALL_FEATURES[1]    # 212
RIGHT_6D = ActionFeature.HUMAN_RIGHT_6D              # (51, 57)
RIGHT_JOINTS = ActionFeature.HUMAN_RIGHT_JOINTS       # (57, 102)

# Arm joints → [51:56] (5 dims within the 6D wrist region)
ARM_START = RIGHT_6D[0]   # 51
ARM_END   = ARM_START + ARM_DIM  # 56

# Gripper → 15 MANO curl axes (every 3rd dim in [57:102])
NUM_FINGER_JOINTS = 15
_CURL_INDICES = [RIGHT_JOINTS[0] + i * 3 for i in range(NUM_FINGER_JOINTS)]


def pad_state(state_6: np.ndarray, active: bool = True):
    """Map 6-dim joint state → (212,) unified state + (212,) mask."""
    s = torch.zeros(UNIFIED_STATE_DIM, dtype=torch.float32)
    m = torch.zeros(UNIFIED_STATE_DIM, dtype=torch.bool)
    if active:
        t = torch.as_tensor(state_6, dtype=torch.float32)
        s[ARM_START:ARM_END] = t[:ARM_DIM]
        for idx in _CURL_INDICES:
            s[idx] = t[ARM_DIM]  # gripper
        m[ARM_START:ARM_END] = True
        m[RIGHT_JOINTS[0]:RIGHT_JOINTS[1]] = True
    return s, m


def pad_action(action_6: np.ndarray, active: bool = True):
    """Map (T,6) joint actions → (T,192) unified + (T,192) mask."""
    if action_6.ndim == 1:
        action_6 = action_6[np.newaxis, :]
    T = action_6.shape[0]
    a = torch.zeros(T, UNIFIED_ACTION_DIM, dtype=torch.float32)
    m = torch.zeros(T, UNIFIED_ACTION_DIM, dtype=torch.bool)
    if active:
        t = torch.as_tensor(action_6, dtype=torch.float32)
        a[:, ARM_START:ARM_END] = t[:, :ARM_DIM]
        for idx in _CURL_INDICES:
            a[:, idx] = t[:, ARM_DIM]  # gripper
        m[:, ARM_START:ARM_END] = True
        m[:, RIGHT_JOINTS[0]:RIGHT_JOINTS[1]] = True
    return a, m


def extract_action(predicted_192: np.ndarray):
    """Extract 6-dim joint action from 192-dim prediction. (T,192)→(T,6)."""
    arm = predicted_192[:, ARM_START:ARM_END]              # (T,5)
    curls = predicted_192[:, _CURL_INDICES]                 # (T,15)
    grip = curls.mean(axis=-1, keepdims=True)               # (T,1)
    return np.concatenate([arm, grip], axis=-1)


class AlohaMiniDatasetCore:
    """Reads AlohaMini LeRobot v3 data (parquet + inline PNG)."""

    def __init__(self, data_dir, statistics_path, chunk_size=16,
                 camera_fov_rad=1.6, image_size=224, use_wrist_cam=False):
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.image_size = image_size
        self.use_wrist_cam = use_wrist_cam

        # Camera FOV (cam_head = 1.6 rad in ManiSkill)
        self.fov = np.array([camera_fov_rad, camera_fov_rad], dtype=np.float32)
        f = image_size / (2.0 * np.tan(camera_fov_rad / 2.0))
        cx = cy = image_size / 2.0
        self.intrinsics = np.array(
            [[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32)

        # Statistics
        self.data_statistics = read_dataset_statistics(statistics_path)
        self.global_data_statistics = None

        # Load parquet
        pq_files = sorted(glob.glob(os.path.join(data_dir, "data", "chunk-*", "*.parquet")))
        assert pq_files, f"No parquet files in {data_dir}/data/"
        self.df = pd.concat([pd.read_parquet(f) for f in pq_files], ignore_index=True)

        # Episode index
        self.episodes = []
        for ep in sorted(self.df["episode_index"].unique()):
            idxs = self.df.index[self.df["episode_index"] == ep].tolist()
            self.episodes.append((idxs[0], idxs[-1]))

        # Flat sample list
        self.samples = []
        for ei, (s, e) in enumerate(self.episodes):
            for fi in range(e - s + 1):
                self.samples.append((ei, fi))

        # Task description
        tasks_path = os.path.join(data_dir, "meta", "tasks.parquet")
        tasks_df = pd.read_parquet(tasks_path)
        if "task" in tasks_df.columns:
            self.task_description = tasks_df["task"].iloc[0]
        else:
            self.task_description = str(tasks_df.index[0])

        print(f"AlohaMiniDataset: {len(self.episodes)} episodes, "
              f"{len(self.samples)} frames, task='{self.task_description}'")

    def __len__(self):
        return len(self.samples)

    def set_global_data_statistics(self, stats):
        self.global_data_statistics = stats

    def _decode_image(self, row, cam_key="observation.images.cam_head"):
        img_data = row[cam_key]
        img_bytes = img_data["bytes"] if isinstance(img_data, dict) else img_data
        return np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"), dtype=np.uint8)

    def __getitem__(self, idx):
        ep_i, frame_i = self.samples[idx]
        start, end = self.episodes[ep_i]
        global_idx = start + frame_i

        row = self.df.iloc[global_idx]
        images = [self._decode_image(row)]
        if self.use_wrist_cam:
            images.append(self._decode_image(row, "observation.images.cam_right_wrist"))
        image = np.stack(images, axis=0)  # (N_views, H, W, 3)

        state = np.array(row["observation.state"], dtype=np.float32)

        # Action chunk
        actions = []
        valid = 0
        for t in range(self.chunk_size):
            fi = global_idx + t
            if fi <= end:
                actions.append(np.array(self.df.iloc[fi]["action"], dtype=np.float32))
                valid += 1
            else:
                actions.append(np.zeros(ACTION_DIM, dtype=np.float32))
        action_list = np.stack(actions, axis=0)  # (T, 6)

        # Masks: per-step validity
        action_mask = np.zeros((self.chunk_size, 2), dtype=bool)
        action_mask[:valid, 1] = True  # right hand active

        instruction = f"Left hand: None. Right hand: {self.task_description}."

        return {
            "instruction": instruction,
            "image_list": image,
            "image_mask": np.array([True] * len(images)),
            "action_list": action_list,
            "action_mask": action_mask,
            "current_state": state,
            "current_state_mask": np.array([False, True], dtype=bool),
            "fov": self.fov.copy(),
            "intrinsics": self.intrinsics.copy(),
        }

    def transform_trajectory(self, sample, normalization=True):
        """Normalize with per-dim stats, then pad to unified space."""
        action_np = sample["action_list"].copy()   # (T, 6)
        state_np = sample["current_state"].copy()  # (6,)

        if normalization and self.global_data_statistics is not None:
            s = self.global_data_statistics
            action_np = (action_np - s["action_right_mean"]) / (s["action_right_std"] + 1e-7)
            state_np  = (state_np  - s["state_right_mean"])  / (s["state_right_std"]  + 1e-7)

        us, usm = pad_state(state_np)
        ua, uam = pad_action(action_np)

        sample["action_list"] = ua
        sample["action_mask"] = uam
        sample["current_state"] = us
        sample["current_state_mask"] = usm
        return sample

    def transform_trajectory_raw(self, sample, normalization=True):
        """Normalize but keep raw dimensions (no padding to 192/212-dim).

        Used with SimpleActionHead where actions stay at robot_action_dim.
        """
        action_np = sample["action_list"].copy()   # (T, 6)
        state_np = sample["current_state"].copy()  # (6,)

        if normalization and self.global_data_statistics is not None:
            s = self.global_data_statistics
            action_np = (action_np - s["action_right_mean"]) / (s["action_right_std"] + 1e-7)
            state_np  = (state_np  - s["state_right_mean"])  / (s["state_right_std"]  + 1e-7)

        sample["action_list"] = torch.as_tensor(action_np, dtype=torch.float32)
        sample["current_state"] = torch.as_tensor(state_np, dtype=torch.float32)
        # Simple masks: all dims active
        sample["action_mask"] = torch.ones(sample["action_list"].shape[:2], dtype=torch.bool) if sample["action_list"].ndim >= 2 else torch.ones(1, dtype=torch.bool)
        sample["current_state_mask"] = torch.ones(state_np.shape[0], dtype=torch.bool)
        return sample
