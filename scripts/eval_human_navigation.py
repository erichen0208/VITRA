#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path
from typing import Optional, Tuple

import cv2
import imageio.v2 as imageio
import numpy as np
import pandas as pd
import torch

from vitra.models.vla_builder import load_model
from vitra.utils.config_utils import load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize one clip side-by-side with exact XY-theta trajectory from gt.json"
    )
    p.add_argument(
        "--clip-path",
        type=Path,
        required=True,
        help="Path to one clip, e.g. hd-epic-dataset-vla/P01/P01-20240202-110250/0.mp4",
    )
    p.add_argument(
        "--gt-path",
        type=Path,
        default=None,
        help="Path to gt.json (default: same session folder as clip)",
    )
    p.add_argument(
        "--out-path",
        type=Path,
        default=None,
        help="Output mp4 path (default: debug/<participant>/<session>/<clip_stem>_with_slam_gt.mp4)",
    )
    p.add_argument(
        "--camera-size",
        type=int,
        default=20,
        help="Current-pose triangle marker size in pixels",
    )
    p.add_argument(
        "--future-camera-scale",
        type=float,
        default=0.7,
        help="Scale factor applied to --camera-size for refined future triangles",
    )
    p.add_argument(
        "--trail-len",
        type=int,
        default=120,
        help="Max number of past frames shown as trail on the right panel",
    )
    p.add_argument(
        "--pose-mode",
        type=str,
        choices=["raw", "aligned"],
        default="aligned",
        help=(
            "Pose transform mode: 'raw' keeps gt.json values; 'aligned' offsets the first point to (0,0) "
            "and rotates all points/headings so the first heading faces +y"
        ),
    )
    p.add_argument(
        "--source",
        type=str,
        choices=["gt", "refined"],
        default="gt",
        help="Visualization source: raw per-frame gt trajectory, or refined future targets",
    )
    p.add_argument(
        "--refined-path",
        type=Path,
        default=None,
        help="Path to refined gt json (default: <session>/refined_gt.json, fallback <session>/refined_t.json)",
    )
    p.add_argument(
        "--split",
        type=str,
        choices=["all", "train", "val"],
        default="val",
        help="Deterministic clip split in the current session. Default 'val' is outside the 95%% train split.",
    )
    p.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Random seed for deterministic clip split.",
    )
    p.add_argument(
        "--train-ratio",
        type=float,
        default=0.95,
        help="Train ratio used by deterministic split.",
    )
    p.add_argument(
        "--model-config",
        type=Path,
        default=Path("vitra/configs/navigation_finetune.json"),
        help="Navigation model config JSON.",
    )
    p.add_argument(
        "--model-checkpoint",
        type=Path,
        default=None,
        help="Path to trained navigation checkpoint (e.g. checkpoints/navigation/step_6000.pt).",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Inference device for model prediction overlay.",
    )
    p.add_argument(
        "--num-ddim-steps",
        type=int,
        default=10,
        help="DDIM denoise steps for model prediction.",
    )
    p.add_argument(
        "--cfg-scale",
        type=float,
        default=5.0,
        help="Classifier-free guidance scale for model prediction.",
    )
    return p.parse_args()


def _resolve_existing_dir(raw_path: str, config_path: Path) -> Path:
    p = Path(raw_path)
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(Path.cwd() / p)
        cfg_dir = config_path.resolve().parent
        candidates.append(cfg_dir / p)
        vitra_root = Path(__file__).resolve().parents[1]
        candidates.append(vitra_root / p)
        workspace_root = Path(__file__).resolve().parents[2]
        candidates.append(workspace_root / p)

    for c in candidates:
        if c.exists() and c.is_dir():
            return c.resolve()
    checked = "\n  - " + "\n  - ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"Could not resolve data_dir: {raw_path}\nChecked:{checked}")


def _to_waypoints_xytheta(action_raw: np.ndarray, waypoint_horizon: int) -> np.ndarray:
    a = np.asarray(action_raw, dtype=np.float32)
    if a.ndim == 1 and a.shape[0] == waypoint_horizon * 3:
        return a.reshape(waypoint_horizon, 3)
    if a.ndim == 2 and a.shape == (waypoint_horizon, 3):
        return a
    raise ValueError(f"Unsupported action shape: {a.shape}")


def compute_waypoint_xy_stats(data_dir: Path, action_col: str, waypoint_horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    files = sorted((data_dir / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files under {data_dir / 'data'}")

    sum_xy = np.zeros((2,), dtype=np.float64)
    sumsq_xy = np.zeros((2,), dtype=np.float64)
    count = 0

    for p in files:
        df = pd.read_parquet(p, columns=[action_col])
        for a in df[action_col].values:
            wp = _to_waypoints_xytheta(a, waypoint_horizon)
            xy = wp[:, :2].astype(np.float64)
            sum_xy += xy.sum(axis=0)
            sumsq_xy += (xy * xy).sum(axis=0)
            count += xy.shape[0]

    if count <= 0:
        raise RuntimeError("No waypoint rows found for XY stats")

    mean = (sum_xy / count).astype(np.float32)
    var = np.maximum((sumsq_xy / count) - (mean.astype(np.float64) ** 2), 1e-12)
    std = np.sqrt(var).astype(np.float32)
    return mean, std


def split_session_clips(session_dir: Path, train_ratio: float, seed: int) -> Tuple[set, set]:
    clips = sorted(session_dir.glob("*.mp4"), key=lambda p: (not p.stem.isdigit(), p.stem))
    if not clips:
        return set(), set()

    idx = np.arange(len(clips), dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)

    train_ratio = min(max(float(train_ratio), 0.0), 1.0)
    n_total = len(clips)
    n_train = int(round(n_total * train_ratio))
    n_train = min(max(n_train, 1), n_total)
    if n_total > 1 and n_train == n_total:
        n_train = n_total - 1

    train_set = {clips[i].name for i in idx[:n_train]}
    val_set = {clips[i].name for i in idx[n_train:]}
    if not val_set:
        val_set = train_set
    return train_set, val_set


class NavModelOverlay:
    def __init__(
        self,
        config_path: Path,
        checkpoint_path: Path,
        device: str,
        num_ddim_steps: int,
        cfg_scale: float,
    ):
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

        cfg = load_config(str(config_path))
        cfg["model_load_path"] = str(checkpoint_path)
        data_cfg = cfg["data"]
        data_dir = _resolve_existing_dir(data_cfg["data_dir"], config_path)

        self.waypoint_horizon = int(data_cfg.get("waypoint_horizon", 16))
        self.action_dim = int(cfg.get("action_model", {}).get("action_dim", 4))
        self.camera_fov_rad = float(data_cfg.get("camera_fov_rad", 1.73))
        self.num_ddim_steps = int(num_ddim_steps)
        self.cfg_scale = float(cfg_scale)

        self.xy_norm_mode = str(data_cfg.get("waypoint_xy_norm", "global"))

        # Prefer training-exported target stats from checkpoint run directory.
        ckpt_stats_path = checkpoint_path.parent / "statistics.json"
        if ckpt_stats_path.exists():
            ckpt_stats = json.loads(ckpt_stats_path.read_text(encoding="utf-8"))
            mean = ckpt_stats.get("target_xy_mean")
            std = ckpt_stats.get("target_xy_std")
            if mean is not None and std is not None:
                self.xy_mean = np.asarray(mean, dtype=np.float32)
                self.xy_std = np.maximum(np.asarray(std, dtype=np.float32), 1e-6)
                self.xy_norm_mode = str(ckpt_stats.get("waypoint_xy_norm", self.xy_norm_mode))
            else:
                action_col = data_cfg.get("action_col", "action")
                self.xy_mean, self.xy_std = compute_waypoint_xy_stats(data_dir, action_col, self.waypoint_horizon)
        else:
            action_col = data_cfg.get("action_col", "action")
            self.xy_mean, self.xy_std = compute_waypoint_xy_stats(data_dir, action_col, self.waypoint_horizon)

        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.model = load_model(cfg).to(self.device).eval()

        self.state_dim = int(cfg.get("state_encoder", {}).get("state_dim", self.action_dim))

    def predict_local_future(self, frame_bgr: np.ndarray, instruction: str) -> np.ndarray:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        state = torch.zeros((1, self.state_dim), dtype=torch.float32, device=self.device)
        state_mask = torch.zeros((1, self.state_dim), dtype=torch.float32, device=self.device)
        action_mask = torch.ones(
            (1, self.waypoint_horizon, self.action_dim),
            dtype=torch.float32,
            device=self.device,
        )
        fov = torch.tensor([[self.camera_fov_rad, self.camera_fov_rad]], dtype=torch.float32, device=self.device)

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            pred = self.model.predict_action(
                image=frame_rgb,
                instruction=instruction,
                current_state=state,
                current_state_mask=state_mask,
                action_mask_torch=action_mask,
                fov=fov,
                num_ddim_steps=self.num_ddim_steps,
                cfg_scale=self.cfg_scale,
                sample_times=1,
            )

        pred = np.asarray(pred[0], dtype=np.float32)  # [H, 4]
        if pred.ndim != 2 or pred.shape[0] != self.waypoint_horizon or pred.shape[1] < 4:
            raise RuntimeError(f"Unexpected prediction shape: {pred.shape}")

        if self.xy_mean.ndim == 2 and self.xy_std.ndim == 2:
            if self.xy_mean.shape[0] != pred.shape[0] or self.xy_mean.shape[1] != 2:
                raise RuntimeError(
                    f"Per-step xy stats shape {self.xy_mean.shape} incompatible with prediction {pred.shape}"
                )
            xy = pred[:, :2] * self.xy_std + self.xy_mean
        else:
            xy = pred[:, :2] * self.xy_std[None, :] + self.xy_mean[None, :]
        yaw = np.arctan2(pred[:, 2], pred[:, 3])
        out = np.concatenate([xy, yaw[:, None]], axis=1)
        return out


def load_clip_xytheta_strict(gt_path: Path, clip_path: Path) -> np.ndarray:
    data = json.loads(gt_path.read_text(encoding="utf-8"))
    clips = data.get("clips", [])
    if not isinstance(clips, list):
        raise RuntimeError(f"Invalid GT format in {gt_path}")

    clip_name = clip_path.name
    clip_stem = clip_path.stem
    has_index = clip_stem.isdigit()
    clip_index = int(clip_stem) if has_index else None

    for item in clips:
        if not isinstance(item, dict):
            continue

        if has_index:
            if str(item.get("index", "")) != str(clip_index):
                continue
        else:
            if str(item.get("clip", "")) != clip_name:
                continue

        arr = item.get("xytheta")
        if arr is None:
            key_name = "index" if has_index else "clip"
            key_val = clip_index if has_index else clip_name
            raise RuntimeError(f"GT entry found by {key_name}={key_val} but missing xytheta in {gt_path}")

        poses = np.asarray(arr, dtype=float)
        if poses.ndim != 2 or poses.shape[1] != 3:
            raise RuntimeError(f"Expected xytheta to have shape [N,3], got {poses.shape} for {clip_name}")
        if not np.all(np.isfinite(poses)):
            raise RuntimeError(f"xytheta contains non-finite values for {clip_name} in {gt_path}")
        return poses

    if has_index:
        raise KeyError(f"Clip index {clip_index} not found in {gt_path}")
    raise KeyError(f"Clip name {clip_name} not found in {gt_path}")


def load_clip_refined_local_future_strict(refined_path: Path, clip_path: Path) -> np.ndarray:
    data = json.loads(refined_path.read_text(encoding="utf-8"))
    clips = data.get("clips", [])
    if not isinstance(clips, list):
        raise RuntimeError(f"Invalid refined format in {refined_path}: missing clips list")

    clip_name = clip_path.name
    clip_stem = clip_path.stem
    has_index = clip_stem.isdigit()
    clip_index = int(clip_stem) if has_index else None

    for item in clips:
        if not isinstance(item, dict):
            continue

        if has_index:
            if str(item.get("index", "")) != str(clip_index):
                continue
        else:
            if str(item.get("clip", "")) != clip_name:
                continue

        arr = item.get("local_future_xytheta")
        if arr is None:
            key_name = "index" if has_index else "clip"
            key_val = clip_index if has_index else clip_name
            raise RuntimeError(f"Refined entry found by {key_name}={key_val} but missing local_future_xytheta in {refined_path}")

        fut = np.asarray(arr, dtype=float)
        if fut.ndim != 3 or fut.shape[2] != 3:
            raise RuntimeError(f"Expected local_future_xytheta to have shape [N,H,3], got {fut.shape} for {clip_name}")
        if not np.all(np.isfinite(fut)):
            raise RuntimeError(f"local_future_xytheta contains non-finite values for {clip_name} in {refined_path}")
        return fut

    if has_index:
        raise KeyError(f"Clip index {clip_index} not found in {refined_path}")
    raise KeyError(f"Clip name {clip_name} not found in {refined_path}")


def load_clip_instruction(session_dir: Path, clip_path: Path) -> str:
    nav_path = session_dir / "navigation_clips.json"
    if not nav_path.exists():
        return ""

    try:
        rows = json.loads(nav_path.read_text(encoding="utf-8"))
    except Exception:
        return ""

    if not isinstance(rows, list):
        return ""

    clip_name = clip_path.name
    clip_stem = clip_path.stem
    has_index = clip_stem.isdigit()
    clip_index = int(clip_stem) if has_index else None

    for row in rows:
        if not isinstance(row, dict):
            continue

        if has_index:
            if str(row.get("index", "")) != str(clip_index):
                continue
        else:
            if str(row.get("clip", "")) != clip_name:
                continue

        llm = row.get("llm")
        if not isinstance(llm, dict):
            return ""
        action = str(llm.get("action", "")).strip()
        return action

    return ""


def draw_instruction(img: np.ndarray, text: str) -> None:
    content = text.strip() if text else ""
    line_prefix = "instruction: "
    if not content:
        content = "(none)"

    max_chars = 52
    words = content.split()
    lines = []
    cur = ""
    for w in words:
        nxt = w if not cur else f"{cur} {w}"
        if len(nxt) <= max_chars:
            cur = nxt
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    if not lines:
        lines = [content]

    rendered = [f"{line_prefix}{lines[0]}"] + [f"  {ln}" for ln in lines[1:]]
    line_h = 24
    pad_x = 14
    pad_y = 12
    box_w = min(img.shape[1] - 2 * pad_x, 980)
    box_h = pad_y * 2 + line_h * len(rendered)

    overlay = img.copy()
    cv2.rectangle(overlay, (pad_x, pad_y), (pad_x + box_w, pad_y + box_h), (248, 248, 248), -1)
    cv2.rectangle(overlay, (pad_x, pad_y), (pad_x + box_w, pad_y + box_h), (80, 80, 80), 1)
    cv2.addWeighted(overlay, 0.88, img, 0.12, 0.0, img)

    y = pad_y + 22
    for ln in rendered:
        cv2.putText(img, ln, (pad_x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (25, 25, 25), 2, cv2.LINE_AA)
        y += line_h


def validate_exact_alignment(poses: np.ndarray, n_frames: int, clip_path: Path, gt_path: Path) -> None:
    if n_frames <= 0:
        raise RuntimeError(f"Video has invalid frame count: {n_frames} for {clip_path}")
    if len(poses) != n_frames:
        raise RuntimeError(
            f"Strict mode requires exact frame alignment: video has {n_frames} frames, "
            f"but xytheta has {len(poses)} rows for {clip_path.name} in {gt_path}"
        )


def wrap_to_pi(theta: np.ndarray) -> np.ndarray:
    return (theta + np.pi) % (2.0 * np.pi) - np.pi


def transform_poses(poses: np.ndarray, mode: str) -> np.ndarray:
    if mode == "raw":
        return poses.copy()

    if mode != "aligned":
        raise ValueError(f"Unknown pose mode: {mode}")

    if len(poses) == 0:
        return poses.copy()

    out = poses.copy()
    x0, y0, theta0 = float(out[0, 0]), float(out[0, 1]), float(out[0, 2])

    # Translate so the first point is exactly at the origin.
    xy = out[:, :2] - np.array([x0, y0], dtype=float)

    # Rotate all positions/headings so the first heading becomes +y (pi/2).
    rot = (np.pi / 2.0) - theta0
    c = math.cos(rot)
    s = math.sin(rot)
    r = np.array([[c, -s], [s, c]], dtype=float)
    out[:, :2] = xy @ r.T
    out[:, 2] = wrap_to_pi(out[:, 2] + rot)
    return out


def to_panel_xy(x_m: float, y_m: float, cx: int, cy: int, scale: float) -> Tuple[int, int]:
    px = int(round(cx + x_m * scale))
    py = int(round(cy - y_m * scale))
    return px, py


def refined_local_to_aligned_global(poses_aligned: np.ndarray, future_local: np.ndarray) -> np.ndarray:
    if future_local.ndim != 3 or future_local.shape[2] != 3:
        raise ValueError(f"Expected future_local [N,H,3], got {future_local.shape}")
    if len(poses_aligned) != future_local.shape[0]:
        raise ValueError(
            f"Frame mismatch between aligned poses ({len(poses_aligned)}) and refined futures ({future_local.shape[0]})"
        )

    out = np.zeros_like(future_local)
    for i in range(len(poses_aligned)):
        cx, cy, ctheta = float(poses_aligned[i, 0]), float(poses_aligned[i, 1]), float(poses_aligned[i, 2])
        rot = (np.pi / 2.0) - ctheta
        c = math.cos(rot)
        s = math.sin(rot)
        r = np.array([[c, -s], [s, c]], dtype=float)

        dxy = future_local[i, :, :2] @ r
        out[i, :, :2] = dxy + np.array([cx, cy], dtype=float)
        # Inverse of refine local yaw convention where +y is yaw 0:
        # local_yaw = global_yaw - current_yaw.
        out[i, :, 2] = wrap_to_pi(future_local[i, :, 2] + ctheta)
    return out


def draw_camera_eye(img: np.ndarray, px: int, py: int, heading: float, size: int, color: Tuple[int, int, int]) -> None:
    # Anchor the marker apex at (px, py) so the rendered current point sits
    # exactly on the true XY path point for any heading.
    p_apex = np.array([0.0, 0.0], dtype=float)
    p_left = np.array([1.75 * size, 0.9 * size], dtype=float)
    p_right = np.array([1.75 * size, -0.9 * size], dtype=float)
    base = np.stack([p_apex, p_left, p_right], axis=0)

    c = math.cos(heading)
    s = math.sin(heading)
    r = np.array([[c, -s], [s, c]], dtype=float)
    rot = base @ r.T

    pts = np.zeros((3, 2), dtype=np.int32)
    pts[:, 0] = np.round(px + rot[:, 0]).astype(np.int32)
    pts[:, 1] = np.round(py - rot[:, 1]).astype(np.int32)

    line_w = max(1, int(round(size * 0.22)))
    dot_r = max(1, int(round(size * 0.14)))
    cv2.line(img, tuple(pts[0]), tuple(pts[1]), color, line_w, cv2.LINE_AA)
    cv2.line(img, tuple(pts[0]), tuple(pts[2]), color, line_w, cv2.LINE_AA)
    cv2.circle(img, tuple(pts[0]), dot_r, (20, 20, 20), -1, cv2.LINE_AA)


def render_right_panel(
    panel_w: int,
    panel_h: int,
    poses: np.ndarray,
    frame_idx: int,
    scale: float,
    camera_size: int,
    trail_len: int,
    pose_mode: str,
    source: str,
    future_global: np.ndarray,
    future_camera_scale: float,
    model_future_global: Optional[np.ndarray],
) -> np.ndarray:
    panel = np.full((panel_h, panel_w, 3), 245, dtype=np.uint8)
    cx = panel_w // 2
    cy = panel_h // 2

    meters_span_x = panel_w / max(scale, 1e-6)
    meters_span_y = panel_h / max(scale, 1e-6)
    max_m = int(max(2, math.ceil(max(meters_span_x, meters_span_y) * 0.5)))
    y0 = int(round(cy))
    x0 = int(round(cx))
    for m in range(-max_m, max_m + 1):
        x = int(round(cx + m * scale))
        y = int(round(cy - m * scale))
        col = (220, 220, 220) if m != 0 else (180, 180, 180)
        cv2.line(panel, (x, 0), (x, panel_h - 1), col, 1, cv2.LINE_AA)
        cv2.line(panel, (0, y), (panel_w - 1, y), col, 1, cv2.LINE_AA)

        if 16 <= x <= panel_w - 42:
            cv2.putText(panel, f"{m}m", (x - 12, min(panel_h - 8, y0 + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1, cv2.LINE_AA)
        if 16 <= y <= panel_h - 12:
            cv2.putText(panel, f"{m}m", (max(4, x0 + 6), y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1, cv2.LINE_AA)

    # Draw exact full path so current/future context is always visible.
    if len(poses) >= 2:
        full_pts = [to_panel_xy(float(p[0]), float(p[1]), cx, cy, scale) for p in poses]
        cv2.polylines(panel, [np.array(full_pts, dtype=np.int32)], False, (180, 180, 180), 1, cv2.LINE_AA)

    i = frame_idx
    s = max(0, i - max(1, trail_len) + 1)
    trail = poses[s : i + 1]
    if len(trail) >= 2:
        pts = [to_panel_xy(float(p[0]), float(p[1]), cx, cy, scale) for p in trail]
        cv2.polylines(panel, [np.array(pts, dtype=np.int32)], False, (50, 120, 230), 2, cv2.LINE_AA)

    px, py = to_panel_xy(float(poses[i, 0]), float(poses[i, 1]), cx, cy, scale)
    draw_camera_eye(panel, px, py, float(poses[i, 2]), camera_size, (30, 180, 70))

    if source == "refined" and future_global is not None:
        fut = future_global[i]
        fut_pts = [to_panel_xy(float(p[0]), float(p[1]), cx, cy, scale) for p in fut]
        if len(fut_pts) >= 2:
            cv2.polylines(panel, [np.array(fut_pts, dtype=np.int32)], False, (210, 70, 90), 2, cv2.LINE_AA)
        future_size = max(6, int(round(camera_size * max(0.1, float(future_camera_scale)))))
  
        for j, (fx, fy) in enumerate(fut_pts):
            c = 90 + int(120.0 * (j + 1) / max(1, len(fut_pts)))
            color = (c, 60, 100)
            draw_camera_eye(panel, fx, fy, float(fut[j, 2]), future_size, color)

    if model_future_global is not None:
        pred = model_future_global
        pred_pts = [to_panel_xy(float(p[0]), float(p[1]), cx, cy, scale) for p in pred]
        if len(pred_pts) >= 2:
            cv2.polylines(panel, [np.array(pred_pts, dtype=np.int32)], False, (0, 0, 255), 2, cv2.LINE_AA)
        pred_size = max(6, int(round(camera_size * max(0.1, float(future_camera_scale)))))
        for j, (fx, fy) in enumerate(pred_pts):
            red_v = 120 + int(135.0 * (j + 1) / max(1, len(pred_pts)))
            draw_camera_eye(panel, fx, fy, float(pred[j, 2]), pred_size, (0, 0, red_v))

    n = len(poses)
    txt0 = f"frame: {i + 1}/{n}"
    txt1 = f"x: {float(poses[i, 0]):+.3f} m"
    txt2 = f"y: {float(poses[i, 1]):+.3f} m"
    txt3 = f"yaw: {float(poses[i, 2]):+.3f} rad"
    cv2.putText(panel, txt0, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(panel, txt1, (14, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(panel, txt2, (14, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(panel, txt3, (14, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)
    label = "refined_t" if source == "refined" else "gt.json"
    cv2.putText(panel, f"XY plane ({label}), mode={pose_mode}", (14, panel_h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2, cv2.LINE_AA)
    return panel


def main() -> None:
    args = parse_args()

    clip_path = args.clip_path
    if not clip_path.exists():
        raise FileNotFoundError(f"Clip not found: {clip_path}")

    session_dir = clip_path.parent

    if args.split != "all":
        train_set, val_set = split_session_clips(session_dir, args.train_ratio, args.split_seed)
        selected = train_set if args.split == "train" else val_set
        if clip_path.name not in selected:
            raise RuntimeError(
                f"Clip {clip_path.name} is not in split='{args.split}' for session {session_dir.name} "
                f"(train_ratio={args.train_ratio}, seed={args.split_seed})."
            )

    gt_path = args.gt_path if args.gt_path is not None else (session_dir / "gt.json")
    if not gt_path.exists():
        raise FileNotFoundError(f"GT file not found: {gt_path}")

    refined_path = None
    if args.source == "refined":
        if args.pose_mode != "aligned":
            raise RuntimeError("--source refined requires --pose-mode aligned")
        if args.refined_path is not None:
            refined_path = args.refined_path
        else:
            p1 = session_dir / "refined_gt.json"
            p2 = session_dir / "refined_t.json"
            refined_path = p1 if p1.exists() else p2
        if not refined_path.exists():
            raise FileNotFoundError(
                f"Refined file not found: {refined_path}. Expected refined_gt.json or refined_t.json in session dir"
            )

    if args.out_path is not None:
        out_path = args.out_path
    else:
        repo_root = Path(__file__).resolve().parent.parent
        participant = session_dir.parent.name
        session = session_dir.name
        out_path = repo_root / "debug" / participant / session / f"{clip_path.stem}_with_slam_gt.mp4"

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open clip: {clip_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 1e-6:
        raise RuntimeError(f"Invalid FPS from video metadata: {fps} for {clip_path}")

    w = int(round(float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))))
    h = int(round(float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    n_frames = int(round(float(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

    poses_raw = load_clip_xytheta_strict(gt_path, clip_path)
    validate_exact_alignment(poses_raw, n_frames, clip_path, gt_path)
    poses = transform_poses(poses_raw, args.pose_mode)
    instruction_text = load_clip_instruction(session_dir, clip_path)

    future_global = None
    if args.source == "refined":
        future_local = load_clip_refined_local_future_strict(refined_path, clip_path)
        if future_local.shape[0] != len(poses):
            raise RuntimeError(
                f"Refined frame count {future_local.shape[0]} does not match gt/video frame count {len(poses)} for {clip_path}"
            )
        future_global = refined_local_to_aligned_global(poses, future_local)

    predictor = None
    if args.model_checkpoint is not None:
        predictor = NavModelOverlay(
            config_path=args.model_config,
            checkpoint_path=args.model_checkpoint,
            device=args.device,
            num_ddim_steps=args.num_ddim_steps,
            cfg_scale=args.cfg_scale,
        )

    max_extent = float(np.max(np.abs(poses[:, :2]))) if len(poses) else 1.0
    if future_global is not None and future_global.size > 0:
        max_extent = max(max_extent, float(np.max(np.abs(future_global[:, :, :2]))))
    max_extent = max(max_extent, 0.5)
    scale = 0.42 * min(w, h) / max_extent

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(out_path),
        fps=fps,
        codec="libx264",
        macro_block_size=1,
    )

    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if i >= len(poses):
            raise RuntimeError(
                f"Video produced more frames than GT rows ({i + 1} > {len(poses)}) for {clip_path}"
            )

        pred_global = None
        if predictor is not None:
            pred_local = predictor.predict_local_future(frame, instruction_text)
            pred_global = refined_local_to_aligned_global(poses[i : i + 1], pred_local[None, :, :])[0]

        right = render_right_panel(
            panel_w=w,
            panel_h=h,
            poses=poses,
            frame_idx=i,
            scale=scale,
            camera_size=args.camera_size,
            trail_len=args.trail_len,
            pose_mode=args.pose_mode,
            source=args.source,
            future_global=future_global,
            future_camera_scale=args.future_camera_scale,
            model_future_global=pred_global,
        )

        combo = np.hstack([frame, right])
        draw_instruction(combo, instruction_text)
        combo_rgb = cv2.cvtColor(combo, cv2.COLOR_BGR2RGB)
        writer.append_data(combo_rgb)
        i += 1

    cap.release()
    writer.close()

    if i != len(poses):
        raise RuntimeError(
            f"Rendered frame count {i} does not match GT row count {len(poses)} for {clip_path}"
        )

    print("Done.")
    print(f"clip: {clip_path}")
    print(f"gt: {gt_path}")
    if refined_path is not None:
        print(f"refined: {refined_path}")
    print(f"source: {args.source}")
    print(f"pose_mode: {args.pose_mode}")
    print(f"split: {args.split} (train_ratio={args.train_ratio}, seed={args.split_seed})")
    print(f"model_overlay: {'on' if predictor is not None else 'off'}")
    if predictor is not None:
        print(f"model_checkpoint: {args.model_checkpoint}")
        print(f"model_xy_norm: {predictor.xy_norm_mode}")
    print(f"instruction: {instruction_text if instruction_text else '(none)'}")
    print(f"frames: {i}")
    print(f"out: {out_path}")


if __name__ == "__main__":
    main()
