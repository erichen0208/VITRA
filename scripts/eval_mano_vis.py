"""
eval_mano_vis.py — Evaluate VITRA and record side-by-side video with raw MANO output.

Left panel  : robot simulation (head camera RGB)
Right panel : 3D MANO hand mesh rendered from the model's raw 192-dim output
              (before gripper retargeting — shows exactly what the DiT predicts)

Usage:
    python scripts/eval_mano_vis.py \
        --checkpoint ./checkpoints/aloha_mini/final.pt \
        --output_dir ./checkpoints/aloha_mini/mano_vis \
        --num_episodes 2 \
        --object 006_mustard_bottle
"""

import argparse, json, os, sys, time
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
import imageio

# ── repo paths ────────────────────────────────────────────────────────────
_HERE = os.path.dirname(__file__)
_VITRA = os.path.join(_HERE, "..")
_DC = os.path.join(_HERE, "../../data_collection_maniskill")
for p in [_VITRA, _DC]:
    if p not in sys.path:
        sys.path.insert(0, p)

from vitra.utils.config_utils import load_config
from vitra.models.vla_builder import load_model
from vitra.datasets.grasp_dataset import ACTION_DIM, STATE_DIM, ActionNormalizer
from vitra.utils.gripper_retarget import (
    gripper_to_mano_state, gripper_to_mano_state_mask,
    gripper_to_mano_mask, mano_action_to_gripper,
    RIGHT_TRANS, RIGHT_ROT, RIGHT_JOINTS, MANO_REST_HAND_POSE,
    mano_wrist_rotation, MANO_REST_R, R_WORLD_TO_MANO_CAM,
    FOUR_FINGER_MCP_SLICES, MCP_FLEX_OFFSET,
    tcp_rot_to_mano_orient,
)
from coordinate_transforms import tcp_in_cam_frame, build_ee_state
from ik_controller import IKController
from robot_registry import ROBOT_CONFIGS, make_instruction

# Rx(+90deg): rotate scene so dorsal side (+Y hand) faces camera (+Z).
_R_TOP_VIEW = R.from_euler('x', 90, degrees=True).as_matrix().astype(np.float32)

# Scales robot world-frame TCP offset (metres) to MANO camera-space offset.
# Robot world frame: +X forward, +Y left, +Z up
# MANO camera frame (OpenCV): +X right, +Y down, +Z depth (into scene)
#   world +X (fwd)  → MANO +Z  (depth)
#   world +Y (left) → MANO -X  (left = negative right)
#   world +Z (up)   → MANO -Y  (up = negative down)
WORLD_TO_MANO_SCALE = 2.0


# ═══════════════════════════════════════════════════════════════════════════
#  MANO renderer  (front + top-down views, with full 6D wrist pose)
# ═══════════════════════════════════════════════════════════════════════════

class MANORenderer:
    """Render a single right-hand MANO mesh from a 192-dim action vector.

    Produces TWO views following analysis/visualize_retarget_video.py:
      • Front view  — fingers into screen, palm facing left
      • Top view    — dorsal (back of hand) facing camera

    The model-predicted 6D wrist pose (trans + rot) is reflected:
      • Translation dims [51:54] drive the MANO `transl` parameter.
      • Rotation dims [54:57] drive `global_orient` via
        `mano_wrist_rotation() @ MANO_REST_R`.
    """

    PANEL_W = 448
    PANEL_H = 448
    FOCAL   = 420.0
    DEPTH   = 0.38   # base z-distance from camera (m) — closer = bigger hand
    HAND_COLOR = np.array([0.4078, 0.4980, 0.7451], dtype=np.float32)

    def __init__(self, mano_path: str, device: str = "cuda"):
        self.device = torch.device(device)
        from libs.models.mano_wrapper import MANO
        from visualization.render_utils import Renderer

        self.mano = MANO(
            model_path=mano_path,
            is_rhand=True,
            use_pca=False,
            flat_hand_mean=False,
        ).to(self.device)

        self.faces = torch.from_numpy(self.mano.faces.astype(np.int64)).to(self.device)
        self.betas = torch.zeros(1, 10, device=self.device)
        self.renderer = Renderer(
            width=self.PANEL_W,
            height=self.PANEL_H,
            focal_length=(self.FOCAL, self.FOCAL),
            device=self.device,
        )
        self._K = np.array([
            [self.FOCAL, 0, self.PANEL_W / 2],
            [0, self.FOCAL, self.PANEL_H / 2],
            [0, 0, 1],
        ], dtype=np.float64)

    # ── helpers ───────────────────────────────────────────────────────

    def _proj(self, pt3d):
        uvw = self._K @ pt3d
        if abs(uvw[2]) < 1e-6:
            return None
        return (int(round(uvw[0] / uvw[2])), int(round(uvw[1] / uvw[2])))

    def _draw_axes(self, img, origin_3d, R_orient, length=0.04):
        """Draw XYZ axes on an RGB image (not BGR)."""
        rgb_colors = [(220, 0, 0), (0, 180, 0), (0, 0, 220)]
        h, w = img.shape[:2]
        o = self._proj(origin_3d)
        if o is None:
            return
        for k, lbl in enumerate(['X', 'Y', 'Z']):
            ep = self._proj(origin_3d + R_orient[:, k] * length)
            if ep is None:
                continue
            cv2.arrowedLine(img, o, ep, rgb_colors[k], 2, tipLength=0.28)
            cv2.putText(img, lbl,
                        (max(3, min(w - 16, ep[0])), max(12, min(h - 3, ep[1]))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, rgb_colors[k], 1)

    @staticmethod
    def _world_to_mano_transl(tcp_offset_world):
        """Robot world-frame offset (m) → MANO camera-space offset.

        Robot world frame: +X forward, +Y left, +Z up
        MANO camera (OpenCV): +X right, +Y down, +Z depth (into scene)
            world +X (fwd)  → MANO +Z
            world +Y (left) → MANO -X
            world +Z (up)   → MANO -Y
        """
        ox, oy, oz = np.asarray(tcp_offset_world, dtype=np.float32)
        return np.array([-oy, -oz, ox], dtype=np.float32) * WORLD_TO_MANO_SCALE

    def _prepare_hand(self, mano_action):
        """Extract and clean hand pose from 192-dim action vector.

        Returns: (hp45_euler [45], rot_euler [3], cam_trans [3])
        """
        cam_trans   = mano_action[RIGHT_TRANS]   # [3] camera-frame translation
        rot_euler   = mano_action[RIGHT_ROT]     # [3] wrist euler [roll, flex, yaw]
        joints_flat = mano_action[RIGHT_JOINTS]  # [45] 15 joint × 3 Euler XYZ

        # Keep ONLY the 4 trained MCP-flex dims (others are random noise)
        hp45_clean = np.zeros(45, dtype=np.float32)
        for mcp in FOUR_FINGER_MCP_SLICES:
            dim = mcp.start + MCP_FLEX_OFFSET
            hp45_clean[dim] = joints_flat[dim]
        # Add rest pose for natural thumb spread and slight PIP curl
        hp45 = hp45_clean + MANO_REST_HAND_POSE
        return hp45, rot_euler, cam_trans

    @torch.no_grad()
    def _render_with_orient(self, hp45, R_go, transl):
        """Core render: hp45 (45 Euler), R_go (3×3), transl (3,) → RGB uint8."""
        go_t = torch.from_numpy(R_go.astype(np.float32))[None, None].to(self.device)

        R_hp = R.from_euler('xyz', hp45.reshape(15, 3)).as_matrix().astype(np.float32)
        hp_t = torch.from_numpy(R_hp)[None].to(self.device)

        transl_t = torch.tensor(transl, dtype=torch.float32, device=self.device).unsqueeze(0)

        out = self.mano(
            global_orient=go_t, hand_pose=hp_t, betas=self.betas,
            transl=transl_t, pose2rot=False,
        )
        verts = out.vertices[0]  # [V, 3]

        n_v = verts.shape[0]
        colors = torch.tensor(self.HAND_COLOR, device=self.device).expand(n_v, 3)
        rend, mask = self.renderer.render(
            verts_list=[verts], faces_list=[self.faces], colors_list=[colors],
        )
        bg = np.full((self.PANEL_H, self.PANEL_W, 3), 220, dtype=np.uint8)
        bg[mask] = rend[mask]

        # Draw orientation axes at wrist
        try:
            wrist_3d = verts.cpu().numpy().mean(0)
            wrist_3d[2] = max(wrist_3d[2], 0.05)
            self._draw_axes(bg, wrist_3d, R_go.astype(np.float64))
        except Exception:
            pass
        return bg

    def render_views(self, mano_action: np.ndarray,
                     tcp_offset_world: np.ndarray,
                     R_tcp_world: np.ndarray,
                     R_tcp_rest: np.ndarray,
                     mano_offset_override: np.ndarray | None = None):
        """Render front + top-down views using actual TCP pose from simulator.

        Args:
            mano_action:  [192] model output for finger joints only.
            tcp_offset_world:  [3] world-frame TCP offset from rest (metres).
            R_tcp_world:  [3,3] actual TCP rotation in world frame.
            R_tcp_rest:   [3,3] TCP rotation at rest/neutral pose.
            mano_offset_override: Optional pre-normalised [3] offset in MANO
                camera space.  If None, computed from tcp_offset_world.
        Returns:
            front_rgb: [H, W, 3] uint8 RGB — front view
            top_rgb:   [H, W, 3] uint8 RGB — dorsal/top-down view
        """
        hp45, _, _ = self._prepare_hand(mano_action)

        # Global orient from actual TCP rotation (exact 6-DOF retarget)
        R_go = tcp_rot_to_mano_orient(R_tcp_world, R_tcp_rest)

        # Translation: base depth + world-to-MANO-camera offset
        base_transl = np.array([0., 0., self.DEPTH], dtype=np.float32)
        if mano_offset_override is not None:
            mano_offset = mano_offset_override
        else:
            mano_offset = self._world_to_mano_transl(tcp_offset_world)

        # ── Front view ────────────────────────────────────────────────
        front_transl = base_transl + mano_offset
        front = self._render_with_orient(hp45, R_go, front_transl)

        # ── Top-down (dorsal) view ────────────────────────────────────
        R_go_top = _R_TOP_VIEW @ R_go
        # Rotate the offset into top-cam frame: Rx(90) @ [x,y,z] = [x, -z, y]
        off_top = np.array([mano_offset[0], -mano_offset[2], mano_offset[1]],
                           dtype=np.float32)
        top_transl = base_transl + off_top
        top = self._render_with_orient(hp45, R_go_top, top_transl)

        return front, top

    def blank_frame(self):
        """Return dark placeholder pair when no prediction is available."""
        img = np.full((self.PANEL_H, self.PANEL_W, 3), 220, dtype=np.uint8)
        cv2.putText(img, "Waiting...", (60, self.PANEL_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1, cv2.LINE_AA)
        return img, img.copy()


# ═══════════════════════════════════════════════════════════════════════════
#  Episode runner
# ═══════════════════════════════════════════════════════════════════════════

def _overlay_label(img: np.ndarray, text: str, color=(255, 255, 80)) -> np.ndarray:
    img = img.copy()
    cv2.putText(img, text, (6, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img


def run_episode(
    env, ik, model, action_norm, state_norm,
    instruction, chunk_size, execute_steps, ddim_steps, cfg_scale,
    cam_head_key, cam_wrist_key, tcp_pose_attr,
    use_retarget, use_left, fov, device,
    mano_renderer: MANORenderer,
    out_path: str,
    max_steps: int = 200,
) -> bool:
    """Run one evaluation episode; write side-by-side video to out_path.

    Two-pass strategy
    -----------------
    Pass 1 — robot control: run the episode step-by-step, buffer each step's
             sim RGB, raw 192-dim MANO prediction, and gripper position.
             No MANO rendering yet.
    Normalise: compute global translation centre & scale across the full
             trajectory so the hand always stays within the render frame.
    Pass 2 — render: replay buffered data, render MANO with globally-
             normalised offsets, compose and write video.
    """
    # Maximum hand displacement from centre in MANO camera space (metres).
    # Keeps the hand within ~quarter of the frame width on either side.
    TRANS_LIMIT = 0.07

    use_wrist_rgbd = model.use_wrist_cam_train
    fps = env.unwrapped.control_freq

    raw_obs, _ = env.reset()
    buf, buf_i = None, 0
    cur_mano192_chunk = None   # [T, 192] current prediction chunk

    # ── Helper: read TCP world-frame pose from simulator ──────────────
    def _read_tcp_world_pose(env, tcp_pose_attr):
        tcp_pose = getattr(env.unwrapped.agent, tcp_pose_attr)
        pos = tcp_pose.p[0].cpu().numpy().astype(np.float32)     # [3]
        quat = tcp_pose.q[0].cpu().numpy()                       # [4] wxyz
        R_tcp = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()  # scipy uses xyzw
        return pos, R_tcp.astype(np.float32)

    # ── Pass 1: robot control ─────────────────────────────────────────
    # Each entry: (sim_rgb, mano192, cur_grip, tcp_pos_world, R_tcp_world)
    recorded: list[tuple] = []
    success = False

    # Record the initial (rest) TCP world pose before any actions
    tcp_pos_rest, R_tcp_rest = _read_tcp_world_pose(env, tcp_pose_attr)

    for step in range(max_steps):
        # ── Inference step ──
        if buf is None or buf_i >= execute_steps:
            head = raw_obs["sensor_data"][cam_head_key]["rgb"][0].cpu().numpy()
            wrist_rgbd_t = None
            if use_wrist_rgbd:
                wrist_rgb  = raw_obs["sensor_data"][cam_wrist_key]["rgb"][0].cpu().numpy()
                depth_raw  = raw_obs["sensor_data"][cam_wrist_key]["depth"][0, :, :, 0].cpu()
                depth_m    = (depth_raw.float() / 1000.0).clamp(0.0, 10.0)
                rgb_t      = torch.from_numpy(wrist_rgb).float().permute(2, 0, 1) / 255.0
                wrist_rgbd_t = torch.cat([rgb_t, depth_m.unsqueeze(0)], dim=0).unsqueeze(0).to(device)

            tcp_pos, tcp_euler = tcp_in_cam_frame(env, cam_head_key, tcp_pose_attr, env_idx=0)
            qpos_now  = env.unwrapped.agent.robot.get_qpos().cpu().numpy()
            grip_now  = qpos_now[0, ik.gripper_qidx].astype(np.float32)
            state_7   = build_ee_state(tcp_pos, tcp_euler, grip_now)
            s_norm    = state_norm.normalize(state_7)

            s_mano   = gripper_to_mano_state(torch.tensor(s_norm, dtype=torch.float32), use_left=use_left)
            s_t      = s_mano.unsqueeze(0).to(device)
            s_mask_7 = torch.ones(1, STATE_DIM, dtype=torch.float32, device=device)
            s_mask   = gripper_to_mano_state_mask(s_mask_7, use_left=use_left).to(device)
            a_mask_7 = torch.ones(1, chunk_size, ACTION_DIM, dtype=torch.float32, device=device)
            a_mask   = gripper_to_mano_mask(a_mask_7, use_left=use_left).to(device)

            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred = model.predict_action(
                    image=head, instruction=instruction,
                    current_state=s_t, current_state_mask=s_mask,
                    action_mask_torch=a_mask,
                    num_ddim_steps=ddim_steps, cfg_scale=cfg_scale,
                    fov=fov, sample_times=1, wrist_rgbd=wrist_rgbd_t,
                )

            # pred[0] shape: [T, 192]  — raw MANO output before any retargeting
            cur_mano192_chunk = pred[0]  # [T, 192] numpy

            # Retarget back to 7-dim gripper for robot control
            if use_retarget:
                pred_7d = mano_action_to_gripper(cur_mano192_chunk, use_left=use_left)  # [T, 7]
                chunk   = action_norm.denormalize(pred_7d)
            else:
                chunk   = action_norm.denormalize(cur_mano192_chunk[:, :7])
            chunk[:, 6] = np.clip(chunk[:, 6], -1.1, 0.0)
            buf, buf_i = chunk, 0

        ee_action         = buf[buf_i]
        mano192_this_step = cur_mano192_chunk[buf_i]  # [192] raw for this timestep
        buf_i += 1

        # ── Gripper: absolute target → delta ──
        cur_grip   = float(env.unwrapped.agent.robot.get_qpos()[0, ik.gripper_qidx])
        ee_exec    = ee_action.copy()
        ee_exec[6] = ee_action[6] - cur_grip

        act_t = ik.step(ee_exec, env, env_idx=0)
        raw_obs, reward, terminated, truncated, info = env.step(act_t)

        if info.get("success", torch.tensor([False]))[0].item():
            success = True

        # Read actual TCP world pose AFTER the physics step (smooth)
        tcp_pos_w, R_tcp_w = _read_tcp_world_pose(env, tcp_pose_attr)

        sim_rgb = raw_obs["sensor_data"][cam_head_key]["rgb"][0].cpu().numpy()
        recorded.append((sim_rgb, mano192_this_step.copy(), cur_grip,
                         tcp_pos_w.copy(), R_tcp_w.copy()))

        if terminated.any() or truncated.any():
            break

    # ── Compute global translation normalisation ───────────────────────
    # Convert all world-frame TCP offsets to MANO camera space.
    all_mano_offsets = np.stack(
        [mano_renderer._world_to_mano_transl(rec[3] - tcp_pos_rest)
         for rec in recorded], axis=0
    )  # [T, 3]
    trans_center = all_mano_offsets.mean(axis=0)            # [3]
    max_dev = np.abs(all_mano_offsets - trans_center).max()  # scalar
    # Scale so the largest excursion maps to TRANS_LIMIT; never scale up.
    trans_scale = min(TRANS_LIMIT / max(max_dev, 1e-4), 1.0)

    # ── Pass 2: render + compose frames ───────────────────────────────
    frames: list[np.ndarray] = []
    for step, (sim_rgb, mano192, cur_grip, tcp_pos_w, R_tcp_w) in enumerate(recorded):
        # Globally-normalised MANO-space offset: hand stays centred with
        # proportional but bounded movement.
        raw_offset  = mano_renderer._world_to_mano_transl(tcp_pos_w - tcp_pos_rest)
        norm_offset = (raw_offset - trans_center) * trans_scale  # [3]

        mano_front, mano_top = mano_renderer.render_views(
            mano192,
            tcp_offset_world=(tcp_pos_w - tcp_pos_rest),
            R_tcp_world=R_tcp_w,
            R_tcp_rest=R_tcp_rest,
            mano_offset_override=norm_offset,
        )

        # ── Compose 3-panel side-by-side frame ──
        H_sim, W_sim = sim_rgb.shape[:2]
        H_out = mano_front.shape[0]  # PANEL_H (448)

        # Upscale sim to match MANO panel height so hands appear large.
        if H_sim != H_out:
            w_new  = int(W_sim * H_out / H_sim)
            sim_up = cv2.resize(sim_rgb, (w_new, H_out), interpolation=cv2.INTER_LINEAR)
        else:
            sim_up = sim_rgb
        W_sim_up = sim_up.shape[1]

        sim_labeled   = _overlay_label(sim_up,     f"Robot sim  step={step:03d}", (255, 220, 80))
        front_labeled = _overlay_label(mano_front,  "MANO front",                  (80, 220, 255))
        top_labeled   = _overlay_label(mano_top,    "MANO dorsal",                 (80, 255, 180))

        frame = np.concatenate([sim_labeled, front_labeled, top_labeled], axis=1)

        # ── Gripper closing indicator overlay ──
        grip_pct = abs(cur_grip) / 1.1
        bar_w    = int(grip_pct * (W_sim_up - 20))
        cv2.rectangle(frame, (10, H_out - 14), (10 + bar_w, H_out - 5),
                      (80, 200, 80) if grip_pct < 0.3 else (200, 80, 80), -1)
        cv2.putText(frame, f"grip {cur_grip:.3f}", (10, H_out - 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        frames.append(frame)

    # ── Write video ──
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if frames:
        try:
            imageio.mimsave(out_path, frames, fps=fps, codec='libx264', quality=8)
            print(f"  Saved SUCCESS → {out_path} ({len(frames)} frames)")
        except Exception as e:
            print(f"  Saved FAIL → {e}")

    return success


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="VITRA eval with MANO hand visualization")
    parser.add_argument("--checkpoint",   required=True,  help="Path to .pt checkpoint")
    parser.add_argument("--output_dir",   default="/tmp/mano_vis")
    parser.add_argument("--num_episodes", type=int, default=2, help="Episodes per object")
    parser.add_argument("--max_steps",    type=int, default=200)
    parser.add_argument("--object",       default=None,
                        help="Single YCB object ID (e.g. 004_sugar_box). If omitted, run all 8 eval objects.")
    parser.add_argument("--seed",         type=int, default=1)
    parser.add_argument("--no_cuda",      action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

    # ── Load config, stats, model ──
    ckpt_dir    = os.path.dirname(args.checkpoint)
    config_path = os.path.join(ckpt_dir, "config.json")
    stats_path  = os.path.join(ckpt_dir, "statistics.json")

    configs = load_config(config_path)
    configs["model_load_path"] = args.checkpoint
    with open(stats_path) as f:
        stats = json.load(f)

    print("Loading model …")
    model = load_model(configs).to(device).eval()
    model.use_bf16 = configs.get("use_bf16", True)
    print(f"Model loaded  ({sum(p.numel() for p in model.parameters())/1e6:.1f} M params)")

    # ── MANO renderer ──
    mano_path = os.path.join(_VITRA, "weights", "mano")
    mano_renderer = MANORenderer(mano_path=mano_path, device=device)

    # ── Eval config ──
    eval_cfg    = configs.get("eval", {})
    data_cfg    = configs.get("data", {})
    retarget_cfg = configs.get("retarget", {})
    rcfg        = ROBOT_CONFIGS[eval_cfg.get("robot", "aloha_mini")]
    use_retarget = configs.get("use_retarget", True)
    use_left    = retarget_cfg.get("use_left", False)
    chunk_size  = configs.get("fwd_pred_next_n", 16)
    fov_rad     = data_cfg.get("camera_fov_rad", 1.6)
    execute_steps = eval_cfg.get("execute_steps", 8)
    ddim_steps  = eval_cfg.get("ddim_steps", 10)
    cfg_scale   = eval_cfg.get("cfg_scale", 1.5)
    frozen      = rcfg["frozen_joint_indices"]
    cam_head_key = rcfg["cam_head"]
    cam_wrist_key = rcfg["cam_wrist"]
    tcp_pose_attr = rcfg["tcp_pose_attr"]
    exclude_cams  = list(rcfg["cam_exclude"])
    fov = torch.tensor([[fov_rad, fov_rad]], dtype=torch.float32)

    action_norm = ActionNormalizer(stats["action_mean"], stats["action_std"],
                                   vmin=stats.get("action_min"), vmax=stats.get("action_max"))
    state_norm  = ActionNormalizer(stats["state_mean"],  stats["state_std"],
                                   vmin=stats.get("state_min"),  vmax=stats.get("state_max"))

    # ── Object list ──
    import mani_skill.envs  # noqa: F401
    from mani_skill.envs.tasks.digital_twins.grasp.grasp_ycb_random import (
        YCB_EVAL_OBJECTS, YCB_EVAL_SEEN_IDX, YCB_EVAL_UNSEEN_IDX,
    )
    import mani_skill.envs.tasks.digital_twins.grasp.grasp_ycb_random as _ycb_mod
    import gymnasium as gym

    save_train = list(_ycb_mod._YCB_TRAIN)
    if args.object:
        obj_list = [args.object]
    else:
        obj_list = list(YCB_EVAL_OBJECTS)

    results = {}

    for obj_id in obj_list:
        _ycb_mod._YCB_TRAIN = [obj_id]
        instruction = make_instruction(obj_id)
        tag = "seen" if YCB_EVAL_OBJECTS.index(obj_id) in YCB_EVAL_SEEN_IDX else "unseen"
        print(f"\n[{tag}] {obj_id} — {instruction}")

        env = gym.make(
            rcfg["env_ids"]["ycb_random"], num_envs=1,
            obs_mode="rgbd", render_mode="rgb_array", sim_backend="auto",
            robot_uids=rcfg["robot_uid"], control_mode=rcfg["control_mode"],
            frozen_joint_indices=frozen, exclude_cameras=exclude_cams,
            sim_config=dict(scene_config=dict(
                solver_position_iterations=25,
                enable_ccd=True,
            )),
        )
        env.reset(seed=args.seed)
        ik = IKController(env, rcfg, frozen,
                          damping=eval_cfg.get("ik_damping", 0.05),
                          gripper_scale=eval_cfg.get("gripper_scale", 1.0),
                          rotation_weight=eval_cfg.get("rotation_weight", 0.3))
        torch.cuda.empty_cache()

        obj_dir  = os.path.join(args.output_dir, obj_id)
        n_ok = 0
        for ep in range(args.num_episodes):
            out_path = os.path.join(obj_dir, f"ep{ep}.mp4")
            ok = run_episode(
                env=env, ik=ik, model=model,
                action_norm=action_norm, state_norm=state_norm,
                instruction=instruction,
                chunk_size=chunk_size, execute_steps=execute_steps,
                ddim_steps=ddim_steps, cfg_scale=cfg_scale,
                cam_head_key=cam_head_key, cam_wrist_key=cam_wrist_key,
                tcp_pose_attr=tcp_pose_attr,
                use_retarget=use_retarget,
                use_left=use_left, fov=fov, device=device,
                mano_renderer=mano_renderer,
                out_path=out_path,
                max_steps=args.max_steps,
            )
            n_ok += int(ok)

        env.close()
        rate = 100 * n_ok / args.num_episodes
        results[obj_id] = {"tag": tag, "success": n_ok, "total": args.num_episodes, "rate": rate}
        print(f"  → {n_ok}/{args.num_episodes} ({rate:.0f}%)")

    _ycb_mod._YCB_TRAIN = save_train

    # ── Summary ──
    print(f"\n{'='*50}")
    print(f"{'Object':<28} {'Tag':<8} {'Rate':>6}")
    print("-" * 50)
    for obj_id, r in results.items():
        print(f"{obj_id:<28} {r['tag']:<8} {r['rate']:>5.0f}%")
    overall = 100 * sum(r["success"] for r in results.values()) / max(sum(r["total"] for r in results.values()), 1)
    print(f"\nOverall: {overall:.1f}%")
    print(f"Videos  : {args.output_dir}")

    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
