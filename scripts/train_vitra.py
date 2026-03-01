"""
train_vitra.py — Train / Evaluate VITRA on single-arm grasp tasks.

Native 7-dim state and action (no 192/212-dim padding).
Wrist camera uses RGBD (4-channel) encoder.

Usage:
    # Train
    python scripts/train_vitra.py train --config vitra/configs/aloha_mini_finetune.json

    # Evaluate (auto-discovers config/stats from checkpoint dir)
    python scripts/train_vitra.py eval --checkpoint ./checkpoints/aloha_mini/final.pt

    # Print dataset statistics
    python scripts/train_vitra.py stats --data_dir <lerobot_data_dir>
"""

import argparse, json, math, os, sys, time
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from vitra.datasets.grasp_dataset import (
    ACTION_DIM, STATE_DIM, GraspDatasetCore, load_or_compute_statistics,
    decode_depth_from_rgb, ActionNormalizer,
)
from vitra.models.vla_builder import load_model
from vitra.utils.config_utils import load_config
from vitra.utils.data_utils import PaddedCollatorForHandPrediction

# ─── Robot registry (single source of truth) ─────────────────────────────
_DC_DIR = os.path.join(os.path.dirname(__file__), "../../data_collection_maniskill")
if _DC_DIR not in sys.path:
    sys.path.insert(0, _DC_DIR)
from robot_registry import ROBOT_CONFIGS, TASK_DESCRIPTIONS, EE_ACTION_DIM, JOINT_ACTION_DIM, make_instruction  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════
#  Statistics  (simple 7-dim Gaussian — no left/right split)
# ═══════════════════════════════════════════════════════════════════════════

def compute_statistics_from_lerobot(data_dir: str) -> dict:
    """Read LeRobot meta/stats.json → simple 7-dim statistics dict."""
    stats_path = os.path.join(data_dir, "meta", "stats.json")
    assert os.path.exists(stats_path), f"Not found: {stats_path}"
    with open(stats_path) as f:
        lr = json.load(f)
    return {
        "dataset_name": os.path.basename(data_dir),
        "action_dim": len(lr["action"]["mean"]),
        "state_dim": len(lr["observation.state"]["mean"]),
        "state_mean": lr["observation.state"]["mean"],
        "state_std":  lr["observation.state"]["std"],
        "state_min":  lr["observation.state"]["min"],
        "state_max":  lr["observation.state"]["max"],
        "action_mean": lr["action"]["mean"],
        "action_std":  lr["action"]["std"],
        "action_min":  lr["action"]["min"],
        "action_max":  lr["action"]["max"],
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def _ensure_maniskill():
    import mani_skill.envs  # noqa: F401


@torch.no_grad()
def run_eval(model, stats, configs, record_dir=None, num_episodes=None, seed=1):
    """Evaluate on 8 deterministic YCB objects (4 seen + 4 unseen).

    Uses native 7-dim state/action with per-dim normalization:
    Z-score for position/rotation, symmetric min-max for gripper
    (see ActionNormalizer).

    Returns dict with overall/seen/unseen rates and per-object breakdown.
    """
    import gymnasium as gym
    _ensure_maniskill()
    from mani_skill.envs.tasks.digital_twins.grasp.grasp_ycb_random import (
        YCB_EVAL_OBJECTS, YCB_EVAL_SEEN_IDX, YCB_EVAL_UNSEEN_IDX,
    )
    import mani_skill.envs.tasks.digital_twins.grasp.grasp_ycb_random as _ycb_mod
    from coordinate_transforms import tcp_in_cam_frame, build_ee_state
    from ik_controller import IKController

    eval_cfg = configs.get("eval", {})
    data_cfg = configs.get("data", {})
    robot_name = eval_cfg.get("robot", "aloha_mini")
    rcfg = ROBOT_CONFIGS[robot_name]

    num_episodes = num_episodes or eval_cfg.get("num_episodes", 25)
    max_steps = eval_cfg.get("max_steps", 200)
    execute_steps = eval_cfg.get("execute_steps", 4)
    gripper_scale = eval_cfg.get("gripper_scale", 1.0)
    rotation_weight = eval_cfg.get("rotation_weight", 0.0)
    ik_damping = eval_cfg.get("ik_damping", 0.05)
    chunk_size = configs.get("fwd_pred_next_n", 16)
    fov_rad = data_cfg.get("camera_fov_rad", 1.6)
    ddim_steps = eval_cfg.get("ddim_steps", 10)
    cfg_scale = eval_cfg.get("cfg_scale", 5.0)

    frozen = rcfg["frozen_joint_indices"]
    cam_head_key = rcfg["cam_head"]
    cam_wrist_key = rcfg["cam_wrist"]
    tcp_pose_attr = rcfg["tcp_pose_attr"]
    exclude_cams = list(rcfg["cam_exclude"])

    fov = torch.tensor([[fov_rad, fov_rad]], dtype=torch.float32)
    device = next(model.parameters()).device

    # ── Per-dim normalizers ────────────────────────────────────────────────
    # Z-score for position/rotation (dims 0–5).
    # Symmetric min-max for gripper (dim 6): raw 0 → norm 0 (diffusion
    # prior centre), so the model naturally predicts "no movement" when
    # uncertain.  Works for any task (open + close).
    action_norm = ActionNormalizer(
        stats["action_mean"], stats["action_std"],
        vmin=stats.get("action_min"), vmax=stats.get("action_max"),
    )
    state_norm = ActionNormalizer(
        stats["state_mean"], stats["state_std"],
        vmin=stats.get("state_min"), vmax=stats.get("state_max"),
    )
    use_wrist_rgbd = model.use_wrist_cam_train

    from mani_skill.envs.tasks.digital_twins.grasp.grasp_base_env import NUM_GRID_POSITIONS
    N_GRID = NUM_GRID_POSITIONS  # 9 (3×3 grid)

    n_objects = len(YCB_EVAL_OBJECTS)
    per_obj_success = [0] * n_objects
    save_train = list(_ycb_mod._YCB_TRAIN)

    for obj_i, obj_id in enumerate(YCB_EVAL_OBJECTS):
        tag = "seen  " if obj_i in YCB_EVAL_SEEN_IDX else "unseen"
        instruction = make_instruction(obj_id)
        _ycb_mod._YCB_TRAIN = [obj_id]
        torch.cuda.empty_cache()

        env = gym.make(
            rcfg["env_ids"]["ycb_random"], num_envs=1,
            obs_mode="rgbd", render_mode="rgb_array", sim_backend="auto",
            robot_uids=rcfg["robot_uid"], control_mode=rcfg["control_mode"],
            frozen_joint_indices=rcfg["frozen_joint_indices"],
            exclude_cameras=exclude_cams,
        )
        env.reset(seed=seed)

        ik = IKController(env, rcfg, frozen,
                          damping=ik_damping, gripper_scale=gripper_scale,
                          rotation_weight=rotation_weight)

        if record_dir:
            from mani_skill.utils.wrappers.record import RecordEpisode
            obj_dir = os.path.join(record_dir, obj_id)
            os.makedirs(obj_dir, exist_ok=True)
            env = RecordEpisode(env, output_dir=obj_dir, save_trajectory=False,
                                max_steps_per_video=max_steps,
                                video_fps=env.unwrapped.control_freq)

        obj_successes = 0
        for ep in range(num_episodes):
            grid_idx = ep % N_GRID
            reset_opts = {"grid_positions_idx": [grid_idx]}
            raw_obs, _ = env.reset(options=reset_opts)
            buf, buf_i = None, 0
            success = False

            for step in range(max_steps):
                if buf is not None and buf_i < execute_steps:
                    ee_action = buf[buf_i]; buf_i += 1
                else:
                    # ── Head camera RGB ──
                    head = raw_obs["sensor_data"][cam_head_key]["rgb"][0].cpu().numpy()

                    # ── Wrist camera RGBD ──
                    wrist_rgbd_t = None
                    if use_wrist_rgbd:
                        wrist_rgb = raw_obs["sensor_data"][cam_wrist_key]["rgb"][0].cpu().numpy()
                        depth_raw = raw_obs["sensor_data"][cam_wrist_key]["depth"][0, :, :, 0].cpu()
                        depth_m = (depth_raw.float() / 1000.0).clamp(0.0, 10.0)
                        rgb_t = torch.from_numpy(wrist_rgb).float().permute(2, 0, 1) / 255.0
                        depth_t = depth_m.unsqueeze(0)
                        wrist_rgbd_t = torch.cat([rgb_t, depth_t], dim=0).unsqueeze(0).to(device)

                    # ── State: 7-dim [pos, euler_xyz, gripper] ──
                    tcp_pos_cam, tcp_euler_cam = tcp_in_cam_frame(
                        env, cam_head_key, tcp_pose_attr, env_idx=0,
                    )
                    qpos = env.unwrapped.agent.robot.get_qpos().cpu().numpy()
                    gripper_pos = qpos[0, ik.gripper_qidx].astype(np.float32)
                    state_7 = build_ee_state(tcp_pos_cam, tcp_euler_cam, gripper_pos)

                    # Normalise state with ActionNormalizer
                    # (symmetric_minmax keeps gripper in [-1,1] even
                    #  if the raw value drifts slightly beyond the
                    #  training range — no separate OOD clamp needed)
                    s_norm = state_norm.normalize(state_7)
                    s_t = torch.tensor(s_norm, dtype=torch.float32).unsqueeze(0).to(device)
                    s_mask = torch.ones(1, STATE_DIM, dtype=torch.float32, device=device)
                    a_mask = torch.ones(1, chunk_size, ACTION_DIM, dtype=torch.float32, device=device)

                    # Predict
                    pred = model.predict_action(
                        image=head, instruction=instruction,
                        current_state=s_t,
                        current_state_mask=s_mask,
                        action_mask_torch=a_mask,
                        num_ddim_steps=ddim_steps, cfg_scale=cfg_scale,
                        fov=fov, sample_times=1,
                        wrist_rgbd=wrist_rgbd_t,
                    )

                    # Denormalise with ActionNormalizer
                    # No task-specific gripper clamping — the model
                    # should learn when to open/close from data.
                    # symmetric_minmax places "no movement" at norm 0.0
                    # (the diffusion prior centre), so the model
                    # naturally defaults to "stay still" when uncertain.
                    chunk = action_norm.denormalize(pred[0])  # [T, 7]
                    buf, buf_i = chunk, 1
                    ee_action = chunk[0]

                # Apply EE delta via IK controller (NR IK + PD gripper)
                act_t = ik.step(ee_action, env, env_idx=0)
                raw_obs, reward, terminated, truncated, info = env.step(act_t)
                if info.get("success", torch.tensor([False]))[0].item():
                    success = True
                if terminated.any() or truncated.any():
                    break

            obj_successes += int(success)

        per_obj_success[obj_i] = obj_successes
        env.close()
        r = 100 * obj_successes / max(num_episodes, 1)
        print(f"  [{tag}] {obj_id:25s}: {obj_successes:2d}/{num_episodes} ({r:.0f}%)")

    _ycb_mod._YCB_TRAIN = save_train

    seen_total = num_episodes * len(YCB_EVAL_SEEN_IDX)
    unseen_total = num_episodes * len(YCB_EVAL_UNSEEN_IDX)
    overall = 100 * sum(per_obj_success) / max(num_episodes * n_objects, 1)
    seen_rate = 100 * sum(per_obj_success[i] for i in YCB_EVAL_SEEN_IDX) / max(seen_total, 1)
    unseen_rate = 100 * sum(per_obj_success[i] for i in YCB_EVAL_UNSEEN_IDX) / max(unseen_total, 1)

    results = {
        "robot": robot_name,
        "overall_pct": round(overall, 1),
        "seen_pct": round(seen_rate, 1),
        "unseen_pct": round(unseen_rate, 1),
        "num_episodes_per_object": num_episodes,
        "grid_positions": N_GRID,
        "per_object": {
            obj_id: {
                "split": "seen" if i in YCB_EVAL_SEEN_IDX else "unseen",
                "successes": per_obj_success[i],
                "success_rate_pct": round(100 * per_obj_success[i] / max(num_episodes, 1), 1),
            }
            for i, obj_id in enumerate(YCB_EVAL_OBJECTS)
        },
    }
    print(f"\n  Overall: {overall:.1f}%  Seen: {seen_rate:.1f}%  Unseen: {unseen_rate:.1f}%")
    if record_dir:
        os.makedirs(record_dir, exist_ok=True)
        json_path = os.path.join(record_dir, "eval_results.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Results → {json_path}")
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  Training dataset wrapper
# ═══════════════════════════════════════════════════════════════════════════

class GraspTrainDataset(Dataset):
    """Thin PyTorch wrapper around GraspDatasetCore.

    Handles:
      - Normalisation of state/action (Gaussian, 7-dim)
      - Head camera RGB → PaliGemma processor input
      - Wrist camera RGBD → 4-channel [B, 4, H, W] tensor
    """

    def __init__(self, core: GraspDatasetCore, processor, camera_fov_rad=1.6):
        self.core = core
        self.processor = processor
        self.camera_fov_rad = camera_fov_rad

    def __len__(self):
        return len(self.core)

    def __getitem__(self, idx):
        sample = self.core[idx]
        sample = self.core.normalize(sample)
        return self._to_collator_format(sample)

    def _to_collator_format(self, data):
        # ── Head camera → VLM input ──────────────────────────────────────
        head_img = Image.fromarray(data["head_rgb"])
        text = "<image>" + data["instruction"]
        inputs = self.processor(text=text, images=[head_img], return_tensors="pt").to(torch.float32)

        out = dict(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"].squeeze(0),
            labels=None,
            dataset_name="grasp",
            actions=data["action_list"],               # [T, 7]
            action_masks=data["action_mask"],           # [T, 7]
            current_state_mask=data["current_state_mask"],  # [7]
            current_state=data["current_state"],        # [7]
            fov=torch.tensor([self.camera_fov_rad, self.camera_fov_rad], dtype=torch.float32),
        )

        # ── Wrist camera RGBD → 4-channel tensor ────────────────────────
        wrist_rgb = data["wrist_rgb"]      # [H, W, 3] uint8
        wrist_depth = data["wrist_depth"]  # [H, W] float32 metres

        # Convert to float [0, 1] and stack
        rgb_t = torch.from_numpy(wrist_rgb).float().permute(2, 0, 1) / 255.0  # [3, H, W]
        depth_t = torch.from_numpy(wrist_depth).float().unsqueeze(0)           # [1, H, W]
        out["wrist_rgbd"] = torch.cat([rgb_t, depth_t], dim=0)                # [4, H, W]

        return out


# ═══════════════════════════════════════════════════════════════════════════
#  Optimizer / scheduler helpers
# ═══════════════════════════════════════════════════════════════════════════

def make_optimizer(model, t_cfg):
    """Create AdamW with 3 parameter groups for fine-grained LR control.

    Group 1 – Pretrained DiT blocks (self-attn, FFN, adaLN, cross-attn):
              Slower LR to preserve pretrained features.
    Group 2 – New I/O layers (ActionEmbedder, StateEmbedder, FinalLayer,
              wrist_projector, missing_wrist_tokens, cognition_token, fov_encoder,
              TimestepEmbedder, LabelEmbedder, positional_embedding):
              Higher LR since they're randomly initialised.
    Group 3 – LoRA adapters + multi_modal_projector (P_g):
              Moderate LR for adapting the VLM to the robot domain.
    """
    from vitra.utils.lora import lora_params as _lora_params

    # Collect param IDs per group
    # --- Group 2: new I/O layers in the action model ---
    new_io_names = {
        "x_embedder", "state_embedder", "final_layer",
        "t_embedder", "z_embedder", "positional_embedding",
    }
    new_io_ids = set()
    dit = model.act_model.net  # the DiT module
    for name, param in dit.named_parameters():
        top = name.split(".")[0]
        if top in new_io_names or name == "positional_embedding":
            new_io_ids.add(id(param))

    # Wrist modules + cognition token + fov_encoder → also new I/O
    for attr in ("wrist_projector", "missing_wrist_tokens", "cognition_token", "fov_encoder"):
        obj = getattr(model, attr, None)
        if obj is None:
            continue
        if isinstance(obj, nn.Parameter):
            new_io_ids.add(id(obj))
        elif isinstance(obj, nn.Module):
            for p in obj.parameters():
                new_io_ids.add(id(p))

    # Wrist encoder input projection (trainable part of frozen backbone)
    if hasattr(model, "wrist_rgbd_encoder"):
        if getattr(model, "wrist_encoder_type", None) == "resnet34":
            input_proj = model.wrist_rgbd_encoder.conv1
        else:
            input_proj = model.wrist_rgbd_encoder.patch_embed
        for p in input_proj.parameters():
            new_io_ids.add(id(p))

    # --- Group 3: LoRA + P_g ---
    lora_pg_ids = set()
    for p in _lora_params(model):
        lora_pg_ids.add(id(p))
    if hasattr(model.model, "multi_modal_projector"):
        for p in model.model.multi_modal_projector.parameters():
            lora_pg_ids.add(id(p))

    # --- Group 1: pretrained DiT blocks (everything in act_model not in new_io) ---
    dit_block_ids = set()
    for p in model.act_model.parameters():
        pid = id(p)
        if pid not in new_io_ids:
            dit_block_ids.add(pid)

    # Build groups from trainable params only
    g_dit, g_io, g_lora, g_other = [], [], [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        pid = id(p)
        if pid in new_io_ids:
            g_io.append(p)
        elif pid in lora_pg_ids:
            g_lora.append(p)
        elif pid in dit_block_ids:
            g_dit.append(p)
        else:
            g_other.append(p)

    lr_dit  = t_cfg.get("lr_dit_blocks", 5e-5)
    lr_io   = t_cfg.get("lr_new_io", 2e-4)
    lr_lora = t_cfg.get("lr_lora_pg", 1e-4)
    lr_base = t_cfg.get("learning_rate", 1e-4)

    groups = []
    if g_dit:
        groups.append({"params": g_dit,   "lr": lr_dit,  "name": "dit_blocks"})
    if g_io:
        groups.append({"params": g_io,    "lr": lr_io,   "name": "new_io"})
    if g_lora:
        groups.append({"params": g_lora,  "lr": lr_lora, "name": "lora_pg"})
    if g_other:
        groups.append({"params": g_other, "lr": lr_base, "name": "other"})

    for g in groups:
        n = sum(p.numel() for p in g["params"])
        print(f"  optim group '{g['name']}': {n/1e6:.2f}M params, lr={g['lr']:.1e}")

    return torch.optim.AdamW(groups, weight_decay=t_cfg.get("weight_decay", 0.01))


def make_scheduler(optimizer, warmup_steps, total_phase_steps):
    """Warmup + cosine decay (min 10% of peak LR)."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_phase_steps - warmup_steps, 1)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ═══════════════════════════════════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════════════════════════════════

def train(args):
    configs = load_config(args.config)
    data_cfg = configs["data"]
    t_cfg = configs["trainer"]
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda")
    writer = SummaryWriter(log_dir=os.path.join(output_dir, "tb_logs"))

    robot_name = configs.get("eval", {}).get("robot", "aloha_mini")
    rcfg = ROBOT_CONFIGS[robot_name]

    # ── Statistics (simple 7-dim Gaussian) ────────────────────────────────
    print("Computing statistics …")
    stats_dict = compute_statistics_from_lerobot(data_cfg["data_dir"])
    stats_path = os.path.join(output_dir, "statistics.json")
    with open(stats_path, "w") as f:
        json.dump(stats_dict, f, indent=2)

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(configs, f, indent=2)

    # ── Model ─────────────────────────────────────────────────────────────
    print("Building model …")
    model = load_model(configs).to(device).train()

    # ── Training phases ───────────────────────────────────────────────────
    phases_cfg = configs.get("training_phases", {})
    if phases_cfg:
        p1 = phases_cfg.get("phase1_steps", 2000)
        p2 = p1 + phases_cfg.get("phase2_steps", 4000)
        phase_boundaries = [("phase1", 0), ("phase2", p1)]
        max_steps = p2
        print(f"Two-phase training: P1={p1}, P2={p2-p1}  (total={max_steps})")
    else:
        max_steps = t_cfg.get("max_steps", 6000)
        phase_boundaries = [("legacy", 0)]

    phase_name = phase_boundaries[0][0]
    if phase_name.startswith("phase"):
        model.set_training_phase(phase_name)
    else:
        model.trainable_params_setup()
    model.use_bf16 = configs.get("use_bf16", False)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {trainable/1e6:.1f}M trainable / {total/1e6:.1f}M total")

    # ── Dataset (native 7-dim, RGBD wrist) ────────────────────────────────
    cam_wrist_rgb_col = data_cfg.get("cam_wrist_rgb_col",
                                     f"observation.images.{rcfg['cam_wrist']}")
    cam_wrist_depth_col = data_cfg.get("cam_wrist_depth_col",
                                       f"observation.depth.{rcfg['cam_wrist']}")
    cam_head_col = data_cfg.get("cam_head_col",
                                f"observation.images.{rcfg['cam_head']}")

    core = GraspDatasetCore(
        data_dir=data_cfg["data_dir"],
        chunk_size=configs.get("fwd_pred_next_n", 16),
        cam_head_col=cam_head_col,
        cam_wrist_rgb_col=cam_wrist_rgb_col,
        cam_wrist_depth_col=cam_wrist_depth_col,
        stats_path=stats_path,
    )

    dataset = GraspTrainDataset(
        core, model.processor,
        camera_fov_rad=data_cfg.get("camera_fov_rad", 1.6),
    )
    collator = PaddedCollatorForHandPrediction(
        model.processor.tokenizer.model_max_length,
        model.processor.tokenizer.pad_token_id,
        padding_side="right",
    )
    loader = DataLoader(dataset, batch_size=configs["batch_size"], shuffle=True,
                        num_workers=4, collate_fn=collator, drop_last=True, pin_memory=True)
    print(f"Dataset: {len(dataset)} samples, batch={configs['batch_size']}")

    # ── Optimizer ─────────────────────────────────────────────────────────
    optimizer = make_optimizer(model, t_cfg)
    warmup = t_cfg.get("warmup_steps", 100)
    phase1_steps = phase_boundaries[1][1] if len(phase_boundaries) > 1 else max_steps
    scheduler = make_scheduler(optimizer, warmup, phase1_steps)
    grad_accum = t_cfg.get("grad_accum", 4)
    scaler = torch.amp.GradScaler("cuda", enabled=configs.get("use_bf16", False))

    save_every = t_cfg.get("save_every", 1000)
    log_every = t_cfg.get("log_every", 50)
    eval_every = t_cfg.get("eval_every", 0)
    eval_episodes = t_cfg.get("eval_episodes", 8)

    # ── Loop ──────────────────────────────────────────────────────────────
    step, micro, loss_sum = 0, 0, 0.0
    t0 = time.time()
    print(f"Training {max_steps} steps (grad_accum={grad_accum}) …")

    while step < max_steps:
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                     enabled=configs.get("use_bf16", False)):
                out = model.forward(
                    pixel_values=batch["pixel_values"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    action_labels=batch["actions"],
                    action_masks=batch["action_masks"],
                    current_state_mask=batch["current_state_mask"],
                    current_state=batch["current_state"],
                    fov=batch["fov"],
                    wrist_rgbd=batch.get("wrist_rgbd"),
                )
            loss = out["loss"] / grad_accum
            scaler.scale(loss).backward()
            loss_sum += out["loss"].item()
            micro += 1

            if micro % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), t_cfg.get("gradient_clip_val", 1.0))
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(); scheduler.step()
                step += 1

                # Phase transition
                for pname, pstart in phase_boundaries:
                    if pstart == step and pname != phase_name:
                        phase_name = pname
                        print(f"\n{'='*60}\n  Switching to {phase_name} at step {step}\n{'='*60}")
                        model.set_training_phase(phase_name)
                        optimizer = make_optimizer(model, t_cfg)
                        scheduler = make_scheduler(optimizer, warmup, max_steps - step)
                        scaler = torch.amp.GradScaler("cuda", enabled=configs.get("use_bf16", False))
                        break

                if step % log_every == 0:
                    avg = loss_sum / (log_every * grad_accum)
                    lr_strs = []
                    for gi, pg in enumerate(optimizer.param_groups):
                        gn = pg.get("name", f"g{gi}")
                        writer.add_scalar(f"lr/{gn}", pg["lr"], step)
                        lr_strs.append(f"{gn}={pg['lr']:.2e}")
                    writer.add_scalar("train/loss", avg, step)
                    print(f"[{phase_name}] step {step}/{max_steps}  loss={avg:.4f}  "
                          f"lr=[{', '.join(lr_strs)}]  t={time.time()-t0:.0f}s")
                    loss_sum = 0.0

                if step % save_every == 0:
                    p = os.path.join(output_dir, f"step_{step}.pt")
                    torch.save(model.state_dict(), p)
                    print(f"  ✓ Saved → {p}")

                if eval_every > 0 and step % eval_every == 0:
                    print(f"\n── Eval @ step {step} ──")
                    model.eval()
                    edir = os.path.join(output_dir, "eval", f"step_{step}")
                    res = run_eval(model, stats_dict, configs,
                                   record_dir=edir, num_episodes=eval_episodes, seed=step)
                    writer.add_scalar("eval/overall", res["overall_pct"], step)
                    writer.add_scalar("eval/seen", res["seen_pct"], step)
                    writer.add_scalar("eval/unseen", res["unseen_pct"], step)
                    model.train()

                if step >= max_steps:
                    break

    final = os.path.join(output_dir, "final.pt")
    torch.save(model.state_dict(), final)
    writer.close()
    print(f"Done → {final}  (TensorBoard: {os.path.join(output_dir, 'tb_logs')})")


# ═══════════════════════════════════════════════════════════════════════════
#  Standalone evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate(args):
    ckpt_dir = os.path.dirname(args.checkpoint)
    config_path = args.config or os.path.join(ckpt_dir, "config.json")
    stats_path = os.path.join(ckpt_dir, "statistics.json")
    record_dir = args.record_dir or os.path.join(ckpt_dir, "eval_results")

    configs = load_config(config_path)
    configs["model_load_path"] = args.checkpoint

    print(f"Loading model from {args.checkpoint} …")
    model = load_model(configs).to("cuda").eval()
    model.use_bf16 = configs.get("use_bf16", True)

    with open(stats_path) as f:
        stats = json.load(f)

    num_ep = args.num_episodes or configs.get("eval", {}).get("num_episodes", 25)

    print(f"{'='*60}")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Episodes   : {num_ep} per object")
    print(f"  Record     : {record_dir}")
    print(f"{'='*60}")

    run_eval(model, stats, configs, record_dir=record_dir,
             num_episodes=num_ep, seed=args.seed)


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="VITRA grasp training / evaluation")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--config", required=True)
    t.add_argument("--output_dir", default="./checkpoints/aloha_mini")

    e = sub.add_parser("eval")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--config", default=None)
    e.add_argument("--record_dir", default=None)
    e.add_argument("--num_episodes", type=int, default=None)
    e.add_argument("--seed", type=int, default=1)

    s = sub.add_parser("stats")
    s.add_argument("--data_dir", required=True)

    args = p.parse_args()
    if args.cmd == "train":
        train(args)
    elif args.cmd == "eval":
        evaluate(args)
    elif args.cmd == "stats":
        print(json.dumps(compute_statistics_from_lerobot(args.data_dir), indent=2))


if __name__ == "__main__":
    main()
