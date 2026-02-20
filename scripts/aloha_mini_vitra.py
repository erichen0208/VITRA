"""
aloha_mini_vitra.py — Train / Evaluate VITRA on AlohaMini (config-driven).

All parameters live in the config JSON. CLI is minimal.

Supports two action head modes:
  - "simple":    MLP action head, raw 6-dim actions, 2-phase training
  - "diffusion": DiT diffusion head, 192-dim padded actions (original)

Usage:
    # Train (with mid-training eval + video)
    python scripts/aloha_mini_vitra.py train --config vitra/configs/aloha_mini_finetune.json

    # Standalone eval (auto-discovers config/stats from checkpoint dir)
    python scripts/aloha_mini_vitra.py eval --checkpoint ./checkpoints/aloha_mini/final.pt

    # Print dataset statistics
    python scripts/aloha_mini_vitra.py stats --data_dir <lerobot_data_dir>
"""

import argparse, json, os, sys, time
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from vitra.datasets.aloha_mini_dataset import (
    ACTION_DIM, AlohaMiniDatasetCore, pad_state, pad_action, extract_action,
)
from vitra.models.vla_builder import load_model
from vitra.utils.config_utils import load_config
from vitra.utils.data_utils import PaddedCollatorForHandPrediction, read_dataset_statistics

# ─── Constants ────────────────────────────────────────────────────────────
ROBOT_CONFIGS = {
    "aloha_mini": {
        "robot_uid": "aloha_mini_so100_v2",
        "control_mode": "pd_joint_delta_pos_right_arm_only",
        "frozen_joint_indices": [0, 1, 2, 3, 4, 6, 8, 10, 12, 14],
        "env_ids": {
            "cube": "AlohaMiniGraspCube-v1",
            "bottle": "AlohaMiniGraspBottle-v1",
            "ycb_random": "AlohaMiniGraspYCBRandom-v1",
        },
    },
}
TASK_DESCRIPTIONS = {
    "cube": "Grasp the cube and lift it",
    "bottle": "Grasp the bottle and lift it",
    "ycb_random": "Grasp the object and lift it",
}


# ═══════════════════════════════════════════════════════════════════════════
#  Statistics (computed inline from LeRobot data)
# ═══════════════════════════════════════════════════════════════════════════

def compute_statistics_from_lerobot(data_dir: str) -> dict:
    """Read LeRobot meta/stats.json → VITRA statistics dict."""
    stats_path = os.path.join(data_dir, "meta", "stats.json")
    assert os.path.exists(stats_path), f"Not found: {stats_path}"
    with open(stats_path) as f:
        lr = json.load(f)
    dim = len(lr["action"]["mean"])
    d = 1e-4
    return {
        "dataset_name": os.path.basename(data_dir),
        "action_dim": dim, "state_dim": dim,
        "state_right":  {"mean": lr["observation.state"]["mean"],
                         "std":  lr["observation.state"]["std"]},
        "action_right": {"mean": lr["action"]["mean"],
                         "std":  lr["action"]["std"]},
        "state_left":   {"mean": [d]*dim, "std": [d]*dim},
        "action_left":  {"mean": [d]*dim, "std": [d]*dim},
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Shared eval utilities
# ═══════════════════════════════════════════════════════════════════════════

def _ensure_maniskill():
    """Lazy-import ManiSkill envs (registers gym envs)."""
    dc_dir = os.path.join(os.path.dirname(__file__), "../../data_collection_maniskill")
    if dc_dir not in sys.path:
        sys.path.insert(0, dc_dir)
    import mani_skill.envs  # noqa: F401


def get_obs_images(raw_obs, use_wrist_cam):
    """Extract camera images from ManiSkill obs → np array."""
    head = raw_obs["sensor_data"]["cam_head"]["rgb"].cpu().numpy()
    if head.ndim == 4: head = head[0]
    if not use_wrist_cam:
        return head  # (H, W, 3)
    wrist = raw_obs["sensor_data"]["cam_right_wrist"]["rgb"].cpu().numpy()
    if wrist.ndim == 4: wrist = wrist[0]
    return np.stack([head, wrist], axis=0)  # (2, H, W, 3)


def get_active_qpos(env, active_indices):
    return env.unwrapped.agent.robot.get_qpos().cpu().numpy().flatten()[active_indices].astype(np.float32)


@torch.no_grad()
def run_eval(model, stats, configs, record_dir=None, num_episodes=None, seed=1):
    """Run VITRA in ManiSkill. Returns (success_rate, avg_reward)."""
    import gymnasium as gym
    _ensure_maniskill()

    eval_cfg = configs.get("eval", {})
    data_cfg = configs.get("data", {})
    use_wrist_cam = data_cfg.get("use_wrist_cam", False)
    is_simple = configs.get("action_head_type", "diffusion") == "simple"

    rcfg = ROBOT_CONFIGS[eval_cfg.get("robot", "aloha_mini")]
    task = eval_cfg.get("task", "ycb_random")
    env_id = rcfg["env_ids"][task]
    task_desc = TASK_DESCRIPTIONS.get(task, "Grasp the object and lift it")

    num_episodes = num_episodes or eval_cfg.get("num_episodes", 20)
    max_steps = eval_cfg.get("max_steps", 200)
    execute_steps = eval_cfg.get("execute_steps", 4)
    gripper_scale = eval_cfg.get("gripper_scale", 1.0)
    chunk_size = configs.get("fwd_pred_next_n", 16)
    fov_rad = data_cfg.get("camera_fov_rad", 1.6)

    # Diffusion-only params
    ddim_steps = eval_cfg.get("ddim_steps", 10)
    cfg_scale = eval_cfg.get("cfg_scale", 5.0)

    frozen = set(rcfg["frozen_joint_indices"])
    exclude_cams = ["cam_left_wrist"]
    if not use_wrist_cam:
        exclude_cams.append("cam_right_wrist")

    # ── Create env ────────────────────────────────────────────────────────
    env = gym.make(
        env_id, num_envs=1, obs_mode="rgb", render_mode="sensors",
        sim_backend="auto", robot_uids=rcfg["robot_uid"],
        control_mode=rcfg["control_mode"],
        frozen_joint_indices=rcfg["frozen_joint_indices"],
        exclude_cameras=exclude_cams,
    )
    env.reset(seed=seed)
    active_indices = [i for i in range(env.unwrapped.agent.robot.get_qpos().shape[-1])
                      if i not in frozen]

    if record_dir:
        from mani_skill.utils.wrappers.record import RecordEpisode
        os.makedirs(record_dir, exist_ok=True)
        env = RecordEpisode(env, output_dir=record_dir, save_trajectory=False,
                            max_steps_per_video=max_steps,
                            video_fps=env.unwrapped.control_freq)

    # ── Inference setup ───────────────────────────────────────────────────
    fov = torch.tensor([[fov_rad, fov_rad]], dtype=torch.float32)
    instruction = f"Left hand: None. Right hand: {task_desc}."
    device = next(model.parameters()).device

    successes, total_reward = 0, 0.0
    for ep in range(num_episodes):
        raw_obs, _ = env.reset()
        buf, buf_i = None, 0
        ep_reward, success = 0.0, False

        for step in range(max_steps):
            images = get_obs_images(raw_obs, use_wrist_cam)
            state_6 = get_active_qpos(env, active_indices)

            # action chunking
            if buf is not None and buf_i < execute_steps:
                action = buf[buf_i]; buf_i += 1
            else:
                s_norm = (state_6 - stats["state_right_mean"]) / (stats["state_right_std"] + 1e-7)

                if is_simple:
                    # Simple head: raw 6-dim state, direct 6-dim output
                    pred = model.predict_action(
                        image=images, instruction=instruction,
                        current_state=torch.tensor(s_norm, dtype=torch.float32).unsqueeze(0),
                        fov=fov, sample_times=1,
                    )
                    chunk = pred[0] * (stats["action_right_std"] + 1e-7) + stats["action_right_mean"]
                else:
                    # Diffusion head: pad to unified space, extract from 192-dim
                    us, usm = pad_state(s_norm)
                    _, uam = pad_action(np.zeros((chunk_size, ACTION_DIM)))
                    pred = model.predict_action(
                        image=images, instruction=instruction,
                        current_state=us.unsqueeze(0),
                        current_state_mask=usm.unsqueeze(0),
                        action_mask_torch=uam.unsqueeze(0),
                        num_ddim_steps=ddim_steps, cfg_scale=cfg_scale,
                        fov=fov, sample_times=1,
                    )
                    p6 = extract_action(pred[0])
                    chunk = p6 * (stats["action_right_std"] + 1e-7) + stats["action_right_mean"]

                chunk[:, 5] *= gripper_scale
                buf, buf_i = chunk, 1
                action = chunk[0]

            act_t = torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
            raw_obs, reward, terminated, truncated, info = env.step(act_t)
            ep_reward += reward.item() if hasattr(reward, "item") else float(reward)
            if info.get("success", torch.tensor([False]))[0].item():
                success = True
            if terminated.any() or truncated.any():
                break

        successes += int(success)
        total_reward += ep_reward
        rate = 100 * successes / (ep + 1)
        print(f"  ep {ep+1:3d}/{num_episodes}: {'✓' if success else '✗'}  "
              f"rew={ep_reward:.2f}  steps={step+1}  rate={rate:.1f}%")

    env.close()
    return 100 * successes / num_episodes, total_reward / num_episodes


# ═══════════════════════════════════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════════════════════════════════

class AlohaMiniTrainDataset(Dataset):
    """Thin PyTorch wrapper around AlohaMiniDatasetCore."""

    def __init__(self, core: AlohaMiniDatasetCore, processor, normalization=True,
                 raw_output=False):
        self.core = core
        self.processor = processor
        self.normalization = normalization
        self.raw_output = raw_output

    def __len__(self):
        return len(self.core)

    def __getitem__(self, idx):
        sample = self.core[idx]
        if self.raw_output:
            sample = self.core.transform_trajectory_raw(sample, self.normalization)
        else:
            sample = self.core.transform_trajectory(sample, self.normalization)
        return self._to_collator_format(sample)

    def _to_collator_format(self, data):
        imgs = [Image.fromarray(img) for img in data["image_list"]]
        text = "<image>" * len(imgs) + data["instruction"]
        inputs = self.processor(text=text, images=imgs, return_tensors="pt").to(torch.float32)
        return dict(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"].squeeze(0),
            labels=None,
            dataset_name="aloha_mini",
            actions=data["action_list"],
            action_masks=data["action_mask"],
            current_state_mask=data["current_state_mask"],
            current_state=data["current_state"],
            fov=torch.tensor(data["fov"], dtype=torch.float32),
            intrinsics=torch.tensor(data["intrinsics"], dtype=torch.float32),
        )


# ═══════════════════════════════════════════════════════════════════════════
#  Optimizer helper — recreated at each phase transition
# ═══════════════════════════════════════════════════════════════════════════

def make_optimizer(model, t_cfg):
    """Create AdamW over currently-trainable parameters only."""
    act_ids = {id(p) for p in model.act_model.parameters() if p.requires_grad}
    backbone = [p for p in model.parameters() if p.requires_grad and id(p) not in act_ids]
    act_params = [p for p in model.act_model.parameters() if p.requires_grad]

    groups = []
    if backbone:
        groups.append({"params": backbone, "lr": t_cfg["learning_rate"]})
    if act_params:
        groups.append({"params": act_params,
                       "lr": t_cfg.get("action_model_learning_rate", t_cfg["learning_rate"])})
    return torch.optim.AdamW(groups, weight_decay=t_cfg.get("weight_decay", 0.01))


def train(args):
    configs = load_config(args.config)
    data_cfg = configs["data"]
    t_cfg = configs["trainer"]
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda")

    is_simple = configs.get("action_head_type", "diffusion") == "simple"

    # ── Inline statistics ─────────────────────────────────────────────────
    print("Computing statistics …")
    stats_dict = compute_statistics_from_lerobot(data_cfg["data_dir"])
    stats_path = os.path.join(output_dir, "statistics.json")
    with open(stats_path, "w") as f:
        json.dump(stats_dict, f, indent=2)
    data_cfg["statistics_path"] = stats_path

    # Save config for eval
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(configs, f, indent=2)

    # ── Model ─────────────────────────────────────────────────────────────
    print("Building model …")
    model = load_model(configs).to(device).train()

    # ── Training phases ───────────────────────────────────────────────────
    phases_cfg = configs.get("training_phases", {})
    if is_simple and phases_cfg:
        p1 = phases_cfg.get("phase1_steps", 2000)
        p2 = p1 + phases_cfg.get("phase2_steps", 4000)
        phase_boundaries = [("phase1", 0), ("phase2", p1)]
        max_steps = p2
        print(f"Two-phase training: P1={p1}, P2={p2-p1}  (total={max_steps})")
    else:
        max_steps = t_cfg.get("max_steps", 6000)
        phase_boundaries = [("legacy", 0)]

    # Start first phase
    phase_name = phase_boundaries[0][0]
    if phase_name.startswith("phase"):
        model.set_training_phase(phase_name)
    else:
        model.trainable_params_setup()
    model.use_bf16 = configs.get("use_bf16", False)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {trainable/1e6:.1f}M trainable / {total/1e6:.1f}M total")

    # ── Dataset ───────────────────────────────────────────────────────────
    core = AlohaMiniDatasetCore(
        data_dir=data_cfg["data_dir"],
        statistics_path=stats_path,
        chunk_size=configs["fwd_pred_next_n"],
        camera_fov_rad=data_cfg.get("camera_fov_rad", 1.6),
        use_wrist_cam=data_cfg.get("use_wrist_cam", False),
    )
    core.set_global_data_statistics(core.data_statistics)
    stats_data = core.data_statistics  # for mid-training eval

    dataset = AlohaMiniTrainDataset(
        core, model.processor, normalization=True, raw_output=is_simple,
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
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(s / max(warmup, 1), 1.0))
    grad_accum = t_cfg.get("grad_accum", 4)
    scaler = torch.amp.GradScaler("cuda", enabled=configs["use_bf16"])

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
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=configs["use_bf16"]):
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

            if micro % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), t_cfg.get("gradient_clip_val", 1.0))
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(); scheduler.step()
                step += 1

                # ── Phase transition ──────────────────────────────────────
                for pname, pstart in phase_boundaries:
                    if pstart == step and pname != phase_name:
                        phase_name = pname
                        print(f"\n{'='*60}")
                        print(f"  Switching to {phase_name} at step {step}")
                        print(f"{'='*60}")
                        model.set_training_phase(phase_name)
                        optimizer = make_optimizer(model, t_cfg)
                        scheduler = torch.optim.lr_scheduler.LambdaLR(
                            optimizer, lambda s: min(s / max(warmup, 1), 1.0))
                        scaler = torch.amp.GradScaler("cuda", enabled=configs["use_bf16"])
                        break

                if step % log_every == 0:
                    avg = loss_sum / (log_every * grad_accum)
                    print(f"[{phase_name}] step {step}/{max_steps}  loss={avg:.4f}  "
                          f"lr={optimizer.param_groups[0]['lr']:.2e}  t={time.time()-t0:.0f}s")
                    loss_sum = 0.0

                if step % save_every == 0:
                    p = os.path.join(output_dir, f"step_{step}.pt")
                    torch.save(model.state_dict(), p)
                    print(f"  ✓ Saved → {p}")

                # ── Mid-training eval ─────────────────────────────────────
                if eval_every > 0 and step % eval_every == 0:
                    print(f"\n── Eval @ step {step} ──")
                    model.eval()
                    edir = os.path.join(output_dir, "videos")
                    rate, avg_r = run_eval(model, stats_data, configs,
                                           record_dir=edir, num_episodes=eval_episodes,
                                           seed=step)
                    print(f"  → {rate:.1f}% success, avg_rew={avg_r:.2f}\n")
                    model.train()

                if step >= max_steps:
                    break

    final = os.path.join(output_dir, "final.pt")
    torch.save(model.state_dict(), final)
    print(f"Done → {final}")


# ═══════════════════════════════════════════════════════════════════════════
#  Standalone evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate(args):
    ckpt_dir = os.path.dirname(args.checkpoint)
    config_path = args.config or os.path.join(ckpt_dir, "config.json")
    stats_path = os.path.join(ckpt_dir, "statistics.json")
    record_dir = args.record_dir or os.path.join(ckpt_dir, "eval_videos")

    configs = load_config(config_path)
    configs["model_load_path"] = args.checkpoint

    print(f"Loading model from {args.checkpoint} …")
    model = load_model(configs).to("cuda").eval()
    model.use_bf16 = configs.get("use_bf16", True)
    stats = read_dataset_statistics(stats_path)

    num_ep = args.num_episodes or configs.get("eval", {}).get("num_episodes", 20)

    print(f"{'='*60}")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Episodes   : {num_ep}")
    print(f"  Record     : {record_dir}")
    print(f"{'='*60}")

    rate, avg_r = run_eval(model, stats, configs, record_dir=record_dir,
                           num_episodes=num_ep, seed=args.seed)
    print(f"\n{'='*60}")
    print(f"  Success: {rate:.1f}%  Avg reward: {avg_r:.2f}")
    print(f"{'='*60}")


# ═══════════════════════════════════════════════════════════════════════════
#  CLI (minimal — all params in config JSON)
# ═══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="VITRA × AlohaMini")
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
        st = compute_statistics_from_lerobot(args.data_dir)
        print(json.dumps(st, indent=2))


if __name__ == "__main__":
    main()
