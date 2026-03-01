"""Quick debug: see what actions VITRA predicts during eval."""
import sys, os
_dc = os.path.join(os.path.dirname(__file__), "../../data_collection_maniskill")
if _dc not in sys.path:
    sys.path.insert(0, _dc)

import numpy as np
import torch
import json
import gymnasium as gym

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from vitra.models.vla_builder import load_model
from vitra.utils.config_utils import load_config
from vitra.datasets.grasp_dataset import ACTION_DIM, STATE_DIM, ActionNormalizer

import mani_skill.envs
import mani_skill.envs.tasks.digital_twins.grasp.grasp_ycb_random as _ycb_mod
from robot_registry import ROBOT_CONFIGS
from coordinate_transforms import tcp_in_cam_frame, build_ee_state
from ik_controller import IKController

def main():
    ckpt_dir = "./checkpoints/aloha_mini"
    configs = load_config(os.path.join(ckpt_dir, "config.json"))
    configs["model_load_path"] = os.path.join(ckpt_dir, "step_2000.pt")
    
    model = load_model(configs).to("cuda").eval()
    model.use_bf16 = configs.get("use_bf16", True)
    
    with open(os.path.join(ckpt_dir, "statistics.json")) as f:
        stats = json.load(f)
    
    a_mean = np.array(stats["action_mean"], dtype=np.float32)
    a_std = np.array(stats["action_std"], dtype=np.float32)
    s_mean = np.array(stats["state_mean"], dtype=np.float32)
    s_std = np.array(stats["state_std"], dtype=np.float32)

    action_norm = ActionNormalizer(
        stats["action_mean"], stats["action_std"],
        vmin=stats.get("action_min"), vmax=stats.get("action_max"),
    )
    state_norm = ActionNormalizer(
        stats["state_mean"], stats["state_std"],
        vmin=stats.get("state_min"), vmax=stats.get("state_max"),
    )
    
    eval_cfg = configs.get("eval", {})
    rcfg = ROBOT_CONFIGS["aloha_mini"]
    frozen = rcfg["frozen_joint_indices"]
    cam_head = rcfg["cam_head"]
    cam_wrist = rcfg["cam_wrist"]
    tcp_attr = rcfg["tcp_pose_attr"]
    exclude_cams = list(rcfg["cam_exclude"])
    
    chunk_size = configs.get("fwd_pred_next_n", 16)
    execute_steps = eval_cfg.get("execute_steps", 4)
    fov = torch.tensor([[1.6, 1.6]], dtype=torch.float32)
    instruction = "Left hand: None. Right hand: Grasp the object and lift it."
    
    _ycb_mod._YCB_TRAIN = ["013_apple"]
    
    env = gym.make(
        rcfg["env_ids"]["ycb_random"], num_envs=1,
        obs_mode="rgbd", render_mode="rgb_array", sim_backend="auto",
        robot_uids=rcfg["robot_uid"], control_mode=rcfg["control_mode"],
        frozen_joint_indices=frozen, exclude_cameras=exclude_cams,
    )
    env.reset(seed=42)
    
    ik = IKController(env, rcfg, frozen,
                      damping=eval_cfg.get("ik_damping", 0.05),
                      gripper_scale=eval_cfg.get("gripper_scale", 1.0),
                      rotation_weight=eval_cfg.get("rotation_weight", 0.0))
    
    raw_obs, _ = env.reset(options={"grid_positions_idx": [4]})
    
    print("\n=== VITRA Eval Debug ===")
    print(f"Action stats: mean={a_mean}, std={a_std}")
    print(f"State stats: mean={s_mean}, std={s_std}")
    labels = ["Δtx", "Δty", "Δtz", "Δrx", "Δry", "Δrz", "Δgrip"]
    
    buf, buf_i = None, 0
    
    for step in range(50):
        if buf is not None and buf_i < execute_steps:
            ee_action = buf[buf_i]; buf_i += 1
        else:
            head = raw_obs["sensor_data"][cam_head]["rgb"][0].cpu().numpy()
            
            wrist_rgb = raw_obs["sensor_data"][cam_wrist]["rgb"][0].cpu().numpy()
            depth_raw = raw_obs["sensor_data"][cam_wrist]["depth"][0, :, :, 0].cpu()
            depth_m = (depth_raw.float() / 1000.0).clamp(0.0, 10.0)
            rgb_t = torch.from_numpy(wrist_rgb).float().permute(2, 0, 1) / 255.0
            depth_t = depth_m.unsqueeze(0)
            wrist_rgbd_t = torch.cat([rgb_t, depth_t], dim=0).unsqueeze(0).to("cuda")
            
            tcp_pos_cam, tcp_euler_cam = tcp_in_cam_frame(env, cam_head, tcp_attr, env_idx=0)
            qpos = env.unwrapped.agent.robot.get_qpos().cpu().numpy()
            gripper_pos = qpos[0, ik.gripper_qidx].astype(np.float32)
            state_7 = build_ee_state(tcp_pos_cam, tcp_euler_cam, gripper_pos)
            
            s_norm = state_norm.normalize(state_7)
            s_t = torch.tensor(s_norm, dtype=torch.float32).unsqueeze(0).to("cuda")
            s_mask = torch.ones(1, STATE_DIM, dtype=torch.float32, device="cuda")
            a_mask = torch.ones(1, chunk_size, ACTION_DIM, dtype=torch.float32, device="cuda")
            
            with torch.no_grad():
                pred = model.predict_action(
                    image=head, instruction=instruction,
                    current_state=s_t, current_state_mask=s_mask,
                    action_mask_torch=a_mask,
                    num_ddim_steps=10, cfg_scale=5.0,
                    fov=fov, sample_times=1,
                    wrist_rgbd=wrist_rgbd_t,
                )
            
            pred_7 = pred[0]  # [T, 7]
            chunk = action_norm.denormalize(pred_7)
            
            if step == 0:
                print(f"\nRaw state_7: {state_7}")
                print(f"Normalized state: {s_norm}")
                print(f"\nPred raw (first step of chunk): {pred_7[0]}")
                print(f"Denorm action (first step): {chunk[0]}")
                print(f"\nFull chunk denorm:")
                for t in range(min(4, len(chunk))):
                    print(f"  t={t}: {chunk[t]}")
            
            buf, buf_i = chunk, 1
            ee_action = chunk[0]
        
        # Print action being applied
        qpos_before = env.unwrapped.agent.robot.get_qpos().cpu().numpy()[0].copy()
        ik.step(ee_action, env, env_idx=0)
        qpos_after = env.unwrapped.agent.robot.get_qpos().cpu().numpy()[0]
        dq = qpos_after - qpos_before
        
        active_idx = [i for i in range(len(qpos_before)) if i not in set(frozen)]
        dq_active = dq[active_idx]
        
        act_t = torch.zeros(1, 6, dtype=torch.float32, device="cuda")
        raw_obs, reward, terminated, truncated, info = env.step(act_t)
        
        tcp_pos_cam_new, _ = tcp_in_cam_frame(env, cam_head, tcp_attr, env_idx=0)
        grip_new = env.unwrapped.agent.robot.get_qpos().cpu().numpy()[0, ik.gripper_qidx]
        
        if step < 20 or step % 10 == 0:
            print(f"\nStep {step}: ee_action={ee_action}")
            print(f"  IK dq (active joints): {dq_active}")
            print(f"  TCP cam pos: {tcp_pos_cam_new}")
            print(f"  Gripper qpos: {grip_new:.4f}")
    
    env.close()

if __name__ == "__main__":
    main()
