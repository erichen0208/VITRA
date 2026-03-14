"""
gripper_retarget.py — Retarget single-jaw gripper to MANO dexterous hand format.

Maps a 7-dim gripper state/action [tx, ty, tz, rx, ry, rz, gripper] into the
original VITRA 192-dim (action) / 212-dim (state) format to leverage pretrained
dexterous-hand weights.

Retargeting analogy:
  - Fixed jaw  → Thumb (CMC/MCP/IP all straight = zeros)
  - Moving jaw → Four fingers (Index, Middle, Pinky, Ring)
    Only the MCP joint (closest to wrist) moves; PIP/DIP stay at zero.
    The single gripper scalar drives all four MCP joints equally.

MANO 45-dim joint layout per hand (Euler xyz per joint):
  Index  MCP [0:3],   PIP [3:6],   DIP [6:9]
  Middle MCP [9:12],  PIP [12:15], DIP [15:18]
  Pinky  MCP [18:21], PIP [21:24], DIP [24:27]
  Ring   MCP [27:30], PIP [30:33], DIP [33:36]
  Thumb  CMC [36:39], MCP [39:42], IP  [42:45]

Per-hand 51-dim: [trans(3), rot(3), joints(45)]
Full 192-dim:  [left_hand(51), right_hand(51), padding(90)]
Full 212-dim:  [192-dim action, left_beta(10), right_beta(10)]

Hardware / display constants (Aloha Mini + MANO visual config):
  GRIPPER_ACTION_SCALE   — rad/unit for pd_joint_delta_pos_right_arm_only
  GRIPPER_INIT_Q6        — neutral arm configuration (6-dim right arm)
  GRIPPER_AXIS_DISPLAY_R — 3×3 rotation to apply before drawing TCP axes
  MANO_REST_R            — MANO global-orient at rest (fingers forward, palm left)
  MANO_REST_HAND_POSE    — 45-dim thumb rest shape (jaw-like spread)

Gripper ↔ MANO curl normalization:
  GRIPPER_QPOS_OPEN   = 0.0  — normalised open  (symmetric_minmax preserves 0)
  GRIPPER_QPOS_CLOSE  = -1.0 — normalised closed (symmetric_minmax maps min → -1)
  MANO_MCP_FULL_CURL  = 2.0  rad — MCP-z that makes fingers touch the thumb

  Linear mapping (training retarget + display helpers):
    norm grip =  0.0  → MCP-z = 0.0                (fingers straight)
    norm grip = -1.0  → MCP-z = MANO_MCP_FULL_CURL (fingers touch thumb)

High-level display helpers:
  gripper_euler_to_mano_euler(wrist_euler) → strips yaw
  mano_wrist_rotation(wrist_euler)         → R_wrist (3×3 float32)
  gripper_to_mano_display(qpos, euler)     → (hp45, eu_display)
  mano_to_gripper_display(mcp_val, euler)  → (qpos, eu_display, hp45)
"""

import numpy as np
import torch
from scipy.spatial.transform import Rotation as _SciRot


# ── MANO joint indices within the 45-dim hand_pose ──────────────────────
# MCP joints (closest to wrist) for the four "moving" fingers:
# Each MCP has 3 Euler components [x, y, z] = [abduction, twist, flexion].
# Flexion/extension (curl) is the Z-component (offset +2 within each 3-dim slice).
INDEX_MCP  = slice(0, 3)     # dims 0-2
MIDDLE_MCP = slice(9, 12)    # dims 9-11
PINKY_MCP  = slice(18, 21)   # dims 18-20
RING_MCP   = slice(27, 30)   # dims 27-30
MCP_FLEX_OFFSET = 2          # z-component = flexion axis within each MCP's 3-dim Euler

# Thumb joints (all stay at zero = fixed jaw):
THUMB_CMC = slice(36, 39)
THUMB_MCP = slice(39, 42)
THUMB_IP  = slice(42, 45)

# Indices of MCP joints in 45-dim space
FOUR_FINGER_MCP_SLICES = [INDEX_MCP, MIDDLE_MCP, PINKY_MCP, RING_MCP]

# Per-hand offsets in the 192-dim action vector (right hand only for single-arm)
RIGHT_HAND_OFFSET = 51
RIGHT_TRANS = slice(51, 54)
RIGHT_ROT   = slice(54, 57)
RIGHT_JOINTS = slice(57, 102)  # 45 dims

LEFT_HAND_OFFSET = 0
LEFT_TRANS = slice(0, 3)
LEFT_ROT   = slice(3, 6)
LEFT_JOINTS = slice(6, 51)


def gripper_to_mano_action(gripper_action, use_left=False):
    """Convert 7-dim gripper action to 192-dim MANO action.

    Args:
        gripper_action: [..., 7] tensor or ndarray.
            [Δtx, Δty, Δtz, Δrx, Δry, Δrz, Δgripper]
        use_left: if True, map to left hand instead of right.

    Returns:
        mano_action: [..., 192] tensor or ndarray (same type as input).
    """
    is_tensor = isinstance(gripper_action, torch.Tensor)
    if is_tensor:
        device = gripper_action.device
        dtype = gripper_action.dtype
        shape = gripper_action.shape[:-1]
        out = torch.zeros(*shape, 192, device=device, dtype=dtype)
    else:
        shape = gripper_action.shape[:-1]
        out = np.zeros((*shape, 192), dtype=np.float32)

    if use_left:
        trans_slice = LEFT_TRANS
        rot_slice = LEFT_ROT
        joints_slice = LEFT_JOINTS
    else:
        trans_slice = RIGHT_TRANS
        rot_slice = RIGHT_ROT
        joints_slice = RIGHT_JOINTS

    # Translation and rotation map directly
    out[..., trans_slice] = gripper_action[..., :3]
    out[..., rot_slice] = gripper_action[..., 3:6]

    # Gripper → four-finger MCP joints
    # Scale normalised gripper [OPEN=0, CLOSE=-1] → MCP flex [0, FULL_CURL=+2].
    # Linear map:  t = (grip - OPEN) / (CLOSE - OPEN)   ∈ [0, 1]
    #              flex = t * MANO_MCP_FULL_CURL          ∈ [0, 2]
    # Clamp t to [0, 1] so flex stays in the valid range even if the input
    # is slightly out-of-bounds (numerical noise or OOD samples).
    grip_val = gripper_action[..., 6:7]  # [..., 1]
    t = (grip_val - GRIPPER_QPOS_OPEN) / (GRIPPER_QPOS_CLOSE - GRIPPER_QPOS_OPEN)
    if is_tensor:
        t = t.clamp(0.0, 1.0)
    else:
        t = np.clip(t, 0.0, 1.0)
    flex = t * MANO_MCP_FULL_CURL  # [..., 1]

    joints_start = joints_slice.start
    for mcp in FOUR_FINGER_MCP_SLICES:
        # Set the z-component (flexion) of each MCP
        out[..., joints_start + mcp.start + MCP_FLEX_OFFSET] = flex[..., 0]

    return out


def gripper_to_mano_state(gripper_state, use_left=False):
    """Convert 7-dim gripper state to 212-dim MANO state.

    Args:
        gripper_state: [..., 7] tensor or ndarray.
            [tx, ty, tz, rx, ry, rz, gripper_pos]
        use_left: if True, map to left hand.

    Returns:
        mano_state: [..., 212] tensor or ndarray.
    """
    is_tensor = isinstance(gripper_state, torch.Tensor)
    if is_tensor:
        device = gripper_state.device
        dtype = gripper_state.dtype
        shape = gripper_state.shape[:-1]
        out = torch.zeros(*shape, 212, device=device, dtype=dtype)
    else:
        shape = gripper_state.shape[:-1]
        out = np.zeros((*shape, 212), dtype=np.float32)

    # Reuse action mapping for the first 192 dims
    out[..., :192] = gripper_to_mano_action(gripper_state, use_left=use_left)
    # Beta params (shape) stay at zero — we don't have hand shape info
    return out


def mano_action_to_gripper(mano_action, use_left=False):
    """Convert 192-dim MANO action back to 7-dim gripper action.

    Extracts translation/rotation from the active hand and averages the
    four MCP flexion values to reconstruct the gripper scalar.

    Args:
        mano_action: [..., 192] tensor or ndarray.
        use_left: if True, extract from left hand.

    Returns:
        gripper_action: [..., 7] tensor or ndarray.
    """
    is_tensor = isinstance(mano_action, torch.Tensor)
    if is_tensor:
        device = mano_action.device
        dtype = mano_action.dtype
        shape = mano_action.shape[:-1]
        out = torch.zeros(*shape, 7, device=device, dtype=dtype)
    else:
        shape = mano_action.shape[:-1]
        out = np.zeros((*shape, 7), dtype=np.float32)

    if use_left:
        trans_slice = LEFT_TRANS
        rot_slice = LEFT_ROT
        joints_slice = LEFT_JOINTS
    else:
        trans_slice = RIGHT_TRANS
        rot_slice = RIGHT_ROT
        joints_slice = RIGHT_JOINTS

    out[..., :3] = mano_action[..., trans_slice]
    out[..., 3:6] = mano_action[..., rot_slice]

    # Average four MCP flexion values (z-component), then scale back
    # to normalised gripper space:
    #   t = clamp(flex / MANO_MCP_FULL_CURL, 0, 1)
    #   grip = OPEN + t * (CLOSE - OPEN)   ∈ [-1, 0]
    joints_start = joints_slice.start
    if is_tensor:
        mcp_vals = torch.stack([
            mano_action[..., joints_start + mcp.start + MCP_FLEX_OFFSET]
            for mcp in FOUR_FINGER_MCP_SLICES
        ], dim=-1)
        avg_flex = mcp_vals.mean(dim=-1)
        t = (avg_flex / MANO_MCP_FULL_CURL).clamp(0.0, 1.0)
    else:
        mcp_vals = np.stack([
            mano_action[..., joints_start + mcp.start + MCP_FLEX_OFFSET]
            for mcp in FOUR_FINGER_MCP_SLICES
        ], axis=-1)
        avg_flex = mcp_vals.mean(axis=-1)
        t = np.clip(avg_flex / MANO_MCP_FULL_CURL, 0.0, 1.0)
    out[..., 6] = GRIPPER_QPOS_OPEN + t * (GRIPPER_QPOS_CLOSE - GRIPPER_QPOS_OPEN)

    return out


def gripper_to_mano_mask(gripper_mask, use_left=False):
    """Convert 7-dim mask to 192-dim MANO mask.

    Active dims: translation (3) + rotation (3) + four MCP flexion (4) = 10.
    All other dims are masked out (zero).

    Args:
        gripper_mask: [..., 7] tensor or ndarray (0/1 values).
        use_left: if True, map to left hand.

    Returns:
        mano_mask: [..., 192] tensor or ndarray.
    """
    is_tensor = isinstance(gripper_mask, torch.Tensor)
    if is_tensor:
        device = gripper_mask.device
        dtype = gripper_mask.dtype
        shape = gripper_mask.shape[:-1]
        out = torch.zeros(*shape, 192, device=device, dtype=dtype)
    else:
        shape = gripper_mask.shape[:-1]
        out = np.zeros((*shape, 192), dtype=np.float32)

    if use_left:
        trans_slice = LEFT_TRANS
        rot_slice = LEFT_ROT
        joints_slice = LEFT_JOINTS
    else:
        trans_slice = RIGHT_TRANS
        rot_slice = RIGHT_ROT
        joints_slice = RIGHT_JOINTS

    # Translation and rotation masks
    out[..., trans_slice] = gripper_mask[..., :3]
    out[..., rot_slice] = gripper_mask[..., 3:6]

    # Gripper mask → four MCP joints (z-component only)
    joints_start = joints_slice.start
    for mcp in FOUR_FINGER_MCP_SLICES:
        out[..., joints_start + mcp.start + MCP_FLEX_OFFSET] = gripper_mask[..., 6]

    return out


def gripper_to_mano_state_mask(gripper_mask, use_left=False):
    """Convert 7-dim state mask to 212-dim MANO state mask."""
    is_tensor = isinstance(gripper_mask, torch.Tensor)
    if is_tensor:
        device = gripper_mask.device
        dtype = gripper_mask.dtype
        shape = gripper_mask.shape[:-1]
        out = torch.zeros(*shape, 212, device=device, dtype=dtype)
    else:
        shape = gripper_mask.shape[:-1]
        out = np.zeros((*shape, 212), dtype=np.float32)

    out[..., :192] = gripper_to_mano_mask(gripper_mask, use_left=use_left)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  Hardware & display configuration (Aloha Mini + MANO visualisation)
# ═══════════════════════════════════════════════════════════════════════════════

# ── Aloha Mini right-arm joint layout (6-dim) ─────────────────────────────────
#   [0] shoulder_pan  [1] shoulder_lift  [2] elbow_flex
#   [3] wrist_flex    [4] wrist_roll     [5] gripper_open

# Radians-per-unit scale for control mode pd_joint_delta_pos_right_arm_only.
# action = clip(delta / SCALE, -1, 1).
GRIPPER_ACTION_SCALE = np.array(
    [0.03924, 0.03917, 0.00140, 0.03924, 0.03925, 0.18290], dtype=np.float64)

# Neutral arm pose: elbow up, wrist straight, gripper open.
GRIPPER_INIT_Q6 = np.array(
    [0.0, -np.pi / 2, np.pi / 2, 0.0, -np.pi / 2, 0.0], dtype=np.float64)

# ── Axis display convention ────────────────────────────────────────────────────
# VITRA readme: +x backward (opposite finger direction), +y right, +z up.
# Apply this rotation to the TCP frame before projecting axes into the image.
GRIPPER_AXIS_DISPLAY_R = np.array(
    [[0, -1, 0],
     [1,  0, 0],
     [0,  0, 1]], dtype=np.float64)

# ── MANO visual configuration ──────────────────────────────────────────────────
# REST_R: Rx(150°) · Rz(-90°) — orients the MANO canonical frame so that
#   fingers point into the screen and palm faces left, matching the gripper view.
# Axis semantics at rest (columns of REST_R):
#   col 0 (+X) = backward (opposite finger direction)
#   col 1 (+Y) = right    (opposite palm facing)
#   col 2 (+Z) = upward
MANO_REST_R = (
    _SciRot.from_euler('x', 90, degrees=True) *
    _SciRot.from_euler('y', 0, degrees=True) *
    _SciRot.from_euler('z', -120, degrees=True)
).as_matrix().astype(np.float32)

# Thumb rest shape: opens a jaw-like gap mirroring the gripper geometry.
# 45-dim hand_pose, Euler xyz per joint:
#   [36] Thumb CMC x = abduction  (+= wider jaw gap)
#   [37] Thumb CMC y = CMC flex
#   [38] Thumb CMC z = axial rotation
MANO_REST_HAND_POSE = np.zeros(45, np.float32)
MANO_REST_HAND_POSE[36] = 1.5 
MANO_REST_HAND_POSE[37] = 0.0
MANO_REST_HAND_POSE[38] = 0.0   

for i in [2, 11, 20, 29]:  
    MANO_REST_HAND_POSE[i] = -0.5

# Four-finger MCP-z stays at 0 so that qpos=0 (open) renders as straight fingers.
# Dynamic curl is added by the display helpers via MANO_MCP_FULL_CURL normalization.


# ── Gripper ↔ MANO curl normalization ────────────────────────────────────────
# Maps the gripper range linearly to the MANO MCP flexion range.
# Used by BOTH training-level retarget and display helpers.
#
# After ActionNormalizer symmetric_minmax, the normalized gripper lives in
# [-1, 0] where 0 = open, -1 = fully closed.  The retarget functions scale
# this to MANO MCP flex [0, MANO_MCP_FULL_CURL] (0 = straight, positive =
# curled) so the action head sees the same sign convention as the pretrained
# human-hand data.
GRIPPER_QPOS_OPEN  = 0.0    # normalised open  (unchanged by symmetric_minmax)
GRIPPER_QPOS_CLOSE = -1.0   # normalised closed (symmetric_minmax maps min → -1)
MANO_MCP_FULL_CURL = 2.0    # rad — MCP-z that makes finger tips touch the thumb
                             #        (tune this constant if the closure looks off)

# ── World → MANO camera frame rotation ────────────────────────────────────────
# Maps a vector in robot world frame (+X fwd, +Y left, +Z up) to the MANO
# rendering camera frame (OpenCV convention: +X right, +Y down, +Z depth).
#   world +X (fwd)  → MANO +Z:   col-Z of M = world +X
#   world +Y (left) → MANO -X:   col-X of M = -world +Y  → M[0,:] = [0,-1,0]
#   world +Z (up)   → MANO -Y:   col-Y of M = -world +Z  → M[1,:] = [0,0,-1]
# M is a proper rotation (det=1, verified).
R_WORLD_TO_MANO_CAM = np.array(
    [[ 0., -1.,  0.],   # MANO x = -world y
     [ 0.,  0., -1.],   # MANO y = -world z
     [ 1.,  0.,  0.]],  # MANO z =  world x
    dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
#  Wrist euler ↔ MANO conversion helpers
# ═══════════════════════════════════════════════════════════════════════════════

def gripper_euler_to_mano_euler(wrist_euler):
    """Strip the yaw component from gripper wrist euler [roll, flex, yaw].

    The Aloha Mini has no wrist-yaw DOF, so euler[2] is always zero in practice.

    Args:
        wrist_euler: array-like [roll, flex, yaw]

    Returns:
        ndarray (3,) float32: [roll, flex, 0.0]
    """
    eu = np.asarray(wrist_euler, dtype=np.float32)
    return np.array([eu[0], eu[1], 0.0], dtype=np.float32)


def mano_wrist_rotation(wrist_euler):
    """Build the MANO global_orient rotation matrix from gripper wrist euler.

    Sign & axis mapping (determined empirically to match gripper motion):
        euler[0] roll → Ry           (arm-axis spin)
        euler[1] flex → Rx, negated  (wrist flex, direction-corrected)
        euler[2] yaw  → ignored

    Euler order 'yxz': Ry(roll) * Rx(-flex).

    Args:
        wrist_euler: array-like [roll, flex, _]

    Returns:
        ndarray (3, 3) float32 rotation matrix.
    """
    eu = np.asarray(wrist_euler, dtype=np.float32)
    eu_mano = np.array([-eu[0], -eu[1], 0.0], dtype=np.float32)
    return _SciRot.from_euler('zxy', eu_mano).as_matrix().astype(np.float32)


def tcp_rot_to_mano_orient(R_tcp_world, R_tcp_rest_world):
    """Convert actual TCP world-frame rotation to MANO global_orient matrix.

    Uses the full 6-DOF TCP rotation read back from the simulator (not just the
    commanded Euler angles), giving an exact retarget for every FORWARD / LATERAL
    / ROLL / FLEX motion where IK produces a non-trivial wrist rotation.

    Derivation:
        1. Compute the delta rotation from rest in world frame:
               R_delta = R_tcp_world @ R_tcp_rest_world.T
        2. Express that delta in MANO camera frame:
               R_go = M @ R_delta @ M.T @ MANO_REST_R
           where M = R_WORLD_TO_MANO_CAM  maps world axes → MANO camera axes.
        3. MANO_REST_R is pre-multiplied so that R_delta = I gives the
           resting hand appearance.

    Args:
        R_tcp_world:      (3,3) float — actual TCP rotation in world frame.
        R_tcp_rest_world: (3,3) float — TCP rotation at the rest/neutral pose.

    Returns:
        ndarray (3,3) float32 — MANO global_orient rotation matrix.
    """
    M = R_WORLD_TO_MANO_CAM
    R_delta = np.asarray(R_tcp_world,      dtype=np.float32) @ \
              np.asarray(R_tcp_rest_world, dtype=np.float32).T
    return (M @ R_delta @ M.T @ MANO_REST_R).astype(np.float32)


def gripper_to_mano_display(gripper_qpos, wrist_euler):
    """Convert a gripper command to MANO display representation.

    Takes RAW gripper qpos (0 = open, -1.1 = closed) and maps to MANO.
    Internally normalises to [-1, 0] before using the training-level retarget.

    Args:
        gripper_qpos: float — Aloha convention: 0 = open, negative = closed.
        wrist_euler:  array-like [roll, flex, yaw].

    Returns:
        hp45:       ndarray (45,) float32 — raw MANO joint angles
                    (add MANO_REST_HAND_POSE in the renderer before use).
        eu_display: ndarray (3,) float32 — [roll, flex, 0], yaw stripped.
    """
    eu = gripper_euler_to_mano_euler(wrist_euler)
    # Map raw qpos [0, -1.1] → normalised [0, -1] for the training retarget
    RAW_CLOSE = -1.1  # physical limit, not the normalised constant
    norm_grip = np.clip(float(gripper_qpos) / abs(RAW_CLOSE), -1.0, 0.0)
    g7 = np.zeros(7, np.float32)
    g7[3:6] = eu
    g7[6] = norm_grip
    m = gripper_to_mano_action(g7)
    return m[RIGHT_JOINTS].copy(), eu


def mano_to_gripper_display(mcp_val, wrist_euler):
    """Convert MANO MCP flexion value to a gripper command for display.

    Inverse of gripper_to_mano_display normalisation:
        mcp_val = 0.0                → qpos = 0.0   (open)
        mcp_val = MANO_MCP_FULL_CURL → qpos = -1.1  (closed)

    Args:
        mcp_val:     float — MANO convention: 0 = straight, positive = curl in.
        wrist_euler: array-like [roll, flex, yaw].

    Returns:
        gripper_qpos: float — Aloha raw qpos (0 = open, negative = closed).
        eu_display:   ndarray (3,) float32 — [roll, flex, 0].
        hp45:         ndarray (45,) float32 — raw MANO joint angles.
    """
    eu = gripper_euler_to_mano_euler(wrist_euler)
    # Build hp45 directly (mcp_val is already in MANO space)
    m = np.zeros(192, np.float32)
    m[RIGHT_ROT] = eu
    for mcp in FOUR_FINGER_MCP_SLICES:
        m[RIGHT_JOINTS.start + mcp.start + MCP_FLEX_OFFSET] = float(mcp_val)
    hp45 = m[RIGHT_JOINTS].copy()
    # Inverse normalise: [0, MANO_MCP_FULL_CURL] → raw qpos [0, -1.1]
    RAW_CLOSE = -1.1  # physical limit
    t = np.clip(float(mcp_val) / MANO_MCP_FULL_CURL, 0.0, 1.0)
    gripper_qpos = t * RAW_CLOSE
    return gripper_qpos, eu, hp45
