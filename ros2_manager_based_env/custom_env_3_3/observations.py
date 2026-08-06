from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils


def eef_pose_axis_angle(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_name: str = "panda_hand",
) -> torch.Tensor:
    """Current end-effector pose in the robot's root frame, matching the
    lerobot/libero observation.state convention: 3D position followed by
    3D orientation in axis-angle (rotation-vector) form -- NOT a
    quaternion, NOT Euler angles, NOT a commanded/goal pose. This is the
    robot's REAL current pose at this simulation step.

    Concatenate with gripper_pos() (2 dims: mirrored finger positions) to
    get the full 8-dim observation.state used by lerobot/libero.

    Returns:
        Tensor of shape (num_envs, 6): [x, y, z, ax, ay, az].
    """
    robot: Articulation = env.scene[robot_cfg.name]
    body_idx = robot.find_bodies(body_name)[0][0]

    ee_pos_w = robot.data.body_pos_w[:, body_idx]
    ee_quat_w = robot.data.body_quat_w[:, body_idx]

    # World frame -> robot root frame (root is fixed to the table for a
    # stationary-base Franka, so this is effectively also "table frame").
    ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, ee_pos_w, ee_quat_w
    )
    ee_axis_angle_b = math_utils.axis_angle_from_quat(ee_quat_b)

    return torch.cat([ee_pos_b, ee_axis_angle_b], dim=-1)


def gripper_pos(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Parallel-gripper finger positions (2 dims, mirrored), matching the
    lerobot/libero robot0_gripper_qpos convention.

    Requires env.cfg.gripper_joint_names to be set to the two finger
    joint names, e.g. ["panda_finger_joint1", "panda_finger_joint2"].
    """
    robot: Articulation = env.scene[robot_cfg.name]

    if not hasattr(env.cfg, "gripper_joint_names"):
        raise NotImplementedError(
            "[Error] Cannot find gripper_joint_names in the environment config"
        )

    gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
    assert len(gripper_joint_ids) == 2, "Observation gripper_pos only supports a parallel gripper"
    finger_joint_1 = robot.data.joint_pos[:, gripper_joint_ids[0]].clone().unsqueeze(1)
    finger_joint_2 = -1 * robot.data.joint_pos[:, gripper_joint_ids[1]].clone().unsqueeze(1)
    return torch.cat((finger_joint_1, finger_joint_2), dim=1)
