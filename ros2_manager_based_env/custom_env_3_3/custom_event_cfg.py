from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events


@configclass
class EventCfg:
    """Configuration for events.

    reset_base (random root pose/velocity teleport) and push_robot
    (periodic random velocity kicks) were removed: both are
    locomotion-specific domain randomization for a mobile/legged robot's
    floating base. Franka is fixed to the table (fixed_base=True) --
    randomly teleporting or "kicking" its root doesn't correspond to
    anything physical and would just fight the ROS2-driven OSC controller.
    """

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    

    # reset -- randomizes starting joint configuration within a scale of
    # the default pose; kept, since a varied start state is still useful
    # for evaluation/rollout (does not move the fixed robot base).
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )


#    # Reset cube 1 to initial position (inside the bin)
#    reset_cube_1_pose = EventTerm(
#        func=franka_stack_events.randomize_object_pose,
#        mode="reset",
#        params={
#            "pose_range": {"x": (0.4, 0.4), "y": (0.0, 0.0), "z": (0.0203, 0.0203), "yaw": (0.0, 0.0)},
#            "min_separation": 0.0,
#            "asset_cfgs": [SceneEntityCfg("cube_1")],
#        },
#    )

    # Reset cube 1, 2 and 3 to initial position (outside the bin, to the left and right)
    reset_cube_pose = EventTerm(
        func=franka_stack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {"x": (0.4, 0.70), "y": (-0.18, 0.18), "z": (0.0203, 0.0203), "yaw": (-1.0, 1.0, 0)},
            "min_separation": 0.1,
            "asset_cfgs": [SceneEntityCfg("cube_1"), SceneEntityCfg("cube_2"), SceneEntityCfg("cube_3")],
        },
    )
    
