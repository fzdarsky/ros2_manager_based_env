from isaaclab.utils import configclass


@configclass
class CommandsCfg:
    """Command specifications for the MDP.

    Intentionally empty: actions are driven externally by a VLA policy over
    ROS2 (see ros2_bridge.py), not by an internally-generated goal the env
    resamples on its own. There is no "target pose" for the env to track --
    the policy decides where to move based on vision + language, so a
    UniformPoseCommand here would just be an unused, confusing artifact
    (and, worse, it was previously being fed into the policy's own
    observations as if it were real proprioception -- see
    custom_observations_cfg.py).
    """
