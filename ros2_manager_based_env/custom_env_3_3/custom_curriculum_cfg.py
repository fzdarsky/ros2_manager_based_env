from isaaclab.utils import configclass


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP.

    Intentionally empty: no RL training curriculum needed for ROS2-driven
    VLA policy rollout/inference.
    """
