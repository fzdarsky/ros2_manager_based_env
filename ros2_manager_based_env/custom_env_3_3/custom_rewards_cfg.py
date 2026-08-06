from isaaclab.utils import configclass


@configclass
class Rewards:
    """Reward terms for the MDP.

    Intentionally empty: this env is used for ROS2-driven VLA policy
    rollout/inference, not RL training -- there is nothing to optimize a
    reward signal for here. (Previously extended the locomotion RewardsCfg
    and nulled out most terms one by one; simpler and less error-prone to
    just not carry that baggage at all.)
    """
