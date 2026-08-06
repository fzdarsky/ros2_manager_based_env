from isaaclab.envs.mdp.observations import image
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .observations import eef_pose_axis_angle, gripper_pos


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        table_cam = ObsTerm(
            func=image, params={"sensor_cfg": SceneEntityCfg("table_cam"), "data_type": "rgb", "normalize": False}
        )
        wrist_cam = ObsTerm(
            func=image, params={"sensor_cfg": SceneEntityCfg("wrist_cam"), "data_type": "rgb", "normalize": False}
        )
        # Real current end-effector pose (NOT a commanded/goal pose) --
        # matches lerobot/libero's observation.state convention. Order
        # matters for concatenate_terms=True below: eef_pose (6) then
        # gripper_pos (2) = 8 dims total, same order as the dataset.
        eef_pose = ObsTerm(func=eef_pose_axis_angle)
        gripper_pos = ObsTerm(func=gripper_pos)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
