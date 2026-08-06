from isaaclab.utils import configclass
from isaaclab.managers import TerminationTermCfg as DoneTerm
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp


@configclass
class TerminationsCfg:
    """Termination terms for the MDP.

    Only time_out (episode length cutoff) is kept -- locomotion-specific
    base_contact/root_height terminations don't apply to a fixed-base arm.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
