from src.dafs.panda.common import RelabelGoalBase

class RelabelGoal(RelabelGoalBase):
    """
    Base data augmentation function (DAF) class.
    """
    def __init__(self, env=None, **kwargs):
        super().__init__(env, **kwargs)

        self.desired_goal_mask[3:6] = True
        # desired_goal = object's position
        self.achieved_goal_mask[:3] = True

