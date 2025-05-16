from omnigibson.reward_functions.reward_function_base import BaseRewardFunction


class GraspingReward(BaseRewardFunction):
    """
    Point goal reward
    Success reward for reaching the goal with the robot's base

    Args:
        pointgoal (PointGoal): Termination condition for checking whether a point goal is reached
        r_pointgoal (float): Reward for reaching the point goal
    """

    def __init__(self, r_grasping=0.01):
        # Store internal vars
        self._r_grasping = r_grasping

        # Run super
        super().__init__()

    def _step(self, task, env, action):
        # Reward received the pointgoal success condition is met
        reward = 0
        if task._fetch_name is not None:
            reward = self._r_grasping
        return reward, {}
