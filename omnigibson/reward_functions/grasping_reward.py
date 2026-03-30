from omnigibson.reward_functions.reward_function_base import BaseRewardFunction


class GraspingReward(BaseRewardFunction):
    """
    Point goal reward
    Success reward for reaching the goal with the robot's base

    Args:
        pointgoal (PointGoal): Termination condition for checking whether a point goal is reached
        r_pointgoal (float): Reward for reaching the point goal
    """

    def __init__(self, r_grasping=0.01, grasp_reward_decay=0.5, max_grasp_reward_count=4):
        # Store internal vars
        self._r_grasping = r_grasping
        self._grasp_reward_decay = grasp_reward_decay
        self._max_grasp_reward_count = max_grasp_reward_count

        # Run super
        super().__init__()

    def _step(self, task, env, action):
        reward = 0.0
        reward_count = 0

        if task._fetch_if == 1 and task._fetch_name is not None:
            reward_count = sum(
                event["event_type"] == "grasp"
                for event in task._object_event_history.get(task._fetch_name, [])
            )
            if reward_count < self._max_grasp_reward_count:
                reward = self._r_grasping * (self._grasp_reward_decay ** reward_count)

        return reward, {
            "grasp_object": task._fetch_name,
            "previous_grasp_count": reward_count,
            "rewarded": reward > 0,
        }
