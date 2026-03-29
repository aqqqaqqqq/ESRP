from omnigibson.reward_functions.reward_function_base import BaseRewardFunction


class RearrangeReachingReward(BaseRewardFunction):
    """
    Dense reaching reward for rearrangement.

    When the robot is not grasping an object, reward is computed from the decrease
    in distance between the robot and the nearest candidate object to rearrange.
    Moving closer gives positive reward, moving farther gives negative reward.
    """

    def __init__(self, r_reaching=0.1):
        self._r_reaching = r_reaching
        self._prev_distance = None
        self._prev_name = None

        super().__init__()

    def reset(self, task, env):
        self._prev_distance, self._prev_name = task.get_reaching_target_info(env)
        super().reset(task, env)

    def _step(self, task, env, action):
        if task._fetch_name is not None:
            self._prev_distance = None
            self._prev_name = None
            return 0.0, {"target_object": None, "distance": None}

        current_distance, current_name = task.get_reaching_target_info(env)
        if current_distance is None:
            self._prev_distance = None
            self._prev_name = None
            return 0.0, {"target_object": None, "distance": None}

        reward = 0.0
        if self._prev_distance is not None and self._prev_name == current_name:
            reward = (self._prev_distance - current_distance) * self._r_reaching

        self._prev_distance = current_distance
        self._prev_name = current_name
        return reward, {"target_object_name": current_name, "distance": current_distance}
