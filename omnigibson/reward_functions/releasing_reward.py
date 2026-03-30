from omnigibson.reward_functions.reward_function_base import BaseRewardFunction


class ReleasingReward(BaseRewardFunction):
    """
    Penalty applied when the robot releases an object too far from its target.
    """

    def __init__(self, r_releasing=0.1, release_distance_thresh=0.5):
        self._r_releasing = r_releasing
        self._release_distance_thresh = release_distance_thresh

        super().__init__()

    def _step(self, task, env, action):
        reward = 0.0
        release_distance = None
        penalized = False

        if task._fetch_if == 0 and task._last_name is not None:
            release_distance = task._get_object_distance_to_target(task._last_name)
            if release_distance is not None and release_distance > self._release_distance_thresh:
                reward = -self._r_releasing
                penalized = True

        return reward, {
            "released_object": task._last_name,
            "release_distance": release_distance,
            "distance_threshold": self._release_distance_thresh,
            "penalized": penalized,
        }
