from omnigibson.reward_functions.reward_function_base import BaseRewardFunction
import omnigibson.utils.transform_utils as T
from copy import deepcopy

class RearrangePotentialReward(BaseRewardFunction):
    """
    Potential reward
    Assume task has get_potential implemented; Low potential is preferred
    (e.g. a common potential for goal-directed task is the distance to goal)

    """

    def __init__(self, r_potential=0.4):
        # Store internal vars
        self._potential = {}
        self._r_potential = r_potential
        self._initial_poential = {}

        # Run super
        super().__init__()

    def reset(self, task, env):
        self._initial_poential = deepcopy(task.objects_initial_potential)
        self._potential = deepcopy(task.objects_initial_potential)

    def _step(self, task, env, action):
        # Reward is proportional to the potential difference between the current and previous timestep
        reward = 0
        for i, v in enumerate(task.objects_to_rearrange):
            new_potential = task.objects_current_potential[v]['pos']
            reward += (self._potential[v]['pos'] - new_potential).numpy() * self._r_potential
            self._potential[v]['pos'] = new_potential
            
        return reward, {'ini':self._initial_poential, 'end': self._potential}

