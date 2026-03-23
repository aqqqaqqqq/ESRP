from omnigibson.reward_functions.reward_function_base import BaseRewardFunction
import omnigibson.utils.transform_utils as T
import torch as th
import numpy as np
import math


class ArrivalReward(BaseRewardFunction):
    """
    Arrival reward
    The reward for each object to rearrange if it reaches the target point

    Args:
        obj_num (int): The number of objects to rearrange
        first_arrival (float): Reward for the first time for reaching the point goal 
        multi_arrival (float): Reward for the multiple time for reaching the point goal 
    """

    def __init__(self, obj_num, first_arrival):
        # Store internal vars
        self._num = obj_num
        self._farrival = first_arrival
        self._arrival_num = 0

        # Run super
        super().__init__()

    def reset(self, task, env):
        # Call super first
        self._arrival_num = 0
        super().reset(task, env)

    def _step(self, task, env, action):
        # Calculate each object one by one
        reward = 0
        
        if task._fetch_if == 0 :
            if task.check_target(task._last_name):
                reward = self._farrival
                self._arrival_num += 1
        elif task._fetch_if == 1:
            if task.check_target(task._fetch_name):
                reward = -self._farrival
                self._arrival_num += -1
        return reward, {self._arrival_num} 
