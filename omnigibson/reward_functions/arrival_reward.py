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
        self._flag = True

        # Run super
        super().__init__()

    def reset(self, task, env):
        # Call super first
        self._arrival_num = 0
        self._flag = True
        super().reset(task, env)

    def calculate_arrival_object(self, task):
        arrival_num = 0
        for i in task.objects_to_rearrange:
            arrival_num += task.check_target(i)
        return arrival_num

    def _step(self, task, env, action):
        # Calculate each object one by one
        reward = 0
        if self._flag:
            self._arrival_num = self.calculate_arrival_object(task)
            self._flag = False
        
        if task._fetch_if == 0 :
            if task.check_target(task._last_name):
                reward = self._farrival
                self._arrival_num += 1
        elif task._fetch_if == 1:
            if task.check_target(task._fetch_name):
                reward = -self._farrival
                self._arrival_num += -1
        return reward, {self._arrival_num} 
