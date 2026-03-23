import omnigibson.utils.transform_utils as T
from omnigibson.termination_conditions.termination_condition_base import SuccessCondition
from omnigibson.object_states import Pose
import torch as th


class RearrangeGoal(SuccessCondition):
    """
    PointGoal (success condition) used for PointNavFixed/RandomTask
    Episode terminates if point goal is reached within @distance_tol by the @robot_idn robot's base

    Args:
        robot_idn (int): robot identifier to evaluate point goal with. Default is 0, corresponding to the first
            robot added to the scene
        distance_tol (float): Distance (m) tolerance between goal position and @robot_idn's robot base position
            that is accepted as a success
        distance_axes (str): Which axes to calculate distances when calculating the goal. Any combination of "x",
            "y", and "z" is valid (e.g.: "xy" or "xyz" or "y")
    """

    def __init__(self, obj_num):

        # Run super init
        super().__init__()

    def _step(self, task, env, action):
        result = 0
        # Terminate if point goal is reached (distance below threshold)
        for i in task.objects_to_rearrange:
            result += task.check_target(i)
        return (result == task._obj_num) and (task._fetch_if == 0)
