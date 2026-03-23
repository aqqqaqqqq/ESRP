from omnigibson.object_states.contact_bodies import ContactBodies
from omnigibson.reward_functions.reward_function_base import BaseRewardFunction


class RobotCollisionReward(BaseRewardFunction):
    """
    Collision reward
    Penalize robot collision. Typically collision_reward_weight is negative. Note that we ignore collisions with any
    floor objects.

    Args:
        robot_idn (int): robot identifier to evaluate collision penalty with. Default is 0, corresponding to the first
            robot added to the scene
        ignore_self_collisions (bool): Whether to ignore robot self-collisions or not
        r_collision (float): Penalty value (>0) to penalize collisions
    """

    def __init__(self, robot_idn=0, ignore_self_collisions=True, r_collision=0.1):
        # Store internal vars
        assert r_collision > 0, f"r_collision must be positive, got: {r_collision}!"
        self._r_collision = r_collision

        # Run super
        super().__init__()

    def _step(self, task, env, action):
        # Penalty is Reward is -self._r_collision if there were any collisions in the last timestep
        reward = float(task._in_contact) * -self._r_collision

        return reward, {}

