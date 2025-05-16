import os
import random
import torch as th
import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import gm
from omnigibson.robots import REGISTERED_ROBOTS
from omnigibson.utils.ui_utils import KeyboardRobotController, choose_from_options
import omnigibson.utils.transform_utils as T
from omnigibson.utils.asset_utils import get_available_g_scenes, get_available_og_scenes, get_available_3dfront_scenes, get_available_3dfront_rooms, get_available_3dfront_room, get_available_3dfront_target_scenes
import math
import argparse
import time
import yaml
from omnigibson.utils.python_utils import meets_minimum_version
from omnigibson.utils.gym_utils import (
    GymObservable,
    maxdim,
    recursively_generate_flat_dict,
)


try:
    import gymnasium as gym
    import tensorboard
    import torch as th
    import torch.nn as nn
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.preprocessing import maybe_transpose
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from omnigibson.object_states.contact_bodies import ContactBodies
except ModuleNotFoundError:
    og.log.error(
        "torch, stable-baselines3, or tensorboard is not installed. "
        "See which packages are missing, and then run the following for any missing packages:\n"
        "pip install stable-baselines3[extra]\n"
        "pip install tensorboard\n"
        "pip install shimmy>=0.2.1\n"
        "Also, please update gym to >=0.26.1 after installing sb3: pip install gym>=0.26.1"
    )
    exit(1)

assert meets_minimum_version(gym.__version__, "0.28.1"), "Please install/update gymnasium to version >= 0.28.1"

# We don't need object states nor transitions rules, so we disable them now, and also enable flatcache for maximum speed
gm.ENABLE_OBJECT_STATES = False
gm.ENABLE_TRANSITION_RULES = False
gm.ENABLE_FLATCACHE = True
gm.RENDER_VIEWER_CAMERA = False


ROBOT_ACTION_DIM = 13

class RearrangementEnv(gym.Env, GymObservable):
    """符合gym接口的机器人重排环境"""
    
    ROBOT_ACTION_DIM = 13
    
    def __init__(self, env, robot):
        self.env = env
        self.robot = robot
        self.grasping_state = 1.0
        self.last_step_result = None
        self._current_step = 0
        self.timeout_limit = 100
        self.post_play_load()
    
    def reset(self, get_obs=True, **kwargs):
        obs, info = self.env.reset(get_obs, **kwargs)
        obs = list(obs.items())[0][1]
        obs = {'robot': obs}
        # self.robot.reset()
        self._current_step = 0
        self.grasping_state = 1.0

        self.robot = self.env.robots[0]
        return obs, info
    
    def get_obs(self):
        obs, info = self.env.get_obs()
        return obs, info
    
    def _collision_detection(self):
        a = time.time()
        robot_in_contact = (
            len(self.env.robots[0].states[ContactBodies].get_value()) > 1
        )

        b = time.time()
        grasping_obj_in_contact = False
        grasping_obj = self.robot._ag_obj_in_hand['0']
        c = time.time()
        if grasping_obj is not None:
            grasping_obj_in_contact = (
            len(grasping_obj.states[ContactBodies].get_value()) > 0)
        in_contact = robot_in_contact or grasping_obj_in_contact
        return in_contact
    
    def _post_process_step(self):
        # obs
        obs = self.last_obs
        # import pdb;pdb.set_trace()
        obs = list(obs.items())[0][1]
        # obs = list(obs.items())[0][1]
        # obs = list(obs.items())[0][1]
        obs = {'robot': obs}

        # done
        done = self.done_flag

        # info 
        info = {}

        # truncated
        truncated = False
        if self._current_step >= self.timeout_limit:
            truncated = True

        # rewards 
        assert set(self.info_list[0]['reward']['reward_breakdown'].keys()) == {'pointgoal', 'arrival'}
        pointgoal_rewards = [info['reward']['reward_breakdown']['pointgoal'] for info in self.info_list]
        arrival_rewards = [info['reward']['reward_breakdown']['arrival'] for info in self.info_list]
        pointgoal_reward = pointgoal_rewards[-1]
        arrival_reward = sum(arrival_rewards)
        reward = pointgoal_reward + arrival_reward
        return obs, reward, done, truncated, info

    def step(self, action):
        a = time.time()
        self.last_obs = None
        self.reward_list = []
        self.done_flag = False
        self.info_list = []
        self.grasping_state = self._generate_action_tensor(action, self.grasping_state)
        self._current_step += 1
        obs, reward, done, truncated, info = self._post_process_step()
        return obs, reward, done, truncated, info
    
    def _generate_action_tensor(self, action, grasping_state):
        if not isinstance(action, int):
            action = action.item()
        if action == 0:
            self._translate(-0.25, grasping_state)
        elif action == 1:
            self._translate(0.25, grasping_state)
        elif action == 2:
            self._rotate(math.pi / 8, grasping_state)
        elif action == 3:
            self._rotate(-math.pi / 8, grasping_state)
        elif action == 4:
            grasping_state = self._fetch()
        elif action == 5:
            grasping_state = self._release()
            
        return grasping_state
    
    def _rotate(self, end_yaw, grasping_state, angle_threshold=0.01):
        all_env_step_time = 0

        world_end_pos, world_end_quat = self.robot.get_position_orientation()
        world_end_quat = T.quat_multiply(T.euler2quat(th.tensor([0.0, end_yaw, 0.0])), world_end_quat)
        diff_yaw = end_yaw
        action = th.zeros(self.ROBOT_ACTION_DIM)
        in_collision_steps = 0
        for idx in range(500):
            if abs(diff_yaw) < angle_threshold:
                action = 0.0 * action
                action[self.robot.controller_action_idx["gripper_0"]] = grasping_state
                obs, reward, done, truncated, info = self.env.step(action, n_render_iterations = 3)
                self.last_obs = obs
                self.reward_list.append(reward)
                self.done_flag = done
                self.info_list.append(info)
                break

            direction = -1.0 if diff_yaw < 0.0 else 1.0
            ang_vel = 0.2 * direction
            action = 0.0 * action 
            base_action = action[self.robot.controller_action_idx["base"]]
            assert (base_action.numel() == 2)
            base_action[0] = -ang_vel
            base_action[1] = ang_vel
            action[self.robot.controller_action_idx["base"]] = base_action
            action[self.robot.controller_action_idx["gripper_0"]] = grasping_state

            a = time.time()
            _render_on_step = True
            if diff_yaw <= 0.03:
                _render_on_step = True
            with og.sim.render_on_step(_render_on_step): 
                obs, reward, done, truncated, info = self.env.step(action)
            all_env_step_time += time.time() - a
            self.last_obs = obs
            self.reward_list.append(reward)
            self.done_flag = done
            self.info_list.append(info)

            # The timeout handle should be within the RearrangementEnv 
            assert not truncated
            if done:
                return
            
            if idx % 5 == 0:
                if self._collision_detection():
                    in_collision_steps += 1
            if in_collision_steps > 4:
                return
            
            world_pose, world_quat = self.robot.get_position_orientation()
            body_target_pose = T.relative_pose_transform(world_end_pos, world_end_quat, world_pose, world_quat)
            diff_yaw = T.quat2euler(body_target_pose[1])[2].item()

    def _translate(self, end_x, grasping_state, angle_threshold=0.04):
        end_pos_robot_frame = th.tensor([end_x, 0.0, 0.0])
        robot_pos, robot_quat = self.robot.get_position_orientation()
        inv_pos, inv_quat = T.invert_pose_transform(robot_pos, robot_quat)
        world_end_pos, _ = T.relative_pose_transform(end_pos_robot_frame, th.tensor([1., 0., 0., 0.]), inv_pos, inv_quat)
        world_end_quat = robot_quat
        diff_pos = end_x
        distance_pos = th.norm(th.tensor([end_x, 0.0, 0.0]))
        all_env_step_time = 0
        world_pose, world_quat = self.robot.get_position_orientation()

        in_collision_steps = 0
        action = th.zeros(self.ROBOT_ACTION_DIM)
        for idx in range(500):
            if abs(distance_pos) < angle_threshold:
                action = 0.0 * action
                action[self.robot.controller_action_idx["gripper_0"]] = grasping_state
                obs, reward, done, truncated, info = self.env.step(action, n_render_iterations = 3)
                self.last_obs = obs
                self.reward_list.append(reward)
                self.done_flag = done
                self.info_list.append(info)
                break

            direction = -1.0 if diff_pos < 0.0 else 1.0
            lin_vel = 0.2 * direction
            action = 0.0 * action
            base_action = action[self.robot.controller_action_idx["base"]]
            assert (base_action.numel() == 2)
            base_action[0] = lin_vel
            base_action[1] = lin_vel
            action[self.robot.controller_action_idx["base"]] = base_action
            action[self.robot.controller_action_idx["gripper_0"]] = grasping_state

            a = time.time()
            _render_on_step = True
            if diff_pos <= 0.08:
                _render_on_step = True
            with og.sim.render_on_step(_render_on_step):
                obs, reward, done, truncated, info = self.env.step(action)
            all_env_step_time += time.time() - a
            self.last_obs = obs
            self.reward_list.append(reward)
            self.done_flag = done
            self.info_list.append(info)

            b = time.time()
            # The timeout handle should be within the RearrangementEnv 
            assert not truncated
            if done:
                return 

            if idx % 5 == 0:
                if self._collision_detection():
                    in_collision_steps += 1
            if in_collision_steps > 4:
                return

            c = time.time()
            world_pose, world_quat = self.robot.get_position_orientation()
            robot_frame_end_pos, _ = T.relative_pose_transform(world_end_pos, world_end_quat, world_pose, world_quat)
            diff_pos = robot_frame_end_pos[0]
            distance_pos = th.norm(robot_frame_end_pos)
            world_pose, world_quat = self.robot.get_position_orientation()

    def _fetch(self):
        action = th.zeros(self.ROBOT_ACTION_DIM)
        base_action = action[self.robot.controller_action_idx["gripper_0"]]
        assert (base_action.numel() == 1)
        base_action[0] = -1.0
        action[self.robot.controller_action_idx["gripper_0"]] = base_action
        obs, reward, done, truncated, info = self.env.step(action)
        self.last_obs = obs
        self.reward_list.append(reward)
        self.done_flag = done
        self.info_list.append(info)

        # The timeout handle should be within the RearrangementEnv 
        assert not truncated
        if done:
            # nothing to do
            pass
        
        if self.robot._ag_obj_in_hand['0'] is None:
            return 1.0
        return -1.0

    def _release(self):
        action = th.zeros(self.ROBOT_ACTION_DIM)
        base_action = action[self.robot.controller_action_idx["gripper_0"]]
        assert (base_action.numel() == 1)
        base_action[0] = 1.0
        action[self.robot.controller_action_idx["gripper_0"]] = base_action
        obs, reward, done, truncated, info = self.env.step(action)
        self.last_obs = obs
        self.reward_list.append(reward)
        self.done_flag = done
        self.info_list.append(info)

        # The timeout handle should be within the RearrangementEnv 
        assert not truncated
        if done:
            # nothing to do
            pass
        
        return 1.0
    
    def _load_observation_space(self):
        obs_space = dict()

        assert len(self.env.robots) == 1
        robot_name = 'robot'

        for robot in self.env.robots:
            # Load the observation space for the robot
            robot_obs = robot.load_observation_space()
            if maxdim(robot_obs) > 0:
                obs_space[robot_name] = next(iter(robot_obs.values()))['rgb']
        return obs_space
    
    def _load_action_space(self):
        self.action_space = gym.spaces.Discrete(6)
    
    def load_observation_space(self):
        obs_space = super().load_observation_space()

        if self.env._flatten_obs_space:
            self.observation_space = gym.spaces.Dict(recursively_generate_flat_dict(dic=obs_space))

        return self.observation_space
    
    def post_play_load(self):
        # Load the obs / action spaces
        self.load_observation_space()
        self._load_action_space()

gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True

class ReplaceInfAndLargeValues(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x:th.Tensor) -> th.Tensor:
        mask = th.isinf(x) | (x>20)
        x_clean = th.where(mask, th.full_like(x, 20.0), x)
        return x_clean

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        super().__init__(observation_space, features_dim=1)
        self.step_index = 0
        self.img_save_dir = "img_save_dir"
        os.makedirs(self.img_save_dir, exist_ok=True)
        total_concat_size = 0
        feature_size = 128
        n_input_channels = 4
        cnn = nn.Sequential(
            ReplaceInfAndLargeValues(),
            nn.Conv2d(n_input_channels, 1, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(1, 2, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        test_tensor = th.zeros(observation_space['robot'].shape).unsqueeze(0)
        with th.no_grad():
            n_flatten = cnn(test_tensor).size()[1]
        fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
        self.net = nn.Sequential(cnn, fc)
        total_concat_size += feature_size

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []
        self.step_index += 1

        # self.extractors contain nn.Modules that do all the processing.
        feature = self.net(observations['robot'])
        return feature


def make_env(config, seed, idx):
    """Helper function to create a single environment instance with proper seeding"""
    def _init():
        # Create the environment
        _env = og.Environment(configs=config)
        
        # Get the robot
        _robot = _env.robots[0]
        
        # Setup controllers
        controller_config = {
            "base": {"name": "JointController"},
            "arm_0": {"name": "JointController", "motor_type": "effort"},
            "gripper_0": {"name": "MultiFingerGripperController"}
        }
        _robot.reload_controllers(controller_config=controller_config)
        
        # Update initial state
        _env.scene.update_initial_state()
        
        # Reset environment and robot
        _env.reset()
        _robot.reset()
        # import pdb; pdb.set_trace()
        # Create and return wrapped environment
        return RearrangementEnv(_env, _robot)
    
    return _init

def main(random_selection=False):
    """
    Robot control demo with selection
    Queries the user to select a robot, the controllers, a scene and a type of input (random actions or teleop)
    """

    tensorboard_log_dir = os.path.join("log_dir", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    prefix = ""
    seed = 0
    num_envs = 4  # Number of parallel environments

    # Choose scene to load

    config_filename = config_filename = os.path.join(og.example_config_path, f"rearrange.yaml")
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    # Make sure flattened obs and action space is used
    config["env"]["flatten_action_space"] = True
    config["env"]["flatten_obs_space"] = True

    # Only use RGB obs
    config["robots"] = [{}]
    config["robots"][0]["obs_modalities"] = ["rgb"]

    # scene_type = "Threed_FRONTScene"
    # config["scene"]["type"] = scene_type

    # scenes = get_available_3dfront_scenes()
    # scene = choose_from_options(options=scenes, name="scene model", random_selection=random_selection)
    # rooms = get_available_3dfront_rooms(scene)
    # room_type = choose_from_options(options=rooms, name="room type", random_selection=random_selection)
    # room = get_available_3dfront_room(scene, room_type)
    # room_model = choose_from_options(options=room, name="room model", random_selection=random_selection)
    # room_model_path = os.path.join(gm.ThreeD_FRONT_DATASET_PATH, "scenes", scene, room_type, room_model)

    # config["scene"]["scene_model"] = room_model
    # config["scene"]["scene_type_path"] = room_model_path
    scene_type = "Threed_FRONTScene"
    config["scene"]["type"] = scene_type
    # Choose the scene model to load
    scenes = get_available_3dfront_scenes()
    scene = choose_from_options(options=scenes, name="scene model", random_selection=random_selection)
    threed_front_path = gm.ThreeD_FRONT_DATASET_PATH
    scene_path = os.path.join(threed_front_path, "scenes", scene)

    room = get_available_3dfront_target_scenes(scene)
    room_model = choose_from_options(options=room, name="room model", random_selection=random_selection) # filename(with .json)

    config["scene"]["scene_model"] = room_model
    config["scene"]["scene_type_path"] = scene_path

    # Choose robot to create
    robot_name = 'Test'

    # Add the robot we want to load
    config["robots"][0]["type"] = robot_name
    config["robots"][0]["obs_modalities"] = ["rgb"]
    config["robots"][0]["action_type"] = "continuous"
    config["robots"][0]["action_normalize"] = True
    config["robots"][0]["grasping_mode"] = 'sticky'
    config["robots"][0]["sensor_config"] = {"VisionSensor": {"sensor_kwargs": {"image_height": 256, "image_width": 256}}}

    num_envs = 2
    env_fns = [make_env(config.copy(), seed + i, i) for i in range(num_envs)]
    # vec_env = DummyVecEnv(env_fns)
    vec_env = SubprocVecEnv(env_fns)
    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
    )

    os.makedirs(tensorboard_log_dir, exist_ok=True)

    model = PPO(
        "MultiInputPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=tensorboard_log_dir,
        policy_kwargs=policy_kwargs,
        n_steps=20 * 10 // num_envs,  # Adjust steps to account for parallel envs
        batch_size=8,
        device="cuda",
    )
    
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=tensorboard_log_dir, name_prefix=prefix)

    og.log.debug(model.policy)
    og.log.info(f"model: {model}")

    og.log.info("Starting training...")
    model.learn(
        total_timesteps=10000000,
        callback=checkpoint_callback
    )
    og.log.info("Finished training!")

    # Always shut down the environment cleanly at the end
    og.clear()
    # Always close the environments
    vec_env.close()


if __name__ == "__main__":
    main()

# import os
# import random
# import torch as th
# import omnigibson as og
# import omnigibson.lazy as lazy
# from omnigibson.macros import gm
# from omnigibson.robots import REGISTERED_ROBOTS
# from omnigibson.utils.ui_utils import KeyboardRobotController, choose_from_options
# import omnigibson.utils.transform_utils as T
# from omnigibson.utils.asset_utils import get_available_g_scenes, get_available_og_scenes, get_available_3dfront_scenes, get_available_3dfront_rooms, get_available_3dfront_room, get_available_3dfront_target_scenes
# import math
# import argparse
# import time
# import yaml
# from omnigibson.utils.python_utils import meets_minimum_version
# from omnigibson.utils.gym_utils import (
#     GymObservable,
#     maxdim,
#     recursively_generate_flat_dict,
# )
# from omnigibson.object_states.contact_bodies import ContactBodies

# try:
#     import gymnasium as gym
#     import tensorboard
#     import torch as th
#     import torch.nn as nn
#     from stable_baselines3 import PPO
#     from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
#     from stable_baselines3.common.evaluation import evaluate_policy
#     from stable_baselines3.common.preprocessing import maybe_transpose
#     from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
#     from stable_baselines3.common.utils import set_random_seed
#     from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
# except ModuleNotFoundError:
#     og.log.error(
#         "torch, stable-baselines3, or tensorboard is not installed. "
#         "See which packages are missing, and then run the following for any missing packages:\n"
#         "pip install stable-baselines3[extra]\n"
#         "pip install tensorboard\n"
#         "pip install shimmy>=0.2.1\n"
#         "Also, please update gym to >=0.26.1 after installing sb3: pip install gym>=0.26.1"
#     )
#     exit(1)

# assert meets_minimum_version(gym.__version__, "0.28.1"), "Please install/update gymnasium to version >= 0.28.1"

# # We don't need object states nor transitions rules, so we disable them now, and also enable flatcache for maximum speed
# gm.ENABLE_OBJECT_STATES = False
# gm.ENABLE_TRANSITION_RULES = False
# gm.ENABLE_FLATCACHE = True
# gm.RENDER_VIEWER_CAMERA = False

# class RearrangementEnv(gym.Env, GymObservable):
#     """符合gym接口的机器人重排环境"""
    
#     ROBOT_ACTION_DIM = 13
    
#     def __init__(self, env, robot):
#         self.env = env
#         self.robot = robot
#         self.grasping_state = 1.0
#         self.last_step_result = None
#         self._current_step = 0
#         self.timeout_limit = 100
#         self.post_play_load()
    
#     def reset(self, get_obs=True, **kwargs):
#         obs, info = self.env.reset(get_obs, **kwargs)
#         obs = list(obs.items())[0][1]
#         obs = {'robot': obs}
#         # self.robot.reset()
#         self._current_step = 0
#         self.grasping_state = 1.0
#         return obs, info
    
#     def get_obs(self):
#         obs, info = self.env.get_obs()
#         return obs, info
    
#     def _collision_detection(self):
#         robot_in_contact = (
#             len(self.env.robots[0].states[ContactBodies].get_value()) > 1
#         )

#         grasping_obj_in_contact = False
#         grasping_obj = self.robot._ag_obj_in_hand['0']
#         if grasping_obj is not None:
#             grasping_obj_in_contact = (
#             len(grasping_obj.states[ContactBodies].get_value()) > 1
#         )
            
#         in_contact = robot_in_contact or grasping_obj_in_contact
#         #print(robot_in_contact, grasping_obj_in_contact)
#         return in_contact
    
#     def _post_process_step(self):
#         # obs
#         obs = self.last_obs
#         obs = list(obs.items())[0][1]
#         obs = {'robot': obs}

#         # done
#         done = self.done_flag

#         # info 
#         info = {}

#         # truncated
#         truncated = False
#         if self._current_step >= self.timeout_limit:
#             truncated = True

#         # rewards 
#         assert set(self.info_list[0]['reward']['reward_breakdown'].keys()) == {'pointgoal', 'arrival', 'potential', 'collision'}
#         pointgoal_rewards = [_info['reward']['reward_breakdown']['pointgoal'] for _info in self.info_list]
#         arrival_rewards = [_info['reward']['reward_breakdown']['arrival'] for _info in self.info_list]
#         potential_rewards = [_info['reward']['reward_breakdown']['potential'] for _info in self.info_list]
#         collision_rewards = [_info['reward']['reward_breakdown']['collision'] for _info in self.info_list]
#         pointgoal_reward = pointgoal_rewards[-1]
#         arrival_reward = arrival_rewards[-1]
#         potential_reward = sum(potential_rewards)
#         collision_reward = next((x for x in collision_rewards if x != 0), 0.0)
#         reward = pointgoal_reward + arrival_reward + potential_reward + collision_reward
#         print(pointgoal_reward, arrival_reward, potential_reward, collision_reward)
#         return obs, reward, done, truncated, info

#     def step(self, action):
#         a = time.time()
#         self.last_obs = None
#         self.reward_list = []
#         self.done_flag = False
#         self.info_list = []
#         self.grasping_state = self._generate_action_tensor(action, self.grasping_state)
#         self._current_step += 1
#         obs, reward, done, truncated, info = self._post_process_step()
#         # print(time.time() - a)
#         # print(action)
#         # print("--- step() ---")
#         return obs, reward, done, truncated, info
    
#     def _generate_action_tensor(self, action, grasping_state):
#         if not isinstance(action, int):
#             action = action.item()
#         if action == 0:
#             self._translate(-0.25, grasping_state)
#         elif action == 1:
#             self._translate(0.25, grasping_state)
#         elif action == 2:
#             self._rotate(math.pi / 8, grasping_state)
#         elif action == 3:
#             self._rotate(-math.pi / 8, grasping_state)
#         elif action == 4:
#             grasping_state = self._fetch()
#         elif action == 5:
#             grasping_state = self._release()
            
#         return grasping_state
    
#     def _rotate(self, end_yaw, grasping_state, angle_threshold=0.01):
#         all_env_step_time = 0

#         world_end_pos, world_end_quat = self.robot.get_position_orientation()
#         world_end_quat = T.quat_multiply(T.euler2quat(th.tensor([0.0, end_yaw, 0.0])), world_end_quat)
#         diff_yaw = end_yaw
#         action = th.zeros(self.ROBOT_ACTION_DIM)
#         in_collision_steps = 0
#         for idx in range(500):
#             if abs(diff_yaw) < angle_threshold:
#                 action = 0.0 * action
#                 action[self.robot.controller_action_idx["gripper_0"]] = grasping_state
#                 obs, reward, done, truncated, info = self.env.step(action, n_render_iterations = 3)
#                 self.last_obs = obs
#                 self.reward_list.append(reward)
#                 self.done_flag = done
#                 self.info_list.append(info)
#                 break

#             direction = -1.0 if diff_yaw < 0.0 else 1.0
#             ang_vel = 0.2 * direction
#             action = 0.0 * action 
#             base_action = action[self.robot.controller_action_idx["base"]]
#             assert (base_action.numel() == 2)
#             base_action[0] = -ang_vel
#             base_action[1] = ang_vel
#             action[self.robot.controller_action_idx["base"]] = base_action
#             action[self.robot.controller_action_idx["gripper_0"]] = grasping_state

#             a = time.time()
#             _render_on_step = False
#             if diff_yaw <= 0.03:
#                 _render_on_step = True
#             with og.sim.render_on_step(_render_on_step): 
#                 self.env.task.get_incontact(self._collision_detection())
#                 obs, reward, done, truncated, info = self.env.step(action)
#             all_env_step_time += time.time() - a
#             self.last_obs = obs
#             self.reward_list.append(reward)
#             self.done_flag = done
#             self.info_list.append(info)

#             # The timeout handle should be within the RearrangementEnv 
#             assert not truncated
#             if done:
#                 return
#             if idx % 5 == 0:
#                 if self._collision_detection():
#                     in_collision_steps += 1
#             if in_collision_steps > 4:
#                 return
#             world_pose, world_quat = self.robot.get_position_orientation()
#             body_target_pose = T.relative_pose_transform(world_end_pos, world_end_quat, world_pose, world_quat)
#             diff_yaw = T.quat2euler(body_target_pose[1])[2].item()
#         #print(f'in rotate {all_env_step_time}')

#     def _translate(self, end_x, grasping_state, angle_threshold=0.04):
#         end_pos_robot_frame = th.tensor([end_x, 0.0, 0.0])
#         robot_pos, robot_quat = self.robot.get_position_orientation()
#         inv_pos, inv_quat = T.invert_pose_transform(robot_pos, robot_quat)
#         world_end_pos, _ = T.relative_pose_transform(end_pos_robot_frame, th.tensor([1., 0., 0., 0.]), inv_pos, inv_quat)
#         world_end_quat = robot_quat
#         diff_pos = end_x
#         distance_pos = th.norm(th.tensor([end_x, 0.0, 0.0]))
#         all_env_step_time = 0
#         world_pose, world_quat = self.robot.get_position_orientation()

#         in_collision_steps = 0
#         action = th.zeros(self.ROBOT_ACTION_DIM)
#         for idx in range(500):
#             #print(distance_pos)
#             if abs(distance_pos) < angle_threshold:
#                 action = 0.0 * action
#                 action[self.robot.controller_action_idx["gripper_0"]] = grasping_state
#                 obs, reward, done, truncated, info = self.env.step(action, n_render_iterations = 3)
#                 self.last_obs = obs
#                 self.reward_list.append(reward)
#                 self.done_flag = done
#                 self.info_list.append(info)
#                 break

#             direction = -1.0 if diff_pos < 0.0 else 1.0
#             lin_vel = 0.2 * direction
#             action = 0.0 * action
#             base_action = action[self.robot.controller_action_idx["base"]]
#             assert (base_action.numel() == 2)
#             base_action[0] = lin_vel
#             base_action[1] = lin_vel
#             action[self.robot.controller_action_idx["base"]] = base_action
#             action[self.robot.controller_action_idx["gripper_0"]] = grasping_state

#             a = time.time()
#             _render_on_step = False
#             if diff_pos <= 0.08:
#                 _render_on_step = True
#             with og.sim.render_on_step(_render_on_step): 
#                 self.env.task.get_incontact(self._collision_detection())
#                 obs, reward, done, truncated, info = self.env.step(action)
#             all_env_step_time += time.time() - a
#             #print(f'in translate {all_env_step_time}')
#             self.last_obs = obs
#             self.reward_list.append(reward)
#             self.done_flag = done
#             self.info_list.append(info)

#             # The timeout handle should be within the RearrangementEnv 
#             assert not truncated
#             if done:
#                 return 

#             if idx % 5 == 0:
#                 if self._collision_detection():
#                     in_collision_steps += 1
#             if in_collision_steps > 4:
#                 return

#             world_pose, world_quat = self.robot.get_position_orientation()
#             robot_frame_end_pos, _ = T.relative_pose_transform(world_end_pos, world_end_quat, world_pose, world_quat)
#             diff_pos = robot_frame_end_pos[0]
#             distance_pos = th.norm(robot_frame_end_pos)
#             world_pose, world_quat = self.robot.get_position_orientation()

#     def _fetch(self):
#         action = th.zeros(self.ROBOT_ACTION_DIM)
#         base_action = action[self.robot.controller_action_idx["gripper_0"]]
#         assert (base_action.numel() == 1)
#         base_action[0] = -1.0
#         action[self.robot.controller_action_idx["gripper_0"]] = base_action
#         obs, reward, done, truncated, info = self.env.step(action)
#         grasping_obj = self.robot._ag_obj_in_hand['0']
#         if grasping_obj is not None:
#             self.env.task.get_fetch(grasping_obj, 1)
#         self.last_obs = obs
#         self.reward_list.append(reward)
#         self.done_flag = done
#         self.info_list.append(info)

#         # The timeout handle should be within the RearrangementEnv 
#         assert not truncated
#         if done:
#             # nothing to do
#             pass
        
#         if self.robot._ag_obj_in_hand['0'] is None:
#             return 1.0
#         return -1.0

#     def _release(self):
#         action = th.zeros(self.ROBOT_ACTION_DIM)
#         base_action = action[self.robot.controller_action_idx["gripper_0"]]
#         assert (base_action.numel() == 1)
#         base_action[0] = 1.0
#         action[self.robot.controller_action_idx["gripper_0"]] = base_action
#         obs, reward, done, truncated, info = self.env.step(action)
#         self.env.task.get_fetch(None, -1)
#         self.last_obs = obs
#         self.reward_list.append(reward)
#         self.done_flag = done
#         self.info_list.append(info)

#         # The timeout handle should be within the RearrangementEnv 
#         assert not truncated
#         if done:
#             # nothing to do
#             pass
        
#         return 1.0
    
#     def _load_observation_space(self):
#         obs_space = dict()

#         assert len(self.env.robots) == 1
#         robot_name = 'robot'

#         for robot in self.env.robots:
#             # Load the observation space for the robot
#             robot_obs = robot.load_observation_space()
#             if maxdim(robot_obs) > 0:
#                 obs_space[robot_name] = next(iter(robot_obs.values()))['rgb']
#         return obs_space
    
#     def _load_action_space(self):
#         self.action_space = gym.spaces.Discrete(6)
    
#     def load_observation_space(self):
#         obs_space = super().load_observation_space()

#         if self.env._flatten_obs_space:
#             self.observation_space = gym.spaces.Dict(recursively_generate_flat_dict(dic=obs_space))

#         return self.observation_space
    
#     def post_play_load(self):
#         # Load the obs / action spaces
#         self.load_observation_space()
#         self._load_action_space()

# class ReplaceInfAndLargeValues(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, x:th.Tensor) -> th.Tensor:
#         mask = th.isinf(x) | (x>20)
#         x_clean = th.where(mask, th.full_like(x, 20.0), x)
#         return x_clean

# class CustomCombinedExtractor(BaseFeaturesExtractor):
#     def __init__(self, observation_space: gym.spaces.Dict):
#         super().__init__(observation_space, features_dim=1)
#         self.step_index = 0
#         self.img_save_dir = "img_save_dir"
#         os.makedirs(self.img_save_dir, exist_ok=True)
#         total_concat_size = 0
#         feature_size = 128
#         n_input_channels = 4
#         cnn = nn.Sequential(
#             #ReplaceInfAndLargeValues(),
#             nn.Conv2d(n_input_channels, 1, kernel_size=8, stride=4, padding=0),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Conv2d(1, 2, kernel_size=4, stride=2, padding=0),
#             nn.ReLU(),
#             nn.Flatten(),
#         )
#         test_tensor = th.zeros(observation_space['robot'].shape).unsqueeze(0)
#         with th.no_grad():
#             n_flatten = cnn(test_tensor).size()[1]
#         fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
#         self.net = nn.Sequential(cnn, fc)
#         total_concat_size += feature_size

#         # Update the features dim manually
#         self._features_dim = total_concat_size

#     def forward(self, observations) -> th.Tensor:
#         encoded_tensor_list = []
#         self.step_index += 1

#         # self.extractors contain nn.Modules that do all the processing.
#         feature = self.net(observations['robot'])
#         return feature


# def make_env(config, seed, idx):
#     """Helper function to create a single environment instance with proper seeding"""
#     def _init():
#         # Create the environment
#         _env = og.Environment(configs=config)
        
#         # Get the robot
#         _robot = _env.robots[0]
        
#         # Setup controllers
#         controller_config = {
#             "base": {"name": "JointController"},
#             "arm_0": {"name": "JointController", "motor_type": "effort"},
#             "gripper_0": {"name": "MultiFingerGripperController"}
#         }
#         _robot.reload_controllers(controller_config=controller_config)
        
#         # Update initial state
#         _env.scene.update_initial_state()
        
#         # Reset environment and robot
#         _env.reset()
#         _robot.reset()
#         # import pdb; pdb.set_trace()
#         # Create and return wrapped environment
#         return RearrangementEnv(_env, _robot)
    
#     return _init

# def main(random_selection=False):
#     """
#     Robot control demo with selection
#     Queries the user to select a robot, the controllers, a scene and a type of input (random actions or teleop)
#     """

#     tensorboard_log_dir = os.path.join("log_dir", time.strftime("%Y%m%d-%H%M%S"))
#     os.makedirs(tensorboard_log_dir, exist_ok=True)
#     prefix = ""
#     seed = 0

#     # Choose scene to load

#     config_filename = config_filename = os.path.join(og.example_config_path, f"rearrange.yaml")
#     config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

#     # Make sure flattened obs and action space is used
#     config["env"]["flatten_action_space"] = True
#     config["env"]["flatten_obs_space"] = True

#     # scene_type = "Threed_FRONTScene"
#     # config["scene"]["type"] = scene_type

#     # scenes = get_available_3dfront_scenes()
#     # scene = choose_from_options(options=scenes, name="scene model", random_selection=random_selection)
#     # rooms = get_available_3dfront_rooms(scene)
#     # room_type = choose_from_options(options=rooms, name="room type", random_selection=random_selection)
#     # room = get_available_3dfront_room(scene, room_type)
#     # room_model = choose_from_options(options=room, name="room model", random_selection=random_selection)
#     # room_model_path = os.path.join(gm.ThreeD_FRONT_DATASET_PATH, "scenes", scene, room_type, room_model)

#     # config["scene"]["scene_model"] = room_model
#     # config["scene"]["scene_type_path"] = room_model_path
#     scene_type = "Threed_FRONTScene"
#     config["scene"]["type"] = scene_type
#     # Choose the scene model to load
#     scenes = get_available_3dfront_scenes()
#     scene = choose_from_options(options=scenes, name="scene model", random_selection=random_selection)
#     threed_front_path = gm.ThreeD_FRONT_DATASET_PATH
#     scene_path = os.path.join(threed_front_path, "scenes", scene)

#     room = get_available_3dfront_target_scenes(scene)
#     room_model = choose_from_options(options=room, name="room model", random_selection=random_selection) # filename(with .json)

#     config["scene"]["scene_model"] = room_model
#     config["scene"]["scene_type_path"] = scene_path

#     # Choose robot to create
#     robot_name = 'Test'

#     # Add the robot we want to load
#     # Only use RGB obs
#     config["robots"] = [{}]
#     config["robots"][0]["type"] = robot_name
#     config["robots"][0]["obs_modalities"] = ["rgb"]
#     config["robots"][0]["action_type"] = "continuous"
#     config["robots"][0]["action_normalize"] = True
#     config["robots"][0]["grasping_mode"] = 'sticky'
#     config["robots"][0]["sensor_config"] = {"VisionSensor": {"sensor_kwargs": {"image_height": 512, "image_width": 512}}}

#     num_envs = 1
#     env_fns = [make_env(config.copy(), seed + i, i) for i in range(num_envs)]
#     vec_env = DummyVecEnv(env_fns)
#     # vec_env = SubprocVecEnv(env_fns)
#     policy_kwargs = dict(
#         features_extractor_class=CustomCombinedExtractor,
#     )

#     os.makedirs(tensorboard_log_dir, exist_ok=True)

#     model = PPO(
#         "MultiInputPolicy",
#         vec_env,
#         verbose=1,
#         tensorboard_log=tensorboard_log_dir,
#         policy_kwargs=policy_kwargs,
#         n_steps=20 * 10 // num_envs,  # Adjust steps to account for parallel envs
#         batch_size=8,
#         device="cuda",
#     )
    
#     checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=tensorboard_log_dir, name_prefix=prefix)

#     og.log.debug(model.policy)
#     og.log.info(f"model: {model}")

#     og.log.info("Starting training...")
#     model.learn(
#         total_timesteps=10000000,
#         callback=checkpoint_callback
#     )
#     og.log.info("Finished training!")

#     # Always shut down the environment cleanly at the end
#     og.clear()
#     # Always close the environments
#     vec_env.close()


# if __name__ == "__main__":
#     main()