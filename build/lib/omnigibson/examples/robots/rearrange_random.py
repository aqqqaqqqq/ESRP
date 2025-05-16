import os
import random
import torch as th
import omnigibson as og
from omnigibson.examples.environments.new_env import FastEnv
from omnigibson.macros import gm
import time
import yaml

gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_TRANSITION_RULES = True
gm.ENABLE_FLATCACHE = True
gm.RENDER_VIEWER_CAMERA = True

# We don't need object states nor transitions rules, so we disable them now, and also enable flatcache for maximum speed
# class VecEnvironment:
#     def __init__(self, num_envs, config):
#         self.num_envs = num_envs
#         if og.sim is not None:
#             og.sim.stop()

#         # First we create the environments. We can't let DummyVecEnv do this for us because of the play call
#         # needing to happen before spaces are available for it to read things from.
#         self.envs = [
#             og.Environment(configs=copy.deepcopy(config), in_vec_env=True)
#             for _ in trange(num_envs, desc="Loading environments")
#         ]

#         top_down_position = th.tensor([30, 37.0, 0])
#         top_down_orientation = th.tensor([0.0, 0.7, 0.7, 0.0])
#         cam = og.sim.viewer_camera
#         cam.set_position_orientation(top_down_position, top_down_orientation)

#         # Play, and finish loading all the envs
#         og.sim.play()
#         for env in self.envs:
#             env.post_play_load()
        
#         self.grasping_obj_list = [None for _ in range(self.num_envs)]

#     def step(self, actions):
#         for idx, (_env, _action) in enumerate(zip(self.envs, actions)):
#             self.grasping_obj_list[idx] = _step(_env, _action, self.grasping_obj_list[idx])

#         action_empty = th.zeros(13)
#         actions_empty = [action_empty for i in range(self.num_envs)]
#         for i, action in enumerate(actions_empty):
#             self.envs[i]._pre_step(action)

#         og.sim.step()

#         observations, rewards, terminates, truncates, infos = [], [], [], [], []
#         for i, action in enumerate(actions):
#             obs, reward, terminated, truncated, info = self.envs[i]._post_step(action)
#             pointgoal_rewards = info['reward']['reward_breakdown']['pointgoal']
#             arrival_rewards = info['reward']['reward_breakdown']['arrival']
#             potential_rewards = info['reward']['reward_breakdown']['potential']
#             print(pointgoal_rewards, arrival_rewards, potential_rewards)
#             observations.append(obs)
#             rewards.append(reward)
#             terminates.append(terminated)
#             truncates.append(truncated)
#             infos.append(info)

#         # import pdb; pdb.set_trace()
#         return observations, rewards, terminates, truncates, infos

#     def reset(self):
#         for env in self.envs:
#             env.reset()
#         self.grasping_obj_list = [None for _ in range(self.num_envs)]

#     def close(self):
#         pass

#     def __len__(self):
#         return self.num_envs

def main():
    """
    Robot control demo with selection
    Queries the user to select a robot, the controllers, a scene and a type of input (random actions or teleop)
    """

    tensorboard_log_dir = os.path.join("log_dir", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(tensorboard_log_dir, exist_ok=True)

    config_filename = os.path.join(og.example_config_path, f"rearrange.yaml")
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    #config['env']['scene_names'] = ["d8f50afc-d93f-49f8-9170-b7b9fe880152_Bedroom-22570", "d8f50afc-d93f-49f8-9170-b7b9fe880152_Bedroom-23309", "d8f50afc-d93f-49f8-9170-b7b9fe880152_Library-20559", "d8f50afc-d93f-49f8-9170-b7b9fe880152_LivingDiningRoom-16926"]
    config['env']['scene_names'] = ["0dbb30c0-3770-472f-a938-58412841c855_Library-66976"]
    env = og.Environment(configs=config)
    rearrangement_env = FastEnv(env)
    num_envs = 1
    max_iterations = 1
    NUM_STEPS = 500
    # vec_env = VecEnvironment(num_envs, config)

    for _ in range(max_iterations):
        start_time = time.time()
        for i in range(NUM_STEPS):
            action = random.randint(0, 5)
            # profiler.enable()
            print(action)
            observations, rewards, terminates, truncates, infos = rearrangement_env.step(action)
            if terminates or truncates:
                exit(0)
            print(infos)
        import pdb;pdb.set_trace()
        
        #     profiler.disable()
        
        # profiler.dump_stats('output.prof')
            
            # for i, info in enumerate(infos):
            #     if info["done"]["success"] and done_list[i]:
            #         infos_d[i] = infos[i]
            #         done_list[i] = False
            # infos_t = infos
        # import pdb; pdb.set_trace()
        step_time = time.time() - start_time
        fps = NUM_STEPS / step_time
        # effective_fps = NUM_STEPS * len(vec_env.envs) / step_time
        effective_fps = NUM_STEPS * 1 / step_time
        print("fps", fps)
        print("effective fps", effective_fps)
        # vec_env.reset()
        rearrangement_env.reset(42)

    rearrangement_env.close()
    import pdb; pdb.set_trace()
    # Always shut down the environment cleanly at the end
    og.clear()
    # Always close the environments
    # vec_env.close()


if __name__ == "__main__":

    main()
