import os
import torch as th
import omnigibson as og
from omnigibson.macros import gm
import argparse
import yaml
import matplotlib
import cProfile
from omnigibson.examples.environments.new_env import FastEnv

matplotlib.use('Agg')
profiler = cProfile.Profile() 

# We don't need object states nor transitions rules, so we disable them now, and also enable flatcache for maximum speed
gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_TRANSITION_RULES = True
gm.ENABLE_FLATCACHE = True
gm.RENDER_VIEWER_CAMERA = True

CONTROL_MODES = dict(
    random="Use autonomous random actions (default)",
    teleop="Use keyboard control",
)

# SCENES = dict(
#     Rs_int="Realistic interactive home environment (default)",
#     empty="Empty environment with no objects",
# )

def main(random_selection=False, headless=False, short_exec=False, quickstart=False):
    """
    Robot control demo with selection
    Queries the user to select a robot, the controllers, a scene and a type of input (random actions or teleop)
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    # Choose scene to load

    config_filename = config_filename = os.path.join(og.example_config_path, f"rearrange.yaml")
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    # config['env']['scene_names'] = ["d8f50afc-d93f-49f8-9170-b7b9fe880152_Bedroom-22570", "d8f50afc-d93f-49f8-9170-b7b9fe880152_Bedroom-23309", "d8f50afc-d93f-49f8-9170-b7b9fe880152_Library-20559", "d8f50afc-d93f-49f8-9170-b7b9fe880152_LivingDiningRoom-16926"]
    config['env']['scene_names'] = ["0dbb30c0-3770-472f-a938-58412841c855_Library-66976"]

    env = og.Environment(configs=config)
    rearrangement_env = FastEnv(env)

    top_down_position = th.tensor([10, 20.0, 0])
    top_down_orientation = th.tensor([0.0, 0.7, 0.7, 0.0])
    cam = og.sim.viewer_camera
    cam.set_position_orientation(top_down_position, top_down_orientation)

    # Other helpful user info
    print("Running demo.")
    print("Press ESC to quit")

    # 循环控制
    max_steps = -1 if not short_exec else 100
    step = 0
    action_empty = th.zeros(13)
    grasping_obj = None

    while step != max_steps:
        action = int(input("请输入动作编号 (0-5): "))
        # 执行动作
        obs, reward, terminated, truncated, info = rearrangement_env.step(action)
        pointgoal_rewards = info['reward']['reward_breakdown']['pointgoal']
        # arrival_rewards = info['reward']['reward_breakdown']['arrival']
        # potential_rewards = info['reward']['reward_breakdown']['potential']
        # grasping_rewards = info['reward']['reward_breakdown']['grasping']
        # living_rewards = info['reward']['reward_breakdown']['living']
        print(pointgoal_rewards)
        # import pdb; pdb.set_trace()
        step += 1

    # Always shut down the environment cleanly at the end
    og.clear()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Teleoperate a robot in a BEHAVIOR scene.")

    parser.add_argument(
        "--quickstart",
        action="store_true",
        help="Whether the example should be loaded with default settings for a quick start.",
    )
    args = parser.parse_args()
    main(quickstart=args.quickstart)