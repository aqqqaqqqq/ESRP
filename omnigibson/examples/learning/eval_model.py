"""Example of implementing and configuring a custom (torch) LSTM containing RLModule.

This example:
    - demonstrates how you can subclass the TorchRLModule base class and set up your
    own LSTM-containing NN architecture by overriding the `setup()` method.
    - shows how to override the 3 forward methods: `_forward_inference()`,
    `_forward_exploration()`, and `forward_train()` to implement your own custom forward
    logic(s), including how to handle STATE in- and outputs to and from these calls.
    - explains when each of these 3 methods is called by RLlib or the users of your
    RLModule.
    - shows how you then configure an RLlib Algorithm such that it uses your custom
    RLModule (instead of a default RLModule).

We implement a simple LSTM layer here, followed by a series of Linear layers.
After the last Linear layer, we add fork of 2 Linear (non-activated) layers, one for the
action logits and one for the value function output.

We test the LSTM containing RLModule on the StatelessCartPole environment, a variant
of CartPole that is non-Markovian (partially observable). Only an RNN-network can learn
a decent policy in this environment due to the lack of any velocity information. By
looking at one observation, one cannot know whether the cart is currently moving left or
right and whether the pole is currently moving up or down).


How to run this script
----------------------
`python [script file name].py --enable-new-api-stack`

For debugging, use the following additional command line options
`--no-tune --num-env-runners=0`
which should allow you to set breakpoints anywhere in the RLlib code and
have the execution stop there for inspection and debugging.

For logging to your WandB account, use:
`--wandb-key=[your WandB API key] --wandb-project=[some project name]
--wandb-run-name=[optional: WandB run name (within the defined project)]`


Results to expect
-----------------
You should see the following output (during the experiment) in your console:

"""
import numpy as np
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
import torch
from ray.rllib.core.rl_module.rl_module import RLModule
from omnigibson.utils.model_utils import (
    LSTMContainingRLModule
    )
from ray.rllib.core.columns import Columns
import omnigibson as og
import yaml 
import os
import gymnasium as gym
from gymnasium.spaces import Box
import omnigibson as og
from omnigibson.examples.environments.new_env import FastEnv
from omnigibson.macros import gm
from loguru import logger

gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_TRANSITION_RULES = True
gm.ENABLE_FLATCACHE = True
gm.RENDER_VIEWER_CAMERA = True

CONTROL_MODES = dict(
    random="Use autonomous random actions (default)",
    teleop="Use keyboard control",
)

MODEL_PATH = '/home/user/Desktop/saved_models/18000'
NEW_LOG = '//home/user/Desktop/wq/try/more_metric.log'
SPLIT = [(0, 200)]

THIS_SPLIT = 0

def main(random_selection=False, headless=False, short_exec=False, quickstart=False):
    spec = RLModuleSpec(LSTMContainingRLModule, observation_space=Box(low=0, high=255, shape=(98306,), dtype=np.uint8), action_space = gym.spaces.Discrete(6))
    rl_module = spec.build()
    rl_module.restore_from_path(path = MODEL_PATH)

    config_filename = os.path.join(og.example_config_path, f"rearrange.yaml")
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    threed_front_path = gm.ThreeD_FRONT_DATASET_PATH
    scene_path = os.path.join(threed_front_path, "scenes", "8148b1a7-7c15-4b53-9be3-8b5a617ba9d2_Bedroom-29109")

    config["scene"]["scene_model"] = "8148b1a7-7c15-4b53-9be3-8b5a617ba9d2_Bedroom-29109_target.json"
    config["scene"]["scene_type_path"] = scene_path

    scene_names = []
    content = []
    with open(NEW_LOG, 'r') as f:
        content = f.read()  # Entire file content as one string

    with open('/home/user/Desktop/rl/omnigibson/data/test_all_data.txt', 'r') as f:
        for line in f:
            scene_name = line.rstrip('\n')
            scene_names.append(scene_name)
        
    # scene_names = scene_names[SPLIT[THIS_SPLIT][0]:SPLIT[THIS_SPLIT][1]]

    #for scene_name in scene_names:
    #    if scene_name in content:
    #        scene_names.remove(scene_name)
    
    scene_names = [
    	s for s in scene_names
    	if s not in content
    ]


    print(scene_names)
    print("len:", len(scene_names))
    config['env']['scene_names'] = scene_names

    env = og.Environment(configs=config)
    rearrangement_env = FastEnv(env)

    # 循环控制
    step = 0

    success = None
    all_objs = []
    success_num = 0
    each_arrival = []
    each_potential = []

    logger.add(NEW_LOG, encoding="utf-8")

    # iterate over the test set
    while len(rearrangement_env.env.scene_names) > 0:
        eval_dict = {}
        obs, _ = rearrangement_env.reset(seed = None)
        init_state = {'h': torch.tensor(rl_module.get_initial_state()['h']).unsqueeze(0), 'c': torch.tensor(rl_module.get_initial_state()['c']).unsqueeze(0)}
        _init = {
            Columns.STATE_IN: init_state,
            Columns.OBS: obs.unsqueeze(0).unsqueeze(0),
        }

        infos = []

        while True:
            _scene_name = rearrangement_env.env._scene_name
            inference_result = rl_module.forward_inference(_init)
            # sample from action dist
            probabilities = inference_result[Columns.ACTION_DIST_INPUTS]
            action = torch.multinomial(torch.softmax(probabilities.view(-1), 0), 1)
            # print(probabilities, action)
            obs, reward, terminated, truncated, info = rearrangement_env.step(int(action))
            _init = {Columns.OBS: obs.unsqueeze(0).unsqueeze(0),
                        Columns.STATE_IN: inference_result[Columns.STATE_OUT]}
    
            infos.append(info)
            step += 1
            if terminated:
                success = True
                success_num +=1
                print('success!')
                break

            if truncated:
                success = False
                print('not success')
                break

        all_objs.append(len(infos[-1]['reward']['potential']))

        _init_potential = 0
        _final_potential = 0
        for k, v in infos[-1]['reward']['potential']['ini'].items():
            _init_potential += v['pos']

        for k, v in infos[-1]['reward']['potential']['end'].items():
            _final_potential += v['pos']
    

        arrival_num = infos[-1]['reward']['arrival']
        assert len(arrival_num) == 1
        arrival_num = next(iter(arrival_num))

        eval_dict['step'] = infos[-1]['episode_length']
        eval_dict['success'] = success
        eval_dict['scene_name'] = _scene_name
        eval_dict['obj_num'] = len(infos[-1]['reward']['potential']['ini'])
        eval_dict['arrival_num'] = arrival_num
        eval_dict['init_potential'] = _init_potential
        eval_dict['fini_potential'] = _final_potential

        eval_dict['first_grasp_step'] = infos[-1]['eval_metrics']['first_grasp_step']
        eval_dict['released_before_target'] = infos[-1]['eval_metrics']['released_before_target']
        eval_dict['grasp_events'] = infos[-1]['eval_metrics']['grasp_events']
        eval_dict['release_events'] = infos[-1]['eval_metrics']['release_events']
        # eval_dict['object_event_history'] = infos[-1]['eval_metrics']['object_event_history']


        each_arrival.append(arrival_num / eval_dict['obj_num']) 
        each_potential.append(_final_potential / _init_potential)

        logger.info(eval_dict)

        rearrangement_env.env.scene_names.remove(_scene_name)

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
