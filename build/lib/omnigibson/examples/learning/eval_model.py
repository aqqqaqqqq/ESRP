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
    LSTMContainingRLModule,
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

gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_TRANSITION_RULES = True
gm.ENABLE_FLATCACHE = True
gm.RENDER_VIEWER_CAMERA = True

CONTROL_MODES = dict(
    random="Use autonomous random actions (default)",
    teleop="Use keyboard control",
)

def main(random_selection=False, headless=False, short_exec=False, quickstart=False):
    spec = RLModuleSpec(LSTMContainingRLModule, observation_space=Box(low=0, high=255, shape=(98306,), dtype=np.uint8), action_space = gym.spaces.Discrete(6))
    rl_module = spec.build()
    rl_module.restore_from_path(path = '/home/pilab/Desktop/wq/rearrange/saved_models/68')
    import pdb; pdb.set_trace()

    config_filename = os.path.join(og.example_config_path, f"rearrange.yaml")
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    threed_front_path = gm.ThreeD_FRONT_DATASET_PATH
    scene_path = os.path.join(threed_front_path, "scenes", "8148b1a7-7c15-4b53-9be3-8b5a617ba9d2_Bedroom-29109")

    config["scene"]["scene_model"] = "8148b1a7-7c15-4b53-9be3-8b5a617ba9d2_Bedroom-29109_target.json"
    config["scene"]["scene_type_path"] = scene_path

    scene_names = []
    total = 0
    scene_dir = os.path.join(threed_front_path, "scenes")
    for entry in os.listdir(scene_dir):
        total += 1
        scene_names.append(entry)
    config['env']['scene_names'] = scene_names

    env = og.Environment(configs=config)
    rearrangement_env = FastEnv(env)

    # 循环控制
    max_steps = -1 if not short_exec else 100
    step = 0

    success = []
    init_potential = []
    finish_potential = []
    all_objs = []
    finished_objs = []

    # iterate over the test set
    while True:
        obs, _ = rearrangement_env.reset(seed = None)
        init_state = {'h': torch.tensor(rl_module.get_initial_state()['h']).unsqueeze(0), 'c': torch.tensor(rl_module.get_initial_state()['c']).unsqueeze(0)}
        _init = {
            Columns.STATE_IN: init_state,
            Columns.OBS: obs.unsqueeze(0).unsqueeze(0),
        }

        infos = []

        while True:
            inference_result = rl_module.forward_inference(_init)
            # sample from action dist
            probabilities = inference_result[Columns.ACTION_DIST_INPUTS]
            action = torch.multinomial(torch.softmax(probabilities.view(-1), 0), 1)
            # print(probabilities, action)
            # import pdb; pdb.set_trace()
            obs, reward, terminated, truncated, info = rearrangement_env.step(int(action))
            _init = {Columns.OBS: obs.unsqueeze(0).unsqueeze(0),
                        Columns.STATE_IN: inference_result[Columns.STATE_OUT]}
    
            infos.append(info)
            step += 1
            if terminated:
                success.append(True)
                print('success!')
                break

            if truncated:
                success.append(False)
                print('not success')
                break

        all_objs.append(len(infos[-1]['reward']['potential']))
        finished_objs = []

        _init_potential = 0
        _final_potential = 0
        for k, v in infos[0]['reward']['potential'].items():
            _init_potential += v['pos']

        for k, v in infos[-1]['reward']['potential'].items():
            _final_potential += v['pos']

        init_potential.append(_init_potential)
        finish_potential.append(_final_potential)
        print(success, init_potential, finish_potential, all_objs)


            

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