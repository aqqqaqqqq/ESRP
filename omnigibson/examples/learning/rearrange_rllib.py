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
from omnigibson.utils.model_utils import (
    LSTMContainingRLModule,
    LSTMContainingRLModuleWithTopDown,
    NoLSTMRLModule,
    LSTMContainingRLModule_pretrained,
)
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
)
from ray.rllib.utils.test_utils import add_rllib_example_script_args
from ray import tune
from ray.tune.registry import get_trainable_cls
from omnigibson.examples.environments.new_env import make_env
from loguru import logger
from datetime import datetime
import wandb
import json
import os

parser = add_rllib_example_script_args(default_reward=300.0, default_timesteps=2000000)

USE_TOP_DOWN = False

tune.register_env("env", make_env)

def tensor_to_serializable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()  # 转换为Python列表
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (dict, list, tuple, str, int, float, bool, type(None))):
        return obj
    else:
        return str(obj)
    
def log_episode_info(episode, **kwargs):
    all_things = {}
    all_things['infos'] = episode.get_infos()
    all_things['rewards'] = episode.get_rewards()
    all_things['actions'] = episode.get_actions()
    grasping_objs = []
    for obj in episode.get_observations():
        grasping_objs.append(obj[-1].item())
    all_things['grasping_objs'] = grasping_objs

if __name__ == "__main__":
    # --- 手动设置 resume 路径（None 表示不恢复） ---
    # resume_learner_group = "/home/user/Desktop/wq/try/third/saved_learner_group/1340"
    resume_learner_group = None  # 若不想恢复则设为 None
    # ------------------------------------------------

    args = parser.parse_args()
    run_id = f"PPO_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    module_class = LSTMContainingRLModuleWithTopDown if USE_TOP_DOWN else LSTMContainingRLModule

    base_config = (
        get_trainable_cls(args.algo)
        .get_default_config()
        .environment(
            env="env",
            env_config={"run_id": run_id, "use_top_down": USE_TOP_DOWN},
            disable_env_checking=True
        )
        .training(
            train_batch_size_per_learner=2048,
            minibatch_size=64,
            lambda_=0.95,
            kl_coeff=0.5,
            clip_param=0.1,
            vf_clip_param=10.0,
            entropy_coeff=0,
            num_epochs=10,
            lr=0.00015,
            grad_clip=100.0,
            grad_clip_by="global_norm",
            vf_loss_coeff=0.1,
        )
        .env_runners(sample_timeout_s=6000.0, num_gpus_per_env_runner=0.33, num_env_runners=3, create_env_on_local_worker=False)
        .learners(num_gpus_per_learner=0.5)
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=module_class,
                model_config={"max_seq_len": 64},
            ),
        )
        .resources()
        .callbacks(on_episode_end=log_episode_info)
    )

    algo = base_config.build()
    learner_group = algo.learner_group
    module = algo.get_module()

    # --- resume logic ---
    step = 0
    if resume_learner_group is not None:
        learner_group.restore_from_path(resume_learner_group)
        step = int(os.path.basename(resume_learner_group)) + 1
    # --------------------

    wandb.init(
        project="rearrange",
        name=run_id,
        mode='offline',
        dir="/home/user/Desktop/"
    )

    while True:
        result = algo.train()
        result_to_save = {k: v for k, v in result.items() if k != 'config'}
        wandb.log(result_to_save)

        NUM_UPDATES_PER_SAVING = 20
        if step % NUM_UPDATES_PER_SAVING == 0:
            module.save_to_path(os.path.join('/home/user/Desktop/wq/try/third/saved_models/', str(step)))
            learner_group.save_to_path(os.path.join('/home/user/Desktop/wq/try/third/saved_learner_group/', str(step)))

        step += 1
