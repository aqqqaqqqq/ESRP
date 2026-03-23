# evaluate_cnn_only.py

import os
import yaml
import torch

# ONNX fallback compatibility...
if not hasattr(torch.onnx, "_CAFFE2_ATEN_FALLBACK"):
    try:
        from torch.onnx._internal.caffe2.legacy import _CAFFE2_ATEN_FALLBACK
    except ImportError:
        try:
            from torch.onnx.utils import _CAFFE2_ATEN_FALLBACK
        except ImportError:
            _CAFFE2_ATEN_FALLBACK = None
    if _CAFFE2_ATEN_FALLBACK is not None:
        torch.onnx._CAFFE2_ATEN_FALLBACK = _CAFFE2_ATEN_FALLBACK

import numpy as np
import random
from torch.distributions import Categorical
import omnigibson as og
from omnigibson.macros import gm
from omnigibson.examples.environments.new_env import FastEnv
from PIL import Image
import matplotlib
matplotlib.use('Agg')

import sys
vendored = os.path.expanduser(
    "~/.local/share/ov/pkg/isaac-sim-4.1.0/exts/"
    "omni.isaac.ml_archive/pip_prebundle"
)
sys.path = [p for p in sys.path if not p.startswith(vendored)]
from torchvision import transforms

# —— 这里改成你的纯 CNN 策略模型 —— #
from .model.MobileNet import CNNOnlyPolicy  

# 全局常量
gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_TRANSITION_RULES = True
gm.ENABLE_FLATCACHE = True
gm.RENDER_VIEWER_CAMERA = True

# 图像预处理
transform_small = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
transform_large = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def load_cnn_policy(model_path: str, num_actions: int, device: torch.device):
    """
    加载纯 CNN 策略：CNNOnlyPolicy
    """
    model = CNNOnlyPolicy(
        num_actions=num_actions,
        feature_dim=256,
        pretrained=False  # 模型权重马上会被覆盖
    ).to(device)
    state = torch.load(model_path, map_location=device)
    # 如果 checkpoint 是 compile 后带 _orig_mod 前缀，可用 strip_prefix 函数
    model.load_state_dict(state)
    model.eval()
    return model

def load_policy_strip_prefix(model_cls, model_path, *model_args, **model_kwargs):
    """
    通用加载：先 strip 掉 _orig_mod. 前缀，再 load_state_dict(strict=True)
    """
    # 1) 重建网络结构（注意不要再 compile）
    model = model_cls(*model_args, **model_kwargs)

    # 2) 读入原始 checkpoint
    raw_state = torch.load(model_path, map_location='cpu')

    # 3) strip 前缀
    new_state = {}
    for k, v in raw_state.items():
        if k.startswith("_orig_mod."):
            new_key = k[len("_orig_mod."):]
        else:
            new_key = k
        new_state[new_key] = v

    # 4) 加载
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    model.eval()
    return model

def infer_action(model, small_img, large_img, is_fetch, device):
    """
    small_img, large_img: [3,128,128] Tensor
    is_fetch: Python int 或 Tensor(1,)
    返回 sampled action, probs
    """
    small = small_img.unsqueeze(0).to(device)  # [1,3,128,128]
    large = large_img.unsqueeze(0).to(device)
    if not torch.is_tensor(is_fetch):
        is_fetch = torch.tensor([is_fetch], dtype=torch.float32, device=device)

    with torch.no_grad():
        probs = model(small, large, is_fetch)    # [1, num_actions]
    probs = probs.squeeze(0).cpu()               # [num_actions]

    # ϵ-greedy or sampling
    if random.random() < 0.1:
        action = random.randrange(probs.size(0))
    else:
        m = Categorical(probs)
        action = m.sample().item()

    return action, probs

def main():
    og.log.info("Starting online evaluation with SimpleCNN_LSTM policy")

    # Load OmniGibson config
    config_filename = os.path.join(og.example_config_path, f"rearrange.yaml")
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    scene_type = "Threed_FRONTScene"
    config["scene"]["type"] = scene_type

    threed_front_path = gm.ThreeD_FRONT_DATASET_PATH
    scene_path = os.path.join(threed_front_path, "scenes", "8148b1a7-7c15-4b53-9be3-8b5a617ba9d2_Bedroom-29109")

    config["scene"]["scene_model"] = "8148b1a7-7c15-4b53-9be3-8b5a617ba9d2_Bedroom-29109_target.json"
    config["scene"]["scene_type_path"] = scene_path

    config["env"]["modify_reload_model"] = True
    config['env']['scene_names'] = scene_names
    # Choose robot to create
    robot_name = 'Test'

    # Add the robot we want to load
    config["robots"][0]["type"] = robot_name
    config["robots"][0]["obs_modalities"] = ["rgb"]
    config["robots"][0]["action_type"] = "continuous"
    config["robots"][0]["action_normalize"] = True
    config["robots"][0]["grasping_mode"] = 'sticky'

    # Create the environment
    env = og.Environment(configs=config)
    # Setup controllers
    controller_config = {
        "base": {"name": "JointController"},
        "arm_0": {"name": "JointController", "motor_type": "effort"},
        "gripper_0": {"name": "MultiFingerGripperController"}
    }
    env.robots[0].reload_controllers(controller_config=controller_config)
    env.scene.update_initial_state()

    # Reset environment and robot
    rearr_env = FastEnv(env)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "./omnigibson/baseline/IL/checkpoint/best_cnn_policy_mobile_new_v3_no_lstm.pth"
    model = load_policy_strip_prefix(CNNOnlyPolicy, model_path, num_actions=6).to(device)

    # success_count = 0
    # total = len(scene_names)

    # for ep, scene in enumerate(scene_names, 1):
    #     # reset  
    #     obs, info = rearr_env.reset()
    #     # preprocess obs into two views
    #     def preprocess(o):
    #         arr = o[..., :3].cpu().numpy().astype(np.uint8)
    #         return Image.fromarray(arr)

    #     small_pil = preprocess(obs[0])
    #     large_pil = preprocess(obs[1])
    #     small_in  = transform_small(small_pil)
    #     large_in  = transform_large(large_pil)
    #     is_fetch  = 0

    #     action, probs = infer_action(model, small_in, large_in, is_fetch, device)
    #     done = False
    #     ep_reward = 0.0

    #     while not done:
    #         obs, reward, terminated, truncated, info = rearr_env.step(action)
    #         ep_reward += reward
    #         done = terminated or truncated

    #         if done:
    #             if terminated:
    #                 success_count += 1
    #             break

    #         small_pil = preprocess(obs[0])
    #         large_pil = preprocess(obs[1])
    #         small_in  = transform_small(small_pil)
    #         large_in  = transform_large(large_pil)
    #         # 假设 is_fetch 在 obs 中某个位置
    #         is_fetch = int(obs[...,3].mean()>0)  # 示例

    #         action, probs = infer_action(model, small_in, large_in, is_fetch, device)

    #     print(f"Episode {ep}/{total}: reward={ep_reward:.2f}, success so far {success_count}/{ep}")

    # print(f"Final success rate: {success_count}/{total}")
    total_episodes = len(scene_names)
    success_count = 0

    for ep in range(total_episodes):
        
        action_list = []
        success = []
        init_potential = []
        finish_potential = []
        all_objs = []

        try:
            obs, info = rearr_env.reset(seed=42, options=None)
        except Exception as e:
            print("Reset failed:", e)
            continue
        scene = rearr_env.env.scene.scene_model.replace("_initial.json", "")
        # get two observations, drop 4th channel
        def preprocess(obs_tensor):
            arr = obs_tensor[..., :3].cpu().numpy().astype(np.uint8)
            pil = Image.fromarray(arr)
            return pil

        # small_pil = preprocess(list(obs.values())[0])
        # large_pil = preprocess(list(obs.values())[1]) 
        # import pdb; pdb.set_trace()
        small_pil = preprocess(obs[:-1].reshape(128,128,6)[:,:,:-3])
        large_pil = preprocess(obs[:-1].reshape(128,128,6)[:,:,-3:])
        small_in = transform_small(small_pil) # 3*128*128
        large_in = transform_large(large_pil) # 3*128*128
        # import pdb; pdb.set_trace()
        is_fetch = 0

        action, probs = infer_action(model, small_in, large_in, is_fetch, device)
        # print("action:", action, "probs:", probs)
        action_list.append(action)
        done = False
        ep_reward = 0.0

        while not done:
            obs, reward, terminated, truncated, info = rearr_env.step(action)
            # import pdb;pdb.set_trace()

            if terminated:
                success.append(True)
                success_count += 1
                print('success!')
                break

            if truncated:
                success.append(False)
                print('not success')
                break

            # import pdb; pdb.set_trace()
            done = terminated or truncated
            ep_reward += reward

            small_pil = preprocess(obs[:-1].reshape(128,128,6)[:,:,:-3])
            large_pil = preprocess(obs[:-1].reshape(128,128,6)[:,:,-3:])
            small_in = transform_small(small_pil) # 3*128*128
            large_in = transform_large(large_pil) # 3*128*128
            is_fetch = obs[98304].item()
            action, probs = infer_action(model, small_in, large_in, is_fetch, device)
            print("action:", action, "probs:", probs)
            action_list.append(action)
            # reward_info = list(info.items())[0][1]
            # done_info = list(info.items())[1][1]
            # point_goal = list(done_info.items())[0][1]
            # point_goal = list(point_goal.items())[0][1]
            # point_goal = list(point_goal.items())[0][1]
            # import pdb; pdb.set_trace()

        all_objs.append(len(info['reward']['potential']['ini']))

        _init_potential = 0
        _final_potential = 0
        for k, v in info['reward']['potential']['ini'].items():
            _init_potential += v['pos']

        for k, v in info['reward']['potential']['end'].items():
            _final_potential += v['pos']

        arrival_num = info['reward']['arrival']

        init_potential.append(_init_potential)
        finish_potential.append(_final_potential)
        print(success, init_potential, finish_potential, all_objs, arrival_num)

        with open("C:/Users/Admin/Desktop/OmniGibson-Rearrange/evaluate_mobilev3_no_lstm.txt", 'a') as f:
            f.write(f"{scene}: success: {success}, init_potential: {init_potential}, finish_potential: {finish_potential}, all_objs: {all_objs}, arrival_num: {arrival_num}" + '\n')
        # check success from info
        # if point_goal:
        #     success_count += 1
        #     with open("C:/Users/Admin/Desktop/OmniGibson-Rearrange/rearrange_success_scenes.txt", 'a') as f:
        #         f.write(f"{scene}: action: {action_list} ,reward: {ep_reward}" + '\n')
        # else:
        #     with open("C:/Users/Admin/Desktop/OmniGibson-Rearrange/rearrange_unsuccess_scenes.txt", 'a') as f:
        #         f.write(f"{scene}: reward: {ep_reward}" + '\n')


        print(f"Episode {ep+1}/{total_episodes}: reward {ep_reward:.2f}, "
              f"success {success_count}/{ep+1}")
        # remove the completed scene
        scene = rearr_env.env.scene.scene_model.replace("_initial.json", "")
        scene_names.remove(scene)
        rearr_env.env.scene_names = scene_names

    print(f"Final success rate: {success_count}/{total_episodes}")

scene_names = []
# if __name__ == "__main__":
#     # headless mode for faster rendering
#     os.environ["OMNIGIBSON_HEADLESS"] = "1"
#     # build scene_names from your validation folder
#     with open("C:/Users/Admin/Desktop/OmniGibson-Rearrange/train_sampled.txt", 'r') as f:
#         for line in f:
#             parts = line.strip('\n')
#             scene_names.append(parts)
#     with open("C:/Users/Admin/Desktop/OmniGibson-Rearrange/evaluate_mobilev3_no_lstm_train.txt", 'r') as f:
#         for line in f:
#             parts = line.split(':', 1)
#             if len(parts[0]) > 0:
#                 invalid_scene = parts[0].replace("_target.json", "")
#                 # print(invalid_scene)
#                 # if invalid_scene in scene_names:
#                 scene_names.remove(invalid_scene)
#     print("scene_names:", scene_names)
#     print("len:", len(scene_names))
#     main()

if __name__ == "__main__":
    # headless mode for faster rendering
    os.environ["OMNIGIBSON_HEADLESS"] = "1"
    # build scene_names from your validation folder
    scenes_dir = "C:/Users/Admin/Desktop/OmniGibson-Rearrange/imitation_data_val"
    for fn in os.listdir(scenes_dir):
        scene_names.append(fn)
    with open("C:/Users/Admin/Desktop/OmniGibson-Rearrange/evaluate_mobilev3_no_lstm.txt", 'r') as f:
        for line in f:
            parts = line.split(':', 1)
            if len(parts[0]) > 0:
                invalid_scene = parts[0].replace("_target.json", "")
                # print(invalid_scene)
                # if invalid_scene in scene_names:
                scene_names.remove(invalid_scene)
    print("scene_names:", scene_names)
    print("len:", len(scene_names))
    main()