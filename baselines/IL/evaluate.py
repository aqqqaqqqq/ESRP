# import yaml
# import os
# import random
# import torch as th
# import math
# import json
# import numpy as np
# import time
# import shutil
# import omnigibson as og
# from omnigibson.macros import gm
# from omnigibson.examples.environments.new_env import FastEnv
# from torch.distributions import Categorical
# from .model.EfficientNet_b0 import PolicyNet
# import numba
# from PIL import Image
# import matplotlib.pyplot as plt
# from matplotlib.patches import Polygon, Circle, FancyArrow
# import matplotlib
# matplotlib.use('Agg')
# import sys, os
# vendored = os.path.expanduser(
#     "~/.local/share/ov/pkg/isaac-sim-4.1.0/exts/"
#     "omni.isaac.ml_archive/pip_prebundle"
# )
# sys.path = [p for p in sys.path if not p.startswith(vendored)]
# from torchvision import transforms


# def main(random_selection=False, headless=False, short_exec=False):
#     """
#     Robot control demo with selection
#     Queries the user to select a robot, the controllers, a scene and a type of input (random actions or teleop)
#     """
#     og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

#     # Choose scene to load

#     config_filename = os.path.join(og.example_config_path, f"rearrange.yaml")
#     config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

#     scene_type = "Threed_FRONTScene"
#     config["scene"]["type"] = scene_type

#     threed_front_path = gm.ThreeD_FRONT_DATASET_PATH
#     scene_path = os.path.join(threed_front_path, "scenes", "8148b1a7-7c15-4b53-9be3-8b5a617ba9d2_Bedroom-29109")

#     config["scene"]["scene_model"] = "8148b1a7-7c15-4b53-9be3-8b5a617ba9d2_Bedroom-29109_target.json"
#     config["scene"]["scene_type_path"] = scene_path

#     config["env"]["modify_reload_model"] = True
#     # config['env']['scene_names'] = ["d8f50afc-d93f-49f8-9170-b7b9fe880152_Library-20559", "d8f50afc-d93f-49f8-9170-b7b9fe880152_MasterBedroom-24026", "0003d406-5f27-4bbf-94cd-1cff7c310ba1_Bedroom-54672", "3d9f406a-4032-44ed-9f55-064f14fe2250_SecondBedroom-67719"]
#     # config['env']['scene_names'] = ["d8f50afc-d93f-49f8-9170-b7b9fe880152_Library-20559", "d8f50afc-d93f-49f8-9170-b7b9fe880152_MasterBedroom-24026"]
#     config['env']['scene_names'] = scene_names
#     # Choose robot to create
#     robot_name = 'Test'

#     # Add the robot we want to load
#     config["robots"][0]["type"] = robot_name
#     config["robots"][0]["obs_modalities"] = ["rgb"]
#     config["robots"][0]["action_type"] = "continuous"
#     config["robots"][0]["action_normalize"] = True
#     config["robots"][0]["grasping_mode"] = 'sticky'

#     # Create the environment
#     env = og.Environment(configs=config)
#     # Setup controllers
#     controller_config = {
#         "base": {"name": "JointController"},
#         "arm_0": {"name": "JointController", "motor_type": "effort"},
#         "gripper_0": {"name": "MultiFingerGripperController"}
#     }
#     env.robots[0].reload_controllers(controller_config=controller_config)

    
#     # Because the controllers have been updated, we need to update the initial state so the correct controller state
#     # is preservedworld_end_quat
#     env.scene.update_initial_state()

#     # Reset environment and robot
#     rearrangement_env = FastEnv(env)

#     # Other helpful user info
#     print("Running demo.")
#     print("Press ESC to quit")

#     # max_iterations = len(scene_names) if not short_exec else 1

#     device = th.device("cuda" if th.cuda.is_available() else "cpu")
#     model = load_policy("/home/pilab/Siqi/github/OmniGibson-Rearrange/omnigibson/baseline/IL/best_model_2.pth", num_actions=6, device=device)

#     episode = len(scene_names)
#     acc = 0
#     for j in range(episode):
#         print(f"--------------------------{j+1}/{episode}---------------------------------")
#         reward_total = 0  
#         truncated = False
#         terminated = False
#         point_goal = False
#         # action = 5
#         try:
#             obs = rearrangement_env.reset()
#         except ValueError as e:
#             print(e)
#             room_model = rearrangement_env.env.scene.scene_model
#             scene_name = room_model.replace("_initial.json", "")
#             scene_names.remove(scene_name)
#             rearrangement_env.env.scene_names = scene_names
#             continue
#         obs1 = list(obs.items())[0][1] # torch.Size([128, 128, 4])
#         obs2 = list(obs.items())[1][1]
        
#         img_np1 = obs1.cpu().numpy()
#         img_uint81 = img_np1.astype(np.uint8)
#         img_pil1 = Image.fromarray(img_uint81, mode='RGB')
#         img1 = transform_small(img_pil1)

#         img_np2 = obs2.cpu().numpy()
#         img_uint82 = img_np2.astype(np.uint8)
#         img_pil2 = Image.fromarray(img_uint82, mode='RGB')
#         img2 = transform_large(img_pil2)
        
#         action, _ = infer_action(model, img1, img2, device)
#         # import pdb; pdb.set_trace()
#         while not(truncated or terminated):

#             obs, reward, terminated, truncated, info = rearrangement_env.step(action)
#             # import pdb; pdb.set_trace()
#             obs1 = list(obs.items())[0][1] # torch.Size([128, 128, 4])
#             obs2 = list(obs.items())[1][1]
            
#             img_np1 = obs1.cpu().numpy()
#             img_uint81 = img_np1.astype(np.uint8)
#             img_pil1 = Image.fromarray(img_uint81, mode='RGB')
#             img1 = transform_small(img_pil1)

#             img_np2 = obs2.cpu().numpy()
#             img_uint82 = img_np2.astype(np.uint8)
#             img_pil2 = Image.fromarray(img_uint82, mode='RGB')
#             img2 = transform_large(img_pil2)
            
#             action, _ = infer_action(model, img1, img2, device)
#             # import pdb; pdb.set_trace()

#             reward_info = list(info.items())[0][1]
#             done_info = list(info.items())[1][1]
#             point_goal = list(done_info.items())[0][1]
#             point_goal = list(point_goal.items())[0][1]
#             point_goal = list(point_goal.items())[0][1]
#             # import pdb; pdb.set_trace()
            
#             reward_total += reward

#             if point_goal:
#                 acc += 1
#         print(f"episode {j+1}: reward :{reward_total} is_rearrange :{acc}/{episode}")
#         room_model = rearrangement_env.env.scene.scene_model
#         scene_name = room_model.replace("_initial.json", "")
#         scene_names.remove(scene_name)
#         rearrangement_env.env.scene_names = scene_names
        


# def load_policy(model_path: str, num_actions: int, device: th.device):
#     """
#     创建网络、加载权重、切换到 eval 模式。
#     """
#     model = PolicyNet(num_actions=num_actions, pretrained=False).to(device)
#     state_dict = th.load(model_path, map_location=device)
#     model.load_state_dict(state_dict)
#     model.eval()
#     return model

# def infer_action(model, small_img: th.Tensor, large_img: th.Tensor, device: th.device):
#     """
#     small_img: (3, 128,128) FloatTensor, 取值 [0,1]
#     large_img: (3, 224,224) FloatTensor, 取值 [0,1]
#     返回：
#       action: int, 从分布中采样得到的动作
#       probs:  Tensor shape [num_actions], 对应的 softmax 概率
#     """
#     small = small_img.unsqueeze(0).to(device)  # [1,3,128,128]
#     large = large_img.unsqueeze(0).to(device)  # [1,3,224,224]

#     with th.no_grad():
#         probs = model(small, large)             # [1, num_actions]
#     probs = probs.squeeze(0).cpu()              # [num_actions]

#     # 采样
#     m = Categorical(probs)
#     action = m.sample().item()
#     print("action:", action)
#     return action, probs

# transform_small = transforms.Compose([
#                   transforms.ToTensor(),
#                   transforms.Resize((128, 128)),
#                   ])
# transform_large = transforms.Compose([
#                   transforms.ToTensor(),
#                   transforms.Resize((224, 224)), 
#                   ])
# scene_names = []
# total = 0
# if __name__ == "__main__":
#     os.environ["OMNIGIBSON_HEADLESS"] = "1"
#     threed_front_path = gm.ThreeD_FRONT_DATASET_PATH
#     scenes_dir_path = "/home/pilab/Siqi/github/imitation_data_val"


#     for entry in os.listdir(scenes_dir_path):
#         total += 1
#         scene_names.append(entry.replace(".npz", ""))

#     main()


import os
import yaml
import argparse
import torch
# 兼容不同版本：把内部落脚点补齐到老的名字
if not hasattr(torch.onnx, "_CAFFE2_ATEN_FALLBACK"):
    try:
        # PyTorch >= 2.0
        from torch.onnx._internal.caffe2.legacy import _CAFFE2_ATEN_FALLBACK
    except ImportError:
        try:
            # PyTorch 1.11 ~ 1.13
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
import matplotlib.pyplot as plt

# filter out vendored torchvision in Isaac Sim
import sys
vendored = os.path.expanduser(
    "~/.local/share/ov/pkg/isaac-sim-4.1.0/exts/"
    "omni.isaac.ml_archive/pip_prebundle"
)
sys.path = [p for p in sys.path if not p.startswith(vendored)]
from torchvision import transforms

from .model.MobileNet_lstm import SimpleCNN_LSTM  # your simple CNN + LSTM

gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_TRANSITION_RULES = True
gm.ENABLE_FLATCACHE = True
gm.RENDER_VIEWER_CAMERA = True
# Transforms for both views (resize to 128×128)
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


def compute_camera_height_from_polygon(sensor, polygon: np.ndarray) -> float:
    """
    计算俯视相机高度，使给定多边形在画面中尽可能充满  renderProductResolution 指定的图像。

    1) 先根据 renderProductResolution 裁剪 cameraAperture（mm）以匹配图像宽高比，
    2) 再把 mm 单位转换到场景单位，最后
    3) 用 pinhole 模型算出高度。

    Args:
        polygon (np.ndarray): 多边形顶点列表，形状 (N,2)，单位为场景单位。

    Returns:
        float: 建议相机高度，单位为场景单位。
    """
    # —————————————————————————————
    # 1. 读属性
    # —————————————————————————————
    # 物理传感器（aperture）宽/高，单位 mm
    ap_w_mm, ap_h_mm = sensor.camera_parameters["cameraAperture"].tolist()
    print("ap_w_mm, ap_h_mm:", ap_w_mm, ap_h_mm)
    # 渲染产物的像素分辨率
    img_w_px, img_h_px = sensor.camera_parameters["renderProductResolution"].tolist()
    print("img_w_px, img_h_px:", img_w_px, img_h_px)
    # 场景单位到米的转换：1 su = metersPerSceneUnit 米
    m_per_su = sensor.camera_parameters["metersPerSceneUnit"]
    # 焦距 mm
    f_mm = sensor.camera_parameters["cameraFocalLength"]
    print("f_mm:", f_mm)

    # —————————————————————————————
    # 2. 按分辨率裁剪 aperture
    # —————————————————————————————
    img_ar    = img_w_px / img_h_px
    sensor_ar = ap_w_mm  / ap_h_mm

    if sensor_ar > img_ar:
        # 传感器比图像更“宽”，裁掉左右 → 用新的 sensor 宽度
        eff_h_mm = ap_h_mm
        eff_w_mm = ap_h_mm * img_ar
    else:
        # 传感器比图像更“高”，裁掉上下 → 用新的 sensor 高度
        eff_w_mm = ap_w_mm
        eff_h_mm = ap_w_mm / img_ar

    # —————————————————————————————
    # 3. 转换到场景单位
    # —————————————————————————————
    mm_to_su   = (1e-3 / m_per_su)
    sensor_w_su = eff_w_mm * mm_to_su
    sensor_h_su = eff_h_mm * mm_to_su
    focal_su    = f_mm    * mm_to_su

    # —————————————————————————————
    # 4. 多边形世界大小
    # —————————————————————————————
    center       = polygon.mean(axis=0)
    offsets      = polygon - center
    width_world  = np.max(np.abs(offsets[:, 0])) * 2
    height_world = np.max(np.abs(offsets[:, 1])) * 2

    # —————————————————————————————
    # 5. 针孔模型算高度
    # —————————————————————————————
    h_x = focal_su * width_world  / sensor_w_su
    h_y = focal_su * height_world / sensor_h_su

    return float(max(h_x, h_y))

def load_policy(model_path: str, num_actions: int, device: torch.device):
    model = SimpleCNN_LSTM(num_actions=num_actions, lstm_hidden=256).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
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

def infer_action(model, small_img, large_img, is_fetch, hidden, device):
    """
    small_img: Tensor shape (3,128,128)
    large_img: Tensor shape (3,128,128)
    hidden: (h, c) LSTM states
    """
    small = small_img.unsqueeze(0).to(device)  # [1,3,128,128]
    large = large_img.unsqueeze(0).to(device)  # [1,3,128,128]
    # 如果 is_fetch 是 Python int，先转为 Tensor
    if not torch.is_tensor(is_fetch):
        is_fetch = torch.tensor([is_fetch], dtype=torch.float32, device=device)

    with torch.no_grad():
        probs, hidden = model.forward_step(small, large, is_fetch, hidden)
    probs = probs.squeeze(0).cpu()  # [num_actions]
    epsilon = 0.1
    if random.random() < epsilon:
        num_actions = probs.shape[0]
        action = random.randrange(num_actions)
    else:
        m = Categorical(probs)
        action = m.sample().item()

    return action, probs, hidden

def main(exp_name, random_selection=False, headless=True, short_exec=False):
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
    # model = load_policy(
    #     "./omnigibson/baseline/IL/checkpoint/best_lstm_policy_mobile_new_v3_64.pth",
    #     num_actions=6,
    #     device=device
    # )

    model = load_policy_strip_prefix(
        SimpleCNN_LSTM,
        f"./omnigibson/baseline/IL/experiments/checkpoints/{exp_name}/best_model.pth",
        num_actions=6,
        feature_dim=256,
        lstm_hidden=256,
        pretrained=False  # 预训练标志无所谓，因为马上 load 掉
    ).to(device)

    total_episodes = len(scene_names)
    success_count = 0

    for ep in range(total_episodes):

        # reset hidden state at episode start
        hidden = model.init_hidden(batch_size=1, device=device)
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
        scene = rearr_env.env.scene.scene_model.replace("_initial_5.json", "")
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

        action, probs, hidden = infer_action(model, small_in, large_in, is_fetch, hidden, device)
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
            action, probs, hidden = infer_action(model, small_in, large_in, is_fetch, hidden, device)
            # print("action:", action, "probs:", probs)
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

        with open(f"./omnigibson/baseline/IL/experiments/results/{exp_name}/test.txt", 'a') as f:
            f.write(f"{scene}: success: {success}, init_potential: {init_potential}, finish_potential: {finish_potential}, all_objs: {all_objs}, arrival_num: {arrival_num}" + '\n')

        print(f"Episode {ep+1}/{total_episodes}: reward {ep_reward:.2f}, "
              f"success {success_count}/{ep+1}")
        # remove the completed scene
        scene = rearr_env.env.scene.scene_model.replace("_initial_5.json", "")
        scene_names.remove(scene)
        rearr_env.env.scene_names = scene_names

    print(f"Final success rate: {success_count}/{total_episodes}")

scene_names = []

if __name__ == "__main__":
    # headless mode for faster rendering
    # os.environ["OMNIGIBSON_HEADLESS"] = "1"

    parser = argparse.ArgumentParser(description='Train LSTM model for imitation learning')
    parser.add_argument('--exp_name', type=str, required=True,
                        help='Experiment name (used for checkpoint and log directories)')
    args = parser.parse_args()
    exp_name = args.exp_name
    print(f"Experiment name: {exp_name}")

    with open("./test_data.txt", 'r') as f:
        for line in f:
            parts = line.strip('\n')
            scene_names.append(parts)
    test_results_dir = f"./omnigibson/baseline/IL/experiments/results/{exp_name}"
    test_results_path = f"{test_results_dir}/test.txt"
    if not os.path.exists(test_results_dir):
        os.makedirs(test_results_dir, exist_ok=True)
    if not os.path.exists(test_results_path):
        with open(test_results_path, 'w') as f:
            pass
    with open(f"{test_results_path}", 'r') as f:
        for line in f:
            parts = line.split(':', 1)
            if len(parts[0]) > 0:
                invalid_scene = parts[0].replace("_target.json", "")
                # print(invalid_scene)
                if invalid_scene in scene_names:
                    scene_names.remove(invalid_scene)
    print("scene_names:", scene_names)
    print("todo:", len(scene_names))
    main(exp_name)
