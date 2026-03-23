import yaml
import os
import random
import torch as th
import math
import json
import numpy as np
import time
import shutil
import omnigibson as og
from omnigibson.macros import gm
from omnigibson.utils.asset_utils import get_available_g_scenes, get_available_og_scenes, get_available_3dfront_scenes, get_available_3dfront_rooms, get_available_3dfront_room, get_available_3dfront_target_scenes
from omnigibson.utils.bbox_utils import is_candidate_colliding, compute_floor_aabb, find_free_area_on_floor_random, remove_duplicate_vertices, remove_useless_points, is_point_in_polygon, sample_point_around_object, visualize_scene
import omnigibson.lazy as lazy
from omnigibson.robots import REGISTERED_ROBOTS
from omnigibson.utils.ui_utils import KeyboardRobotController, choose_from_options
import omnigibson.utils.transform_utils as T
from omnigibson.examples.robots.control_example_test import choose_controllers, generate_action_tensor, navigate_to, CONTROL_MODES
from omnigibson.utils.bbox_utils import sample_point_around_object, get_obj_bbox
from omnigibson.examples.learning.new_rearrangement_demo import RearrangementEnv
from omnigibson.examples.environments.new_env import FastEnv, OccupancyInfo
from omnigibson.utils.usd_utils import CollisionAPI
from shapely.geometry import Polygon as pol

import numba
from PIL import Image
from omnigibson.utils.bbox_utils import is_candidate_colliding, compute_floor_aabb, find_free_area_on_floor_random, remove_duplicate_vertices, remove_useless_points, is_point_in_polygon, sample_point_around_object, visualize_scene
from omnigibson.examples.environments.nav import find_path
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, FancyArrow
import matplotlib
matplotlib.use('Agg')


def save_img(img_tensor, scene_name, filename):

    img_np = img_tensor.cpu().numpy()
    img_uint8 = img_np.astype(np.uint8)
    img_pil = Image.fromarray(img_uint8, mode='RGBA')
    save_dir = os.path.join(os.getcwd(), scene_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{filename}.png")
    img_pil.save(save_path)

def save_scene_data_paired(scene_name, actions, imgs, out_dir='.'):
    """
    将单个场景的 (action, obs) 对打包成一个结构化数组，
    并保存为 {scene_name}.npz。
    """
    # 1) 转为 NumPy
    actions_arr = np.array(actions, dtype=np.int32)  # (N,)
    imgs_arr    = np.stack([img.cpu().numpy() for img in imgs], axis=0)  # (N,128,128,4)

    # 2) 构造结构化数组 dtype：
    #    field 'action' 存 int32，field 'img' 存 uint8 或 float32，视你的 tensor dtype 而定
    dtype = [
        ('action', np.int32),
        ('obs',    imgs_arr.dtype, (128, 128, 4))
    ]
    paired = np.zeros(actions_arr.shape[0], dtype=dtype)
    paired['action'] = actions_arr
    paired['obs']    = imgs_arr

    # 3) 保存
    out_dir = os.path.join(os.getcwd(), "imitation_data")
    os.makedirs(out_dir, exist_ok=True)
    fn = os.path.join(out_dir, f'{scene_name}.npz')
    # 把整个结构化数组存到 key='data'
    np.savez_compressed(fn, data=paired)
    print(f"Saved {scene_name}: {paired.shape[0]} frames → {fn}")

def visualize_path_state(obstacles,
                         boundaries,
                         robot_center,
                         robot_radius,
                         target_idx,
                         yaw,
                         save_dir="visualizations",
                         filename="path_state.png"):
    """
    Visualize the current state for path finding, including robot orientation.

    Args:
        obstacles: List of obstacle bounding boxes (each an array-like of shape [N,2]).
        boundaries: Floor boundaries as an array-like of shape [M,2].
        robot_center: Robot position in XZ plane [x, z].
        robot_radius: Robot radius.
        target_idx: Index of the target object.
        ori: Quaternion of robot orientation (x, y, z, w).
        save_dir: Directory to save the visualization.
        filename: Filename for the saved visualization.
    """


    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')

    # Draw floor boundaries
    boundaries = np.array(boundaries)
    floor_patch = Polygon(boundaries, closed=True, edgecolor='k',
                          facecolor='none', linewidth=2, label='Floor')
    ax.add_patch(floor_patch)

    # Draw obstacles
    for i, obstacle in enumerate(obstacles):
        obstacle = np.array(obstacle)
        color = 'red' if i == target_idx else 'green'
        poly_patch = Polygon(obstacle, closed=True, edgecolor=color,
                             facecolor=color, linewidth=1, alpha=0.5,
                             label=f'Obstacle {i}')
        ax.add_patch(poly_patch)

        # Obstacle index label
        center = obstacle.mean(axis=0)
        ax.text(center[0], center[1], f'{i}', ha='center', va='center',
                fontsize=8, color='black')

    # Draw robot body
    robot_circle = Circle(robot_center, robot_radius, linewidth=2,
                          edgecolor='blue', facecolor='blue', alpha=0.5,
                          label='Robot')
    ax.add_patch(robot_circle)

    # Draw orientation arrow
    arrow_length = robot_radius * 1.5
    dx = arrow_length * np.cos(yaw)
    dy = arrow_length * np.sin(yaw)
    ax.add_patch(FancyArrow(robot_center[0],
                            robot_center[1],
                            dx, dy,
                            width=robot_radius*0.3,
                            length_includes_head=True,
                            head_width=robot_radius*0.6,
                            head_length=robot_radius*0.6,
                            color='blue'))

    # Set plot limits with padding
    all_x = np.concatenate([boundaries[:, 0], [robot_center[0]]])
    all_y = np.concatenate([boundaries[:, 1], [robot_center[1]]])
    padding = 1.0
    ax.set_xlim(all_x.min() - padding, all_x.max() + padding)
    ax.set_ylim(all_y.min() - padding, all_y.max() + padding)

    # Labels and title
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Z Coordinate')
    ax.set_title('Path Finding State Visualization')

    # Annotate robot position and orientation
    ax.text(all_x.min(), all_y.max() + 0.5,
            f'Robot position: ({robot_center[0]:.2f}, {robot_center[1]:.2f})',
            fontsize=12, color='blue')
    ax.text(all_x.min(), all_y.max() + 0.8,
            f'Robot yaw (deg): {np.degrees(yaw):.1f}',
            fontsize=12, color='blue')
    ax.text(all_x.min(), all_y.max() + 1.1,
            f'Target object: {target_idx}',
            fontsize=12, color='red')

    # Save figure
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

    print(f"Saved path state visualization to {save_path}")

label_reverse = {0:1, 1:0, 2:3, 3:2, 4:5, 5:4}

def main(random_selection=False, headless=False, short_exec=False):
    """
    Robot control demo with selection
    Queries the user to select a robot, the controllers, a scene and a type of input (random actions or teleop)
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    # Choose scene to load
    config_filename = os.path.join(og.example_config_path, f"generate_initial_layout.yaml")
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    scene_type = "Threed_FRONTScene"
    config["scene"]["type"] = scene_type
    # Choose the scene model to load
    # scenes = get_available_3dfront_scenes()
    # scene = choose_from_options(options=scenes, name="scene model", random_selection=random_selection)
    threed_front_path = gm.ThreeD_FRONT_DATASET_PATH
    scene_path = os.path.join(threed_front_path, "scenes", "0a8d471a-2587-458a-9214-586e003e9cf9_LivingDiningRoom-4017")

    # room = get_available_3dfront_target_scenes(scene)
    # room_model = choose_from_options(options=room, name="room model", random_selection=random_selection) # filename(with .json)

    config["scene"]["scene_model"] = "0a8d471a-2587-458a-9214-586e003e9cf9_LivingDiningRoom-4017_target.json"
    config["scene"]["scene_type_path"] = scene_path

    config["env"]["modify_reload_model"] = True
    # config['env']['scene_names'] = ["d8f50afc-d93f-49f8-9170-b7b9fe880152_Library-20559", "d8f50afc-d93f-49f8-9170-b7b9fe880152_MasterBedroom-24026", "0003d406-5f27-4bbf-94cd-1cff7c310ba1_Bedroom-54672", "3d9f406a-4032-44ed-9f55-064f14fe2250_SecondBedroom-67719"]
    # config['env']['scene_names'] = ["d8f50afc-d93f-49f8-9170-b7b9fe880152_Library-20559", "d8f50afc-d93f-49f8-9170-b7b9fe880152_MasterBedroom-24026"]
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
    # Choose robot controller to use
    robot = env.robots[0]
    env.scene.update_initial_state()

    # Reset environment and robot
    rearrangement_env = FastEnv(env)
    # Other helpful user info
    print("Running demo.")
    print("Press ESC to quit")

    max_iterations = len(scene_names) if not short_exec else 1
    # max_iterations = 100
    for j in range(max_iterations):
        # action = [4, 3, 1, 1, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 5, 0, 0, 0]
        # for ii in range(1000):
        #     rearrangement_env.step(4)
            
        # rearrangement_env.reset()
        print(f"@@@@@@@@@@@@@@@@@@@@@process: {total - len(scene_names) + 1}/{total}@@@@@@@@@@@@@@@@@@@@@@")
        print(f"scene:{rearrangement_env.env.scene.scene_model}")
        total_step = 0
        is_rearranged_num = 0
        total_obs = []
        total_actions = []
        room_model = rearrangement_env.env.scene.scene_model
        scene_name = room_model.replace("_target.json", "")

        robot = rearrangement_env._get_robot()
        # Loop control until user quits
        max_steps = -1 if not short_exec else 100

        objects_to_rearrange = rearrangement_env.env.task.get_rearrange_objects_names(env)
        # random.shuffle(objects_to_rearrange)
        print("objects_to_rearrange", objects_to_rearrange)
        obstacles_target_bbox = OccupancyInfo.get_obstacles(rearrangement_env.env)
        not_disrearrange_objects = []
        # import pdb; pdb.set_trace()
        for obj_n in objects_to_rearrange:
            print("----------------------------------")
            print("obj_to_rearrange:", obj_n)
            # get floor
            floor = rearrangement_env.env.task.get_floor(rearrangement_env.env)
            floor_xyz = floor.floor_xyz

            num_points = len(floor_xyz) // 3
            floor_vertices = []
            for i in range(num_points):
                x_floor = floor_xyz[3*i]
                y_floor = floor_xyz[3*i+1]
                z_floor = floor_xyz[3*i+2]
                floor_vertices.append([x_floor, y_floor, z_floor])
            floor_vertices = remove_duplicate_vertices(floor_vertices)
            floor_poly = [[v[0], v[2]] for v in floor_vertices]
            boundaries = remove_useless_points(floor_poly)

            # get all obstacles
            object_names = rearrangement_env.env.task.get_all_objects_names(rearrangement_env.env)
            obstacles = [get_obj_bbox(rearrangement_env.env, object_name).tolist() for object_name in object_names]

            # get target id
            target_idx = object_names.index(obj_n)

            # get robot center
            x, _, z = robot.get_position_orientation(frame="scene")[0]
            robot_center = (x.item(), z.item())

            # get robot yaw
            current_yaw = robot.get_yaw().item()

            robot_radius = 0.4

            # Visualize the current path state before finding path
            # visualize_path_state(
            #     obstacles=obstacles,
            #     boundaries=boundaries,
            #     robot_center=robot_center,
            #     robot_radius=robot_radius,
            #     target_idx=target_idx,
            #     yaw=current_yaw,
            #     save_dir="visualizations",
            #     filename=f"path_state_before_{obj_n}.png"
            # )

            actions = []
            try:
                # import pdb; pdb.set_trace()
                target_point, actions = find_path(
                    obstacles=obstacles,
                    boundaries=boundaries,
                    robot_center=robot_center,
                    robot_radius=robot_radius,
                    target=target_idx,
                    target_min_distance=0.5,
                    target_max_distance=1.0,
                    initial_yaw=current_yaw)
            except (ValueError, AttributeError) as e:
                print(f"Cannot rearrange {obj_n}")
                print(e)
                not_disrearrange_objects.append(obj_n)
                continue
            navigate_step = 0
            navi_success = False
            while actions and navigate_step < 100:
                navigate_step += 1
                # Visualize the path state before each step
                # visualize_path_state(
                #     obstacles=obstacles,
                #     boundaries=boundaries,
                #     robot_center=robot_center,
                #     robot_radius=robot_radius,
                #     target_idx=target_idx,
                #     yaw=current_yaw,
                #     save_dir="visualizations",
                #     filename=f"path_state_step_{obj_n}_{len(actions)}.png"
                # )
                # import pdb;pdb.set_trace()
                try:
                    target_point, actions = find_path(
                        obstacles=obstacles,
                        boundaries=boundaries,
                        robot_center=robot_center,
                        robot_radius=robot_radius,
                        target=target_idx,
                        target_min_distance=0.5,
                        target_max_distance=1.0,
                        initial_yaw=current_yaw)
                except ValueError:
                    
                    continue
                print(actions)

                if len(actions)==0:
                    navi_success = True
                    print("Navigate success!")
                    break
                obs, reward, done, truncated, info = rearrangement_env.step(actions[0])
                if info["valid_move"]:
                    total_step += 1
                    obs = list(obs.items())[0][1]
                    obs = list(obs.items())[0][1]
                    obs = list(obs.items())[0][1]
                    total_actions.append(actions[0])
                    total_obs.append(obs)

                # Update robot position after the step
                x, _, z = robot.get_position_orientation(frame="scene")[0]
                robot_center = (x.item(), z.item())
                current_yaw = robot.get_yaw().item()
            if not navi_success:
                print(f"Cannot rearrange {obj_n}")
                not_disrearrange_objects.append(obj_n)
                continue
            # grap the object 
            step = 0
            max_steps = 10
            print("----------------------------------")
            print(f"Start grabing {obj_n}")

            while step != max_steps:
                print("Grab step:", step)
                if rearrangement_env.grasping_obj is not None:
                    print('Grab Success!')
                    break 
                obs, _, _, _, info = rearrangement_env.step(4)
                if info["valid_move"]:
                    total_step += 1
                    obs = list(obs.items())[0][1]
                    obs = list(obs.items())[0][1]
                    obs = list(obs.items())[0][1]
                    total_actions.append(4)
                    total_obs.append(obs)
                # save_img(obs, scene_name, f"step_{total_step}")
                step += 1
                
            if rearrangement_env.grasping_obj is None:
                print('Grab Failed!')
                continue
            # randomly walk
            print("----------------------------------")
            print(f"Start walking with {obj_n}")

            # move backward until invalid
            print("starting rule walking.")
            obs, _, _, _, info = rearrangement_env.step(0)
            if info["valid_move"]:
                print("rule walking: 0")
                total_step += 1
                obs = list(obs.items())[0][1]
                obs = list(obs.items())[0][1]
                obs = list(obs.items())[0][1]
                total_actions.append(0)
                total_obs.append(obs)
            while info["valid_move"]:
                obs, _, _, _, info = rearrangement_env.step(0)
                if info["valid_move"]:
                    print("rule walking: 0")
                    total_step += 1
                    obs = list(obs.items())[0][1]
                    obs = list(obs.items())[0][1]
                    obs = list(obs.items())[0][1]
                    total_actions.append(0)
                    total_obs.append(obs)
            # turn right or left twice
            choices = [2, 3]
            random_action = random.choice(choices)
            obs, _, _, _, info = rearrangement_env.step(random_action)
            if info["valid_move"]:
                print(f"rule walking: {random_action}")
                total_step += 1
                obs = list(obs.items())[0][1]
                obs = list(obs.items())[0][1]
                obs = list(obs.items())[0][1]
                total_actions.append(random_action)
                total_obs.append(obs)
            obs, _, _, _, info = rearrangement_env.step(random_action)
            if info["valid_move"]:
                print(f"rule walking: {random_action}")
                total_step += 1
                obs = list(obs.items())[0][1]
                obs = list(obs.items())[0][1]
                obs = list(obs.items())[0][1]
                total_actions.append(random_action)
                total_obs.append(obs)
            # move forward until invalid
            obs, _, _, _, info = rearrangement_env.step(1)
            if info["valid_move"]:
                print("rule walking: 1")
                total_step += 1
                obs = list(obs.items())[0][1]
                obs = list(obs.items())[0][1]
                obs = list(obs.items())[0][1]
                total_actions.append(1)
                total_obs.append(obs)
            while info["valid_move"]:
                obs, _, _, _, info = rearrangement_env.step(1)
                if info["valid_move"]:
                    print("rule walking: 1")
                    total_step += 1
                    obs = list(obs.items())[0][1]
                    obs = list(obs.items())[0][1]
                    obs = list(obs.items())[0][1]
                    total_actions.append(1)
                    total_obs.append(obs)
            obstacles_bbox = OccupancyInfo.get_obstacles(rearrangement_env.env)
            # import pdb; pdb.set_trace()
            obj_n_bbox = obstacles_bbox[obj_n] 
            obj_n_tar_bbox = obstacles_target_bbox[obj_n]
            poly1 = pol(obj_n_bbox)
            poly2 = pol(obj_n_tar_bbox)
            # if intersect , random walk until no overlap
            if poly1.intersects(poly2):
                
                step = 0
                max_steps = 20
                pre_action = -1
                
                while step != max_steps and poly1.intersects(poly2):
                    print("Randomly walk step:", step)
                    # import pdb;pdb.set_trace()
                    choices = [0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3]
                    random_action = random.choice(choices)
                    reverse_current_action = label_reverse[random_action]
                    while reverse_current_action == pre_action:
                        random_action = random.choice(choices)
                        reverse_current_action = label_reverse[random_action]
                    pre_action = random_action

                    obs, _, _, _, info = rearrangement_env.step(random_action)
                    if info["valid_move"]:
                        total_step += 1
                        obs = list(obs.items())[0][1]
                        obs = list(obs.items())[0][1]
                        obs = list(obs.items())[0][1]
                        total_actions.append(random_action)
                        total_obs.append(obs)

                        obstacles_bbox = OccupancyInfo.get_obstacles(rearrangement_env.env)
                        obj_n_bbox = obstacles_bbox[obj_n] 
                        poly1 = pol(obj_n_bbox)
                    
                    step += 1
                
            # release the object 
            print("----------------------------------")
            print(f"Start releasing {obj_n}")
            step = 0
            max_steps = 10
            while step != max_steps:
                print("Release step:", step)
                if rearrangement_env.grasping_obj is None:
                # if grasping_state == 1.0:
                    print('Release Success!')
                    break 
                obs, _, _, _, info = rearrangement_env.step(5)
                if info["valid_move"]:
                    total_step += 1
                    obs = list(obs.items())[0][1]
                    obs = list(obs.items())[0][1]
                    obs = list(obs.items())[0][1]
                    total_actions.append(5)
                    total_obs.append(obs)
                # save_img(obs, scene_name, f"step_{total_step}")
                step += 1
                
            if rearrangement_env.grasping_obj is not None:
                print('Release Failed!')
            print("----------------------------------")
            if poly1.intersects(poly2):
                print(f"Cannot rearrange {obj_n}")
                not_disrearrange_objects.append(obj_n)
            else:
                is_rearranged_num += 1
                print(f"{obj_n} rearrange success!")
            

        if is_rearranged_num == 0:
            with open("C:/Users/Admin/Desktop/OmniGibson-Rearrange/invalid_scenes_new.txt", 'a') as f:
                f.write(f"{rearrangement_env.env.scene.scene_model}: Not disarrange objects." + '\n')

            # invalid_scene_names.append(env.scene.scene_model)
            if rearrangement_env.env.scene.scene_model.replace("_target.json", "") in scene_names:
                scene_names.remove(rearrangement_env.env.scene.scene_model.replace("_target.json", ""))
            rearrangement_env.env.scene_names = scene_names
            reset_success = False
            while not reset_success:
                try:
                    rearrangement_env.reset()
                    reset_success = True
                except (ValueError, AssertionError, TypeError, AttributeError) as e:
                    with open("C:/Users/Admin/Desktop/OmniGibson-Rearrange/invalid_scenes_new.txt", 'a') as f:
                        f.write(f"{rearrangement_env.env.scene.scene_model}: No space to place a robot." + '\n')
                    if rearrangement_env.env.scene.scene_model.replace("_target.json", "") in scene_names:
                        scene_names.remove(rearrangement_env.env.scene.scene_model.replace("_target.json", ""))
                    rearrangement_env.env.scene_names = scene_names
            continue
        # Save!!!!!!!!!!!!!!!!
        room_model = rearrangement_env.env.scene.scene_model
        scene_path = rearrangement_env.env.scene.scene_dir

        # save robot final pos and ori
        robot_pos, robot_ori = rearrangement_env._get_robot().get_position_orientation(frame='scene')
        # import pdb;pdb.set_trace()
        scene_robot_pos_ori_path = os.path.join(scene_path, 'robot_pos_ori.txt')
        with open(scene_robot_pos_ori_path, 'w') as f:
            f.write(f"pos:{robot_pos.tolist()}" + "\n" + f"ori:{robot_ori.tolist()}")

        # Save the imitation learning data as .npz
        save_scene_data_paired(scene_name, total_actions, total_obs)

        # Save the current layout as the initial layout (.json)
        scene_target_path = os.path.join(scene_path, room_model)
        with open(scene_target_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # update pos and ori of objects to be rearranged and fixed_base of not rearranged objects
        object_registry = data["state"]["object_registry"]
        object_info = data["objects_info"]["init_info"]

        for obj_name, obj_info in object_info.items():
            if "furniture" in obj_name:
                obj_info["args"]["fixed_base"] = True
                # set is_to_rearrange=false if it is not disrearranged!
                if obj_name in not_disrearrange_objects:
                    obj_info["args"]["is_to_rearrange"] = False
                if obj_info["args"]["is_to_rearrange"]:

                    obj_info["args"]["target_bbox"] = obstacles_target_bbox[obj_name].tolist()
                    new_pos, new_ori = rearrangement_env.env.task.get_object_pos_ori(rearrangement_env.env, obj_name)
                    new_pos = new_pos.tolist()
                    new_ori = new_ori.tolist()
                    object_registry[obj_name]["root_link"]["pos"] = new_pos
                    object_registry[obj_name]["root_link"]["ori"] = new_ori

        modified_room_model = room_model.replace("target", "initial")
        scene_initial_path = os.path.join(scene_path, modified_room_model)
        # print("scene_initial_path:", scene_initial_path)
        with open(scene_initial_path, 'w') as f:
                json.dump(data, f, indent=4)
        print(f"Initial JSON file saved as {scene_initial_path}")
        if scene_name in scene_names:
            scene_names.remove(scene_name)
        rearrangement_env.env.scene_names = scene_names

        reset_success = False
        while not reset_success:
            try:
                rearrangement_env.reset()
                reset_success = True
            except (ValueError, AssertionError, TypeError) as e:
                with open("C:/Users/Admin/Desktop/OmniGibson-Rearrange/invalid_scenes_new.txt", 'a') as f:
                    f.write(f"{rearrangement_env.env.scene.scene_model}: No space to place a robot." + '\n')
                if rearrangement_env.env.scene.scene_model.replace("_target.json", "") in scene_names:
                    scene_names.remove(rearrangement_env.env.scene.scene_model.replace("_target.json", ""))
                rearrangement_env.env.scene_names = scene_names
                

scene_names = []
total = 0
valid_scene = 0

if __name__ == "__main__":
    os.environ["OMNIGIBSON_HEADLESS"] = "1"
    threed_front_path = gm.ThreeD_FRONT_DATASET_PATH
    scenes_dir_path = os.path.join(threed_front_path, "scenes")
    parent_folder = scenes_dir_path

    for file in os.listdir("C:/Users/Admin/Desktop/OmniGibson-Rearrange/imitation_data"):
        scene_name = os.path.splitext(os.path.basename(file))[0]
        scene_names.append(scene_name)

    for entry in os.listdir(scenes_dir_path):
        total += 1
        entry_path = os.path.join(scenes_dir_path, entry)
        if os.path.isdir(entry_path):
            # scene_names.append(entry)
            is_valid = False
            fils = 0
            for file in os.listdir(entry_path):
                fils += 1
                if "initial" in file and entry in scene_names:
                    is_valid = True
                    file_path = os.path.join(entry_path, file)
                    # print(entry)
                    scene_names.remove(entry)
                    # os.remove(file_path)
            if fils == 4:
                valid_scene += 1
            # if is_valid:
            #     valid_scene += 1
            # else:
            #     shutil.rmtree(entry_path)
    print("valid num:", valid_scene)

    # remove invalid scenes 
    file_path = "C:/Users/Admin/Desktop/OmniGibson-Rearrange/invalid_scenes_new.txt"
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.split(':', 1)
            if len(parts[0]) > 0:
                invalid_scene = parts[0].replace("_target.json", "")
                # print(invalid_scene)
                if invalid_scene in scene_names:
                    scene_names.remove(invalid_scene)
    # # print(scene_names)
    # print("to-do:", len(scene_names))
    # print(f"process: {total - len(scene_names) + 1}/{total}")
    # import pdb;pdb.set_trace()
    # scene_names = ["d5d0600c-07dd-47dc-bd61-d18c0d464904_MasterBedroom-22479"]
    # scene_names= ["9929e0fc-e018-434a-a51e-65ee7459182c_OtherRoom-11678"]
    main()