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
from omnigibson.examples.environments.new_env import FastEnv
from omnigibson.utils.usd_utils import CollisionAPI
import numba
from PIL import Image
from omnigibson.utils.bbox_utils import is_candidate_colliding, compute_floor_aabb, find_free_area_on_floor_random, remove_duplicate_vertices, remove_useless_points, is_point_in_polygon, sample_point_around_object, visualize_scene
from omnigibson.examples.environments.nav import find_path
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, FancyArrow
import matplotlib
matplotlib.use('Agg')


def compute_angle(p1, p2):
    """
    计算在xz平面上,从点 p1 到点 p2 的向量相对于正x轴的角度。
    坐标系规定:正x轴为0度,正z轴为 -pi/2,负z轴为 pi/2。
    
    参数:
        p1: (x1, z1)
        p2: (x2, z2)
    返回:
        angle: 弧度制角度
    """
    dx = p2[0] - p1[0]
    dz = p2[1] - p1[1]
    angle = math.atan2(dz, dx)
    return angle

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

# def main(random_selection=False, headless=False, short_exec=False, quickstart=False):
#     """
#     Prompts the user to select a type of scene and loads a turtlebot into it, generating a Point-Goal navigation
#     task within the environment.

#     It steps the environment 100 times with random actions sampled from the action space,
#     using the Gym interface, resetting it 10 times.
#     """
#     og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

#     # Load the config
#     config_filename = os.path.join(og.example_config_path, f"generate_initial_layout.yaml")
#     config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

#     scene_type = "Threed_FRONTScene"
#     config["scene"]["type"] = scene_type
#     # Choose the scene model to load
#     # scenes = get_available_3dfront_scenes()
#     # scene = choose_from_options(options=scenes, name="scene model", random_selection=random_selection)
#     threed_front_path = gm.ThreeD_FRONT_DATASET_PATH
#     scene_path = os.path.join(threed_front_path, "scenes", "8148b1a7-7c15-4b53-9be3-8b5a617ba9d2_Bedroom-29109")

#     # room = get_available_3dfront_target_scenes(scene)
#     # room_model = choose_from_options(options=room, name="room model", random_selection=random_selection) # filename(with .json)

#     config["scene"]["scene_model"] = "8148b1a7-7c15-4b53-9be3-8b5a617ba9d2_Bedroom-29109_target.json"
#     config["scene"]["scene_type_path"] = scene_path

#     # Choose robot to create
#     # robot_name = 'Test'

#     # # Add the robot we want to load
#     # # Only use RGB obs
#     # config["robots"] = [{}]
#     # config["robots"][0]["type"] = robot_name
#     # config["robots"][0]["obs_modalities"] = ["rgb"]
#     # config["robots"][0]["action_type"] = "continuous"
#     # config["robots"][0]["action_normalize"] = True
#     # config["robots"][0]["grasping_mode"] = 'sticky'
#     # config["robots"][0]["sensor_config"] = {"VisionSensor": {"sensor_kwargs": {"image_height": 512, "image_width": 512}}}

#     # Load the environment
#     env = og.Environment(configs=config)

#     # Setup controllers
#     # controller_config = {
#     #     "base": {"name": "JointController"},
#     #     "arm_0": {"name": "JointController", "motor_type": "effort"},
#     #     "gripper_0": {"name": "MultiFingerGripperController"}
#     # }
#     # env.robots[0].reload_controllers(controller_config=controller_config)

#     # Run a simple loop and reset periodically
#     max_iterations = 10 if not short_exec else 1
#     for j in range(max_iterations):
#         og.log.info("Resetting environment")
#         env.reset()
#         for i in range(1500):
#             x, _, z = env.robots[0].get_position_orientation()[0]
#             action = env.action_space.sample()
#             state, reward, terminated, truncated, info = env.step(action)
#             if terminated or truncated:
#                 og.log.info("Episode finished after {} timesteps".format(i + 1))
#                 break
#             # Test get_rearrange_objects_names
#             # objects_to_rearrange = env.task.get_rearrange_objects_names(env)
#             # all_objs = env.task.get_all_objects_names(env)
#             # print(all_objs)
#             # for obj in all_objs:
#             #     pos, _ = env.task.get_object_pos_ori(env, obj)
#             #     print(f"{obj}:", pos)
#             # print(objects_to_rearrange)
#             # if i == 200:
#             #     _, _ = env.task.sample_no_collision_point_around_object(env, objects_to_rearrange[0])
#             #     _, _ = env.task.sample_no_collision_point_around_object(env, objects_to_rearrange[1])
#             #     _, _ = env.task.sample_no_collision_point_around_object(env, objects_to_rearrange[2])
#             #     _, _ = env.task.sample_no_collision_point_around_object(env, objects_to_rearrange[3])

#             # Test get_object_pos_ori
#             # pos, ori = env.task.get_object_pos_ori(env, objects_to_rearrange[0])
#             # print("pos",pos,"ori",ori)

#             # Test get_initial_pos_ori
#             # pos, ori = env.task.get_initial_pos_ori(env, objects_to_rearrange[0])
#             # print("pos",pos,"ori",ori)

#             # Test generate_layout
            
#             # env.task.generate_layout(env)
#             # print(env.task.objects_target_pos_ori)

#             # Test get_target_pos_ori
#             # for obj_n in objects_to_rearrange:

#             #     pos, ori = env.task.get_initial_pos_ori(env, obj_n)
#             #     print("obj_n:", obj_n)
#             #     print(f"initial pos :{pos}, ori :{ori}")
#             #     pos, ori = env.task.get_target_pos_ori(env, obj_n)
#             #     print(f"target pos :{pos}, ori :{ori}")
            

#             # Test is_robot_collision
#             # robot_pos = env.task.get_robot_pos(env)
#             # print("robot_pos:", robot_pos)
#             # print(env.task.is_robot_collision(env, robot_pos[0], robot_pos[2]))


#     # Always close the environment at the end
#     env.close()


# def main(random_selection=False, headless=False, short_exec=False, quickstart=False):
#     """
#     Robot control demo with selection
#     Queries the user to select a robot, the controllers, a scene and a type of input (random actions or teleop)
#     """
#     og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

#     # Choose scene to load

#     config_filename = os.path.join(og.example_config_path, f"generate_initial_layout.yaml")
#     config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

#     scene_type = "Threed_FRONTScene"
#     config["scene"]["type"] = scene_type

#     scenes = get_available_3dfront_scenes()
#     scene = choose_from_options(options=scenes, name="scene model", random_selection=random_selection)
#     rooms = get_available_3dfront_rooms(scene)
#     room_type = choose_from_options(options=rooms, name="room type", random_selection=random_selection)
#     room = get_available_3dfront_room(scene, room_type)
#     room_model = choose_from_options(options=room, name="room model", random_selection=random_selection)
#     room_model_path = os.path.join(gm.ThreeD_FRONT_DATASET_PATH, "scenes", scene, room_type, room_model)

#     config["scene"]["scene_model"] = room_model
#     config["scene"]["scene_type_path"] = room_model_path

#     # Choose robot to create
#     robot_name = 'Test'
#     # if not quickstart:
#     #     robot_name = choose_from_options(
#     #         options=list(sorted(REGISTERED_ROBOTS.keys())), name="robot", random_selection=random_selection
#     #     )
#     # import pdb; pdb.set_trace()


#     # Add the robot we want to load
    
#     config["robots"][0]["type"] = robot_name
#     config["robots"][0]["obs_modalities"] = ["rgb"]
#     config["robots"][0]["action_type"] = "continuous"
#     config["robots"][0]["action_normalize"] = True
#     config["robots"][0]["grasping_mode"] = 'sticky'

#     # Create the environment
#     env = og.Environment(configs=config)
#     og.sim.enable_viewer_camera_teleoperation()

#     # Choose robot controller to use
#     robot = env.robots[0]
#     controller_choices = {
#         "base": "DifferentialDriveController",
#         "arm_0": "InverseKinematicsController",
#         "gripper_0": "MultiFingerGripperController",
#         "camera": "JointController",
#     }
#     if not quickstart:
#         controller_choices = choose_controllers(robot=robot, random_selection=random_selection)

#     # Choose control mode
#     # if random_selection:
#     #     control_mode = "random"
#     # elif quickstart:
#     #     control_mode = "teleop"
#     # else:
#     #     control_mode = choose_from_options(options=CONTROL_MODES, name="control mode")
#     control_mode = "teleop"
#     # Update the control mode of the robot
#     controller_config = {component: {"name": name} for component, name in controller_choices.items()}
#     controller_config['arm_0']['motor_type'] = 'effort'
#     controller_config['camera']['motor_type'] = 'effort'
#     robot.reload_controllers(controller_config=controller_config)
#     # import pdb; pdb.set_trace()
#     # import pdb; pdb.set_trace()
#     # Because the controllers have been updated, we need to update the initial state so the correct controller state
#     # is preservedworld_end_quat
#     env.scene.update_initial_state()

#     # Reset environment and robot
#     env.reset()
    
#     objects_to_rearrange = env.task.get_rearrange_objects_names(env)
#     for obj_n in objects_to_rearrange:
#         obj_rearrange = env.scene._init_objs[obj_n]
#         # obj_rearrange.fixed_base = True
#         # for link in obj_rearrange.links.values():
#         #     CollisionAPI.add_to_collision_group(
#         #         col_group=(
#         #             "fixed_base_root_links" if link == obj_rearrange.root_link else "fixed_base_nonroot_links"
#         #         ),
#         #         prim_path=link.prim_path,
#         #     )
#         # obj_rearrange.links["raw_model"].kinematic_only = True
#         obj_rearrange.links["raw_model"].set_attribute("physics:kinematicEnabled", True)
#         # import pdb;pdb.set_trace()

#     rearrangement_env = RearrangementEnv(env, robot)
#     # Create teleop controller
#     action_generator = KeyboardRobotController(robot=robot)

#     # Register custom binding to reset the environment
#     action_generator.register_custom_keymapping(
#         key=lazy.carb.input.KeyboardInput.R,
#         description="Reset the robot",
#         callback_fn=lambda: env.reset(),
#     )

#     # Print out relevant keyboard info if using keyboard teleop
#     if control_mode == "teleop":
#         action_generator.print_keyboard_teleop_info()

#     # Other helpful user info
#     print("Running demo.")
#     print("Press ESC to quit")

#     # Loop control until user quits
#     max_steps = -1 if not short_exec else 100
#     step = 0

#     # _rotate(env, robot, math.pi / 2)
#     # _translate(env, robot, 0.5)
#     # import pdb; pdb.set_trace()

#     # controller = StarterSemanticActionPrimitives(env, enable_head_tracking=False)
#     # execute_controller(controller.apply_ref(StarterSemanticActionPrimitiveSet.NAVIGATE_TO, (10, 10, 0)), env)
#     grasping_state = 1.0
#     while step != max_steps:
#         # action = (
#         #    action_generator.get_random_action() if control_mode == "random" else action_generator.get_teleop_action()
#         # )
#         if step % 400 == 0:
#             action_input = int(input("请输入动作编号 (0-5): "))
#             # action_input = random.randint(0, 4)
#             grasping_state= rearrangement_env._generate_action_tensor(action_input, grasping_state)
#             import pdb; pdb.set_trace()
#             # env.step(action=action)
#             step += 1
#         else:
#             action_input = 6
#             grasping_state= generate_action_tensor(env, robot, action_input, grasping_state)
#             robot_pos = env.task.get_robot_pos(env)
#             print(env.task.is_robot_collision(env, robot_pos[0], robot_pos[2]))
#             step += 1

#     # Always shut down the environment cleanly at the end
#     og.clear()

# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(description="Teleoperate a robot in a BEHAVIOR scene.")

#     parser.add_argument(
#         "--quickstart",
#         action="store_true",
#         help="Whether the example should be loaded with default settings for a quick start.",
#     )
#     args = parser.parse_args()
#     main(quickstart=args.quickstart)


# CONTROL_MODES = dict(
#     random="Use autonomous random actions (default)",
#     teleop="Use keyboard control",
# )
# # Don't use GPU dynamics and use flatcache for performance boost
# gm.USE_GPU_DYNAMICS = False
# gm.ENABLE_FLATCACHE = True


# def visualize_path_state(obstacles, boundaries, robot_center, robot_radius, target_idx, save_dir="visualizations", filename="path_state.png"):
#     """
#     Visualize the current state for path finding using the same parameters as find_path.
    
#     Args:
#         obstacles: List of obstacle bounding boxes
#         boundaries: Floor boundaries
#         robot_center: Robot position in XZ plane [x, z]
#         robot_radius: Robot radius
#         target_idx: Index of the target object
#         save_dir: Directory to save the visualization
#         filename: Filename for the saved visualization
#     """
#     # Create figure and axis
#     fig, ax = plt.subplots(figsize=(10, 10))
#     ax.set_aspect('equal')
    
#     # Draw floor boundaries
#     boundaries = np.array(boundaries)
#     floor_patch = Polygon(boundaries, closed=True, edgecolor='k', facecolor='none', linewidth=2, label='Floor')
#     ax.add_patch(floor_patch)
    
#     # Draw obstacles
#     for i, obstacle in enumerate(obstacles):
#         obstacle = np.array(obstacle)
#         color = 'red' if i == target_idx else 'green'
#         poly_patch = Polygon(obstacle, closed=True, edgecolor=color, facecolor=color, linewidth=1, alpha=0.5, label=f'Obstacle {i}')
#         ax.add_patch(poly_patch)
        
#         # Add obstacle index label
#         center = np.mean(obstacle, axis=0)
#         ax.text(center[0], center[1], f'{i}', ha='center', va='center', fontsize=8, color='black')
    
#     # Draw robot
#     robot_circle = Circle(robot_center, robot_radius, linewidth=2, edgecolor='blue', facecolor='blue', alpha=0.5, label='Robot')
#     ax.add_patch(robot_circle)
    
#     # Set plot limits
#     all_x = np.concatenate([boundaries[:,0], np.array([robot_center[0]])])
#     all_y = np.concatenate([boundaries[:,1], np.array([robot_center[1]])])
#     padding = 1.0
#     ax.set_xlim(all_x.min()-padding, all_x.max()+padding)
#     ax.set_ylim(all_y.min()-padding, all_y.max()+padding)
    
#     # Add labels and title
#     ax.set_xlabel('X Coordinate')
#     ax.set_ylabel('Z Coordinate')
#     ax.set_title('Path Finding State Visualization')
    
#     # Add robot position text
#     robot_text = f'Robot position ({robot_center[0]:.2f}, {robot_center[1]:.2f})'
#     ax.text(all_x.min(), all_y.max()+ 0.5, robot_text, fontsize=12, color='blue')
    
#     # Add target object text
#     target_text = f'Target object: {target_idx}'
#     ax.text(all_x.min(), all_y.max()+ 0.8, target_text, fontsize=12, color='red')
    
#     # Save the visualization
#     os.makedirs(save_dir, exist_ok=True)
#     save_path = os.path.join(save_dir, filename)
#     plt.savefig(save_path, dpi=150)
#     plt.close(fig)
#     print(f"Saved path state visualization to {save_path}")

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
    scene_path = os.path.join(threed_front_path, "scenes", "8148b1a7-7c15-4b53-9be3-8b5a617ba9d2_Bedroom-29109")

    # room = get_available_3dfront_target_scenes(scene)
    # room_model = choose_from_options(options=room, name="room model", random_selection=random_selection) # filename(with .json)

    config["scene"]["scene_model"] = "8148b1a7-7c15-4b53-9be3-8b5a617ba9d2_Bedroom-29109_target.json"
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
    
    # Because the controllers have been updated, we need to update the initial state so the correct controller state
    # is preservedworld_end_quat
    env.scene.update_initial_state()

    # Reset environment and robot
    rearrangement_env = FastEnv(env)

    # Other helpful user info
    print("Running demo.")
    print("Press ESC to quit")

    # max_iterations = len(scene_names) if not short_exec else 1
    max_iterations = 100
    for j in range(max_iterations):
        for ii in range(1000):
            rearrangement_env.step(2)
        rearrangement_env.reset()
        # print(f"@@@@@@@@@@@@@@@@@@@@@process: {total - len(scene_names) + 1}/{total}@@@@@@@@@@@@@@@@@@@@@@")
        # print(f"scene:{rearrangement_env.env.scene.scene_model}")
        # total_step = 0
        # is_rearranged_num = 0
        # total_obs = []
        # total_actions = []
        # room_model = rearrangement_env.env.scene.scene_model
        # scene_name = room_model.replace("_target.json", "")

        # robot = rearrangement_env._get_robot()
        # # Loop control until user quits
        # max_steps = -1 if not short_exec else 100

        # objects_to_rearrange = rearrangement_env.env.task.get_rearrange_objects_names(env)
        # random.shuffle(objects_to_rearrange)
        # print("objects_to_rearrange", objects_to_rearrange)

        # for obj_n in objects_to_rearrange:
        #     print("----------------------------------")
        #     print("obj_to_rearrange:", obj_n)
        #     # get floor
        #     floor = rearrangement_env.env.task.get_floor(rearrangement_env.env)
        #     floor_xyz = floor.floor_xyz

        #     num_points = len(floor_xyz) // 3
        #     floor_vertices = []
        #     for i in range(num_points):
        #         x_floor = floor_xyz[3*i]
        #         y_floor = floor_xyz[3*i+1]
        #         z_floor = floor_xyz[3*i+2]
        #         floor_vertices.append([x_floor, y_floor, z_floor])
        #     floor_vertices = remove_duplicate_vertices(floor_vertices)
        #     floor_poly = [[v[0], v[2]] for v in floor_vertices]
        #     boundaries = remove_useless_points(floor_poly)

        #     # get all obstacles
        #     object_names = rearrangement_env.env.task.get_all_objects_names(rearrangement_env.env)
        #     obstacles = [get_obj_bbox(rearrangement_env.env, object_name).tolist() for object_name in object_names]

        #     # get target id
        #     target_idx = object_names.index(obj_n)

        #     # get robot center
        #     x, _, z = robot.get_position_orientation(frame="scene")[0]
        #     robot_center = (x.item(), z.item())

        #     # get robot yaw
        #     current_yaw = robot.get_yaw().item()

        #     robot_radius = 0.4

        #     # Visualize the current path state before finding path
        #     # visualize_path_state(
        #     #     obstacles=obstacles,
        #     #     boundaries=boundaries,
        #     #     robot_center=robot_center,
        #     #     robot_radius=robot_radius,
        #     #     target_idx=target_idx,
        #     #     yaw=current_yaw,
        #     #     save_dir="visualizations",
        #     #     filename=f"path_state_before_{obj_n}.png"
        #     # )

        #     actions = []
        #     try:
        #         target_point, actions = find_path(
        #             obstacles=obstacles,
        #             boundaries=boundaries,
        #             robot_center=robot_center,
        #             robot_radius=robot_radius,
        #             target=target_idx,
        #             target_min_distance=0.5,
        #             target_max_distance=1.0,
        #             initial_yaw=current_yaw)
        #     except (ValueError, AttributeError):
        #         print(f"Cannot rearrange {obj_n}")
        #         continue
        #     navigate_step = 0
        #     navi_success = False
        #     while actions and navigate_step < 100:
        #         navigate_step += 1
        #         # Visualize the path state before each step
        #         # visualize_path_state(
        #         #     obstacles=obstacles,
        #         #     boundaries=boundaries,
        #         #     robot_center=robot_center,
        #         #     robot_radius=robot_radius,
        #         #     target_idx=target_idx,
        #         #     yaw=current_yaw,
        #         #     save_dir="visualizations",
        #         #     filename=f"path_state_step_{obj_n}_{len(actions)}.png"
        #         # )
        #         # import pdb;pdb.set_trace()
        #         try:
        #             target_point, actions = find_path(
        #                 obstacles=obstacles,
        #                 boundaries=boundaries,
        #                 robot_center=robot_center,
        #                 robot_radius=robot_radius,
        #                 target=target_idx,
        #                 target_min_distance=0.5,
        #                 target_max_distance=1.0,
        #                 initial_yaw=current_yaw)
        #         except ValueError:
                    
        #             continue
        #         print(actions)

        #         if len(actions)==0:
        #             navi_success = True
        #             print("Navigate success!")
        #             break
        #         obs, reward, done, truncated, info = rearrangement_env.step(actions[0])
        #         if info["valid_move"]:
        #             total_step += 1
        #             obs = list(obs.items())[0][1]
        #             obs = list(obs.items())[0][1]
        #             obs = list(obs.items())[0][1]
        #             total_actions.append(actions[0])
        #             total_obs.append(obs)
        #         # save_img(obs, scene_name, f"step_{total_step}")
        #         # Update robot position after the step
        #         x, _, z = robot.get_position_orientation(frame="scene")[0]
        #         robot_center = (x.item(), z.item())
        #         current_yaw = robot.get_yaw().item()
        #     if not navi_success:
        #         print(f"Cannot rearrange {obj_n}")
        #         continue
        #     # grap the object 
        #     step = 0
        #     max_steps = 10
        #     print("----------------------------------")
        #     print(f"Start grabing {obj_n}")
        #     # obj_rearrange = env.scene._init_objs[obj_n]
        #     # obj_rearrange.fixed_base = False
        #     # obj_rearrange.links["raw_model"].set_attribute("physics:kinematicEnabled", False)

        #     while step != max_steps:
        #         print("Grab step:", step)
        #         if rearrangement_env.grasping_obj is not None:
        #         # if grasping_state == -1.0:
        #             print('Grab Success!')
        #             is_rearranged_num += 1
        #             break 
        #         obs, _, _, _, info = rearrangement_env.step(4)
        #         if info["valid_move"]:
        #             total_step += 1
        #             obs = list(obs.items())[0][1]
        #             obs = list(obs.items())[0][1]
        #             obs = list(obs.items())[0][1]
        #             total_actions.append(4)
        #             total_obs.append(obs)
        #         # save_img(obs, scene_name, f"step_{total_step}")
        #         step += 1
                
        #     if rearrangement_env.grasping_obj is None:
        #         print('Grab Failed!')
        #         continue
        #     # randomly walk
        #     print("----------------------------------")
        #     print(f"Start walking with {obj_n}")
        #     step = 0
        #     max_steps = 20
        #     # firstly backward move 1 step
        #     rearrangement_env.enable_collision_detection = False
        #     obs, _, _, _, info = rearrangement_env.step(0)
        #     total_step += 1
        #     obs = list(obs.items())[0][1]
        #     obs = list(obs.items())[0][1]
        #     obs = list(obs.items())[0][1]
        #     total_actions.append(0)
        #     total_obs.append(obs)

        #     rearrangement_env.enable_collision_detection = True
        #     obs, _, _, _, info = rearrangement_env.step(0)
        #     if info["valid_move"]:
        #         total_step += 1
        #         obs = list(obs.items())[0][1]
        #         obs = list(obs.items())[0][1]
        #         obs = list(obs.items())[0][1]
        #         total_actions.append(0)
        #         total_obs.append(obs)
        #     # save_img(obs, scene_name, f"step_{total_step}")
            
        #     while step != max_steps:
        #         print("Randomly walk step:", step)
        #         # import pdb;pdb.set_trace()
        #         choices = [0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3]
        #         random_action = random.choice(choices)
        #         obs, _, _, _, info = rearrangement_env.step(random_action)
        #         if info["valid_move"]:
        #             total_step += 1
        #             obs = list(obs.items())[0][1]
        #             obs = list(obs.items())[0][1]
        #             obs = list(obs.items())[0][1]
        #             total_actions.append(random_action)
        #             total_obs.append(obs)
        #         # save_img(obs, scene_name, f"step_{total_step}")
        #         step += 1
                
        #     # release the object 
        #     print("----------------------------------")
        #     print(f"Start releasing {obj_n}")
        #     step = 0
        #     max_steps = 10
        #     while step != max_steps:
        #         print("Release step:", step)
        #         if rearrangement_env.grasping_obj is None:
        #         # if grasping_state == 1.0:
        #             print('Release Success!')
        #             break 
        #         obs, _, _, _, info = rearrangement_env.step(5)
        #         if info["valid_move"]:
        #             total_step += 1
        #             obs = list(obs.items())[0][1]
        #             obs = list(obs.items())[0][1]
        #             obs = list(obs.items())[0][1]
        #             total_actions.append(5)
        #             total_obs.append(obs)
        #         # save_img(obs, scene_name, f"step_{total_step}")
        #         step += 1
                
        #     if rearrangement_env.grasping_obj is not None:
        #     # if grasping_state == -1.0:
        #         print('Release Failed!')
        #     # obj_rearrange = env.scene._init_objs[obj_n]
        #     # obj_rearrange.fixed_base = True
        #     # obj_rearrange.links["raw_model"].set_attribute("physics:kinematicEnabled", True)
        #     print("----------------------------------")
        #     print(f"{obj_n} rearrange success!")
            

        # if is_rearranged_num == 0:
        #     with open("/home/pilab/Siqi/github/OmniGibson-Rearrange/invalid_scenes2.txt", 'a') as f:
        #         f.write(f"{rearrangement_env.env.scene.scene_model}: Not disarrange objects." + '\n')

        #     # invalid_scene_names.append(env.scene.scene_model)
        #     scene_names.remove(rearrangement_env.env.scene.scene_model.replace("_target.json", ""))
        #     rearrangement_env.env.scene_names = scene_names
        #     reset_success = False
        #     while not reset_success:
        #         try:
        #             rearrangement_env.reset()
        #             reset_success = True
        #         except (ValueError, AssertionError, TypeError) as e:
        #             with open("/home/pilab/Siqi/github/OmniGibson-Rearrange/invalid_scenes2.txt", 'a') as f:
        #                 f.write(f"{rearrangement_env.env.scene.scene_model}: No space to place a robot." + '\n')
        #             scene_names.remove(rearrangement_env.env.scene.scene_model.replace("_target.json", ""))
        #             rearrangement_env.env.scene_names = scene_names
        #     continue

        # # Save the imitation learning data as .npz
        # save_scene_data_paired(scene_name, total_actions, total_obs)
        # # Save the current layout as the initial layout (.json)
        # room_model = rearrangement_env.env.scene.scene_model
        # scene_path = rearrangement_env.env.scene.scene_dir
        # scene_target_path = os.path.join(scene_path, room_model)
        # with open(scene_target_path, 'r', encoding='utf-8') as f:
        #     data = json.load(f)
        # # update pos and ori of objects to be rearranged and fixed_base of not rearranged objects
        # object_registry = data["state"]["object_registry"]
        # object_info = data["objects_info"]["init_info"]
        # for obj_name, obj_info in object_info.items():
        #     if "furniture" in obj_name:
        #         obj_info["args"]["fixed_base"] = True
        #         if obj_info["args"]["is_to_rearrange"]:
        #             obj_info["args"]["fixed_base"] = False
        #             new_pos, new_ori = rearrangement_env.env.task.get_object_pos_ori(rearrangement_env.env, obj_name)
        #             new_pos = new_pos.tolist()
        #             new_ori = new_ori.tolist()
        #             object_registry[obj_name]["root_link"]["pos"] = new_pos
        #             object_registry[obj_name]["root_link"]["ori"] = new_ori

        # modified_room_model = room_model.replace("target", "initial")
        # scene_initial_path = os.path.join(scene_path, modified_room_model)
        # # print("scene_initial_path:", scene_initial_path)
        # with open(scene_initial_path, 'w') as f:
        #         json.dump(data, f, indent=4)
        # print(f"Initial JSON file saved as {scene_initial_path}")
        # scene_names.remove(scene_name)
        # # import pdb;pdb.set_trace()
        # # for ii in range(1000):
        # #     rearrangement_env.step(0)
        # rearrangement_env.env.scene_names = scene_names

        # reset_success = False
        # while not reset_success:
        #     try:
        #         rearrangement_env.reset()
        #         reset_success = True
        #     except (ValueError, AssertionError, TypeError) as e:
        #         with open("/home/pilab/Siqi/github/OmniGibson-Rearrange/invalid_scenes2.txt", 'a') as f:
        #             f.write(f"{rearrangement_env.env.scene.scene_model}: No space to place a robot." + '\n')
        #         scene_names.remove(rearrangement_env.env.scene.scene_model.replace("_target.json", ""))
        #         rearrangement_env.env.scene_names = scene_names
                
        # og.clear()

scene_names = []
total = 0
valid_scene = 0

if __name__ == "__main__":
    os.environ["OMNIGIBSON_HEADLESS"] = "1"
    threed_front_path = gm.ThreeD_FRONT_DATASET_PATH
    scenes_dir_path = os.path.join(threed_front_path, "scenes")
    parent_folder = scenes_dir_path

    for entry in os.listdir(scenes_dir_path):
        total += 1
        entry_path = os.path.join(scenes_dir_path, entry)
        if os.path.isdir(entry_path):
            scene_names.append(entry)
            is_valid = False
            fils = 0
            for file in os.listdir(entry_path):
                fils += 1
                if "initial" in file:
                    is_valid = True
                    file_path = os.path.join(entry_path, file)
                    # print(entry)
                    # scene_names.remove(entry)
                    # os.remove(file_path)
            if fils == 3:
                valid_scene += 1
            # if is_valid:
            #     valid_scene += 1
            # else:
            #     shutil.rmtree(entry_path)
    print("valid num:", valid_scene)
    print("invalid nums:", len(scene_names))

    # remove invalid scenes 
    # file_path = "/home/pilab/Siqi/github/OmniGibson-Rearrange/invalid_scenes2.txt"
    # with open(file_path, 'r', encoding='utf-8') as file:
    #     for line in file:
    #         parts = line.split(':', 1)
    #         if len(parts[0]) > 0:
    #             invalid_scene = parts[0].replace("_target.json", "")
    #             scene_names.remove(invalid_scene)
    # print(scene_names)
    # print(f"process: {total - len(scene_names) + 1}/{total}")
    # import pdb;pdb.set_trace()
    # scene_names = ["f1e40a9c-1791-4411-9b5f-5383ecee8c65_Bedroom-8601"]
    
    main()