import yaml
import os
import random
import torch as th
import math
import json
import omnigibson as og
from omnigibson.macros import gm
from omnigibson.utils.asset_utils import get_available_g_scenes, get_available_og_scenes, get_available_3dfront_scenes, get_available_3dfront_rooms, get_available_3dfront_room
import omnigibson.lazy as lazy
from omnigibson.robots import REGISTERED_ROBOTS
from omnigibson.utils.ui_utils import KeyboardRobotController, choose_from_options
import omnigibson.utils.transform_utils as T
from omnigibson.examples.robots.control_example_test import choose_controllers, generate_action_tensor, navigate_to, CONTROL_MODES
from omnigibson.utils.bbox_utils import sample_point_around_object
import math

def remove_roll_pitch(q):
    """
    去除输入四元数 q 中除绕 y 轴旋转外的分量，返回一个只包含 yaw 的四元数。
    假设 q 的格式为 (x, y, z, w)。
    """
    x, y, z, w = q
    # 提取旋转矩阵中对应的分量
    r00 = 1 - 2*y*y - 2*z*z
    r02 = 2*x*z + 2*y*w
    # 计算 yaw 角（以弧度为单位），满足：正 x 轴为 0 度
    yaw = math.atan2(r02, r00)
    # 构造仅绕 y 轴旋转的四元数
    new_y = math.sin(yaw / 2)
    new_w = math.cos(yaw / 2)
    return (0.0, new_y, 0.0, new_w)

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
#     robot.reset()

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
#             grasping_state= generate_action_tensor(env, robot, action_input, grasping_state)
#             # import pdb; pdb.set_trace()
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

#     # # Choose control mode
#     # if random_selection:
#     #     control_mode = "random"
#     # elif quickstart:
#     #     control_mode = "teleop"
#     # else:
#     #     control_mode = choose_from_options(options=CONTROL_MODES, name="control mode")

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

#     # Update the simulator's viewer camera's pose so it points towards the robot
#     og.sim.viewer_camera.set_position_orientation(
#         position=th.tensor([1.46949, -3.97358, 2.21529]),
#         orientation=th.tensor([0.56829048, 0.09569975, 0.13571846, 0.80589577]),
#     )

#     # Reset environment and robot
#     env.reset()
#     robot.reset()

#     # # Create teleop controller
#     # action_generator = KeyboardRobotController(robot=robot)

#     # # Register custom binding to reset the environment
#     # action_generator.register_custom_keymapping(
#     #     key=lazy.carb.input.KeyboardInput.R,
#     #     description="Reset the robot",
#     #     callback_fn=lambda: env.reset(),
#     # )

#     # # Print out relevant keyboard info if using keyboard teleop
#     # if control_mode == "teleop":
#     #     action_generator.print_keyboard_teleop_info()

#     # Other helpful user info
#     print("Running demo.")
#     print("Press ESC to quit")

#     # Loop control until user quits
#     max_steps = -1 if not short_exec else 100
#     grasping_state = 1.0

#     objects_to_rearrange = env.task.get_rearrange_objects_names(env)
#     random.shuffle(objects_to_rearrange)
#     print("objects_to_rearrange", objects_to_rearrange)

#     for obj_n in objects_to_rearrange:
#         x, z = env.task.sample_no_collision_point_around_object(env, obj_n)
#         print(f"no-collision point around {obj_n} is: ({x},{z}).")
#         step = 0
#         # get pos around the object to be rearranged
#         while step != max_steps:
            
#             a = navigate_to(env, robot, x, z, 0)
#             if a is not None and len(a) == 0:
#                 print('Navigate Success!')
#                 break 
#             print("action:", a)
#             # og.log.info(f"action: {a}")
#             # og.log.info(f"robot.get_position_orientation(): {robot.get_position_orientation()[0][0], robot.get_position_orientation()[0][2], T.quat2euler(robot.get_position_orientation()[1])[1]}")
#             grasping_state = generate_action_tensor(env, robot, a[0], grasping_state)
#             # og.log.info(f"robot.get_position_orientation(): {robot.get_position_orientation()[0][0], robot.get_position_orientation()[0][2], T.quat2euler(robot.get_position_orientation()[1])[1]}")

#             step += 1
#             print("Navigate step:", step)
#         # grap the object 
#         step = 0
#         while step != max_steps:
#             if grasping_state == -1.0:
#                 print('Grap Success!')
#                 break 
#             grasping_state = generate_action_tensor(env, robot, 4, grasping_state)
#             step += 1
#             print("Grap step:", step)
#         # randomly walk
#         step = 0
#         max_steps = 10
#         while step != max_steps:
#             random_action = random.randint(0, 3)
#             grasping_state = generate_action_tensor(env, robot, random_action, grasping_state)
#             step += 1
#             print("Randomly walk step:", step)
#         # release the object 
#         step = 0
#         while step != max_steps:
#             if grasping_state == 1.0:
#                 print('Release Success!')
#                 break 
#             grasping_state = generate_action_tensor(env, robot, 5, grasping_state)
#             step += 1
#             print("Release step:", step)
#         print(f"{obj_n} rearrange success!")
        

#     # Save the current layout as the initial layout (.json)
#     scene_target_path = os.path.join(room_model_path, "json", f"{room_model}_target.json")
#     with open(scene_target_path, 'r', encoding='utf-8') as f:
#         data = json.load(f)
#     # update pos and ori of objects to be rearranged and fixed_base of not rearranged objects
#     object_registry = data["state"]["object_registry"]
#     object_info = data["objects_info"]["init_info"]
#     for obj_name, obj_info in object_info.items():
#         if "furniture" in obj_name:
#             obj_info["args"]["fixed_base"] = False
#             if obj_info["args"]["is_to_rearrange"]:
#                 new_pos, new_ori = env.task.get_object_pos_ori(env, obj_name)
#                 new_pos = new_pos.tolist()
#                 new_ori = new_ori.tolist()
#                 object_registry[obj_name]["root_link"]["pos"], object_registry[obj_name]["root_link"]["ori"] = new_pos, new_ori
    

#     scene_initial_path = os.path.join(room_model_path, "json", f"{room_model}_initial.json")
#     # print("scene_initial_path:", scene_initial_path)
#     with open(scene_initial_path, 'w') as f:
#             json.dump(data, f, indent=4)
#     print(f"Modified JSON file saved as {scene_initial_path}")
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

CONTROL_MODES = dict(
    random="Use autonomous random actions (default)",
    teleop="Use keyboard control",
)
# Don't use GPU dynamics and use flatcache for performance boost
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True

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
    angle = math.atan2(-dz, dx)
    return angle

def main(random_selection=False, headless=False, short_exec=False, quickstart=False):
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

    scenes = get_available_3dfront_scenes()
    scene = choose_from_options(options=scenes, name="scene model", random_selection=random_selection)
    rooms = get_available_3dfront_rooms(scene)
    room_type = choose_from_options(options=rooms, name="room type", random_selection=random_selection)
    room = get_available_3dfront_room(scene, room_type)
    room_model = choose_from_options(options=room, name="room model", random_selection=random_selection)
    room_model_path = os.path.join(gm.ThreeD_FRONT_DATASET_PATH, "scenes", scene, room_type, room_model)

    config["scene"]["scene_model"] = room_model
    config["scene"]["scene_type_path"] = room_model_path

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
    og.sim.enable_viewer_camera_teleoperation()

    # Choose robot controller to use
    robot = env.robots[0]
    controller_choices = {
        "base": "DifferentialDriveController",
        "arm_0": "InverseKinematicsController",
        "gripper_0": "MultiFingerGripperController",
        "camera": "JointController",
    }
    if not quickstart:
        controller_choices = choose_controllers(robot=robot, random_selection=random_selection)

    # # Choose control mode
    # if random_selection:
    #     control_mode = "random"
    # elif quickstart:
    #     control_mode = "teleop"
    # else:
    #     control_mode = choose_from_options(options=CONTROL_MODES, name="control mode")

    # Update the control mode of the robot
    controller_config = {component: {"name": name} for component, name in controller_choices.items()}
    controller_config['arm_0']['motor_type'] = 'effort'
    controller_config['camera']['motor_type'] = 'effort'
    robot.reload_controllers(controller_config=controller_config)
    # import pdb; pdb.set_trace()
    # import pdb; pdb.set_trace()
    # Because the controllers have been updated, we need to update the initial state so the correct controller state
    # is preservedworld_end_quat
    env.scene.update_initial_state()

    # Update the simulator's viewer camera's pose so it points towards the robot
    og.sim.viewer_camera.set_position_orientation(
        position=th.tensor([1.46949, -3.97358, 2.21529]),
        orientation=th.tensor([0.56829048, 0.09569975, 0.13571846, 0.80589577]),
    )

    # Reset environment and robot
    env.reset()
    robot.reset()

    # # Create teleop controller
    # action_generator = KeyboardRobotController(robot=robot)

    # # Register custom binding to reset the environment
    # action_generator.register_custom_keymapping(
    #     key=lazy.carb.input.KeyboardInput.R,
    #     description="Reset the robot",
    #     callback_fn=lambda: env.reset(),
    # )

    # # Print out relevant keyboard info if using keyboard teleop
    # if control_mode == "teleop":
    #     action_generator.print_keyboard_teleop_info()

    # Other helpful user info
    print("Running demo.")
    print("Press ESC to quit")

    # Loop control until user quits
    grasping_state = 1.0

    objects_to_rearrange = env.task.get_rearrange_objects_names(env)
    print("objects_to_rearrange", objects_to_rearrange)
    # ramdomly walk and grab object
    _step = 0
    while _step != 4:
        print("------------------------------------------------------------------")
        print(f"Epoch {_step}")

        print("----------------------------------")
        print(f"Start randomly walking 1")
        step = 0
        max_steps = -1
        while step != max_steps:
            
            if grasping_state == -1.0:
                print('Grab Success!')
                break 
            random_action = random.randint(0, 4)
            grasping_state = generate_action_tensor(env, robot, random_action, grasping_state)
            
            step += 1
            print("Random walk 1 step:", step)
        print("----------------------------------")
        print(f"Start walking with obj")
        step = 0
        max_steps = 20
        while step != max_steps:
            
            random_action = random.randint(0, 3)
            grasping_state = generate_action_tensor(env, robot, random_action, grasping_state)
            
            step += 1
            print("Grab walk step:", step)
        print("----------------------------------")
        print(f"Start releasing")
        step = 0
        max_steps = -1
        while step != max_steps:
            if grasping_state == 1.0:
                print('Release Success!')
                break 
            grasping_state = generate_action_tensor(env, robot, 5, grasping_state)
            step += 1
            print("Release step:", step)
        print("----------------------------------")
        print(f"Start randomly walking 2")
        step = 0
        max_steps = 20
        while step != max_steps:
            
            random_action = random.randint(0, 3)
            grasping_state = generate_action_tensor(env, robot, random_action, grasping_state)
            
            step += 1
            print("Random walk 2 step:", step)

        _step += 1
    

    # Save the current layout as the initial layout (.json)

    new_p = {}
    new_o = {}
    scene_target_path = os.path.join(room_model_path, "json", f"{room_model}_target.json")
    with open(scene_target_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # update pos and ori of objects to be rearranged and fixed_base of not rearranged objects
    object_registry = data["state"]["object_registry"]
    object_info = data["objects_info"]["init_info"]
    for obj_name, obj_info in object_info.items():
        if "furniture" in obj_name:
            obj_info["args"]["fixed_base"] = False
            if obj_info["args"]["is_to_rearrange"]:
                new_pos, new_ori = env.task.get_object_pos_ori(env, obj_name)
                new_pos = new_pos.tolist()
                new_ori = tuple(new_ori.tolist())
                new_ori = list(remove_roll_pitch(new_ori))
                object_registry[obj_name]["root_link"]["pos"], object_registry[obj_name]["root_link"]["ori"] = new_pos, new_ori
                new_p[obj_name] = new_pos
                new_o[obj_name] = new_ori


    scene_initial_path = os.path.join(room_model_path, "json", f"{room_model}_initial.json")
    # print("scene_initial_path:", scene_initial_path)
    with open(scene_initial_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Modified JSON file saved as {scene_initial_path}")

    json_initial_path = os.path.join(room_model_path, "bbox", f"{room_model}_initial.json")
    json_target_path = os.path.join(room_model_path, "bbox", f"{room_model}_target.json")
    import shutil
    shutil.copy(json_target_path, json_initial_path)
    # print("scene_initial_path:", scene_initial_path)
    with open(json_initial_path, 'w') as f:
        data_json = json.load(f)
    objs = data_json["furniture"]
    for obj_name, obj_info in objs.items():
        if obj_name in new_p.keys():
            obj_info["pos"] = new_p[obj_name]
            obj_info["ori"] = new_o[obj_name]
    with open(json_initial_path, 'w') as f:
        json.dump(data_json, f, indent=4)
    print(f"Modified JSON file saved as {json_initial_path}")
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