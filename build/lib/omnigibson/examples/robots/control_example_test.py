import random
import torch as th
import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import gm
from omnigibson.robots import REGISTERED_ROBOTS
from omnigibson.utils.ui_utils import KeyboardRobotController, choose_from_options
import omnigibson.utils.transform_utils as T
from omnigibson.utils.asset_utils import get_available_g_scenes, get_available_og_scenes, get_available_3dfront_scenes, get_available_3dfront_rooms, get_available_3dfront_room
import math
import os
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROBOT_ACTION_DIM = 13


# def _rotate(env, robot, end_yaw, grasping_state, angle_threshold = 0.01):
#     world_end_pos, world_end_quat = robot.get_position_orientation()
#     world_end_quat = T.quat_multiply(T.euler2quat(th.tensor([0.0, 0.0, end_yaw])), world_end_quat)
#     diff_yaw = end_yaw
    
#     # Initialize lists to store trajectory points
#     trajectory_x = []
#     trajectory_z = []
    
#     for _ in range(500):
#         if abs(diff_yaw) < angle_threshold:
#             break

#         print(robot.links['l_wheel_link'].aabb_center[2] - robot.links['r_wheel_link'].aabb_center[2])
#         direction = -1.0 if diff_yaw < 0.0 else 1.0
#         ang_vel = 5 * direction
#         action = th.zeros(ROBOT_ACTION_DIM)
#         base_action = action[robot.controller_action_idx["base"]]
#         assert (base_action.numel() == 2)
#         base_action[0] = -ang_vel
#         base_action[1] = ang_vel
#         action[robot.controller_action_idx["base"]] = base_action
#         action[robot.controller_action_idx["gripper_0"]] = grasping_state

#         env.step(action)

#         world_pose, world_quat = robot.get_position_orientation()
#         # Record trajectory points
#         trajectory_x.append(world_pose[0].item())
#         trajectory_z.append(world_pose[1].item())
        
#         body_target_pose = T.relative_pose_transform(world_end_pos, world_end_quat, world_pose, world_quat)
#         diff_yaw = T.quat2euler(body_target_pose[1])[2].item()

def _rotate(env, robot, end_yaw, grasping_state, angle_threshold = 0.01):
    world_end_pos, world_end_quat = robot.get_position_orientation()
    # import pdb; pdb.set_trace()
    world_end_quat = T.quat_multiply(T.euler2quat(th.tensor([0.0, end_yaw, 0.0])), world_end_quat)
    diff_yaw = end_yaw
    for _ in range(500):
        # print(get_collisions())
        print(diff_yaw)
        if abs(diff_yaw) < angle_threshold:
            break

        direction = -1.0 if diff_yaw < 0.0 else 1.0
        ang_vel = 0.2 * direction
        # print(ang_vel)
        action = th.zeros(ROBOT_ACTION_DIM)
        base_action = action[robot.controller_action_idx["base"]]
        # import pdb; pdb.set_trace()
        assert (base_action.numel() == 2)
        base_action[0] = -ang_vel
        base_action[1] = ang_vel
        action[robot.controller_action_idx["base"]] = base_action
        action[robot.controller_action_idx["gripper_0"]] = grasping_state

        env.step(action)

        world_pose, world_quat = robot.get_position_orientation()
        body_target_pose = T.relative_pose_transform(world_end_pos, world_end_quat, world_pose, world_quat)
        # print(T.quat2euler(body_target_pose[1]))
        diff_yaw = T.quat2euler(body_target_pose[1])[2].item()
        # print(f'diff is {diff_yaw}')

def _translate(env, robot, end_x, grasping_state, angle_threshold = 0.05):
    end_pos_robot_frame = th.tensor([end_x, 0.0, 0.0])
    robot_pos, robot_quat = robot.get_position_orientation()
    # import pdb; pdb.set_trace()
    inv_pos, inv_quat = T.invert_pose_transform(robot_pos, robot_quat)
    world_end_pos, _ = T.relative_pose_transform(end_pos_robot_frame, th.tensor([1., 0., 0., 0.]), inv_pos, inv_quat)
    world_end_quat = robot_quat
    diff_pos = end_x
    diff_yaw = 0
    distance_pos = th.norm(th.tensor([end_x, 0.0, 0.0]))
    for _ in range(500):
        if abs(distance_pos) < angle_threshold:
            _rotate(env, robot, T.quat2euler(world_end_quat)[1] - T.quat2euler(world_quat)[1], grasping_state)
            break

        direction = -1.0 if diff_pos < 0.0 else 1.0
        angle_direction = -1.0 if diff_yaw < 0.0 else 1.0
        lin_vel = 0.2 * direction
        action = th.zeros(ROBOT_ACTION_DIM)
        base_action = action[robot.controller_action_idx["base"]]
        # import pdb; pdb.set_trace()
        assert (base_action.numel() == 2)
        base_action[0] = lin_vel
        base_action[1] = 0.2 * angle_direction
        action[robot.controller_action_idx["base"]] = base_action
        action[robot.controller_action_idx["gripper_0"]] = grasping_state
        # print(action)

        env.step(action)

        world_pose, world_quat = robot.get_position_orientation()
        robot_frame_end_pos, robot_frame_end_quat = T.relative_pose_transform(world_end_pos, world_end_quat, world_pose, world_quat)
        diff_pos = robot_frame_end_pos[0]
        distance_pos = th.norm(robot_frame_end_pos)

        world_pose, world_quat = robot.get_position_orientation()
        # body_target_pose = T.relative_pose_transform(world_end_pos, world_end_quat, world_pose, world_quat)
        # print(robot_frame_end_pos)
        if diff_pos > 0:
            diff_yaw = th.atan2(robot_frame_end_pos[1], robot_frame_end_pos[0])
        else:
            diff_yaw = th.atan2(-robot_frame_end_pos[1], -robot_frame_end_pos[0])

def _fetch(env, robot):
    action = th.zeros(ROBOT_ACTION_DIM)
    # import pdb; pdb.set_trace()
    base_action = action[robot.controller_action_idx["gripper_0"]]
    assert (base_action.numel() == 1)
    base_action[0] = -1.0
    action[robot.controller_action_idx["gripper_0"]] = base_action
    env.step(action)
    # import pdb; pdb.set_trace()
    if robot._ag_obj_in_hand['0'] is None:
        return 1.0
    return -1.0

def _release(env, robot):
    action = th.zeros(ROBOT_ACTION_DIM)
    base_action = action[robot.controller_action_idx["gripper_0"]]
    assert (base_action.numel() == 1)
    base_action[0] = 1.0
    action[robot.controller_action_idx["gripper_0"]] = base_action
    env.step(action)
    return 1.0

def _nothing(env, robot, grasping_state):
    action = th.zeros(ROBOT_ACTION_DIM)
    base_action = action[robot.controller_action_idx["gripper_0"]]
    assert (base_action.numel() == 1)
    base_action[0] = grasping_state
    action[robot.controller_action_idx["gripper_0"]] = base_action
    env.step(action)

def navigate_to(env, robot, target_x, target_z, target_orientation):
    robot_pos, robot_quat = robot.get_position_orientation()
    x, _, z = robot_pos
    # print("robot_pos in navigate_to:", robot_pos)
    assert not env.task.is_robot_collision(env, x, z)
    # _, yaw, _ = T.quat2euler(robot_quat)
    yaw = env.robots[0].get_yaw()
    # Ensure initial yaw is in [-π, π]
    yaw = float(yaw)
    if yaw > math.pi:
        yaw -= 2 * math.pi
    elif yaw < -math.pi:
        yaw += 2 * math.pi
    
    # Ensure target orientation is in [-π, π]
    target_orientation = float(target_orientation)
    if target_orientation > math.pi:
        target_orientation -= 2 * math.pi
    elif target_orientation < -math.pi:
        target_orientation += 2 * math.pi
    
    # Check if target is in collision
    if env.task.is_robot_collision(env, target_x, target_z):
        print("Target position is in collision!")
        return None
        
    # Node class for A* search
    class Node:
        def __init__(self, x, z, yaw, g_cost=0, h_cost=0, parent=None, action=None):
            self.x = float(x)  # Convert to float if tensor
            self.z = float(z)  # Convert to float if tensor
            self.yaw = float(yaw)  # Convert to float if tensor
            # Normalize yaw to [-π, π]
            if self.yaw > math.pi:
                self.yaw -= 2 * math.pi
            elif self.yaw < -math.pi:
                self.yaw += 2 * math.pi
                
            self.g_cost = g_cost  # Cost from start to current node
            self.h_cost = h_cost  # Estimated cost from current node to goal
            self.f_cost = g_cost + h_cost
            self.parent = parent
            self.action = action
            
        def __eq__(self, other):
            if not isinstance(other, Node):
                return False
            # Use half of the action resolution as threshold
            pos_threshold = 0.125  # Half of 0.25m
            ang_threshold = math.pi / 16  # Half of math.pi/8
            
            # Calculate angular difference accounting for circular nature
            yaw_diff = abs(self.yaw - other.yaw)
            if yaw_diff > math.pi:
                yaw_diff = 2 * math.pi - yaw_diff
                
            return (abs(self.x - other.x) < pos_threshold and 
                   abs(self.z - other.z) < pos_threshold and 
                   yaw_diff < ang_threshold)
            
        def __hash__(self):
            # Round to the nearest action resolution
            x_rounded = round(self.x / 0.125) * 0.125
            z_rounded = round(self.z / 0.125) * 0.125
            # Normalize yaw to [-π, π] before rounding
            yaw_normalized = self.yaw
            if yaw_normalized > math.pi:
                yaw_normalized -= 2 * math.pi
            elif yaw_normalized < -math.pi:
                yaw_normalized += 2 * math.pi
            yaw_rounded = round(yaw_normalized / (math.pi/16)) * (math.pi/8)
            # Ensure yaw_rounded stays in [-π, π]
            if yaw_rounded > math.pi:
                yaw_rounded -= 2 * math.pi
            elif yaw_rounded < -math.pi:
                yaw_rounded += 2 * math.pi
            return hash((x_rounded, z_rounded, yaw_rounded))
    
    def heuristic(node, goal_x, goal_z, goal_yaw):
        # Manhattan distance for position
        pos_cost = abs(node.x - goal_x) + abs(node.z - goal_z)
        # Angular difference for orientation, accounting for circular nature
        ang_diff = abs(node.yaw - goal_yaw)
        if ang_diff > math.pi:
            ang_diff = 2 * math.pi - ang_diff
        return pos_cost + 0.5 * ang_diff
    
    def get_successors(node):
        successors = []
        actions = [
            (0, -0.25, 0),  # Backward
            (1, 0.25, 0),   # Forward
            (2, 0, -math.pi/8),  # Rotate left
            (3, 0, math.pi/8)  # Rotate right
        ]
        
        for action_id, dx, dyaw in actions:
            # Calculate new position based on action
            new_x = node.x + dx * math.cos(node.yaw)
            new_z = node.z + dx * math.sin(node.yaw)
            new_yaw = node.yaw + dyaw
            # Normalize new_yaw to [-π, π]
            if new_yaw > math.pi:
                new_yaw -= 2 * math.pi
            elif new_yaw < -math.pi:
                new_yaw += 2 * math.pi
            
            # Check collision
            if env.task.is_robot_collision(env, new_x, new_z):
                continue
                
            successors.append(Node(new_x, new_z, new_yaw, action=action_id))
            
        return successors
    
    # Initialize start and goal nodes
    start_node = Node(x, z, yaw)
    goal_node = Node(target_x, target_z, target_orientation)
    
    # A* search
    open_set = {start_node}
    closed_set = set()
    came_from = {}
    g_score = {start_node: 0}
    f_score = {start_node: heuristic(start_node, target_x, target_z, target_orientation)}
    
    while open_set:
        current = min(open_set, key=lambda n: f_score.get(n, float('inf')))
        # print(current.x, current.z, current.yaw)
        
        if current == goal_node:
            # Reconstruct path
            path = []
            while current.parent is not None:
                path.append(current.action)
                current = current.parent
            return list(reversed(path))
            
        open_set.remove(current)
        closed_set.add(current)
        
        for successor in get_successors(current):
            if successor in closed_set:
                continue
                
            tentative_g_score = g_score[current] + 1  # Each action costs 1
            
            if successor not in open_set:
                open_set.add(successor)
                g_score[successor] = tentative_g_score
                f_score[successor] = tentative_g_score + heuristic(successor, target_x, target_z, target_orientation)
                successor.parent = current
            elif tentative_g_score < g_score[successor]:
                g_score[successor] = tentative_g_score
                f_score[successor] = tentative_g_score + heuristic(successor, target_x, target_z, target_orientation)
                successor.parent = current
    
    print("No path found to target!")
    return None

CONTROL_MODES = dict(
    random="Use autonomous random actions (default)",
    teleop="Use keyboard control",
)

# SCENES = dict(
#     Rs_int="Realistic interactive home environment (default)",
#     empty="Empty environment with no objects",
# )

# Don't use GPU dynamics and use flatcache for performance boost
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True

def generate_action_tensor(env, robot, action, grasping_state):
    # 定义硬编码的平动、转动配置
    # _rotate(env, robot, math.pi / 2)
    # _translate(env, robot, 0.5)
    
    # 根据输入的动作生成不同的tensor
    if action == 0:  # 向后
        _translate(env, robot, -0.25, grasping_state)
    elif action == 1:  # 向前
        _translate(env, robot, 0.25, grasping_state)
    elif action == 2:  # 左转
        _rotate(env, robot, math.pi / 8, grasping_state)
    elif action == 3:  # 右转
        _rotate(env, robot, -math.pi / 8, grasping_state)
    elif action == 4:  # 按下放开
        grasping_state = _fetch(env, robot)
    elif action == 5:  # 按下收缩
        grasping_state = _release(env, robot)
    elif action == 6:
        _nothing(env, robot, grasping_state)

    return grasping_state

def choose_controllers(robot, random_selection=False):
    """
    For a given robot, iterates over all components of the robot, and returns the requested controller type for each
    component.

    :param robot: BaseRobot, robot class from which to infer relevant valid controller options
    :param random_selection: bool, if the selection is random (for automatic demo execution). Default False

    :return dict: Mapping from individual robot component (e.g.: base, arm, etc.) to selected controller names
    """
    # Create new dict to store responses from user
    controller_choices = dict()

    # Grab the default controller config so we have the registry of all possible controller options
    default_config = robot._default_controller_config

    # Iterate over all components in robot
    for component, controller_options in default_config.items():
        # Select controller
        options = list(sorted(controller_options.keys()))
        choice = choose_from_options(
            options=options, name="{} controller".format(component), random_selection=random_selection
        )

        # Add to user responses
        controller_choices[component] = choice

    return controller_choices


def main(random_selection=False, headless=False, short_exec=False, quickstart=False):
    """
    Robot control demo with selection
    Queries the user to select a robot, the controllers, a scene and a type of input (random actions or teleop)
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    # Choose scene to load

    # config_filename = os.path.join(og.example_config_path, f"generate_initial_layout.yaml")
    # config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    # scene_type = "Threed_FRONTScene"
    # config["scene"]["type"] = scene_type

    scenes = get_available_3dfront_scenes()
    scene = choose_from_options(options=scenes, name="scene model", random_selection=random_selection)
    rooms = get_available_3dfront_rooms(scene)
    room_type = choose_from_options(options=rooms, name="room type", random_selection=random_selection)
    room = get_available_3dfront_room(scene, room_type)
    room_model = choose_from_options(options=room, name="room model", random_selection=random_selection)
    room_model_path = os.path.join(gm.ThreeD_FRONT_DATASET_PATH, "scenes", scene, room_type, room_model)

    # config["scene"]["scene_model"] = room_model
    # config["scene"]["scene_type_path"] = room_model_path
    config = {}
    config["scene"] = {"type": "Scene"}

    # Choose robot to create
    robot_name = 'Fetch'
    # if not quickstart:
    #     robot_name = choose_from_options(
    #         options=list(sorted(REGISTERED_ROBOTS.keys())), name="robot", random_selection=random_selection
    #     )
    # import pdb; pdb.set_trace()


    # Add the robot we want to load
    config["robots"] = [{}]
    config["robots"][0]["type"] = robot_name
    config["robots"][0]["obs_modalities"] = ["rgb"]
    config["robots"][0]["action_type"] = "continuous"
    config["robots"][0]["action_normalize"] = True
    config["robots"][0]["grasping_mode"] = 'sticky'

    # Create the environment
    env = og.Environment(configs=config)
    import pdb; pdb.set_trace()

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

    # Choose control mode
    if random_selection:
        control_mode = "random"
    elif quickstart:
        control_mode = "teleop"
    else:
        control_mode = choose_from_options(options=CONTROL_MODES, name="control mode")

    # Update the control mode of the robot
    controller_config = {component: {"name": name} for component, name in controller_choices.items()}
    controller_config['arm_0']['motor_type'] = 'effort'
    # controller_config['camera']['motor_type'] = 'effort'
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

    # Create teleop controller
    action_generator = KeyboardRobotController(robot=robot)

    # Register custom binding to reset the environment
    action_generator.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.R,
        description="Reset the robot",
        callback_fn=lambda: env.reset(),
    )

    # Print out relevant keyboard info if using keyboard teleop
    if control_mode == "teleop":
        action_generator.print_keyboard_teleop_info()

    # Other helpful user info
    print("Running demo.")
    print("Press ESC to quit")

    # Loop control until user quits
    max_steps = -1 if not short_exec else 100
    step = 0

    # _rotate(env, robot, math.pi / 2)
    # _translate(env, robot, 0.5)
    # import pdb; pdb.set_trace()

    # controller = StarterSemanticActionPrimitives(env, enable_head_tracking=False)
    # execute_controller(controller.apply_ref(StarterSemanticActionPrimitiveSet.NAVIGATE_TO, (10, 10, 0)), env)
    grasping_state = 1.0
    while step != max_steps:
        # action = (
        #    action_generator.get_random_action() if control_mode == "random" else action_generator.get_teleop_action()
        # )
        # for test
        # a = [6]
        if True:
            # action_input = int(input("请输入动作编号 (0-5): "))
            # action_input = random.randint(0, 4)
            # a = navigate_to(env, robot, 5, -2, 0)
            a = [random.randint(2, 3)]
            if a is not None and len(a) == 0:
                print('Success!')
                break 
            print(f"action: {a}")
            print(f"robot.get_position_orientation(): {robot.get_position_orientation()[0][0], robot.get_position_orientation()[0][2], T.quat2euler(robot.get_position_orientation()[1])[1]}")
            grasping_state= generate_action_tensor(env, robot, a[0], grasping_state)
            print(f"robot.get_position_orientation(): {robot.get_position_orientation()[0][0], robot.get_position_orientation()[0][2], T.quat2euler(robot.get_position_orientation()[1])[1]}")
            # import pdb; pdb.set_trace()
            # env.step(action=action)
            step += 1
        else:
            action_input = 6
            grasping_state= generate_action_tensor(env, robot, action_input, grasping_state)
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