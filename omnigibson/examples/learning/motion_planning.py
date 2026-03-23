import copy
import itertools
import math
import numpy as np
from shapely.geometry import Point, Polygon
import torch as th
import omnigibson.utils.transform_utils as T

class State:
    """
    2D state representation for motion planning.
    Tracks x, z positions and yaw angle on the ground plane (XZ plane).
    Y coordinate is vertical (height) and remains constant.
    Roll and pitch are stored but not modified (no actual DOF, but needed for consistent transforms).
    """
    def __init__(self, floor_poly, unmovable_object_poly_list, movable_object_poly_list, 
                 robot_x, robot_z, robot_yaw, robot_roll, robot_pitch, robot_radius, grasping_object_id, _robot_yaw_in_quat):
        self.floor_poly = floor_poly
        self.unmovable_object_poly_list = unmovable_object_poly_list
        self.movable_object_poly_list = movable_object_poly_list
        self.robot_x = robot_x
        self.robot_z = robot_z
        self.robot_yaw = robot_yaw
        self.robot_roll = robot_roll  # No DOF, but needed for consistent quaternion
        self.robot_pitch = robot_pitch  # No DOF, but needed for consistent quaternion
        self.robot_radius = robot_radius
        self.grasping_object_id = grasping_object_id
        self._robot_yaw_in_quat = _robot_yaw_in_quat

def simulation_step(state, action, debug=False):
    """
    Execute one step of simulation.
    Action: 0=backward, 1=forward, 2=turn_left, 3=turn_right, 4=fetch, 5=release
    
    Args:
        state: Current simulation state
        action: Action to execute (0-5)
        debug: If True, print detailed information
    
    Returns:
        new_state: Updated state after attempting the action
    """
    if action in [0, 1, 2, 3]:
        return simulation_move(state, action, debug)
    elif action == 4:
        return simulation_fetch(state, debug)
    elif action == 5:
        return simulation_release(state, debug)
    else:
        raise ValueError(f"Invalid action: {action}")

def simulation_move(state, action, debug=False):
    """
    Pure 2D motion simulation on the xz plane.
    Action: 0=backward, 1=forward, 2=turn_left, 3=turn_right
    
    Matches the logic in new_env.py's _move() function:
    - Creates local displacement [diff_x, 0.0, 0.0] and rotation diff_yaw
    - Uses pose_transform to convert to world frame (2D version)
    
    Args:
        state: Current simulation state
        action: Movement action to execute (0-3)
        debug: If True, print detailed collision information
    
    Returns:
        new_state: Updated state after attempting the action
    """
    action2diff = {0: (-0.25, 0.0), 1: (0.25, 0.0), 2: (0.0, math.pi / 8), 3: (0.0, -math.pi / 8)}
    diff_x, diff_yaw = action2diff[action]
    
    robot_x, robot_z, robot_yaw = state.robot_x, state.robot_z, state.robot_yaw
    robot_roll, robot_pitch = state.robot_roll, state.robot_pitch
    _robot_yaw_in_quat = state._robot_yaw_in_quat
    
    # Use TORCH's pose_transform directly to ensure 100% consistency with real environment
    # Key: X, Z are horizontal plane, Y is vertical (height). Yaw rotates around Y axis.
    # Roll and pitch are preserved from the initial state (no DOF, but needed for consistency)
    current_pos_3d = th.tensor([robot_x, 0.0, robot_z])  # Y height doesn't matter for 2D
    current_quat = T.euler2quat(th.tensor([robot_roll, robot_pitch, _robot_yaw_in_quat]))
    _pos = th.tensor([diff_x, 0.0, 0.0])
    _quat = T.euler2quat(th.tensor([0.0, 0.0, diff_yaw]))
    # print('In 2D world:')
    # print(current_pos_3d, current_quat, _pos, _quat)
    # print(T.quat2euler(current_quat))
    next_pos_3d, next_quat = T.pose_transform(current_pos_3d, current_quat, _pos, _quat)
    
    next_robot_x = next_pos_3d[0].item()
    next_robot_z = next_pos_3d[2].item()  # Z coordinate for horizontal plane
    next_euler = T.quat2euler(next_quat)
    next_robot_roll = next_euler[0].item()
    next_robot_pitch = next_euler[1].item()
    next_robot_yaw_in_quat = next_euler[2].item()

    _x = th.tensor([1.0, 0, 0])
    trans_x = T.quat_apply(next_quat, _x)
    next_robot_yaw = th.atan2(trans_x[2], trans_x[0]).item()
    
    # Update grasping object position if holding something
    # This ensures the grasped object moves with the robot, maintaining relative position
    # Use the SAME method as new_env.py
    new_movable_object_poly_list = copy.deepcopy(state.movable_object_poly_list)
    if state.grasping_object_id is not None:
        grasping_obj_poly = state.movable_object_poly_list[state.grasping_object_id]
        new_grasping_obj_poly = []
        
        # Old and new robot poses in 3D (Y=0 for height on XZ plane)
        old_robot_pos = th.tensor([robot_x, 0.0, robot_z])
        old_robot_quat = T.euler2quat(th.tensor([robot_roll, robot_pitch, _robot_yaw_in_quat]))
        new_robot_pos = th.tensor([next_robot_x, 0.0, next_robot_z])
        new_robot_quat = T.euler2quat(th.tensor([next_robot_roll, next_robot_pitch, next_robot_yaw_in_quat]))
        
        for p in grasping_obj_poly:
            # Convert 2D point to 3D pose (X, Y=0, Z)
            _pos = th.tensor([p[0], 0.0, p[1]])
            _quat = T.euler2quat(th.tensor([0.0, 0.0, 0.0]))
            # Transform using the same method as new_env.py
            new_pos, _ = _keep_relative_to_robot(_pos, _quat, old_robot_pos, old_robot_quat, new_robot_pos, new_robot_quat)
            # Extract X and Z coordinates
            new_grasping_obj_poly.append([new_pos[0].item(), new_pos[2].item()])
        
        new_movable_object_poly_list[state.grasping_object_id] = new_grasping_obj_poly

    no_collision = simulation_collision(state, next_robot_x, next_robot_z, next_robot_yaw, 
                                       new_movable_object_poly_list, debug=debug)

    if no_collision:
        new_state = copy.deepcopy(state)
        new_state.robot_x = next_robot_x
        new_state.robot_z = next_robot_z
        new_state.robot_yaw = next_robot_yaw
        new_state.robot_roll = next_robot_roll
        new_state.robot_pitch = next_robot_pitch
        new_state._robot_yaw_in_quat = next_robot_yaw_in_quat
        new_state.movable_object_poly_list = new_movable_object_poly_list
    else:
        new_state = copy.deepcopy(state)

    return new_state

def _keep_relative_to_robot(pos, quat, old_robot_pos, old_robot_quat, new_robot_pos, new_robot_quat):
    """
    Keep object's relative position to robot. This is the SAME function as in new_env.py.
    Uses torch's transform utilities to ensure consistency.
    
    Args:
        pos: Object position in world frame
        quat: Object quaternion in world frame
        old_robot_pos, old_robot_quat: Old robot pose
        new_robot_pos, new_robot_quat: New robot pose
    
    Returns:
        new_pos, new_quat: Updated object pose in world frame
    """
    # Transform object pose to robot's local frame
    pos_in_robot_frame, quat_in_robot_frame = T.relative_pose_transform(pos, quat, old_robot_pos, old_robot_quat)
    # Transform back to world frame with new robot pose
    pos_in_world_frame, quat_in_world_frame = T.pose_transform(new_robot_pos, new_robot_quat, pos_in_robot_frame, quat_in_robot_frame)
    return pos_in_world_frame, quat_in_world_frame


def simulation_collision(state, new_robot_x, new_robot_z, new_robot_yaw, new_movable_object_poly_list, debug=False):
    """
    Pure 2D collision detection on the XZ plane.
    
    Args:
        state: Current state
        new_robot_x, new_robot_z, new_robot_yaw: New robot pose to check
        new_movable_object_poly_list: Updated movable object polygons
        debug: If True, print detailed collision information
    
    Returns:
        valid: True if no collision, False otherwise
    """
    floor_area_poly = Polygon(state.floor_poly)
    
    # Combine all obstacles (unmovable + movable except grasping object)
    all_obstacles = []
    obstacle_labels = []  # For debugging
    
    for i, obj_poly in enumerate(state.unmovable_object_poly_list):
        all_obstacles.append(obj_poly)
        obstacle_labels.append(f"Unmovable_{i}")
    
    for i, obj_poly in enumerate(new_movable_object_poly_list):
        # Skip the grasping object from collision check
        if state.grasping_object_id is not None and i == state.grasping_object_id:
            continue
        all_obstacles.append(obj_poly)
        obstacle_labels.append(f"Movable_{i}")

    # Robot as a circle on 2D plane (XZ plane)
    new_robot_center = Point(new_robot_x, new_robot_z)
    new_robot_circle = new_robot_center.buffer(state.robot_radius)
    
    # Objects to check collision for
    moved_polygons = [new_robot_circle]
    moved_labels = ["Robot"]
    
    # If holding an object, also check collision for that object
    if state.grasping_object_id is not None:
        grasping_obj_poly = new_movable_object_poly_list[state.grasping_object_id]
        new_obj_poly = Polygon(grasping_obj_poly)
        moved_polygons.append(new_obj_poly)
        moved_labels.append(f"Grasping_Obj_{state.grasping_object_id}")

    obstacles_poly = [Polygon(obstacle) for obstacle in all_obstacles]
    
    # Check for intersections and floor coverage
    any_intersection = False
    collision_details = []
    
    for (obs_idx, obs_poly), (mov_idx, mov_poly) in itertools.product(enumerate(obstacles_poly), enumerate(moved_polygons)):
        if obs_poly.intersects(mov_poly):
            any_intersection = True
            collision_details.append((obstacle_labels[obs_idx], moved_labels[mov_idx]))
    
    all_cover = all([floor_area_poly.covers(poly) for poly in moved_polygons])
    not_covered = []
    for i, poly in enumerate(moved_polygons):
        if not floor_area_poly.covers(poly):
            not_covered.append(moved_labels[i])
    
    valid = (not any_intersection and all_cover)
    
    if debug and not valid:
        print(f"\n  [COLLISION DEBUG]")
        print(f"    Robot position: ({new_robot_x:.4f}, {new_robot_z:.4f}, yaw={new_robot_yaw:.4f})")
        print(f"    Total obstacles: {len(obstacles_poly)}")
        print(f"    Grasping object: {state.grasping_object_id}")
        if any_intersection:
            print(f"    [X] Intersections detected: {len(collision_details)}")
            for obs_name, mov_name in collision_details[:5]:  # Show first 5
                print(f"       - {mov_name} collides with {obs_name}")
        if not all_cover:
            print(f"    [X] Not covered by floor: {not_covered}")
    
    return valid

def simulation_fetch(state, debug=False):
    """
    Simulate fetch action. Matches the logic in new_env.py's _fetch() function.
    
    Tries to grasp an object in front of the robot within a visible area:
    - Visible area: 90-degree cone, radius 1.0m
    - Can only grasp if not already holding something
    - Grasps the closest object in visible area
    
    Args:
        state: Current simulation state
        debug: If True, print detailed information
    
    Returns:
        new_state: Updated state with grasping_object_id set if successful
    """
    new_state = copy.deepcopy(state)
    
    # Already holding something
    if state.grasping_object_id is not None:
        if debug:
            print(f"\n  [FETCH] Already grasping object {state.grasping_object_id}, cannot fetch")
        return new_state
    
    # Define visible area (90-degree cone, radius 1.0m)
    THETA = math.pi / 2
    R = 1.0
    VISIBLE_AREA_R = R * 2
    
    robot_x, robot_z, robot_yaw = state.robot_x, state.robot_z, state.robot_yaw
    
    # Create visible area polygon
    point_1 = (robot_x + VISIBLE_AREA_R * math.cos(robot_yaw + THETA / 2),
               robot_z + VISIBLE_AREA_R * math.sin(robot_yaw + THETA / 2))
    point_2 = (robot_x + VISIBLE_AREA_R * math.cos(robot_yaw - THETA / 2),
               robot_z + VISIBLE_AREA_R * math.sin(robot_yaw - THETA / 2))
    visible_area = Polygon([(robot_x, robot_z), point_1, point_2])
    robot_point = Point(robot_x, robot_z)
    # print('In 2D world:')
    # print(robot_x, robot_z, robot_yaw)
    # print(visible_area)
    # print(robot_point)

    # Find movable objects in visible area
    candidates = []
    for i, obj_poly in enumerate(state.movable_object_poly_list):
        obj_polygon = Polygon(obj_poly)
        if obj_polygon.intersects(visible_area) and robot_point.distance(obj_polygon) <= R:
            distance = robot_point.distance(obj_polygon)
            candidates.append((i, distance))
    
    if candidates:
        # Sort by distance and grasp the closest one
        candidates.sort(key=lambda x: x[1])
        grasped_id = candidates[0][0]
        new_state.grasping_object_id = grasped_id
        if debug:
            print(f"\n  [FETCH] Successfully grasped object {grasped_id} (distance: {candidates[0][1]:.4f}m)")
            print(f"    Total candidates: {len(candidates)}")
    else:
        if debug:
            print(f"\n  [FETCH] No objects in visible area")
    
    return new_state

def simulation_release(state, debug=False):
    """
    Simulate release action. Matches the logic in new_env.py's _release() function.
    
    Releases the currently grasped object.
    
    Args:
        state: Current simulation state
        debug: If True, print detailed information
    
    Returns:
        new_state: Updated state with grasping_object_id set to None
    """
    new_state = copy.deepcopy(state)
    
    if state.grasping_object_id is None:
        if debug:
            print(f"\n  [RELEASE] Not holding any object, cannot release")
        return new_state
    
    if debug:
        print(f"\n  [RELEASE] Released object {state.grasping_object_id}")
    
    new_state.grasping_object_id = None
    return new_state