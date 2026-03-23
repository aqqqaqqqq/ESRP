import os
import sys
import yaml
import math
import random
import itertools
import heapq
from collections import deque
from shapely.geometry import Polygon
import numpy as np
import torch as th

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.examples.environments.new_env import FastEnv, OccupancyInfo
import omnigibson.utils.transform_utils as T
from omnigibson.examples.learning.motion_planning import State, simulation_step, simulation_fetch, simulation_release
from omnigibson.examples.learning.vis import visualize_bfs_state, visualize_collision_test, reset_vis_counter

# ============================================================================
# Configuration - All constants declared here
# ============================================================================
DISCRETIZATION_STEP = 0.05              # Grid size for position (meters)
DISCRETIZATION_YAW = math.pi / 32        # Grid size for orientation (radians)
IOU_THRESHOLD = 0.3                     # IoU threshold for goal reaching
CENTER_DISTANCE_THRESHOLD = 0.15        # Center distance threshold (meters)
MAX_BFS_ITERATIONS = 1000            # Max BFS search iterations
TASK_PLANNING_MODE = "exhaustive"           # "random" or "exhaustive"
MAX_EXHAUSTIVE_PERMUTATIONS = 100       # Limit for exhaustive search
ENABLE_VISUALIZATION = False             # Enable BFS visualization for debugging

# Multiprocess configuration
LOCK_TIMEOUT_SECONDS = 900              # 15 minutes lock timeout
LOCK_DIR = 'tamp_locks'                 # Directory for scene locks
RESULT_DIR = 'tamp_results'             # Directory for scene results
PROCESS_PID = os.getpid()               # Current process PID (automatic)

# ============================================================================
# Multiprocess Scene Management (Fully Automatic)
# ============================================================================

def init_multiprocess_dirs():
    """Initialize directories for multiprocess coordination"""
    os.makedirs(LOCK_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

def get_scene_lock_path(scene_name):
    """Get lock file path for a scene"""
    safe_name = scene_name.replace('/', '_').replace('\\', '_')
    return os.path.join(LOCK_DIR, f"{safe_name}.lock")

def get_scene_result_path(scene_name):
    """Get result file path for a scene"""
    safe_name = scene_name.replace('/', '_').replace('\\', '_')
    return os.path.join(RESULT_DIR, f"{safe_name}.result")

def is_lock_expired(lock_path):
    """
    Check if lock file is expired
    Returns True if expired (can be taken over), False if still valid
    """
    if not os.path.exists(lock_path):
        return True
    
    try:
        import time
        lock_age = time.time() - os.path.getmtime(lock_path)
        if lock_age > LOCK_TIMEOUT_SECONDS:
            print(f"[PID {PROCESS_PID}] Lock expired ({lock_age:.0f}s > {LOCK_TIMEOUT_SECONDS}s): {lock_path}")
            return True
        return False
    except Exception as e:
        print(f"[PID {PROCESS_PID}] Error checking lock age: {e}")
        return True

def try_claim_scene(scene_name):
    """
    Try to claim a scene for testing using atomic file creation
    Returns True if successfully claimed, False otherwise
    """
    import time
    lock_path = get_scene_lock_path(scene_name)
    result_path = get_scene_result_path(scene_name)
    
    # Skip if result already exists
    if os.path.exists(result_path):
        return False
    
    # Check if lock exists and is valid
    if os.path.exists(lock_path):
        if not is_lock_expired(lock_path):
            return False
        else:
            # Lock expired, remove it
            try:
                os.remove(lock_path)
                print(f"[PID {PROCESS_PID}] Removed expired lock: {scene_name}")
            except Exception as e:
                print(f"[PID {PROCESS_PID}] Failed to remove expired lock: {e}")
                return False
    
    # Try to create lock file atomically
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        lock_content = f"PID: {PROCESS_PID}\nTime: {time.time()}\nTimestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        os.write(fd, lock_content.encode())
        os.close(fd)
        print(f"[PID {PROCESS_PID}] Successfully claimed scene: {scene_name}")
        return True
    except FileExistsError:
        return False
    except Exception as e:
        print(f"[PID {PROCESS_PID}] Error claiming scene {scene_name}: {e}")
        return False

def release_scene(scene_name):
    """Release scene lock"""
    lock_path = get_scene_lock_path(scene_name)
    try:
        if os.path.exists(lock_path):
            os.remove(lock_path)
            print(f"[PID {PROCESS_PID}] Released lock for scene: {scene_name}")
    except Exception as e:
        print(f"[PID {PROCESS_PID}] Error releasing lock for {scene_name}: {e}")

def save_scene_result(scene_name, success, init_potential, finish_potential, all_objs, arrival_num):
    """
    Save scene test result using atomic file operations
    """
    import time
    result_path = get_scene_result_path(scene_name)
    temp_path = result_path + f".tmp.{PROCESS_PID}.{int(time.time())}"
    
    try:
        with open(temp_path, 'w') as f:
            f.write(f"{scene_name}: success: {success}, init_potential: {init_potential}, "
                   f"finish_potential: {finish_potential}, all_objs: {all_objs}, "
                   f"arrival_num: {arrival_num}\n")
        os.rename(temp_path, result_path)
        print(f"[PID {PROCESS_PID}] Saved result for scene: {scene_name}")
        return True
    except Exception as e:
        print(f"[PID {PROCESS_PID}] Error saving result for {scene_name}: {e}")
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        return False

def get_available_scenes(scene_names):
    """
    Get list of scenes available for testing
    Excludes completed and actively locked scenes
    """
    available = []
    
    for scene in scene_names:
        result_path = get_scene_result_path(scene)
        lock_path = get_scene_lock_path(scene)
        
        # Skip if result exists
        if os.path.exists(result_path):
            continue
        
        # Check lock file
        if os.path.exists(lock_path):
            if not is_lock_expired(lock_path):
                continue
        
        available.append(scene)
    
    return available

def cleanup_temp_files():
    """Clean up leftover temporary files"""
    import time
    try:
        for filename in os.listdir(RESULT_DIR):
            if '.tmp.' in filename:
                filepath = os.path.join(RESULT_DIR, filename)
                file_age = time.time() - os.path.getmtime(filepath)
                if file_age > 3600:
                    os.remove(filepath)
                    print(f"[PID {PROCESS_PID}] Cleaned up old temp file: {filename}")
    except Exception as e:
        print(f"[PID {PROCESS_PID}] Error cleaning temp files: {e}")

def merge_results_to_file(output_file):
    """Merge all individual result files to final output file"""
    all_results = []
    
    for filename in os.listdir(RESULT_DIR):
        if filename.endswith('.result'):
            filepath = os.path.join(RESULT_DIR, filename)
            try:
                with open(filepath, 'r') as f:
                    content = f.read().strip()
                    if content:
                        all_results.append(content)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    
    all_results.sort()
    
    with open(output_file, 'w') as f:
        for result in all_results:
            f.write(result + '\n')
    
    print(f"Merged {len(all_results)} results to {output_file}")
    return len(all_results)

# ============================================================================
# State Extraction and Discretization
# ============================================================================

def extract_environment_state(fast_env):
    """
    Extract current state from real environment
    Returns simulation State object
    """
    print("Extracting environment state...")
    
    floor_poly = OccupancyInfo.get_floor_area(fast_env.env)
    obstacles = OccupancyInfo.get_obstacles(fast_env.env, fast_env.obstacles_cache)
    fast_env.obstacles_cache = obstacles
    
    unmovable_object_poly_list = []
    movable_object_poly_list = []
    movable_object_names = []
    
    for obj_name, obj_poly in obstacles.items():
        obj = fast_env.env.scene._init_objs.get(obj_name)
        poly_list = obj_poly.tolist() if isinstance(obj_poly, np.ndarray) else obj_poly
        if obj and obj.is_to_rearrange:
            movable_object_poly_list.append(poly_list)
            movable_object_names.append(obj_name)
        else:
            unmovable_object_poly_list.append(poly_list)
    
    robot = fast_env._get_robot()
    robot_pos, robot_quat = robot.get_position_orientation(frame='scene')
    robot_x = robot_pos[0].item()
    robot_z = robot_pos[2].item()
    robot_euler = T.quat2euler(robot_quat)
    robot_roll = robot_euler[0].item()
    robot_pitch = robot_euler[1].item()
    _robot_yaw_in_quat = robot_euler[2].item()
    robot_yaw = robot.get_yaw().item()
    robot_radius = fast_env.robot_radius
    
    
    grasping_object_id = None
    if fast_env.grasping_obj is not None:
        grasping_obj_name = fast_env.grasping_obj.split('/')[-1]
        for i, obj_name in enumerate(movable_object_names):
            if obj_name == grasping_obj_name:
                grasping_object_id = i
                break
    
    state = State(
        floor_poly=floor_poly,
        unmovable_object_poly_list=unmovable_object_poly_list,
        movable_object_poly_list=movable_object_poly_list,
        robot_x=robot_x,
        robot_z=robot_z,
        robot_yaw=robot_yaw,
        robot_roll=robot_roll,
        robot_pitch=robot_pitch,
        robot_radius=robot_radius,
        grasping_object_id=grasping_object_id,
        _robot_yaw_in_quat=_robot_yaw_in_quat
    )
    
    print(f"  Movable objects: {len(movable_object_poly_list)}")
    print(f"  Unmovable objects: {len(unmovable_object_poly_list)}")
    print(f"  Robot pose: ({robot_x:.2f}, {robot_z:.2f}, yaw={math.degrees(robot_yaw):.1f}deg)")
    print(f"  Robot grasping: {grasping_object_id}")
    
    # Print details of each movable object
    for i, (obj_name, obj_poly) in enumerate(zip(movable_object_names, movable_object_poly_list)):
        obj_center = np.mean(obj_poly, axis=0)
        print(f"    Movable object {i}: {obj_name}")
        print(f"      Center: ({obj_center[0]:.2f}, {obj_center[1]:.2f})")
    
    return state, movable_object_names

def extract_goal_positions(fast_env, movable_object_names):
    """
    Extract goal positions for all movable objects
    Returns list of goal polygons
    """
    print("Extracting goal positions...")
    
    goal_polys = []
    for i, obj_name in enumerate(movable_object_names):
        obj = fast_env.env.scene._init_objs.get(obj_name)
        if obj and hasattr(obj, 'target_bbox'):
            # Get target polygon from target_bbox
            target_poly = Polygon(np.array(obj.target_bbox))
            goal_polys.append(target_poly)
            goal_center = np.mean(target_poly.exterior.coords[:-1], axis=0)
            print(f"    Object {i}: {obj_name}")
            print(f"      Goal center: ({goal_center[0]:.2f}, {goal_center[1]:.2f})")
        else:
            goal_polys.append(None)
            print(f"    Object {i}: {obj_name} - NO GOAL")
    
    print(f"  Goals extracted: {sum(1 for g in goal_polys if g is not None)}/{len(goal_polys)}")
    return goal_polys

def discretize_pose(x, z, yaw):
    """
    Discretize continuous pose to grid cell
    """
    disc_x = round(x / DISCRETIZATION_STEP) * DISCRETIZATION_STEP
    disc_z = round(z / DISCRETIZATION_STEP) * DISCRETIZATION_STEP
    disc_yaw = round(yaw / DISCRETIZATION_YAW) * DISCRETIZATION_YAW
    # Normalize yaw to [-pi, pi]
    while disc_yaw > math.pi:
        disc_yaw -= 2 * math.pi
    while disc_yaw < -math.pi:
        disc_yaw += 2 * math.pi
    return (disc_x, disc_z, disc_yaw)

def heuristic_to_goal(state, goal_poly):
    """
    A* heuristic: estimate distance from current state to goal
    """
    if state.grasping_object_id is None:
        return float('inf')

    obj_poly = state.movable_object_poly_list[state.grasping_object_id]
    obj_center = np.mean(obj_poly, axis=0)

    if isinstance(goal_poly, Polygon):
        goal_center = np.mean(goal_poly.exterior.coords[:-1], axis=0)
    else:
        goal_center = np.mean(goal_poly, axis=0)

    return np.linalg.norm(obj_center - goal_center)

def calculate_iou(poly1, poly2):
    """
    Calculate IoU between two polygons
    """
    p1 = Polygon(poly1)
    p2 = Polygon(poly2)
    
    if not p1.is_valid or not p2.is_valid:
        return 0.0
    
    intersection = p1.intersection(p2).area
    union = p1.union(p2).area
    
    if union < 1e-6:
        return 0.0
    
    return intersection / union

def calculate_center_distance(poly1, poly2):
    """
    Calculate distance between polygon centers
    Similar to potential-based reward calculation
    """
    if isinstance(poly1, Polygon):
        p1 = poly1
    else:
        p1 = Polygon(poly1)
    
    if isinstance(poly2, Polygon):
        p2 = poly2
    else:
        p2 = Polygon(poly2)
    
    center1 = np.array([p1.centroid.x, p1.centroid.y])
    center2 = np.array([p2.centroid.x, p2.centroid.y])
    
    return np.linalg.norm(center1 - center2)

# ============================================================================
# BFS Search for Motion Planning
# ============================================================================

def bfs_search_to_object(initial_state, target_object_id):
    """
    BFS search to find path to grasp target object
    
    Returns:
        action_sequence: List of actions, or None if not found
    """
    print(f"  BFS search to object {target_object_id}...")
    
    # Reset visualization counter for this search
    if ENABLE_VISUALIZATION:
        reset_vis_counter()
    
    # Print initial information
    target_obj_poly = initial_state.movable_object_poly_list[target_object_id]
    target_center = np.mean(target_obj_poly, axis=0)
    print(f"    [INITIAL STATE]")
    print(f"      Robot position: ({initial_state.robot_x:.2f}, {initial_state.robot_z:.2f})")
    print(f"      Robot yaw: {math.degrees(initial_state.robot_yaw):.1f} deg")
    print(f"      Target object center: ({target_center[0]:.2f}, {target_center[1]:.2f})")
    print(f"      Distance to target: {np.linalg.norm([initial_state.robot_x - target_center[0], initial_state.robot_z - target_center[1]]):.2f}m")
    
    # Visualize initial state
    if ENABLE_VISUALIZATION:
        visualize_bfs_state(initial_state, action_name="Initial State", 
                          target_obj_id=target_object_id, prefix="to_object")
        print(f"      Visualization saved")
    
    # Initial discretized pose
    start_pose = discretize_pose(initial_state.robot_x, initial_state.robot_z, initial_state.robot_yaw)
    
    # BFS data structures
    queue = deque([(initial_state, [])])  # (state, action_sequence)
    visited = {start_pose}
    iterations = 0
    
    # Track exploration
    max_explored_count = 0
    
    while queue and iterations < MAX_BFS_ITERATIONS:
        iterations += 1
        current_state, actions = queue.popleft()
        
        # Print progress every 20 iterations
        if iterations % 20 == 0:
            print(f"    [BFS Progress] Iteration: {iterations}, Queue size: {len(queue)}, Visited states: {len(visited)}")
            print(f"      Current robot: ({current_state.robot_x:.2f}, {current_state.robot_z:.2f}, yaw={math.degrees(current_state.robot_yaw):.1f}deg)")
        
        # Check if we can fetch the target object at current state
        test_fetch_state = simulation_fetch(current_state, debug=False)
        if test_fetch_state.grasping_object_id == target_object_id:
            print(f"    [SUCCESS] Found path! Length: {len(actions)} steps, Iterations: {iterations}")
            print(f"      Final robot position: ({current_state.robot_x:.2f}, {current_state.robot_z:.2f})")
            return actions
        
        # Expand neighbors (try all 4 movement actions: 0,1,2,3)
        expanded_count = 0
        blocked_count = 0
        duplicate_count = 0
        
        action_names = {0: "Back", 1: "Forward", 2: "Left", 3: "Right"}
        
        for action in [0, 1, 2, 3]:
            next_state = simulation_step(current_state, action, debug=False)
            next_pose = discretize_pose(next_state.robot_x, next_state.robot_z, next_state.robot_yaw)
            
            # Check if moved and not visited
            current_pose = discretize_pose(current_state.robot_x, current_state.robot_z, current_state.robot_yaw)
            
            moved = (next_pose != current_pose)
            
            # Visualize first few attempts (limit to avoid too many images)
            if ENABLE_VISUALIZATION and iterations <= 100:
                visualize_collision_test(current_state, next_state, moved, action, 
                                        f"{action_names[action]} (Iter {iterations})")
            
            if next_pose == current_pose:
                blocked_count += 1  # Robot couldn't move (collision or out of bounds)
            elif next_pose in visited:
                duplicate_count += 1  # Already visited this state
            else:
                visited.add(next_pose)
                queue.append((next_state, actions + [action]))
                expanded_count += 1
        
        max_explored_count = max(max_explored_count, expanded_count)
        
        # Print warning if no expansion possible
        if expanded_count == 0 and iterations <= 5:
            print(f"    [WARNING at iter {iterations}] No expansion! Blocked: {blocked_count}, Duplicate: {duplicate_count}")
    
    print(f"    [FAILED] No path found after {iterations} iterations")
    print(f"      Total visited states: {len(visited)}")
    print(f"      Max expansions from one state: {max_explored_count}")
    print(f"      Initial distance to target: {np.linalg.norm([initial_state.robot_x - target_center[0], initial_state.robot_z - target_center[1]]):.2f}m")
    return None

def astar_search_to_goal(initial_state, goal_poly):
    """
    A* search to find path to move grasped object to goal
    Assumes robot is holding the object
    Uses both IoU and center distance for goal checking

    Returns:
        action_sequence: List of actions, or None if not found
    """
    print(f"  A* search to goal position...")

    if initial_state.grasping_object_id is None:
        print("    ERROR: Not holding any object!")
        return None

    # Reset visualization counter for this search
    if ENABLE_VISUALIZATION:
        reset_vis_counter()

    # Print initial information
    obj_poly = initial_state.movable_object_poly_list[initial_state.grasping_object_id]
    obj_center = np.mean(obj_poly, axis=0)
    goal_center = np.mean(goal_poly.exterior.coords[:-1], axis=0) if isinstance(goal_poly, Polygon) else np.mean(goal_poly, axis=0)
    initial_iou = calculate_iou(obj_poly, goal_poly)
    initial_dist = calculate_center_distance(obj_poly, goal_poly)

    print(f"    [INITIAL STATE]")
    print(f"      Robot position: ({initial_state.robot_x:.2f}, {initial_state.robot_z:.2f})")
    print(f"      Robot yaw: {math.degrees(initial_state.robot_yaw):.1f} deg")
    print(f"      Object center: ({obj_center[0]:.2f}, {obj_center[1]:.2f})")
    print(f"      Goal center: ({goal_center[0]:.2f}, {goal_center[1]:.2f})")
    print(f"      Initial IoU: {initial_iou:.3f}, Initial distance: {initial_dist:.3f}m")

    # Visualize initial state with goal
    if ENABLE_VISUALIZATION:
        visualize_bfs_state(initial_state, action_name="Initial State (With Object)",
                          goal_poly=goal_poly, prefix="to_goal")
        print(f"      Visualization saved")

    # A* data structures
    start_pose = discretize_pose(initial_state.robot_x, initial_state.robot_z, initial_state.robot_yaw)

    # Priority queue: (f_score, g_score, state, actions, pose)
    frontier = []
    heapq.heappush(frontier, (0, 0, initial_state, [], start_pose))

    # Track visited states and their best costs
    came_from = {}
    cost_so_far = {}
    came_from[start_pose] = None
    cost_so_far[start_pose] = 0

    visited = {start_pose}
    iterations = 0

    # Track best state found
    best_iou = initial_iou
    best_dist = initial_dist

    while frontier and iterations < MAX_BFS_ITERATIONS:
        iterations += 1

        # Get node with lowest f_score
        f_score, g_score, current_state, actions, current_pose = heapq.heappop(frontier)

        # Check if object reached goal
        obj_poly = current_state.movable_object_poly_list[current_state.grasping_object_id]
        iou = calculate_iou(obj_poly, goal_poly)
        center_dist = calculate_center_distance(obj_poly, goal_poly)

        # Track best found
        if iou > best_iou or center_dist < best_dist:
            best_iou = max(best_iou, iou)
            best_dist = min(best_dist, center_dist)

        # Print progress every 20 iterations
        if iterations % 20 == 0:
            heuristic = heuristic_to_goal(current_state, goal_poly)
            print(f"    [A* Progress] Iteration: {iterations}, Frontier: {len(frontier)}, Visited: {len(visited)}")
            print(f"      Current: IoU={iou:.3f}, Dist={center_dist:.3f}m, g={g_score:.1f}, h={heuristic:.3f}")
            print(f"      Best so far: IoU={best_iou:.3f}, Dist={best_dist:.3f}m")

        # Success if either IoU is high enough OR center distance is small enough
        if iou >= IOU_THRESHOLD or center_dist <= CENTER_DISTANCE_THRESHOLD:
            print(f"    [SUCCESS] Found path! Length: {len(actions)} steps, Iterations: {iterations}")
            print(f"      Final IoU: {iou:.3f}, Final distance: {center_dist:.3f}m")
            return actions

        # Expand neighbors
        action_names = {0: "Back", 1: "Forward", 2: "Left", 3: "Right"}

        for action in [0, 1, 2, 3]:
            next_state = simulation_step(current_state, action, debug=False)
            next_pose = discretize_pose(next_state.robot_x, next_state.robot_z, next_state.robot_yaw)

            # Calculate new cost (uniform cost for all actions)
            new_cost = cost_so_far[current_pose] + 1.0

            # If this path to next_state is better, or first time visiting
            if next_pose not in cost_so_far or new_cost < cost_so_far[next_pose]:
                cost_so_far[next_pose] = new_cost
                priority = new_cost + heuristic_to_goal(next_state, goal_poly)
                heapq.heappush(frontier, (priority, new_cost, next_state, actions + [action], next_pose))
                visited.add(next_pose)

                # Visualize first few attempts
                if ENABLE_VISUALIZATION and iterations <= 50:
                    moved = (next_pose != current_pose)
                    visualize_collision_test(current_state, next_state, moved, action,
                                            f"{action_names[action]} (Iter {iterations}, IoU={iou:.3f})")

    print(f"    [FAILED] No path found after {iterations} iterations")
    print(f"      Total visited states: {len(visited)}")
    print(f"      Best IoU found: {best_iou:.3f}, Best distance: {best_dist:.3f}m")
    return None

# ============================================================================
# Motion Planning - Complete object movement
# ============================================================================

def motion_planning_single_object(initial_state, object_id, goal_poly):
    """
    Plan to move a single object to goal
    
    Returns:
        action_sequence: Complete action sequence, or None if failed
        final_state: Simulated final state, or None if failed
        simulated_states: List of (x, z, yaw) tuples after each action, or None if failed
    """
    print(f"\nMotion planning for object {object_id}...")
    print(f"  [START STATE]")
    print(f"    Robot: ({initial_state.robot_x:.2f}, {initial_state.robot_z:.2f}, yaw={math.degrees(initial_state.robot_yaw):.1f}deg)")
    print(f"    Grasping: {initial_state.grasping_object_id}")
    obj_center = np.mean(initial_state.movable_object_poly_list[object_id], axis=0)
    print(f"    Object {object_id} center: ({obj_center[0]:.2f}, {obj_center[1]:.2f})")
    
    # Phase 1: Navigate to object
    print("\nPhase 1: Navigate to object")
    path_to_object = bfs_search_to_object(initial_state, object_id)
    if path_to_object is None:
        print("  FAILED: Cannot reach object")
        return None, None, None
    
    # Simulate path to object
    state = initial_state
    for action in path_to_object:
        state = simulation_step(state, action, debug=False)
    
    print(f"  [After Phase 1] Robot reached: ({state.robot_x:.2f}, {state.robot_z:.2f}, yaw={math.degrees(state.robot_yaw):.1f}deg)")
    
    # Phase 2: Grasp
    print("\nPhase 2: Grasp object")
    grasp_action = 4
    state = simulation_step(state, grasp_action, debug=False)
    
    if state.grasping_object_id != object_id:
        print("  FAILED: Grasp unsuccessful")
        print(f"    Expected to grasp object {object_id}, but grasping_object_id = {state.grasping_object_id}")
        return None, None, None
    print(f"  Successfully grasped object {object_id}")
    grasped_obj_center = np.mean(state.movable_object_poly_list[object_id], axis=0)
    print(f"  [After Phase 2] Object center: ({grasped_obj_center[0]:.2f}, {grasped_obj_center[1]:.2f})")
    
    # Phase 3: Navigate to goal
    print("\nPhase 3: Navigate to goal")
    path_to_goal = astar_search_to_goal(state, goal_poly)
    if path_to_goal is None:
        print("  FAILED: Cannot reach goal")
        return None, None, None
    
    # Simulate path to goal
    for action in path_to_goal:
        state = simulation_step(state, action, debug=False)
    
    final_obj_center = np.mean(state.movable_object_poly_list[object_id], axis=0)
    print(f"  [After Phase 3] Robot: ({state.robot_x:.2f}, {state.robot_z:.2f})")
    print(f"  [After Phase 3] Object center: ({final_obj_center[0]:.2f}, {final_obj_center[1]:.2f})")
    
    # Phase 4: Release
    print("\nPhase 4: Release object")
    release_action = 5
    state = simulation_step(state, release_action, debug=False)
    
    # Complete action sequence
    full_sequence = path_to_object + [grasp_action] + path_to_goal + [release_action]
    
    # Verify final position
    final_obj_poly = state.movable_object_poly_list[object_id]
    final_iou = calculate_iou(final_obj_poly, goal_poly)
    final_dist = calculate_center_distance(final_obj_poly, goal_poly)
    print(f"  SUCCESS: Object moved, IoU: {final_iou:.3f}, Distance: {final_dist:.3f}m")
    
    # Now simulate the entire sequence again to record states after each action
    print(f"\n  [SIMULATION SEQUENCE] Recording states for {len(full_sequence)} actions...")
    print(f"  [VISUALIZATION] Saving planned path to vis_planned_path/...")
    simulated_states = []
    state = initial_state
    
    # Create visualization directory
    import os
    vis_dir = os.path.join('omnigibson', 'baseline', 'IL', 'vis_planned_path')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Reset visualization counter for clean numbering
    reset_vis_counter()
    
    # Visualize initial state
    action_names = {0: "Back", 1: "Forward", 2: "Left", 3: "Right", 4: "Fetch", 5: "Release"}
    visualize_bfs_state(state, action=None, action_name="Initial_State", 
                       goal_poly=goal_poly, target_obj_id=object_id, prefix="planned_path")
    
    for i, action in enumerate(full_sequence):
        state = simulation_step(state, action, debug=False)
        simulated_states.append((state.robot_x, state.robot_z, state.robot_yaw))
        print(f"    Step {i+1}: Action={action} ({action_names.get(action, action)}), State: x={state.robot_x:.3f}, z={state.robot_z:.3f}, yaw={math.degrees(state.robot_yaw):.1f}deg")
        
        # Visualize each step
        action_desc = f"Step{i+1:03d}_{action_names.get(action, str(action))}"
        visualize_bfs_state(state, action=action, action_name=action_desc,
                           goal_poly=goal_poly, target_obj_id=object_id, prefix="planned_path")
    
    print(f"  [VISUALIZATION] Saved {len(full_sequence) + 1} visualization images to {vis_dir}")
    
    return full_sequence, state, simulated_states

# ============================================================================
# Task Planning
# ============================================================================

def task_planning_random(num_objects):
    """
    Random shuffle of object order
    """
    print("\nTask planning: RANDOM mode")
    order = list(range(num_objects))
    random.shuffle(order)
    print(f"  Object order: {order}")
    return order

def task_planning_exhaustive(initial_state, goal_polys):
    """
    Exhaustive search over object orderings
    Try all permutations and return best one
    """
    print("\nTask planning: EXHAUSTIVE mode")
    
    num_objects = len(goal_polys)
    valid_objects = [i for i, g in enumerate(goal_polys) if g is not None]
    
    if len(valid_objects) > 7:
        print(f"  WARNING: Too many objects ({len(valid_objects)}), limiting to first 7")
        valid_objects = valid_objects[:7]
    
    all_perms = list(itertools.permutations(valid_objects))
    
    if len(all_perms) > MAX_EXHAUSTIVE_PERMUTATIONS:
        print(f"  WARNING: Too many permutations ({len(all_perms)}), sampling {MAX_EXHAUSTIVE_PERMUTATIONS}")
        all_perms = random.sample(all_perms, MAX_EXHAUSTIVE_PERMUTATIONS)
    
    print(f"  Testing {len(all_perms)} permutations...")
    
    best_order = None
    best_count = -1
    
    for perm_idx, perm in enumerate(all_perms):
        if perm_idx % 10 == 0:
            print(f"    Progress: {perm_idx}/{len(all_perms)}")
        
        state = initial_state
        success_count = 0
        
        for obj_id in perm:
            goal_poly = goal_polys[obj_id]
            _, final_state, _ = motion_planning_single_object(state, obj_id, goal_poly)
            
            if final_state is None:
                break
            
            success_count += 1
            state = final_state
        
        if success_count > best_count:
            best_count = success_count
            best_order = list(perm)
            print(f"    New best: {best_order}, success: {best_count}/{len(perm)}")
    
    print(f"  Best order: {best_order}, success count: {best_count}")
    return best_order if best_order else []

# ============================================================================
# Execution in Real Environment
# ============================================================================

def execute_action_sequence(fast_env, actions, simulated_states=None):
    """
    Execute action sequence in real environment
    
    Args:
        fast_env: The FastEnv environment
        actions: List of actions to execute
        simulated_states: Optional list of (x, z, yaw) tuples for comparison
    
    Returns:
        terminated: Whether episode naturally ended with success
        truncated: Whether episode was truncated (failure)
        final_info: Last info dict from environment
    """
    print(f"\nExecuting {len(actions)} actions in real environment...")
    
    # Print initial state
    robot = fast_env._get_robot()
    robot_pos, robot_quat = robot.get_position_orientation(frame='scene')
    robot_yaw = robot.get_yaw().item()
    print(f"  Initial state: x={robot_pos[0].item():.3f}, z={robot_pos[2].item():.3f}, yaw={math.degrees(robot_yaw):.1f}deg")
    
    terminated = False
    truncated = False
    final_info = None
    
    for i, action in enumerate(actions):
        action_names = {0: "Back", 1: "Forward", 2: "Left", 3: "Right", 4: "Fetch", 5: "Release"}
        print(f"\n  Step {i+1}/{len(actions)}: Action={action} ({action_names.get(action, action)})")
        
        # Show simulated state if available
        if simulated_states and i < len(simulated_states):
            sim_x, sim_z, sim_yaw = simulated_states[i]
            print(f"    [SIMULATION] Expected: x={sim_x:.3f}, z={sim_z:.3f}, yaw={math.degrees(sim_yaw):.1f}deg")
        
        # Execute action
        obs, reward, terminated, truncated, info = fast_env.step(action)
        final_info = info
        
        # Print actual state after action
        robot_pos, robot_quat = robot.get_position_orientation(frame='scene')
        robot_yaw = robot.get_yaw().item()
        actual_x = robot_pos[0].item()
        actual_z = robot_pos[2].item()
        print(f"    [REAL ENV]   Actual:   x={actual_x:.3f}, z={actual_z:.3f}, yaw={math.degrees(robot_yaw):.1f}deg")
        
        # Compare if simulated state is available
        if simulated_states and i < len(simulated_states):
            sim_x, sim_z, sim_yaw = simulated_states[i]
            diff_x = abs(actual_x - sim_x)
            diff_z = abs(actual_z - sim_z)
            diff_yaw = abs(robot_yaw - sim_yaw)
            # Normalize yaw difference to [-pi, pi]
            while diff_yaw > math.pi:
                diff_yaw -= 2 * math.pi
            diff_yaw = abs(diff_yaw)
            
            print(f"    [COMPARISON] Diff: dx={diff_x:.4f}m, dz={diff_z:.4f}m, dyaw={math.degrees(diff_yaw):.2f}deg")
            
            if diff_x > 0.01 or diff_z > 0.01 or diff_yaw > math.radians(1):
                print(f"    [WARNING] Large discrepancy detected!")
        
        if terminated or truncated:
            print(f"    Episode ended: terminated={terminated}, truncated={truncated}")
            return terminated, truncated, final_info
    
    # If we finished executing all actions but episode didn't naturally end,
    # try to trigger termination by releasing object and repeating a move action
    print("\n  [POST-EXECUTION] Sequence complete but episode hasn't ended")
    print("  Attempting to trigger episode end: Release object + Turn left")
    
    # Release if holding anything
    if fast_env.grasping_obj is not None:
        print("    Releasing object...")
        obs, reward, terminated, truncated, info = fast_env.step(5)  # Release
        final_info = info
        if terminated or truncated:
            print(f"    Episode ended: terminated={terminated}, truncated={truncated}")
            return terminated, truncated, final_info
    
    # Repeat turn left action to consume remaining steps
    print("    Repeating turn left action...")
    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = fast_env.step(2)  # Turn left
        final_info = info
    
    print(f"  Episode ended: terminated={terminated}, truncated={truncated}")
    return terminated, truncated, final_info

# ============================================================================
# Main TAMP Pipeline
# ============================================================================

def tamp_pipeline_single_episode(fast_env):
    """
    Run TAMP pipeline for a single episode
    
    Returns:
        terminated: Whether episode succeeded
        truncated: Whether episode failed
        info: Final info dict with metrics
    """
    print("="*80)
    print("TAMP BASELINE - Task and Motion Planning")
    print("="*80)
    
    # Step 1: Extract initial state
    print("\n" + "="*80)
    print("STEP 1: Extract Environment State")
    print("="*80)
    initial_state, movable_object_names = extract_environment_state(fast_env)
    goal_polys = extract_goal_positions(fast_env, movable_object_names)
    
    # Step 2: Task planning
    print("\n" + "="*80)
    print("STEP 2: Task Planning")
    print("="*80)
    
    if TASK_PLANNING_MODE == "random":
        object_order = task_planning_random(len(movable_object_names))
    elif TASK_PLANNING_MODE == "exhaustive":
        object_order = task_planning_exhaustive(initial_state, goal_polys)
    else:
        print(f"ERROR: Unknown task planning mode: {TASK_PLANNING_MODE}")
        return None, None, None
    
    # Step 3: Motion planning and execution
    print("\n" + "="*80)
    print("STEP 3: Motion Planning and Execution")
    print("="*80)
    
    # Collect all actions for all objects
    all_actions = []
    all_simulated_states = []
    planned_objects = 0
    
    current_state = initial_state
    
    for idx, object_id in enumerate(object_order):
        print(f"\n{'='*80}")
        print(f"Object {idx+1}/{len(object_order)}: ID={object_id}, Name={movable_object_names[object_id]}")
        print(f"{'='*80}")
        
        goal_poly = goal_polys[object_id]
        if goal_poly is None:
            print("  SKIP: No goal position for this object")
            continue
        
        # Motion planning (do NOT re-sync, use simulated state)
        action_sequence, final_sim_state, simulated_states = motion_planning_single_object(current_state, object_id, goal_poly)
        
        if action_sequence is None:
            print(f"  SKIP: Cannot plan for object {object_id}")
            continue
        
        # Accumulate actions
        all_actions.extend(action_sequence)
        if simulated_states:
            all_simulated_states.extend(simulated_states)
        current_state = final_sim_state
        planned_objects += 1
        print(f"  Planned {len(action_sequence)} actions for this object")
    
    print(f"\n{'='*80}")
    print(f"Total planned objects: {planned_objects}/{len(object_order)}")
    print(f"Total actions: {len(all_actions)}")
    print(f"{'='*80}")
    
    # if len(all_actions) == 0:
    #     print("WARNING: No actions planned, skipping execution")
    #     # Return truncated = True to indicate failure
    #     return False, True, None
    
    # Step 4: Execute all actions in real environment
    print("\n" + "="*80)
    print("STEP 4: Execute in Real Environment")
    print("="*80)
    
    terminated, truncated, info = execute_action_sequence(fast_env, all_actions, all_simulated_states)
    
    return terminated, truncated, info


def tamp_evaluation_loop_multiprocess(fast_env, scene_names):
    """
    Fully automatic multiprocess evaluation loop
    Each process independently selects and tests scenes without manual configuration
    
    Args:
        fast_env: FastEnv instance
        scene_names: List of all scene names to evaluate
    """
    import time
    
    init_multiprocess_dirs()
    cleanup_temp_files()
    
    total_tested = 0
    success_count = 0
    
    print(f"\n[PID {PROCESS_PID}] Starting TAMP evaluation")
    print(f"[PID {PROCESS_PID}] Total scenes in list: {len(scene_names)}")
    print(f"[PID {PROCESS_PID}] Lock timeout: {LOCK_TIMEOUT_SECONDS}s ({LOCK_TIMEOUT_SECONDS/60:.0f} min)")
    
    consecutive_no_scene = 0
    max_consecutive_no_scene = 3
    
    while True:
        # Get available scenes
        available_scenes = get_available_scenes(scene_names)
        
        if not available_scenes:
            consecutive_no_scene += 1
            print(f"\n[PID {PROCESS_PID}] No available scenes (attempt {consecutive_no_scene}/{max_consecutive_no_scene})")
            
            if consecutive_no_scene >= max_consecutive_no_scene:
                print(f"[PID {PROCESS_PID}] No more scenes to test after {max_consecutive_no_scene} attempts. Exiting.")
                break
            
            # Wait for locks to expire
            wait_time = 10 + random.randint(0, 10)
            print(f"[PID {PROCESS_PID}] Waiting {wait_time}s for locks to expire...")
            time.sleep(wait_time)
            continue
        
        # Reset counter
        consecutive_no_scene = 0
        
        print(f"\n[PID {PROCESS_PID}] Available scenes: {len(available_scenes)}")
        
        # Randomly select a scene to avoid process conflicts
        random.shuffle(available_scenes)
        scene_claimed = False
        current_scene = None
        
        for scene in available_scenes:
            if try_claim_scene(scene):
                current_scene = scene
                scene_claimed = True
                break
            time.sleep(0.1)
        
        if not scene_claimed:
            print(f"[PID {PROCESS_PID}] Could not claim any scene, will retry...")
            time.sleep(2 + random.random())
            continue
        
        # Test this scene
        print(f"\n{'='*80}")
        print(f"[PID {PROCESS_PID}] Testing scene: {current_scene}")
        print(f"{'='*80}")
        
        try:
            # Reset environment to this scene
            fast_env.env.scene_names = [current_scene]
            obs, info = fast_env.reset(seed=42, options=None)
            
            actual_scene = fast_env.env.scene.scene_model.replace("_initial_5.json", "")
            print(f"[PID {PROCESS_PID}] Loaded scene: {actual_scene}")
            
            # Run TAMP pipeline
            terminated, truncated, final_info = tamp_pipeline_single_episode(fast_env)
            
            if terminated is None:
                print(f"[PID {PROCESS_PID}] ERROR: TAMP pipeline failed")
                release_scene(current_scene)
                continue
            
            # Extract metrics
            success = [terminated]
            _init_potential = 0
            _final_potential = 0
            
            if final_info and 'reward' in final_info and 'potential' in final_info['reward']:
                for k, v in final_info['reward']['potential']['ini'].items():
                    _init_potential += v['pos']
                for k, v in final_info['reward']['potential']['end'].items():
                    _final_potential += v['pos']
            
            init_potential = [_init_potential]
            finish_potential = [_final_potential]
            
            all_objs = [0]
            if final_info and 'reward' in final_info and 'potential' in final_info['reward']:
                all_objs = [len(final_info['reward']['potential']['ini'])]
            
            arrival_num = 0
            if final_info and 'reward' in final_info and 'arrival' in final_info['reward']:
                arrival_num = final_info['reward']['arrival']
            
            # Save result
            save_scene_result(actual_scene, success, init_potential, 
                            finish_potential, all_objs, arrival_num)
            
            # Release lock
            release_scene(current_scene)
            
            # Update statistics
            total_tested += 1
            if terminated:
                success_count += 1
                print(f"\n[PID {PROCESS_PID}] [SUCCESS] Episode completed successfully!")
            else:
                print(f"\n[PID {PROCESS_PID}] [FAILED] Episode did not complete successfully")
            
            print(f"\n[PID {PROCESS_PID}] Progress: {total_tested} tested, {success_count} succeeded")
            if total_tested > 0:
                print(f"[PID {PROCESS_PID}] Success rate: {success_count/total_tested*100:.1f}%")
            
        except Exception as e:
            print(f"[PID {PROCESS_PID}] ERROR during scene {current_scene}: {e}")
            import traceback
            traceback.print_exc()
            release_scene(current_scene)
            time.sleep(1)
    
    print(f"\n[PID {PROCESS_PID}] Finished! Tested {total_tested} scenes, {success_count} succeeded")

# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """
    Main entry point
    """
    print("Initializing OmniGibson environment...")
    
    # Load config
    config_filename = os.path.join(og.example_config_path, "rearrange.yaml")
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
    
    gm.ENABLE_OBJECT_STATES = False
    gm.ENABLE_TRANSITION_RULES = False
    gm.ENABLE_FLATCACHE = True
    gm.RENDER_VIEWER_CAMERA = False
    gm.USE_GPU_DYNMAICS = True
    
    scene_type = "Threed_FRONTScene"
    config["scene"]["type"] = scene_type
    
    threed_front_path = gm.ThreeD_FRONT_DATASET_PATH
    scene_path = os.path.join(threed_front_path, "scenes", "8148b1a7-7c15-4b53-9be3-8b5a617ba9d2_Bedroom-29109")
    
    config["scene"]["scene_model"] = "8148b1a7-7c15-4b53-9be3-8b5a617ba9d2_Bedroom-29109_target.json"
    config["scene"]["scene_type_path"] = scene_path
    config["env"]["modify_reload_model"] = True
    
    # Load scenes from test data
    scene_names = []
    with open('all.txt', 'r') as f:
        for line in f:
            parts = line.strip('\n')
            scene_names.append(parts)
    
    print(f"\n[PID {PROCESS_PID}] Total scenes to evaluate: {len(scene_names)}")
    for i, scene in enumerate(scene_names[:10]):
        print(f"  {i+1}. {scene}")
    if len(scene_names) > 10:
        print(f"  ... and {len(scene_names) - 10} more")
    
    config['env']['scene_names'] = scene_names
    
    # Robot config
    robot_name = 'Test'
    config["robots"][0]["type"] = robot_name
    config["robots"][0]["obs_modalities"] = ["rgb"]
    config["robots"][0]["action_type"] = "continuous"
    config["robots"][0]["action_normalize"] = True
    config["robots"][0]["grasping_mode"] = 'sticky'
    
    # Create environment
    env = og.Environment(configs=config)
    
    controller_config = {
        "base": {"name": "JointController"},
        "arm_0": {"name": "JointController", "motor_type": "effort"},
        "gripper_0": {"name": "MultiFingerGripperController"}
    }
    env.robots[0].reload_controllers(controller_config=controller_config)
    env.scene.update_initial_state()
    
    # Initialize FastEnv
    fast_env = FastEnv(env)
    obs, info = fast_env.reset(seed=42, options=None)
    
    print(f"\n[PID {PROCESS_PID}] Environment initialized successfully!")
    print(f"[PID {PROCESS_PID}] Scene: {fast_env.env.scene.scene_model}")
    
    # Run multiprocess-friendly TAMP evaluation loop
    tamp_evaluation_loop_multiprocess(fast_env, scene_names)

if __name__ == "__main__":
    # Use PID as part of random seed to ensure different processes have different randomness
    seed = 42 + PROCESS_PID % 1000
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    
    main()

