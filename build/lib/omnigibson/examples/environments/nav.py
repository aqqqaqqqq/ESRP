import numpy as np
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import unary_union
import numba
from typing import List, Tuple
import math

# ---- JIT-compatible functions ----

@numba.jit(nopython=True)
def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle

@numba.jit(nopython=True)
def point_in_polygon(x: float, y: float, polygon: np.ndarray) -> bool:
    """Check if point is inside a polygon using ray casting algorithm."""
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

@numba.jit(nopython=True)
def check_collision_numba(pos: np.ndarray, collision_boundaries: List[np.ndarray]) -> bool:
    """Check if position collides with any boundary."""
    # Last boundary is the environment boundary - point must be inside it
    if not point_in_polygon(pos[0], pos[1], collision_boundaries[-1]):
        return True
        
    # Check collision with obstacles - point must be outside them
    for i in range(len(collision_boundaries) - 1):
        if point_in_polygon(pos[0], pos[1], collision_boundaries[i]):
            return True
    
    return False

@numba.jit(nopython=True)
def compute_distance(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """Compute Euclidean distance between two points."""
    dx = pos2[0] - pos1[0]
    dy = pos2[1] - pos1[1]
    return np.sqrt(dx*dx + dy*dy)

@numba.jit(nopython=True)
def hash_state(pos: np.ndarray, grid_size: float = 0.1) -> Tuple[int, int]:
    """Hash a state to discrete coordinates for efficient visited checking."""
    x_grid = int(pos[0] / grid_size)
    y_grid = int(pos[1] / grid_size)
    return x_grid, y_grid

@numba.jit(nopython=True)
def compute_heuristic(pos: np.ndarray, target_pos: np.ndarray, target_min_distance: float) -> float:
    """Compute heuristic value for A* search (distance to ideal target distance)."""
    current_distance = compute_distance(pos, target_pos)
    return abs(current_distance - target_min_distance)

@numba.jit(nopython=True)
def get_movement_sequence(current_angle: float, target_angle: float) -> Tuple[List[int], float, bool]:
    """Get optimal sequence of actions to move in target direction and return final angle.
    Returns: (action_sequence, final_angle, is_forward)
    Chooses forward/backward based on which requires less rotation."""
    # Try forward movement
    angle_diff_forward = normalize_angle(target_angle - current_angle)
    num_turns_forward = int(abs(angle_diff_forward) / (np.pi / 8) + 0.5)
    
    # Try backward movement
    opposite_angle = normalize_angle(target_angle + np.pi)
    angle_diff_backward = normalize_angle(opposite_angle - current_angle)
    num_turns_backward = int(abs(angle_diff_backward) / (np.pi / 8) + 0.5)
    
    # Choose the direction that requires fewer turns
    actions = []
    if num_turns_forward <= num_turns_backward:
        # Forward movement
        turn_action = 2 if angle_diff_forward < 0 else 3
        for _ in range(num_turns_forward):
            actions.append(turn_action)
        actions.append(1)  # Forward
        return actions, target_angle, True
    else:
        # Backward movement
        turn_action = 2 if angle_diff_backward < 0 else 3
        for _ in range(num_turns_backward):
            actions.append(turn_action)
        actions.append(0)  # Backward
        return actions, opposite_angle, False
    
@numba.jit(nopython=True)
def get_movement_sequence_for_target(current_angle: float, target_angle: float) -> Tuple[List[int], float, bool]:
    """Get optimal sequence of actions to move in target direction and return final angle.
    Returns: (action_sequence, final_angle, is_forward)
    Chooses forward/backward based on which requires less rotation."""
    # Try forward movement
    angle_diff_forward = normalize_angle(target_angle - current_angle)
    num_turns_forward = int(abs(angle_diff_forward) / (np.pi / 8) + 0.5)

    # Choose the direction that requires fewer turns
    actions = []

    # Forward movement
    turn_action = 2 if angle_diff_forward < 0 else 3
    for _ in range(num_turns_forward):
        actions.append(turn_action)
    actions.append(1)  # Forward
    return actions, target_angle, True


@numba.jit(nopython=True)
def generate_actions() -> Tuple[np.ndarray, List[List[int]]]:
    """Generate discretized movement actions and their corresponding basic action sequences."""
    num_directions = 16  # Discretize into 16 directions
    movement_vectors = np.zeros((num_directions * 2, 2), dtype=np.float64)  # *2 for forward/backward
    movement_angles = np.zeros(num_directions * 2, dtype=np.float64)
    
    for i in range(num_directions):
        angle = 2 * np.pi * i / num_directions
        # Forward movement
        movement_vectors[i*2] = [np.cos(angle), np.sin(angle)]
        movement_angles[i*2] = angle
        
        # Backward movement
        movement_vectors[i*2 + 1] = [np.cos(angle), np.sin(angle)]
        movement_angles[i*2 + 1] = angle
    
    return movement_vectors, movement_angles

@numba.jit(nopython=True)
def flood_fill_core(start_pos: np.ndarray, target_pos: np.ndarray,
                   target_min_distance: float, target_max_distance: float,
                   collision_boundaries: List[np.ndarray],
                   initial_yaw: float,
                   movement_vectors: np.ndarray,
                   movement_angles: np.ndarray,
                   max_steps: int = 100000) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Core flood fill algorithm that can be JIT compiled."""
    num_actions = len(movement_vectors)
    
    # Initialize arrays
    states = np.zeros((max_steps, 3), dtype=np.float64)  # [x, y, yaw]
    priorities = np.zeros(max_steps, dtype=np.float64)  # Priority values
    visited_map = np.zeros((1000, 1000, 32), dtype=np.int8)  # Discretized state space
    parent = np.zeros(max_steps, dtype=np.int64)
    action_indices = np.zeros(max_steps, dtype=np.int8)  # Store which action sequence to use
    
    # Initialize first state with provided yaw
    states[0] = [start_pos[0], start_pos[1], initial_yaw]
    priorities[0] = compute_heuristic(start_pos, target_pos, target_min_distance)
    parent[0] = -1  # Root has no parent
    
    # Mark first state as visited
    x_grid, y_grid = hash_state(start_pos)
    angle_grid = int((normalize_angle(initial_yaw) + np.pi) / (np.pi / 16)) % 32
    visited_map[x_grid % 1000, y_grid % 1000, angle_grid] = 1
    
    queue_head = 0  # Points to the current state being processed
    queue_tail = 1  # Points to next available slot
    
    step_size = 0.25  # Robot step size
    steps = 0
    final_idx = -1
    
    while queue_head < queue_tail and steps < max_steps:
        steps += 1
        
        # Get current state
        current = states[queue_head].copy()
        current_parent = parent[queue_head]
        current_action_idx = action_indices[queue_head]
        
        # Move to next state in queue
        queue_head += 1
        
        # Check if we reached the target range
        distance = compute_distance(current[:2], target_pos)
        
        if target_min_distance <= distance <= target_max_distance:
            final_idx = queue_head - 1  # Index of the current state
            break
        
        # Try all possible movement directions
        for action_idx in range(num_actions):
            movement = movement_vectors[action_idx]
            next_pos = current[:2] + movement * step_size
            next_angle = movement_angles[action_idx]
            
            # Check collision
            if check_collision_numba(next_pos, collision_boundaries):
                continue
            
            # Check if state was visited
            x_grid, y_grid = hash_state(next_pos)
            angle_grid = int((normalize_angle(next_angle) + np.pi) / (np.pi / 16)) % 32
            if visited_map[x_grid % 1000, y_grid % 1000, angle_grid] == 1:
                continue
            
            # Add new state to queue
            if queue_tail < max_steps:
                states[queue_tail] = [next_pos[0], next_pos[1], next_angle]
                priorities[queue_tail] = compute_heuristic(next_pos, target_pos, target_min_distance)
                parent[queue_tail] = queue_head - 1  # Parent is the current state
                action_indices[queue_tail] = action_idx
                visited_map[x_grid % 1000, y_grid % 1000, angle_grid] = 1
                queue_tail += 1
        
        # Find next state to explore (minimum priority)
        if queue_head < queue_tail:
            min_idx = queue_head
            min_priority = priorities[queue_head]
            for i in range(queue_head + 1, queue_tail):
                if priorities[i] < min_priority:
                    min_priority = priorities[i]
                    min_idx = i
            
            # Swap with front of queue if needed
            if min_idx != queue_head:
                states[queue_head], states[min_idx] = states[min_idx].copy(), states[queue_head].copy()
                priorities[queue_head], priorities[min_idx] = priorities[min_idx], priorities[queue_head]
                parent[queue_head], parent[min_idx] = parent[min_idx], parent[queue_head]
                action_indices[queue_head], action_indices[min_idx] = action_indices[min_idx], action_indices[queue_head]
    
    return states, parent, action_indices, priorities, movement_angles, final_idx

@numba.jit(nopython=True)
def compute_angle_to_target(robot_pos: np.ndarray, target_pos: np.ndarray) -> float:
    """Compute the angle from robot position to target position."""
    dx = target_pos[0] - robot_pos[0]
    dy = target_pos[1] - robot_pos[1]
    return np.arctan2(dy, dx)

def flood_fill_actions(start_pos: np.ndarray, target_obstacle_pos: np.ndarray,
                      target_min_distance: float, target_max_distance: float,
                      collision_boundaries: List[np.ndarray],
                      initial_yaw: float,
                      max_steps: int = 100000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flood fill algorithm using robot action space to find path to target."""
    
    # Generate discretized movement actions and their angles
    movement_vectors, movement_angles = generate_actions()
    
    # Run core flood fill algorithm
    states, parent, action_indices, priorities, movement_angles, final_idx = flood_fill_core(
        start_pos, target_obstacle_pos, target_min_distance, target_max_distance,
        collision_boundaries, initial_yaw, movement_vectors, movement_angles, max_steps
    )
    
    if final_idx == -1:
        print("No path found!")
        return np.zeros((0, 2), dtype=np.float64), np.zeros(0, dtype=np.int8), np.zeros(2, dtype=np.float64)
    
    # Initialize path reconstruction
    path = np.zeros((max_steps, 2), dtype=np.float64)
    final_actions = []
    path_size = 0
    
    # Collect states and actions in reverse order
    state_sequence = []
    action_idx_sequence = []
    curr_idx = final_idx
    
    print("Path reconstruction:")
    while curr_idx >= 0:
        state_sequence.append(states[curr_idx])
        action_idx_sequence.append(action_indices[curr_idx])
        path[path_size] = states[curr_idx, :2]
        path_size += 1
        # print(states[curr_idx])
        curr_idx = parent[curr_idx]
    
    # Reverse sequences to get correct order
    state_sequence = state_sequence[::-1]
    action_idx_sequence = action_idx_sequence[::-1]
    
    # Process actions in correct order, starting from initial yaw
    current_angle = initial_yaw
    for i in range(len(action_idx_sequence) - 1):  # Skip the first action since it's just the initial state
        action_idx = action_idx_sequence[i + 1]  # Use i + 1 since we want the target state's action
        target_angle = movement_angles[action_idx]
        
        # Get action sequence for this movement
        actions, current_angle, _ = get_movement_sequence(current_angle, target_angle)
        final_actions.extend(actions)
    
    # After reaching the target, make the robot face the target object
    final_pos = states[final_idx, :2]
    target_angle = compute_angle_to_target(final_pos, target_obstacle_pos)
    # print("target_angle:", target_angle)
    # Get action sequence to face the target
    face_actions, _, _ = get_movement_sequence_for_target(current_angle, target_angle)
    
    # Filter out movement actions (0: backward, 1: forward), keep only rotation actions (2: left, 3: right)
    rotation_actions = [action for action in face_actions if action in [2, 3]]
    final_actions.extend(rotation_actions)
    
    print(f"Path found with {path_size} steps, {len(final_actions)} basic actions")
    return path[:path_size][::-1], np.array(final_actions), states[final_idx, :2]

# ---- Non-JIT functions (using Shapely) ----

def get_collision_boundaries(boundaries: List[Tuple[float]], 
                           obstacles: List[List[Tuple[float]]], 
                           robot_radius: float) -> List[List[Tuple[float]]]:
    """Get collision boundaries including inflated obstacles."""
    collision_boundaries = []
    
    # Add inflated obstacles
    for obstacle in obstacles:
        obs_poly = Polygon(obstacle)
        inflated = obs_poly.buffer(robot_radius)
        collision_boundaries.append(list(inflated.exterior.coords))
    
    # Add boundary walls (inward inflation)
    boundary_poly = Polygon(boundaries)
    shrunk = boundary_poly.buffer(-robot_radius)
    collision_boundaries.append(list(shrunk.exterior.coords))
    
    return collision_boundaries

def world_to_grid(point: Tuple[float, float], resolution: float = 0.05) -> Tuple[int, int]:
    """Convert world coordinates to grid coordinates."""
    return (int(point[0] / resolution), int(point[1] / resolution))

def grid_to_world(point: Tuple[int, int], resolution: float = 0.05) -> Tuple[float, float]:
    """Convert grid coordinates to world coordinates."""
    return (point[0] * resolution, point[1] * resolution)

def path_to_actions(path: List[Tuple[int, int]], robot_center: Tuple[float, float]) -> List[int]:
    """Convert path to robot actions."""
    actions = []
    current_pos = np.array(robot_center)
    current_angle = 0
    
    for i in range(len(path) - 1):
        current = path[i]
        next_pos = path[i + 1]
        
        # Convert grid coordinates to world coordinates
        current_world = np.array(grid_to_world(current))
        next_world = np.array(grid_to_world(next_pos))
        
        # Calculate angle to next position
        dx = next_world[0] - current_world[0]
        dy = next_world[1] - current_world[1]
        target_angle = math.atan2(dy, dx)
        
        # Calculate angle difference
        angle_diff = target_angle - current_angle
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
            
        # Add rotation actions
        while abs(angle_diff) > math.pi / 8:
            if angle_diff > 0:
                actions.append(2)  # Left turn
                angle_diff -= math.pi / 8
            else:
                actions.append(3)  # Right turn
                angle_diff += math.pi / 8
                
        # Add forward action
        actions.append(0)
        current_pos = next_world
        current_angle = target_angle
    
    return actions

def find_path(obstacles: List[List[Tuple[float]]], 
              boundaries: List[Tuple[float]], 
              robot_center: Tuple[float, float],
              robot_radius: float,
              target: int,
              target_min_distance: float,
              target_max_distance: float,
              initial_yaw: float = 0.0) -> Tuple[Tuple[float, float], List[int]]:
    """Find path to target using robot action space."""
    # Get target obstacle center
    target_obstacle = Polygon(obstacles[target])
    target_center = np.array(target_obstacle.centroid.coords[0])
    
    # Get collision boundaries
    collision_boundaries = get_collision_boundaries(boundaries, obstacles, robot_radius)
    collision_boundaries_array = [np.array(boundary) for boundary in collision_boundaries]
    
    # Find path using action space flood fill
    path, actions, target_point = flood_fill_actions(
        np.array(robot_center),
        target_center,
        target_min_distance,
        target_max_distance,
        collision_boundaries_array,
        initial_yaw
    )
    
    if len(path) == 0:
        raise ValueError("No path found to target")
    
    return tuple(target_point), actions.tolist()
