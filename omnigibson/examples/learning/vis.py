import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MPLPolygon
from matplotlib.patches import Circle
import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
from shapely.geometry import Polygon, Point

# Global counter for visualization ordering
_vis_counter = 0

def reset_vis_counter():
    """Reset visualization counter"""
    global _vis_counter
    _vis_counter = 0

def visualize_bfs_state(state, action=None, action_name=None, goal_poly=None, target_obj_id=None, prefix="bfs"):
    """
    Visualize BFS search state for debugging
    
    Args:
        state: Current State object from motion_planning
        action: Action taken (0-5) or None
        action_name: Name of action or description
        goal_poly: Goal polygon (Polygon or list) for target visualization
        target_obj_id: ID of target object to highlight
        prefix: Prefix for filename
    """
    global _vis_counter
    _vis_counter += 1
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 1. Visualize floor area
    floor_poly = Polygon(state.floor_poly)
    x, y = floor_poly.exterior.xy
    ax.fill(x, y, alpha=0.3, fc='yellow', ec='black', linewidth=2, label='Floor Area')
    
    # 2. Visualize unmovable obstacles
    for i, obstacle in enumerate(state.unmovable_object_poly_list):
        obs_poly = Polygon(obstacle)
        x, y = obs_poly.exterior.xy
        ax.fill(x, y, alpha=0.6, fc='gray', ec='black', linewidth=1, label='Unmovable' if i == 0 else '')
    
    # 3. Visualize movable objects
    for i, obj_poly_list in enumerate(state.movable_object_poly_list):
        obj_poly = Polygon(obj_poly_list)
        x, y = obj_poly.exterior.xy
        
        # Highlight target object
        if target_obj_id is not None and i == target_obj_id:
            ax.fill(x, y, alpha=0.8, fc='orange', ec='red', linewidth=3, label=f'Target Object {i}')
        # Highlight grasping object
        elif state.grasping_object_id is not None and i == state.grasping_object_id:
            ax.fill(x, y, alpha=0.8, fc='purple', ec='magenta', linewidth=3, label=f'Grasping Object {i}')
        else:
            ax.fill(x, y, alpha=0.5, fc='lightblue', ec='blue', linewidth=1, label='Movable' if i == 0 else '')
        
        # Add object ID label
        center = np.mean(obj_poly_list, axis=0)
        ax.text(center[0], center[1], f'Obj{i}', fontsize=8, ha='center', va='center', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # 4. Visualize goal position (if provided)
    if goal_poly is not None:
        if isinstance(goal_poly, Polygon):
            goal_coords = list(goal_poly.exterior.coords[:-1])
        else:
            goal_coords = goal_poly
        
        goal_shapely = Polygon(goal_coords)
        x, y = goal_shapely.exterior.xy
        ax.plot(x, y, 'g--', linewidth=3, label='Goal Position')
        ax.fill(x, y, alpha=0.2, fc='green')
        
        # Add goal label
        goal_center = np.mean(goal_coords, axis=0)
        ax.text(goal_center[0], goal_center[1], 'GOAL', fontsize=10, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='green', alpha=0.5), color='white', weight='bold')
    
    # 5. Visualize robot
    robot_center = Point(state.robot_x, state.robot_z)
    robot_circle = robot_center.buffer(state.robot_radius)
    x, y = robot_circle.exterior.xy
    ax.fill(x, y, alpha=0.7, fc='red', ec='darkred', linewidth=2, label='Robot')
    
    # Add robot direction indicator (arrow showing yaw)
    arrow_length = state.robot_radius * 1.5
    arrow_dx = arrow_length * np.cos(state.robot_yaw)
    arrow_dy = arrow_length * np.sin(state.robot_yaw)
    ax.arrow(state.robot_x, state.robot_z, arrow_dx, arrow_dy, 
             head_width=0.15, head_length=0.1, fc='darkred', ec='darkred', linewidth=2)
    
    # Add robot position text
    ax.text(state.robot_x, state.robot_z - state.robot_radius - 0.2, 
            f'Robot\n({state.robot_x:.2f}, {state.robot_z:.2f})\nYaw: {np.degrees(state.robot_yaw):.1f}°',
            fontsize=9, ha='center', va='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.7), color='white')
    
    # 6. Add title with state information
    title_parts = [f'Step {_vis_counter}']
    if action_name:
        title_parts.append(f'Action: {action_name}')
    if action is not None and action in [0,1,2,3]:
        action_names = {0: 'Back', 1: 'Forward', 2: 'Left', 3: 'Right'}
        title_parts.append(f'({action_names[action]})')
    
    if state.grasping_object_id is not None:
        title_parts.append(f'[Grasping: Obj{state.grasping_object_id}]')
    else:
        title_parts.append('[Not Grasping]')
    
    ax.set_title(' | '.join(title_parts), fontsize=14, weight='bold')
    
    # 7. Set labels and formatting
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Z (meters)', fontsize=12)
    ax.set_aspect('equal', 'box')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=10)
    
    # 8. Save figure
    # Determine output directory based on prefix
    if prefix == "planned_path":
        vis_dir = os.path.join('omnigibson', 'baseline', 'IL', 'vis_planned_path')
    else:
        vis_dir = os.path.join('omnigibson', 'baseline', 'IL', 'vis')
    os.makedirs(vis_dir, exist_ok=True)
    
    filename = f'{prefix}_{_vis_counter:05d}.png'
    save_path = os.path.join(vis_dir, filename)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    
    return save_path

def visualize_collision_test(old_state, new_state, collision_result, action, action_name):
    """
    Visualize collision detection result
    Shows both old and new robot positions
    
    Args:
        old_state: State before action
        new_state: State after action attempt
        collision_result: True if no collision, False if collision detected
        action: Action ID
        action_name: Action name string
    """
    global _vis_counter
    _vis_counter += 1
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 1. Floor
    floor_poly = Polygon(old_state.floor_poly)
    x, y = floor_poly.exterior.xy
    ax.fill(x, y, alpha=0.3, fc='yellow', ec='black', linewidth=2, label='Floor Area')
    
    # 2. Obstacles (unmovable)
    for i, obstacle in enumerate(old_state.unmovable_object_poly_list):
        obs_poly = Polygon(obstacle)
        x, y = obs_poly.exterior.xy
        ax.fill(x, y, alpha=0.6, fc='gray', ec='black', linewidth=1, label='Unmovable' if i == 0 else '')
    
    # 3. Movable objects (use new state to show updated positions)
    for i, obj_poly_list in enumerate(new_state.movable_object_poly_list):
        obj_poly = Polygon(obj_poly_list)
        x, y = obj_poly.exterior.xy
        
        if new_state.grasping_object_id is not None and i == new_state.grasping_object_id:
            ax.fill(x, y, alpha=0.8, fc='purple', ec='magenta', linewidth=2, label=f'Grasping Obj {i}')
        else:
            ax.fill(x, y, alpha=0.5, fc='lightblue', ec='blue', linewidth=1, label='Movable' if i == 0 else '')
    
    # 4. Old robot position (semi-transparent)
    old_robot_center = Point(old_state.robot_x, old_state.robot_z)
    old_robot_circle = old_robot_center.buffer(old_state.robot_radius)
    x, y = old_robot_circle.exterior.xy
    ax.fill(x, y, alpha=0.3, fc='blue', ec='blue', linewidth=1, linestyle='--', label='Old Robot')
    
    # Old robot direction
    arrow_length = old_state.robot_radius * 1.2
    arrow_dx = arrow_length * np.cos(old_state.robot_yaw)
    arrow_dy = arrow_length * np.sin(old_state.robot_yaw)
    ax.arrow(old_state.robot_x, old_state.robot_z, arrow_dx, arrow_dy,
             head_width=0.1, head_length=0.08, fc='blue', ec='blue', alpha=0.3, linewidth=1)
    
    # 5. New robot position
    new_robot_center = Point(new_state.robot_x, new_state.robot_z)
    new_robot_circle = new_robot_center.buffer(new_state.robot_radius)
    x, y = new_robot_circle.exterior.xy
    
    if collision_result:
        # No collision - green
        ax.fill(x, y, alpha=0.7, fc='green', ec='darkgreen', linewidth=2, label='New Robot (Valid)')
        arrow_color = 'darkgreen'
    else:
        # Collision detected - red
        ax.fill(x, y, alpha=0.7, fc='red', ec='darkred', linewidth=2, label='New Robot (Collision!)')
        arrow_color = 'darkred'
    
    # New robot direction
    arrow_dx = arrow_length * np.cos(new_state.robot_yaw)
    arrow_dy = arrow_length * np.sin(new_state.robot_yaw)
    ax.arrow(new_state.robot_x, new_state.robot_z, arrow_dx, arrow_dy,
             head_width=0.15, head_length=0.1, fc=arrow_color, ec=arrow_color, linewidth=2)
    
    # 6. Title
    status = 'VALID MOVE' if collision_result else 'COLLISION DETECTED'
    status_color = 'green' if collision_result else 'red'
    title = f'Step {_vis_counter} | Action: {action_name} | {status}'
    ax.set_title(title, fontsize=14, weight='bold', color=status_color)
    
    # 7. Add movement arrow
    ax.plot([old_state.robot_x, new_state.robot_x], 
            [old_state.robot_z, new_state.robot_z],
            'k--', linewidth=2, alpha=0.5, label='Movement')
    
    # 8. Formatting
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Z (meters)', fontsize=12)
    ax.set_aspect('equal', 'box')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=10)
    
    # 9. Save
    vis_dir = os.path.join('omnigibson', 'baseline', 'IL', 'vis')
    os.makedirs(vis_dir, exist_ok=True)
    
    filename = f'collision_{_vis_counter:05d}.png'
    save_path = os.path.join(vis_dir, filename)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    
    return save_path

