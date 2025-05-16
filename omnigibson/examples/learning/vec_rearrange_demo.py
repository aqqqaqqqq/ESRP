import os
import random
import torch as th
import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import gm
from omnigibson.robots import REGISTERED_ROBOTS
from omnigibson.utils.ui_utils import KeyboardRobotController, choose_from_options
import omnigibson.utils.transform_utils as T
from omnigibson.utils.asset_utils import get_available_g_scenes, get_available_og_scenes, get_available_3dfront_scenes, get_available_3dfront_rooms, get_available_3dfront_room, get_available_3dfront_target_scenes
import math
import argparse
import time
import yaml
from omnigibson.utils.python_utils import meets_minimum_version
from omnigibson.utils.gym_utils import (
    GymObservable,
    maxdim,
    recursively_generate_flat_dict,
)
from omnigibson.object_states.contact_bodies import ContactBodies
import copy
from tqdm import trange
from shapely.geometry import Point, Polygon
from omnigibson.envs.env_base import Environment
import datetime
import matplotlib.pyplot as plt
import matplotlib
from omnigibson.utils.constants import PrimType
import numpy as np
import itertools
import cv2
import cProfile
import pstats
from collections import defaultdict
matplotlib.use('Agg')
profiler = cProfile.Profile() 

try:
    import gymnasium as gym
    import tensorboard
    import torch as th
    import torch.nn as nn
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.preprocessing import maybe_transpose
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn
except ModuleNotFoundError:
    og.log.error(
        "torch, stable-baselines3, or tensorboard is not installed. "
        "See which packages are missing, and then run the following for any missing packages:\n"
        "pip install stable-baselines3[extra]\n"
        "pip install tensorboard\n"
        "pip install shimmy>=0.2.1\n"
        "Also, please update gym to >=0.26.1 after installing sb3: pip install gym>=0.26.1"
    )
    exit(1)

assert meets_minimum_version(gym.__version__, "0.28.1"), "Please install/update gymnasium to version >= 0.28.1"

# We don't need object states nor transitions rules, so we disable them now, and also enable flatcache for maximum speed
gm.ENABLE_OBJECT_STATES = False
gm.ENABLE_TRANSITION_RULES = False
gm.ENABLE_FLATCACHE = True
gm.RENDER_VIEWER_CAMERA = True

# Keep track of the last used env and what time, to require that others be reset before getting used
last_stepped_env = None
last_stepped_time = None

def visualize_environment(floor_area_poly, obstacles_poly, new_robot_circle, old_robot_circle, new_obj_poly=None, old_obj_poly=None):
    fig, ax = plt.subplots()

    # 1. 可视化 floor polygons
    x, y = floor_area_poly.exterior.xy
    ax.fill(x, y, alpha=0.5, fc='yellow', label='Floor Area')

    # 2. 可视化 obstacles polygons
    for obstacle_poly in obstacles_poly:
        x, y = obstacle_poly.exterior.xy
        ax.fill(x, y, alpha=0.5, fc='red', label='Obstacle')

    # 3. 可视化 机器人新旧位置
    # 旧机器人位置
    x, y = old_robot_circle.exterior.xy
    ax.fill(x, y, alpha=0.5, fc='blue', label='Old Robot')

    # 新机器人位置
    x, y = new_robot_circle.exterior.xy
    ax.fill(x, y, alpha=0.5, fc='green', label='New Robot')

    # 4. 可视化 grasping obj 新旧位置 (如果有)
    if old_obj_poly is not None and new_obj_poly is not None:
        # 旧抓取物体位置
        x, y = old_obj_poly.exterior.xy
        ax.fill(x, y, alpha=0.5, fc='purple', label='Old Grasping Obj')

        # 新抓取物体位置
        x, y = new_obj_poly.exterior.xy
        ax.fill(x, y, alpha=0.5, fc='orange', label='New Grasping Obj')

    # 设置显示
    ax.set_title('Environment Visualization')
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.legend(loc='upper left')
    ax.set_aspect('equal', 'box')

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # 设置保存路径
    save_path = os.path.join('vis', f'visualization_{timestamp}.png')

    # 确保 vis 目录存在
    os.makedirs('vis', exist_ok=True)

    # 保存可视化图像
    plt.savefig(save_path)

_cache = {}
class OccupancyInfo:
    def _sort_axis_aligned_polygon(points, eps=1e-9):
        unique = []
        for p in points:
            if not any(abs(p[0] - q[0]) < eps and abs(p[1] - q[1]) < eps for q in unique):
                unique.append(p)
        points = unique
        candidates_set = set()
        x_groups = defaultdict(list)
        z_groups = defaultdict(list)
        for p in points:
            x, z = p[0], p[1]
            x_groups[round(x, 6)].append(p)
            z_groups[round(z, 6)].append(p)
        for x, plist in x_groups.items():
            plist.sort(key=lambda p: p[1])
            candidates_set.add(tuple(plist[0]))
            candidates_set.add(tuple(plist[-1]))
        for z, plist in z_groups.items():
            plist.sort(key=lambda p: p[0])
            candidates_set.add(tuple(plist[0]))
            candidates_set.add(tuple(plist[-1]))
        candidates = [list(t) for t in candidates_set]

        n = len(candidates)
        neighbors = {i: [] for i in range(n)}
        for i in range(n):
            for j in range(i + 1, n):
                p = candidates[i]
                q = candidates[j]
                if abs(p[0] - q[0]) < eps:
                    valid = True
                    for k in range(n):
                        if k == i or k == j:
                            continue
                        r = candidates[k]
                        if abs(r[0] - p[0]) < eps and (min(p[1], q[1]) < r[1] < max(p[1], q[1])):
                            valid = False
                            break
                    if valid:
                        neighbors[i].append(j)
                        neighbors[j].append(i)
                if abs(p[1] - q[1]) < eps:
                    valid = True
                    for k in range(n):
                        if k == i or k == j:
                            continue
                        r = candidates[k]
                        if abs(r[1] - p[1]) < eps and (min(p[0], q[0]) < r[0] < max(p[0], q[0])):
                            valid = False
                            break
                    if valid:
                        neighbors[i].append(j)
                        neighbors[j].append(i)

        max_z = max(p[1] for p in candidates)
        start_candidates = [(p, i) for i, p in enumerate(candidates) if abs(p[1] - max_z) < eps]
        start_candidates.sort(key=lambda x: x[0][0])
        start_index = start_candidates[0][1]
        start_neis = neighbors[start_index]
        if not start_neis:
            return []
        next_index = None
        for ni in start_neis:
            if abs(candidates[ni][1] - candidates[start_index][1]) < eps:
                next_index = ni
                break
        if next_index is None:
            next_index = start_neis[0]

        order = [start_index]
        prev_index = start_index
        current_index = next_index
        while current_index != start_index:
            order.append(current_index)
            nbrs = neighbors[current_index]
            if len(nbrs) == 1:
                break
            elif len(nbrs) == 2:
                next_index = nbrs[0] if nbrs[0] != prev_index else nbrs[1]
            else:
                next_index = nbrs[0] if nbrs[0] != prev_index else nbrs[1]
            prev_index, current_index = current_index, next_index

        ordered_candidates = [candidates[i] for i in order]

        area = 0
        m = len(ordered_candidates)
        for i in range(m):
            x1, z1 = ordered_candidates[i][0], ordered_candidates[i][1]
            x2, z2 = ordered_candidates[(i + 1) % m][0], ordered_candidates[(i + 1) % m][1]
            area += x1 * z2 - x2 * z1
        if area < 0:
            ordered_candidates.reverse()

        return ordered_candidates

    def _get_floor(env):
        all_objects = env.scene._init_objs
        for obj_name, obj in all_objects.items():
            if obj.mesh_type == "Floor":
                return obj

    def _remove_duplicate_vertices(floor_poly):     
        seen = set()
        unique_poly = []
        for vertex in floor_poly:
            vertex_tuple = tuple(vertex)
            if vertex_tuple not in seen:
                seen.add(vertex_tuple)
                unique_poly.append(vertex)
        return unique_poly

    def _remove_useless_points(_points):
        _points = OccupancyInfo._sort_axis_aligned_polygon(_points)
        j = 0
        while j < (len(_points)):
            a = _points[j]
            b = _points[(j + 1) % len(_points)]
            c = _points[(j + 2) % len(_points)]

            def is_point_colinear(p, a, b, epsilon=1e-8):
                area = (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])
                return abs(area) < epsilon

            if is_point_colinear(b, a, c):
                _points.pop((j + 1) % len(_points))
                continue
            j += 1
        return _points

    def get_floor_area(env):
        floor = OccupancyInfo._get_floor(env)
        floor_xyz = floor.floor_xyz
        num_points = len(floor_xyz) // 3
        floor_vertices = []
        for i in range(num_points):
            x_floor = floor_xyz[3*i]
            y_floor = floor_xyz[3*i+1]
            z_floor = floor_xyz[3*i+2]
            floor_vertices.append([x_floor, y_floor, z_floor])
        floor_vertices = OccupancyInfo._remove_duplicate_vertices(floor_vertices)
        floor_poly = [[v[0], v[2]] for v in floor_vertices]
        boundaries = OccupancyInfo._remove_useless_points(floor_poly)
        return boundaries

    def _get_base_aligned_bbox(_obj):
        assert _obj.prim_type != PrimType.CLOTH
        # Get the base position transform.
        pos, orn = _obj.get_position_orientation()
        base_frame_to_world = T.pose2mat((pos, orn))
        # Prepare the desired frame.
        desired_frame_to_world = base_frame_to_world

        # Compute the world-to-base frame transform.
        world_to_desired_frame = th.linalg.inv_ex(desired_frame_to_world).inverse
        # Grab all the world-frame points corresponding to the object's visual or collision hulls.
        points_in_world = []
        for link_name, link in _obj._links.items():
            hull_points = link.collision_boundary_points_world
            points_in_world.append(hull_points)

        all_points_tensor = th.concat(points_in_world)
        # Move the points to the desired frame
        points = T.transform_points(th.tensor(all_points_tensor, dtype=th.float32), world_to_desired_frame)

        # All points are now in the desired frame: either the base CoM or the xy-plane-aligned base CoM.
        # Now fit a bounding box to all the points by taking the minimum/maximum in the desired frame.
        aabb_min_in_desired_frame = th.amin(points, dim=0)
        aabb_max_in_desired_frame = th.amax(points, dim=0)
        bbox_center_in_desired_frame = (aabb_min_in_desired_frame + aabb_max_in_desired_frame) / 2
        bbox_extent_in_desired_frame = aabb_max_in_desired_frame - aabb_min_in_desired_frame
        # points = th.tensor([])
        return bbox_extent_in_desired_frame, bbox_center_in_desired_frame, pos, orn

    def _get_obj_bbox(env, obj_name):
        """
        Returns 4 points of bbox
        np.array([
        np.array([x1, z1]),
        np.array([x2, z2]),
        np.array([x3, z3]),
        np.array([x4, z4]),
        ])
        """
        _obj = env.scene._init_objs[obj_name]
        bbox_extent_in_desired_frame, bbox_center_in_desired_frame, desired_pos, desired_quat = OccupancyInfo._get_base_aligned_bbox(_obj)
        dx, dy, dz = bbox_extent_in_desired_frame
        x, y, z = bbox_center_in_desired_frame
        # print(a, b)
        # print(dx, dy, dz)

        # only take 4 upper points
        bbox_in_desired_frame = [(x + sx * 0.5 * dx, y + sy * 0.5 * dy, z + sz * 0.5 * dz) for sx in (1, -1) for sy in (1,) for sz in (1, -1)]
        desired_pos, desired_quat = _obj.get_position_orientation(frame = 'scene')
        bbox_in_world = []
        for point in bbox_in_desired_frame:
            bbox_in_world.append((T.quat_apply(desired_quat, th.tensor(point)) + desired_pos).tolist())
        bbox_in_world = [bbox_in_world[0], bbox_in_world[1], bbox_in_world[3], bbox_in_world[2]]
        # print("bbox_in_world:", bbox_in_world)
        bbox_in_world = np.array([np.array([x[0], x[2]]) for x in bbox_in_world])
        # print(f"{obj_name} bbox:", bbox_in_world)
        return bbox_in_world
    
    def _get_mvbb_local(_obj):
        assert _obj.prim_type != PrimType.CLOTH

        # Check if the result is already cached
        if _obj in _cache:
            return _cache[_obj]
        
        # Compute the world-to-base frame transform.
        points_in_world = []
        link = None
        for _, link in _obj._links.items():
            hull_points = link.collision_boundary_points_local
            points_in_world.append(hull_points)

        all_points_tensor = th.concat(points_in_world)
        points_np = all_points_tensor.cpu().numpy()
        res = th.FloatTensor(points_np), link
        
        # Cache the result
        _cache[_obj] = res
        
        return res

    def _get_obj_bbox_new(env, obj_name):
        """
        Returns 4 points of bbox
        np.array([
        np.array([x1, z1]),
        np.array([x2, z2]),
        np.array([x3, z3]),
        np.array([x4, z4]),
        ])
        """
        a = time.perf_counter()
        _obj = env.scene._init_objs[obj_name]
        b = time.perf_counter()
        bbox_in_local, link = OccupancyInfo._get_mvbb_local(_obj)
        c = time.perf_counter()
        bbox_in_world = link.transform_local_points_to_world(bbox_in_local).numpy()
        d = time.perf_counter()
        bbox_in_world = bbox_in_world[:, [0, 2]]
        hull = cv2.convexHull(bbox_in_world)
        e = time.perf_counter()
        # bbox_in_world = bbox_in_world[hull.vertices].tolist()
        hull = hull.reshape(-1, 2)
        if not np.allclose(hull[0], hull[-1]):
            hull = np.vstack([hull, hull[0]])

        bbox_in_world = hull
        f = time.perf_counter()
        print('_get_obj_bbox_new', b - a, c - b, d - c, e - d, f - e)
        return bbox_in_world

    def _get_all_objects_prim_paths(env):
        objects = []
        all_objects = env.scene._init_objs
        for obj_name, obj in all_objects.items():
            if "object" in obj_name:
                objects.append(obj.prim_path)
        # print("num:", len(self.objects_to_rearrange))
        return objects
    
    def _get_all_objects_link_prim_paths(env):
        objects = []
        all_objects = env.scene._init_objs
        for obj_name, obj in all_objects.items():
            if "object" in obj_name:
                for link in obj.links.values():
                    objects.append(link.prim_path)
        # print("num:", len(self.objects_to_rearrange))
        return objects
    
    def get_obstacles(env):
        object_names = env.task.get_all_objects_names(env)
        obstacles = {}
        for object_name in object_names:
            _tensor = OccupancyInfo._get_obj_bbox(env, object_name)
            obstacles[object_name] = _tensor

        return obstacles

def _keep_relative_to_robot(pos, quat, old_robot_pos, old_robot_quat, new_robot_pos, new_robot_quat):
    pos_in_robot_frame, quat_in_robot_frame = T.relative_pose_transform(pos, quat, old_robot_pos, old_robot_quat)
    pos_in_world_frame, quat_in_world_frame = T.pose_transform(new_robot_pos, new_robot_quat, pos_in_robot_frame, quat_in_robot_frame)
    return pos_in_world_frame, quat_in_world_frame

def _check_collision_2d(env, old_robot_pos, old_robot_quat, new_robot_pos, new_robot_quat, grasping_obj):
    a0 = time.perf_counter()
    floor_area = OccupancyInfo.get_floor_area(env)
    floor_area_poly = Polygon(floor_area)
    a1 = time.perf_counter()
    obstacles = OccupancyInfo.get_obstacles(env)
    a2 = time.perf_counter()
    # get robot
    new_robot_center = Point(new_robot_pos[0], new_robot_pos[2])
    new_robot_circle = new_robot_center.buffer(0.5)
    if grasping_obj:
        grasping_obj_name = grasping_obj.split('/')[-1]
        grasping_obj_poly = obstacles[grasping_obj_name]
        new_grasping_obj_poly = []

        for p in grasping_obj_poly:
            _pos = th.tensor([p[0], 0.0, p[1]])
            _quat = T.euler2quat(th.tensor([0.0, 0.0, 0.0]))
            new_pos, _ = _keep_relative_to_robot(_pos, _quat, old_robot_pos, old_robot_quat, new_robot_pos, new_robot_quat)
            new_grasping_obj_poly.append([new_pos[0], new_pos[2]])

        new_obj_poly = Polygon(new_grasping_obj_poly)
        obstacles_without_grasping_obj = [v for k, v in obstacles.items() if k != grasping_obj_name]
    else:
        obstacles_without_grasping_obj = [v for k, v in obstacles.items()]

    a3 = time.perf_counter()
    obstacles_poly = [Polygon(obstacle) for obstacle in obstacles_without_grasping_obj]
    a4 = time.perf_counter()
    
    moved_polygons = [new_robot_circle, new_obj_poly] if grasping_obj is not None else [new_robot_circle]
    any_intersection = any(p1.intersects(p2) for p1, p2 in itertools.product(obstacles_poly, moved_polygons))
    all_cover = all([floor_area_poly.covers(poly) for poly in moved_polygons])
    valid = (not any_intersection and all_cover)
    # a5 = time.perf_counter()
    # print(a1 - a0, a2 - a1, a3 - a2, a4 - a3, a5 - a4)
    # FOR VISULIZATION ONLY
    # old_robot_center = Point(old_robot_pos[0], old_robot_pos[2])
    # old_robot_circle = old_robot_center.buffer(0.5)

    # if grasping_obj:
    #     _grasping_obj_poly = Polygon(grasping_obj_poly)
    #     visualize_environment(floor_area_poly, obstacles_poly, new_robot_circle, old_robot_circle, new_obj_poly, _grasping_obj_poly)
    # else:
    #     visualize_environment(floor_area_poly, obstacles_poly, new_robot_circle, old_robot_circle)
    return valid

def _move(env, diff_x, diff_yaw, grasping_obj):
    _pos = th.tensor([diff_x, 0.0, 0.0])
    _quat = T.euler2quat(th.tensor([0.0, 0.0, diff_yaw]))
    robot_pos, robot_quat = env.robots[0].get_position_orientation(frame = 'scene')
    next_robot_pos, next_robot_quat = T.pose_transform(robot_pos, robot_quat, _pos, _quat)
    if grasping_obj is not None:
        grasping_obj_pos, grasping_obj_quat = env.scene.object_registry('prim_path', grasping_obj).get_position_orientation(frame = 'scene')
        new_grasping_obj_pos, new_grasping_obj_quat = _keep_relative_to_robot(grasping_obj_pos, grasping_obj_quat, robot_pos, robot_quat, next_robot_pos, next_robot_quat)

    if _check_collision_2d(env, robot_pos, robot_quat, next_robot_pos, next_robot_quat, grasping_obj):
        # keep the height
        next_robot_pos[1] = robot_pos[1]
        env.robots[0].set_position_orientation(next_robot_pos, next_robot_quat, frame = 'scene')
        if grasping_obj is not None:
            env.scene.object_registry('prim_path', grasping_obj).set_position_orientation(new_grasping_obj_pos, new_grasping_obj_quat, frame = 'scene')

    
def _fetch(env, grasping_obj):
    if grasping_obj is not None:
        return grasping_obj
    _robot = env.robots[0]
    THETA = math.pi / 2
    R = 1.0
    VISIBLE_AREA_R = R * 2

    bbox_in_world = []
    polygons = []
    for _obj in list(_robot.scene._init_objs.values()):
        if 'test' in _obj.prim_path:
            continue

        if _obj.fixed_base:
                continue

        have_raw_model_prim_path = False
        raw_model_prim = None
        for _, _link in _obj.links.items():
            if 'raw_model' in _link.prim_path:
                raw_model_prim = _link
                have_raw_model_prim_path = True

        if not have_raw_model_prim_path:
            continue

        a, b, bbox_extent_in_desired_frame, bbox_center_in_desired_frame = _obj.get_base_aligned_bbox()
        dx, dy, dz = bbox_extent_in_desired_frame
        x, y, z = bbox_center_in_desired_frame

        # only take 4 upper points
        bbox_in_desired_frame = [(x + sx * 0.5 * dx, y + sy * 0.5 * dy, z + sz * 0.5 * dz) for sx in (1, -1) for sy in (1,) for sz in (1, -1)]
        desired_pos, desired_quat = _obj.get_position_orientation()

        bbox_in_world = []
        for point in bbox_in_desired_frame:
            bbox_in_world.append((T.quat_apply(desired_quat, th.tensor(point)) + desired_pos).tolist())
        
        bbox_in_world = [(x[0], x[2]) for x in bbox_in_world]
        polygons.append((raw_model_prim, Polygon([bbox_in_world[0], bbox_in_world[1], bbox_in_world[3], bbox_in_world[2]])))

    yaw = _robot.get_yaw()
    robot_x, _, robot_z = _robot.get_position_orientation()[0]
    point_1 = (robot_x + VISIBLE_AREA_R * math.cos(yaw + THETA / 2), robot_z + VISIBLE_AREA_R * math.sin(yaw + THETA / 2))
    point_2 = (robot_x + VISIBLE_AREA_R * math.cos(yaw - THETA / 2), robot_z + VISIBLE_AREA_R * math.sin(yaw - THETA / 2))
    visible_area = Polygon([(robot_x, robot_z), point_1, point_2])
    robot_point = Point(robot_x, robot_z)

    filtered_polygons = [p for p in polygons if p[1].intersects(visible_area) and robot_point.distance(p[1]) <= R]

    sorted_polygons = sorted(filtered_polygons, key=lambda x: robot_point.distance(x[1]))
    if len(sorted_polygons):
        ag_prim_path = sorted_polygons[0][0].prim_path
        ag_obj_prim_path = "/".join(ag_prim_path.split("/")[:-1])
        return ag_obj_prim_path
    
    return None 

def _step(env, action, grasping_obj):
    action2diff = {0: (-0.25, 0.0), 1: (0.25, 0.0), 2: (0.0, math.pi / 8), 3: (0.0, -math.pi / 8)}
    if action in {0, 1, 2, 3}:
        _move(env, *action2diff[action], grasping_obj)
    else:
        if action == 4:
            grasping_obj = _fetch(env, grasping_obj)
        elif action == 5:
            grasping_obj = None
        else:
            raise ValueError
    
    env.robots[0].keep_still()
    return grasping_obj

class RearrangeVecEnv(DummyVecEnv):
    """符合gym接口的机器人重排环境"""
    
    def __init__(self, num_envs, config, render_on_step):
        self.num_envs = num_envs
        self.render_on_step = render_on_step

        # First we create the environments. We can't let DummyVecEnv do this for us because of the play call
        # needing to happen before spaces are available for it to read things from.
        tmp_envs = [
            Environment(configs=copy.deepcopy(config), in_vec_env=True)
            for _ in trange(num_envs, desc="Loading environments")
        ]

        top_down_position = th.tensor([5, 40.0, 0])
        # top_down_orientation = th.tensor([0.0, 0.0, -0.70711, 0.70711])
        top_down_orientation = th.tensor([-0.5, -0.5, -0.5, 0.5])
        # top_down_orientation = th.tensor([0.0, -0.70711, 0.70711, 0.0])
        cam = og.sim.viewer_camera
        cam.set_position_orientation(top_down_position, top_down_orientation)
        self.grasping_obj_list = [None for _ in range(self.num_envs)]

        # Play, and finish loading all the envs
        og.sim.play()
        
        for env in tmp_envs:
            env.post_play_load()

        # Now produce some functions that will make DummyVecEnv think it's creating these envs itself
        env_fns = [lambda env_=env: env_ for env in tmp_envs]
        super().__init__(env_fns)

        # Keep track of our last reset time
        self.last_reset_time = time.time()

    def step_async(self, actions: th.tensor) -> None:
        for idx, (_env, _action) in enumerate(zip(self.envs, actions.tolist())):
            self.grasping_obj_list[idx] = _step(_env, _action, self.grasping_obj_list[idx])
        # import pdb; pdb.set_trace()
        # We go into this context in case the pre-step tries to call step / render
        with og.sim.render_on_step(self.render_on_step):
            global last_stepped_env, last_stepped_time

            if last_stepped_env != self:
                # If another environment was used after us, we need to check that we have been reset after that.
                # Consider the common setup where you have a train env and an eval env in the same process.
                # When you step the eval env, the physics state of the train env also gets stepped,
                # despite the train env not taking new actions or outputting new observations.
                # By the time you next step the train env your state has drastically changed.
                # To avoid this from happening, we add a requirement: you can only be stepping
                # one vector env at a time - if you want to step another one, you need to reset it first.
                assert (
                    last_stepped_time is None or self.last_reset_time > last_stepped_time
                ), "You must call reset() before using a different environment."
                last_stepped_env = self
                last_stepped_time = time.time()
            action = th.zeros(13)
            actions = [action for i in range(self.num_envs)]
            self.actions = actions
            for i, action in enumerate(actions):
                self.envs[i]._pre_step(action)

    def step_wait(self) -> VecEnvStepReturn:
        with og.sim.render_on_step(self.render_on_step):
            # Step the entire simulation
            og.sim.step()

            for env_idx in range(self.num_envs):
                # print("--------env_idx:", env_idx)
                # # import pdb;pdb.set_trace()
                # print("scene_prim:", self.envs[env_idx].scene._scene_prim.get_position_orientation())
                # for obj_name, obj in self.envs[env_idx].scene._init_objs.items():
                #     print(f"obj_name:{obj_name}:", obj.get_position_orientation())
                obs, self.buf_rews[env_idx], terminated, truncated, self.buf_infos[env_idx] = self.envs[
                    env_idx
                ]._post_step(self.actions[env_idx])

                if terminated or truncated:
                    self.grasping_obj_list[env_idx] = None

                # import pdb; pdb.set_trace()
                
                # convert to SB3 VecEnv api
                self.buf_dones[env_idx] = terminated or truncated
                # See https://github.com/openai/gym/issues/3102
                # Gym 0.26 introduces a breaking change
                self.buf_infos[env_idx]["TimeLimit.truncated"] = truncated and not terminated

                if self.buf_dones[env_idx]:
                    # save final observation where user can get it, then reset
                    self.buf_infos[env_idx]["terminal_observation"] = obs
                    obs, self.reset_infos[env_idx] = self.envs[env_idx].reset()
                self._save_obs(env_idx, obs)
            print(self.buf_rews)
            return (
                self._obs_from_buf(),
                self.buf_rews,
                self.buf_dones,
                copy.deepcopy(self.buf_infos),
            )

    def reset(self):

        with og.sim.render_on_step(self.render_on_step):
            self.last_reset_time = time.time()

            for env_idx in range(self.num_envs):
                print(f"{env_idx} reset")
                maybe_options = {"options": self._options[env_idx]} if self._options[env_idx] else {}
                self.envs[env_idx].reset(get_obs=False, seed=self._seeds[env_idx], **maybe_options)

            # Settle the environments
            # TODO: fix this once we make the task classes etc. vectorized
            for _ in range(30):
                og.sim.step()

            # Get the new obs
            for env_idx in range(self.num_envs):
                obs, info = self.envs[env_idx].get_obs()
                self._save_obs(env_idx, obs)

            # Seeds and options are only used once
            self._reset_seeds()
            self._reset_options()
            return self._obs_from_buf()

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        super().__init__(observation_space, features_dim=1)
        extractors = {}
        self.step_index = 0
        self.img_save_dir = "img_save_dir"
        os.makedirs(self.img_save_dir, exist_ok=True)
        for key, _ in observation_space.spaces.items():
            if key == "rgb":
                features = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=8, stride=4),     # (32, 31, 31)
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=4, stride=2),    # (64, 14, 14)
                    nn.ReLU(),
                    nn.Conv2d(64, 128, kernel_size=3, stride=2),   # (128, 6, 6)
                    nn.ReLU(),
                    nn.Conv2d(128, 256, kernel_size=3, stride=1),  # (256, 4, 4)
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1))                   # (256, 1, 1)
                    )

                fc = nn.Sequential(
                    nn.Flatten(),                  # (256,)
                    nn.Linear(256, 512),           # 参数：131K
                    nn.ReLU(),
                    nn.Linear(512, 384)            # 参数：197K
                    )
                extractors[key] = nn.Sequential(features, fc)
            if key == "layout":
                features = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=8, stride=4),     # (32, 31, 31)
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=4, stride=2),    # (64, 14, 14)
                    nn.ReLU(),
                    nn.Conv2d(64, 128, kernel_size=3, stride=2),   # (128, 6, 6)
                    nn.ReLU(),
                    nn.Conv2d(128, 256, kernel_size=3, stride=1),  # (256, 4, 4)
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1))                   # (256, 1, 1)
                    )
                fc = nn.Sequential(
                    nn.Flatten(),                  # (256,)
                    nn.Linear(256, 512),           # 参数：131K
                    nn.ReLU(),
                    nn.Linear(512, 128)            # 参数：197K
                    )
                extractors[key] = nn.Sequential(features, fc)
        self.extractors = nn.ModuleDict(extractors)
        # Update the features dim manually
        self._features_dim = 512

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []
        self.step_index += 1
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)

        # self.extractors contain nn.Modules that do all the processing.
        # feature = self.net(observations['rgb'])
        # return feature

def main(random_selection=False):
    """
    Robot control demo with selection
    Queries the user to select a robot, the controllers, a scene and a type of input (random actions or teleop)
    """

    tensorboard_log_dir = os.path.join("log_dir", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    prefix = ""

    config_filename = config_filename = os.path.join(og.example_config_path, f"rearrange.yaml")
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    # Choose the scene model to load
    threed_front_path = gm.ThreeD_FRONT_DATASET_PATH
    scene_path = os.path.join(threed_front_path, "scenes", "8148b1a7-7c15-4b53-9be3-8b5a617ba9d2_Bedroom-29109")

    # room = get_available_3dfront_target_scenes(scene)
    # room_model = choose_from_options(options=room, name="room model", random_selection=random_selection) # filename(with .json)

    config["scene"]["scene_model"] = "8148b1a7-7c15-4b53-9be3-8b5a617ba9d2_Bedroom-29109_target.json"
    config["scene"]["scene_type_path"] = scene_path

    # config['env']['scene_names'] = ["d8f50afc-d93f-49f8-9170-b7b9fe880152_Library-20559", "d8f50afc-d93f-49f8-9170-b7b9fe880152_MasterBedroom-24026", "0a761819-05d1-4647-889b-a726747201b1_KidsRoom-8027", "761a08d2-d407-4398-a3cf-34601f0d5c95_SecondBedroom-172344"]
    config['env']['scene_names'] = ["d8f50afc-d93f-49f8-9170-b7b9fe880152_Library-20559"]
    num_envs = 2
    vec_env = RearrangeVecEnv(num_envs, config, True)

    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
    )

    os.makedirs(tensorboard_log_dir, exist_ok=True)

    model = PPO(
        "MultiInputPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=tensorboard_log_dir,
        policy_kwargs=policy_kwargs,
        n_steps=20 * 10 // num_envs,  # Adjust steps to account for parallel envs
        batch_size=8,
        device="cuda",
    )
    
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=tensorboard_log_dir, name_prefix=prefix)

    og.log.debug(model.policy)
    og.log.info(f"model: {model}")

    og.log.info("Starting training...")
    profiler.enable()
    model.learn(
        total_timesteps=10000,
        callback=checkpoint_callback
    )
    profiler.disable()
    og.log.info("Finished training!")
    profiler.dump_stats("slice2.prof")

    # Always shut down the environment cleanly at the end
    og.clear()
    # Always close the environments
    vec_env.close()


if __name__ == "__main__":
    main()
