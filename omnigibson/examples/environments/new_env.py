from gymnasium.spaces import Box
import yaml
import os
import torch as th
import math
import itertools
import numpy as np
import time
import omnigibson as og
from omnigibson.macros import gm
import omnigibson.utils.transform_utils as T
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib
from collections import defaultdict
from shapely.geometry import Point, Polygon
from datetime import datetime
from omnigibson.utils.constants import PrimType
matplotlib.use('Agg')
import cv2
from collections import OrderedDict
from collections.abc import Iterable
import gymnasium as gym
import pynvml
from omnigibson.examples.environments.get_camera_picture import compute_camera_height_from_polygon

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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
        # import pdb; pdb.set_trace()
        # _pos, _orn = link.get_position_orientation(frame = 'scene')
        bbox_in_world =T.transform_points(bbox_in_local, T.pose2mat((link.get_position_orientation(frame = 'scene')[0], link.get_position_orientation(frame = 'scene')[1]))).numpy()

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
        # print('_get_obj_bbox_new', b - a, c - b, d - c, e - d, f - e)
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
    
    def get_obstacles(env, obstacles_cache):
        object_names = env.task.get_all_objects_names(env)
        obstacles = {}
        for object_name in object_names:
            if object_name in obstacles_cache:
                obstacles[object_name] = obstacles_cache[object_name]
            else:
                _tensor = OccupancyInfo._get_obj_bbox_new(env, object_name)
                obstacles[object_name] = _tensor

        return obstacles
    

def no_physical_step(env, action, n_render_iterations=1):
    # Pre-processing before stepping simulation
    if isinstance(action, Iterable) and not isinstance(action, (dict, OrderedDict)):
        # Convert numpy arrays and lists to tensors
        # Skip dict action
        action = th.as_tensor(action, dtype=th.float).flatten()
    env._pre_step(action)

    # Render any additional times requested
    for _ in range(n_render_iterations):
        og.sim.render()

    # Run final post-processing
    return env._post_step(action)

def monitor_gpu_nvml():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    for p in procs:
        if p.pid == os.getpid():
            return p.usedGpuMemory / (1024 ** 2)

class FastEnv(gym.Env):
    def __init__(self, env, run_id = None):
        self.env = env
        self.grasping_obj = None
        self._frame = 'scene'
        self.robot_radius = 0.4
        self.enable_collision_detection = True
        self.obstacles_cache = {}

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
    
    def _get_robot(self):
        return self.env.robots[0]

    def _position_top_down_camera(self):
        if not getattr(self.env, "_use_top_down", False):
            return
        if not self.env.external_sensors or "top_cam" not in self.env.external_sensors:
            return

        floor_poly = self.env.task.get_floor_poly(self.env)
        polygon = Polygon(floor_poly)
        cam = self.env.external_sensors["top_cam"]
        x = polygon.centroid.x
        z = polygon.centroid.y
        y = compute_camera_height_from_polygon(cam, np.array(floor_poly))
        top_down_position = th.tensor([x, y, z], dtype=th.float32)
        top_down_orientation = th.tensor([-0.5, -0.5, -0.5, 0.5], dtype=th.float32)
        cam.set_position_orientation(top_down_position, top_down_orientation, frame="scene")

    def _keep_relative_to_robot(pos, quat, old_robot_pos, old_robot_quat, new_robot_pos, new_robot_quat):
        pos_in_robot_frame, quat_in_robot_frame = T.relative_pose_transform(pos, quat, old_robot_pos, old_robot_quat)
        pos_in_world_frame, quat_in_world_frame = T.pose_transform(new_robot_pos, new_robot_quat, pos_in_robot_frame, quat_in_robot_frame)
        return pos_in_world_frame, quat_in_world_frame

    def _check_collision_2d(self, old_robot_pos, old_robot_quat, new_robot_pos, new_robot_quat):
        a0 = time.perf_counter()
        floor_area = OccupancyInfo.get_floor_area(self.env)
        floor_area_poly = Polygon(floor_area)
        a1 = time.perf_counter()
        obstacles = OccupancyInfo.get_obstacles(self.env, self.obstacles_cache)
        self.obstacles_cache = obstacles
        a2 = time.perf_counter()
        # get robot
        new_robot_center = Point(new_robot_pos[0], new_robot_pos[2])
        new_robot_circle = new_robot_center.buffer(self.robot_radius)
        if self.grasping_obj:
            grasping_obj_name = self.grasping_obj.split('/')[-1]
            grasping_obj_poly = obstacles[grasping_obj_name]
            new_grasping_obj_poly = []

            for p in grasping_obj_poly:
                _pos = th.tensor([p[0], 0.0, p[1]])
                _quat = T.euler2quat(th.tensor([0.0, 0.0, 0.0]))
                new_pos, _ = FastEnv._keep_relative_to_robot(_pos, _quat, old_robot_pos, old_robot_quat, new_robot_pos, new_robot_quat)
                new_grasping_obj_poly.append([new_pos[0], new_pos[2]])

            new_obj_poly = Polygon(new_grasping_obj_poly)
            obstacles_without_grasping_obj = [v for k, v in obstacles.items() if k != grasping_obj_name]
        else:
            obstacles_without_grasping_obj = [v for k, v in obstacles.items()]

        a3 = time.perf_counter()
        obstacles_poly = [Polygon(obstacle) for obstacle in obstacles_without_grasping_obj]
        a4 = time.perf_counter()
        
        moved_polygons = [new_robot_circle, new_obj_poly] if self.grasping_obj is not None else [new_robot_circle]
        any_intersection = any(p1.intersects(p2) for p1, p2 in itertools.product(obstacles_poly, moved_polygons))
        all_cover = all([floor_area_poly.covers(poly) for poly in moved_polygons])
        valid = (not any_intersection and all_cover)
        a5 = time.perf_counter()
        # print(a1 - a0, a2 - a1, a3 - a2, a4 - a3, a5 - a4)
        # FOR VISULIZATION ONLY
        old_robot_center = Point(old_robot_pos[0], old_robot_pos[2])
        old_robot_circle = old_robot_center.buffer(self.robot_radius)

        # if self.grasping_obj:
        #     _grasping_obj_poly = Polygon(grasping_obj_poly)
        #     visualize_environment(floor_area_poly, obstacles_poly, new_robot_circle, old_robot_circle, new_obj_poly, _grasping_obj_poly)
        # else:
        #     visualize_environment(floor_area_poly, obstacles_poly, new_robot_circle, old_robot_circle)
        #     pass
        return valid

    def _move(self, diff_x, diff_yaw):
        _pos = th.tensor([diff_x, 0.0, 0.0])
        _quat = T.euler2quat(th.tensor([0.0, 0.0, diff_yaw]))
        robot_pos, robot_quat = self._get_robot().get_position_orientation(frame = self._frame)
        next_robot_pos, next_robot_quat = T.pose_transform(robot_pos, robot_quat, _pos, _quat)
        if self.grasping_obj is not None:
            grasping_obj_pos, grasping_obj_quat = self.env.scene.object_registry('prim_path', self.grasping_obj).get_position_orientation(frame = self._frame)
            new_grasping_obj_pos, new_grasping_obj_quat = FastEnv._keep_relative_to_robot(grasping_obj_pos, grasping_obj_quat, robot_pos, robot_quat, next_robot_pos, next_robot_quat)
            
        no_collision = self._check_collision_2d(robot_pos, robot_quat, next_robot_pos, next_robot_quat) if self.enable_collision_detection else True
        if no_collision:
            # keep the height
            next_robot_pos[1] = robot_pos[1]
            self._get_robot().set_position_orientation(next_robot_pos, next_robot_quat, frame = self._frame)
            if self.grasping_obj is not None:
                self.env.scene.object_registry('prim_path', self.grasping_obj).set_position_orientation(new_grasping_obj_pos, new_grasping_obj_quat, frame = self._frame)
                grasping_obj_name = self.grasping_obj.split('/')[-1] 
                if grasping_obj_name in self.obstacles_cache:
                    del self.obstacles_cache[grasping_obj_name]
            return True
        else:
            return False

        
    def _fetch(self):
        if self.grasping_obj is not None:
            return False
        # import pdb;pdb.set_trace()
        _robot = self._get_robot()
        THETA = math.pi / 2
        R = 1.0
        VISIBLE_AREA_R = R * 2

        bbox_in_world = []
        polygons = []

        all_objects = OccupancyInfo.get_obstacles(self.env, self.obstacles_cache)
        for _obj in list(_robot.scene._init_objs.values()):
            if 'test' in _obj.prim_path:
                continue
            
            if not _obj.is_to_rearrange:
                continue

            have_raw_model_prim_path = False
            raw_model_prim = None
            for _, _link in _obj.links.items():
                if 'raw_model' in _link.prim_path:
                    raw_model_prim = _link
                    have_raw_model_prim_path = True

            if not have_raw_model_prim_path:
                continue

            # a, b, bbox_extent_in_desired_frame, bbox_center_in_desired_frame = _obj.get_base_aligned_bbox()
            # dx, dy, dz = bbox_extent_in_desired_frame
            # x, y, z = bbox_center_in_desired_frame

            # # only take 4 upper points
            # bbox_in_desired_frame = [(x + sx * 0.5 * dx, y + sy * 0.5 * dy, z + sz * 0.5 * dz) for sx in (1, -1) for sy in (1,) for sz in (1, -1)]
            # desired_pos, desired_quat = _obj.get_position_orientation()

            # bbox_in_world = []
            # for point in bbox_in_desired_frame:
            #     bbox_in_world.append((T.quat_apply(desired_quat, th.tensor(point)) + desired_pos).tolist())
            
            # bbox_in_world = [(x[0], x[2]) for x in bbox_in_world]
            # polygons.append((raw_model_prim, Polygon([bbox_in_world[0], bbox_in_world[1], bbox_in_world[3], bbox_in_world[2]])))
            boundary_array = all_objects[raw_model_prim.prim_path.split('/')[-2]]
            polygons.append((raw_model_prim, Polygon(boundary_array)))

        yaw = _robot.get_yaw()
        robot_x, _, robot_z = _robot.get_position_orientation(frame = "scene")[0]
        point_1 = (robot_x + VISIBLE_AREA_R * math.cos(yaw + THETA / 2), robot_z + VISIBLE_AREA_R * math.sin(yaw + THETA / 2))
        point_2 = (robot_x + VISIBLE_AREA_R * math.cos(yaw - THETA / 2), robot_z + VISIBLE_AREA_R * math.sin(yaw - THETA / 2))
        visible_area = Polygon([(robot_x, robot_z), point_1, point_2])
        robot_point = Point(robot_x, robot_z)

        filtered_polygons = [p for p in polygons if p[1].intersects(visible_area) and robot_point.distance(p[1]) <= R]

        sorted_polygons = sorted(filtered_polygons, key=lambda x: robot_point.distance(x[1]))
        if len(sorted_polygons):
            ag_prim_path = sorted_polygons[0][0].prim_path
            ag_obj_prim_path = "/".join(ag_prim_path.split("/")[:-1])
            # ag_obj = _robot.scene.object_registry("prim_path", ag_obj_prim_path) # type: ignore
            self.grasping_obj = ag_obj_prim_path
            self.env.task.get_fetch(1, ag_obj_prim_path.split("/")[-1])
            return True
        
        return False

    def _release(self):
        if self.grasping_obj is None:
            return False
        self.grasping_obj = None
        self.env.task.get_fetch(0, None)
        return True

    def step(self, action):
        assert action in {0, 1, 2, 3, 4, 5}
        # print('Received action = ' + str(action))
        a1 = time.perf_counter()
        ROBOT_ACTION_DIM = 13
        no_op_action = th.zeros(ROBOT_ACTION_DIM)
        action2diff = {0: (-0.25, 0.0), 1: (0.25, 0.0), 2: (0.0, math.pi / 8), 3: (0.0, -math.pi / 8)}
        a2 = time.perf_counter()
        _result = None
        if action in {0, 1, 2, 3}:
            _result = self._move(*action2diff[action])
        else:
            if action == 4:
                _result = self._fetch()
            else:
                _result = self._release()
        
        a3 = time.perf_counter()
        self._get_robot().keep_still()
        obstacles = OccupancyInfo.get_obstacles(self.env, self.obstacles_cache)
        obstacles_poly = {k: Polygon(v) for k, v in obstacles.items()}
        self.env.task.get_current_polygon(obstacles_poly)


        observations, rewards, terminates, truncates, infos = no_physical_step(self.env, no_op_action, n_render_iterations=10)

        if self.grasping_obj is not None:
            number = th.tensor(1, dtype=th.float32)
        else:
            number = th.tensor(0, dtype=th.float32)

        if isinstance(observations, dict):
            observations["grasp_flag"] = number
        else:
            number = number.to(observations.dtype)
            observations = th.cat([observations, number.unsqueeze(0)])        

        a4 = time.perf_counter()
        # print(a2 - a1, a3 - a2, a4 - a3)
        infos['valid_move'] = _result
        # print(result[4])
        return observations, rewards, terminates, truncates, infos

    def reset(self, seed, options = None):
        if monitor_gpu_nvml() > 4200:
            os.kill(os.getpid(), 9)
        self.obstacles_cache = {}
        self.env.reset(get_obs=False)
        self._position_top_down_camera()
        og.sim.step()
        obs, info = self.env.get_obs()
        self.grasping_obj = None
        init_grasping_obj = th.tensor(0, dtype=obs.dtype).unsqueeze(0)
        obs = th.cat([obs, init_grasping_obj])
        return obs, info

class FakeEnv(gym.Env):
    def __init__(self, use_top_down=False):
        super().__init__()
        obs_dim = 147457 if use_top_down else 98305
        self.observation_space = Box(low=0, high=255, shape=(obs_dim,), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(6)

    def reset(self, seed = None, options = None):
        raise NotImplementedError

def make_env(cfg):
    import os
    from omnigibson.examples.robots.rearrange_robot import add_top_down_camera

    use_top_down = cfg.get("use_top_down", False)

    if cfg.worker_index == 0:
        return FakeEnv(use_top_down=use_top_down)
    
    config_filename = config_filename = os.path.join(og.example_config_path, f"rearrange.yaml")
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    gm.ENABLE_OBJECT_STATES = False
    gm.ENABLE_TRANSITION_RULES = False
    gm.ENABLE_FLATCACHE = False
    gm.RENDER_VIEWER_CAMERA = False
    gm.USE_GPU_DYNAMICS = True

    threed_front_path = gm.ThreeD_FRONT_DATASET_PATH
    scene_path = os.path.join(threed_front_path, "scenes", "8148b1a7-7c15-4b53-9be3-8b5a617ba9d2_Bedroom-29109")

    config["scene"]["scene_model"] = "8148b1a7-7c15-4b53-9be3-8b5a617ba9d2_Bedroom-29109_target.json"
    config["scene"]["scene_type_path"] = scene_path
    config["env"]["use_top_down"] = use_top_down
    if use_top_down:
        config = add_top_down_camera(config)

    scene_names = []
    with open('/home/user/Desktop/rl/omnigibson/data/3d_front/verified_train_data.txt', 'r') as f:
        for line in f:
            scene_name = line.rstrip('\n')
            scene_names.append(scene_name)
    # print(scene_names)
    # print("len:", len(scene_names))

    config['env']['scene_names'] = scene_names
    env = og.Environment(configs=config)
    rearrangement_env = FastEnv(env, run_id = cfg['run_id'])

    return rearrangement_env

# if __name__ == '__main__':
#     # if cfg.worker_index == 0:
#     #     return FakeEnv()
    
#     config_filename = config_filename = os.path.join(og.example_config_path, f"rearrange.yaml")
#     config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

#     gm.ENABLE_OBJECT_STATES = False
#     gm.ENABLE_TRANSITION_RULES = False
#     gm.ENABLE_FLATCACHE = False
#     gm.RENDER_VIEWER_CAMERA = False
#     gm.USE_GPU_DYNMAICS = True

#     threed_front_path = gm.ThreeD_FRONT_DATASET_PATH
#     scene_path = os.path.join(threed_front_path, "scenes", "8148b1a7-7c15-4b53-9be3-8b5a617ba9d2_Bedroom-29109")

#     config["scene"]["scene_model"] = "8148b1a7-7c15-4b53-9be3-8b5a617ba9d2_Bedroom-29109_target.json"
#     config["scene"]["scene_type_path"] = scene_path

#     scene_names = []
#     with open('/home/user/Desktop/rl/omnigibson/data/3d_front/verified_train_data.txt', 'r') as f:
#         for line in f:
#             scene_name = line.rstrip('\n')
#             scene_names.append(scene_name)
#     # print(scene_names)
#     # print("len:", len(scene_names))

#     config['env']['scene_names'] = scene_names
#     env = og.Environment(configs=config)
#     rearrangement_env = FastEnv(env, run_id = '')
#     rearrangement_env.step(0)
#     import pdb; pdb.set_trace()
