import os
import torch as th
import omnigibson.utils.transform_utils as T
import math
import time
from shapely.geometry import Point, Polygon
import datetime
import matplotlib.pyplot as plt
import matplotlib
from omnigibson.utils.constants import PrimType
import numpy as np
import itertools
import cv2
import cProfile
from collections import defaultdict
matplotlib.use('Agg')
profiler = cProfile.Profile() 

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
    
    def get_obstacles(env):
        object_names = env.task.get_all_objects_names(env)
        obstacles = {}
        for object_name in object_names:
            _tensor = OccupancyInfo._get_obj_bbox(env, object_name)
            obstacles[object_name] = _tensor

        return obstacles
    
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
    a5 = time.perf_counter()
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