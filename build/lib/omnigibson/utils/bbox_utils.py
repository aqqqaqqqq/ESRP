import numpy as np
import json
import math
import torch as th
from collections import defaultdict
import os
import random
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from shapely.geometry import Point as poi
from shapely.geometry import Polygon as pol
from shapely.ops import nearest_points
import omnigibson.utils.transform_utils as T

from shapely.geometry import Polygon as pol
from shapely.geometry import Point as poi
from shapely.geometry import MultiPolygon
from shapely.ops import unary_union, polylabel

try:
    from shapely.errors import GEOSException
except ImportError:
    from shapely.geos import TopologicalError as GEOSException

def quat_to_rotmat(q):
    """
    将四元数 (x, y, z, w) 转换为 3x3 旋转矩阵
    """
    x, y, z, w = q
    xx = x * x; yy = y * y; zz = z * z
    xy = x * y; xz = x * z; yz = y * z
    wx = w * x; wy = w * y; wz = w * z
    return np.array([
        [1 - 2*(yy+zz), 2*(xy-wz),   2*(xz+wy)],
        [2*(xy+wz),     1 - 2*(xx+zz), 2*(yz-wx)],
        [2*(xz-wy),     2*(yz+wx),   1 - 2*(xx+yy)]
    ])

def compute_floor_aabb(floor_vertices):
    """
    根据地板顶点信息(Nx3数组),计算 XZ 平面的 AABB(忽略 Y 轴)
    """
    floor_vertices = np.array(floor_vertices)
    min_x, _, min_z = floor_vertices.min(axis=0)
    max_x, _, max_z = floor_vertices.max(axis=0)
    return min_x, max_x, min_z, max_z

def get_rotated_bbox(obj):
    """
    根据物体的 pos、ori、scale、bbox 计算旋转后的 bounding box 四个角点(XZ 平面）
    
    假定 obj["bbox"] 格式为 [width, height, depth]，其中 width 和 depth 用于平面计算
    """
    pos = np.array(obj["pos"])
    scale = np.array(obj["scale"])
    bbox = np.array(obj["bbox"])
    if bbox.ndim == 2:
        bbox = bbox.ravel()
    bbox = np.array([bbox[0], bbox[2], bbox[1]])
    # 计算物体在 XZ 平面上的有效尺寸
    eff_w = bbox[0] * scale[0]
    eff_d = bbox[2] * scale[2]
    half_w = eff_w / 2.0
    half_d = eff_d / 2.0

    # 定义局部坐标系下的四个角点（顺时针或逆时针均可）
    local_corners = np.array([
        [half_w,  half_d],
        [half_w, -half_d],
        [-half_w, -half_d],
        [-half_w,  half_d]
    ])
    
    # 如果存在旋转信息，则将角点旋转到全局坐标系
    if "ori" in obj:
        rot_mat = quat_to_rotmat(obj["ori"])
        # 提取旋转矩阵的 2x2 部分（对应 XZ 平面）
        R2 = np.array([[rot_mat[0,0], rot_mat[0,2]],
                       [rot_mat[2,0], rot_mat[2,2]]])
        rotated_corners = np.dot(local_corners, R2.T)
    else:
        rotated_corners = local_corners

    # 将局部角点平移到全局位置（取 pos 的 x 和 z）
    center = np.array([pos[0], pos[2]])
    global_corners = rotated_corners + center
    return global_corners

def visualize_scene(env, floor_poly, robot_pos, robot_radius, save_dir, filename="scene.png"):
    """
    可视化地板、物体（旋转后的 bounding box ）、候选区域和机器人圆形
    并将图像保存到指定文件夹
    参数：
      floor_poly: 地板多边形的二维顶点列表，每个顶点 [x, z]
      objs: 字典，键为物体名称，值为包含 pos、ori、scale、bbox 的字典
      candidate_center: 采样到的区域中心点 [x, z]（XZ 平面）
      candidate_size: 采样区域的边长（正方形）
      robot_pos: 机器人的中心点 [x, z]（XZ 平面）
      robot_radius: 机器人的半径
      save_dir: 图片保存的文件夹
      filename: 保存的图片文件名
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    
    # 绘制地板多边形
    floor_poly = np.array(floor_poly)
    floor_patch = Polygon(floor_poly, closed=True, edgecolor='k', facecolor='none', linewidth=2, label='Floor')
    ax.add_patch(floor_patch)
    
    # 绘制所有物体的 rotated bounding box
    # objs == scene._init_objs
    for obj_name, _obj in env.scene._init_objs.items():
        # corners = get_rotated_bbox(obj_info)
        if "mesh" in obj_name:
            continue
        a, b, bbox_extent_in_desired_frame, bbox_center_in_desired_frame = _obj.get_base_aligned_bbox()
        dx, dy, dz = bbox_extent_in_desired_frame
        x, y, z = bbox_center_in_desired_frame
        # print(a, b)
        # print(x, y, z)

        # only take 4 upper points
        bbox_in_desired_frame = [(x + sx * 0.5 * dx, y + sy * 0.5 * dy, z + sz * 0.5 * dz) for sx in (1, -1) for sy in (1,) for sz in (1, -1)]
        desired_pos, desired_quat = _obj.get_position_orientation(frame="scene")

        bbox_in_world = []
        for point in bbox_in_desired_frame:
            bbox_in_world.append((T.quat_apply(desired_quat, th.tensor(point)) + desired_pos).tolist())
        bbox_in_world = [bbox_in_world[0], bbox_in_world[1], bbox_in_world[3], bbox_in_world[2]]
        # print("bbox_in_world:", bbox_in_world)
        
        bbox_in_world = np.array([np.array([x[0], x[2]]) for x in bbox_in_world])


        poly_patch = Polygon(bbox_in_world, closed=True, edgecolor='g', facecolor='g', linewidth=1, alpha=0.5, label=obj_name)
        ax.add_patch(poly_patch)
        # 标注物体名称
        center = np.mean(bbox_in_world, axis=0)
        ax.text(center[0], center[1], obj_name, ha='center', va='center', fontsize=8, color='black')
    
    # 绘制机器人圆形
    robot_circle = Circle(robot_pos, robot_radius, linewidth=2, edgecolor='r', facecolor='r', alpha=0.5, label='Robot')
    ax.add_patch(robot_circle)
    
    # 设置图形范围
    all_x = np.concatenate([floor_poly[:,0], np.array([robot_pos[0]])])
    all_y = np.concatenate([floor_poly[:,1], np.array([robot_pos[1]])])
    padding = 1.0
    ax.set_xlim(all_x.min()-padding, all_x.max()+padding)
    ax.set_ylim(all_y.min()-padding, all_y.max()+padding)
    
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Z Coordinate')
    ax.set_title('Scene Visualization')

    # Add collision indicator
    collision_text = f'Robot position ({robot_pos[0]:.2f}, {robot_pos[1]:.2f})'
    ax.text(all_x.min(), all_y.max()+ 0.5, collision_text, fontsize=12, color='red')

    # ax.legend()
    
    # 保存图片到指定文件夹
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved visualization to {save_path}")

def aabb_overlap_2d(a_min, a_max, b_min, b_max):
    """
    判断二维 AABB 是否重叠
    """
    return not (a_max[0] < b_min[0] or a_min[0] > b_max[0] or
                a_max[1] < b_min[1] or a_min[1] > b_max[1])

def is_point_in_polygon(point, polygon):
    px, py = point
    n = len(polygon)
    if n < 3:
        return False

    count = 0
    for i in range(n):
        v1 = polygon[i]
        v2 = polygon[(i + 1) % n]

        def is_point_on_segment(p, a, b):
            px, py = p
            ax, ay = a
            bx, by = b
            # 叉积为零且坐标在包围盒内
            cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
            if cross != 0:
                return False
            return (min(ax, bx) <= px <= max(ax, bx)) and (min(ay, by) <= py <= max(ay, by))
        
        # 检查点是否在边上
        if is_point_on_segment((px, py), v1, v2):
            return False

        # 处理水平边
        if v1[1] == py and v2[1] == py:
            if min(v1[0], v2[0]) <= px <= max(v1[0], v2[0]):
                return True
            continue

        y1_diff = v1[1] - py
        y2_diff = v2[1] - py

        if y1_diff * y2_diff > 0:
            continue

        # 计算交点x坐标
        if v1[0] == v2[0]:
            x_intersect = v1[0]
        else:
            dy = v2[1] - v1[1]
            dx = v2[0] - v1[0]
            t = (py - v1[1]) / dy
            x_intersect = v1[0] + t * dx

        if x_intersect > px:
            count += 1
        elif x_intersect == px:
            return True

    return count % 2 == 1

def remove_duplicate_vertices(floor_poly):
    """
    去除 floor_poly 中的重复顶点，同时保持顺序
    """
    seen = set()
    unique_poly = []
    for vertex in floor_poly:
        vertex_tuple = tuple(vertex)  # 转换为不可变的 tuple 以用于 set
        if vertex_tuple not in seen:
            seen.add(vertex_tuple)
            unique_poly.append(vertex)  # 仍以 list 形式存储
    return unique_poly

def remove_useless_points(_points):
    _points = sort_axis_aligned_polygon(_points)
    j = 0
    while j < (len(_points)):
        # print((len(_points)))
        # print(f"_points[{j}]:", _points[j])
        a = _points[j]
        b = _points[(j + 1) % len(_points)]
        c = _points[(j + 2) % len(_points)]

        def is_point_colinear(p, a, b, epsilon=1e-8):
            """精确判断三点共线（考虑浮点误差）"""
            area = (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])
            return abs(area) < epsilon

        if is_point_colinear(b, a, c):
            _points.pop((j + 1) % len(_points))
            continue
        j += 1
    return _points

def sort_axis_aligned_polygon(points, eps=1e-9):
    """
    给定一组无序点，这些点构成一个所有角均为90°、边平行坐标轴的图形，
    提取边界候选点、构造邻接关系、遍历构成闭合多边形，
    最后调整为逆时针顺序返回。

    参数：
      points: 二维列表，每个元素为 [x, z]。
      eps: 浮点比较容差。

    返回：
      按逆时针顺序排列的边界顶点列表。
    """
    # 1. 去除重复点
    unique = []
    for p in points:
        if not any(abs(p[0] - q[0]) < eps and abs(p[1] - q[1]) < eps for q in unique):
            unique.append(p)
    points = unique

    # 2. 提取候选边界点：
    # 对于相同 x，取 z 的最小和最大；对于相同 z，取 x 的最小和最大。
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

    # 3. 构建邻接图：只有当两候选点共享相同 x 或相同 z，且两点之间没有其他候选点时，认为它们相邻
    n = len(candidates)
    neighbors = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            p = candidates[i]
            q = candidates[j]
            # 如果 x 坐标相同，则检查 z 坐标之间是否有其他候选点
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
            # 如果 z 坐标相同，则检查 x 坐标之间是否有其他候选点
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

    # 4. 选取起始点：选取 z 最大（即最上）的候选中 x 最小的点
    max_z = max(p[1] for p in candidates)
    start_candidates = [(p, i) for i, p in enumerate(candidates) if abs(p[1] - max_z) < eps]
    start_candidates.sort(key=lambda x: x[0][0])
    start_index = start_candidates[0][1]

    # 在起点处，若存在与其 z 坐标相同的邻居，优先选择该点作为下一个点
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

    # 依邻接关系遍历构造闭合多边形
    order = [start_index]
    prev_index = start_index
    current_index = next_index
    while current_index != start_index:
        order.append(current_index)
        nbrs = neighbors[current_index]
        if len(nbrs) == 1:
            break  # 理论上不应出现
        elif len(nbrs) == 2:
            next_index = nbrs[0] if nbrs[0] != prev_index else nbrs[1]
        else:
            next_index = nbrs[0] if nbrs[0] != prev_index else nbrs[1]
        prev_index, current_index = current_index, next_index

    ordered_candidates = [candidates[i] for i in order]

    # 5. 计算有向面积，若面积为负（顺时针），则反转为逆时针顺序
    area = 0
    m = len(ordered_candidates)
    for i in range(m):
        x1, z1 = ordered_candidates[i][0], ordered_candidates[i][1]
        x2, z2 = ordered_candidates[(i + 1) % m][0], ordered_candidates[(i + 1) % m][1]
        area += x1 * z2 - x2 * z1
    if area < 0:
        ordered_candidates.reverse()

    return ordered_candidates

def point_to_object_distance(env, candidate: np.ndarray, obj_name: str) -> float:
    """
    计算候选点 candidate（二维数组 [x, z]）到物体的距离：
    """
    rect_points = get_obj_bbox(env, obj_name)
    # import pdb;pdb.set_trace()
    # print(obj_name, rect_points)
    rectangle = pol(rect_points)
    point = poi(candidate)

    # 计算点到矩形的欧氏距离
    distance = rectangle.distance(point)
    return distance

def is_candidate_colliding(env, candidate: np.ndarray, obj_name: str, robot_radius: float) -> bool:
    """
    判断候选点 candidate（机器人中心，二维 [x, z]）是否与物体发生“碰撞”
    的标准是：candidate 到物体（经过物体 bbox 和旋转信息确定的矩形）的距离是否小于等于 robot_radius
    """
    dist = point_to_object_distance(env, candidate, obj_name)
    # print(dist)
    return dist <= robot_radius

def get_obj_bbox(env, obj_name):
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

    a, b, bbox_extent_in_desired_frame, bbox_center_in_desired_frame = _obj.get_base_aligned_bbox()
    dx, dy, dz = bbox_extent_in_desired_frame
    x, y, z = bbox_center_in_desired_frame
    # print(a, b)
    # print(dx, dy, dz)

    # only take 4 upper points
    bbox_in_desired_frame = [(x + sx * 0.5 * dx, y + sy * 0.5 * dy, z + sz * 0.5 * dz) for sx in (1, -1) for sy in (1,) for sz in (1, -1)]
    desired_pos, desired_quat = _obj.get_position_orientation(frame="scene")

    bbox_in_world = []
    for point in bbox_in_desired_frame:
        bbox_in_world.append((T.quat_apply(desired_quat, th.tensor(point)) + desired_pos).tolist())
    bbox_in_world = [bbox_in_world[0], bbox_in_world[1], bbox_in_world[3], bbox_in_world[2]]
    # print("bbox_in_world:", bbox_in_world)
    
    bbox_in_world = np.array([np.array([x[0], x[2]]) for x in bbox_in_world])
    # print(f"{obj_name} bbox:", bbox_in_world)
    return bbox_in_world

# def find_free_area_on_floor_random(env, floor_vertices, objs_name, robot_diameter=0.8, max_attempts=10000, floor_y=1.0):
#     """
#     在地板上随机采样一个位置，使得机器人（圆形，直径 robot_diameter）的中心点：
#       1. 位于地板多边形内；
#       2. 对于所有物体，通过计算点到物体矩形（考虑物体旋转）的距离，
#          均大于机器人半径，即 candidate 不落入任何物体膨胀区域内。
#     """
#     # import pdb;pdb.set_trace()
#     robot_radius = robot_diameter / 2.0
#     floor_vertices = remove_duplicate_vertices(floor_vertices)
#     min_x, max_x, min_z, max_z = compute_floor_aabb(floor_vertices)
#     floor_poly = [[v[0], v[2]] for v in floor_vertices]
#     floor_poly = remove_useless_points(floor_poly)
#     print("floor_poly:", floor_poly)
#     for _ in range(max_attempts):
#         # 为保证机器人中心点不会超出地板 AABB，此处进行边界采样
#         if _ < 100:
#             x = random.uniform((4*min_x+max_x)/5 + robot_radius + 0.2, (min_x+4*max_x)/5 - robot_radius - 0.2)
#             z = random.uniform((4*min_z+max_z)/5 + robot_radius + 0.2, (min_z+4*max_z)/5 - robot_radius - 0.2)
#         else:
#             x = random.uniform(min_x + robot_radius + 0.2, max_x - robot_radius - 0.2)
#             z = random.uniform(min_z + robot_radius + 0.2, max_z - robot_radius - 0.2)

#         candidate = np.array([x, z])
#         candidate1 = np.array([x-robot_radius, z-robot_radius])
#         candidate2 = np.array([x-robot_radius, z+robot_radius])
#         candidate3 = np.array([x+robot_radius, z+robot_radius])
#         candidate4 = np.array([x+robot_radius, z-robot_radius])
#         if not (is_point_in_polygon(candidate1, floor_poly) and is_point_in_polygon(candidate2, floor_poly) and is_point_in_polygon(candidate3, floor_poly) and is_point_in_polygon(candidate4, floor_poly)):
#             continue

#         save_dir = os.path.join(os.getcwd(), "load_robot")
#         os.makedirs(save_dir, exist_ok=True)
#         visualize_scene(env, floor_poly, candidate, robot_radius, save_dir, f"{env.scene.idx}_collision_step_{_}.png")

#         collision = False
#         for obj_name in objs_name:
#             if is_candidate_colliding(env, candidate, obj_name, robot_radius):
#                 collision = True
#                 break
#         if collision:
#             continue

#         return [x, floor_y, z], [-0.70711, 0, 0, 0.70711]

#     return None
def find_free_area_on_floor_random(env, floor, obstacles, robot_radius=0.4, tolerance=0.01):
    """
    在 floor 内放置半径为 robot_radius 的圆形机器人，确保不与 obstacles 相交。
    使用最大内切圆中心（Pole of Inaccessibility）方法，并保存可视化结果。
    """
    # 1. 计算 free-space
    try:
        free_space = floor.difference(unary_union(obstacles))
    except GEOSException as e:
        print(f"捕获到拓扑异常，信息：{e}")
        return None
    # 2. 对 free-space 做负缓冲，缩小 robot_radius 区域
    shrunk_space = free_space.buffer(-robot_radius)
    if shrunk_space.is_empty:
        raise ValueError("没有足够空间放置机器人")

    # 3. 将 shrunk_space 拆分为单个多边形列表
    if isinstance(shrunk_space, pol):
        polys = [shrunk_space]
    elif isinstance(shrunk_space, MultiPolygon):
        polys = list(shrunk_space.geoms)
    else:
        raise TypeError(f"Unexpected geometry type: {type(shrunk_space)}")

    # 4. 在每个子多边形上运行 polylabel，选取最大半径对应的中心点
    best_center = None
    best_radius = -1
    for poly in polys:
        center = polylabel(poly, tolerance)  # Pole of Inaccessibility
        radius = poly.exterior.distance(center)
        if radius > best_radius:
            best_radius = radius
            best_center = center

    # # 5. 可视化并保存
    # fig, ax = plt.subplots()
    # # 绘制 floor 边界
    # fx, fy = floor.exterior.xy
    # ax.plot(fx, fy, linewidth=2)
    # # 绘制 obstacles
    # for obs in obstacles:
    #     ox, oy = obs.exterior.xy
    #     ax.fill(ox, oy, alpha=0.3)
    # # 绘制 free-space 边界
    # if free_space.geom_type == 'Polygon':
    #     sx, sy = free_space.exterior.xy
    #     ax.plot(sx, sy, linestyle='-')
    # else:
    #     for poly in free_space.geoms:
    #         sx, sy = poly.exterior.xy
    #         ax.plot(sx, sy, linestyle='-')
    # # 绘制最大内切圆
    # circle = poi(best_center.x, best_center.y).buffer(best_radius)
    # cx, cy = circle.exterior.xy
    # ax.plot(cx, cy, linestyle='--')
    # # 标出圆心
    # ax.scatter([best_center.x], [best_center.y], marker='x')

    # ax.set_aspect('equal', 'box')
    # ax.set_title('Free-space 最大内切圆（考虑 robot_radius）')

    # save_dir = os.path.join(os.getcwd(), "load_robot")
    # os.makedirs(save_dir, exist_ok=True)
    # save_path = os.path.join(save_dir, f"{env.scene.scene_model}.png")
    # plt.savefig(save_path, dpi=150)
    # plt.close(fig)
    # print(f"Saved visualization to {save_path}")

    return best_center

# json_path = "./Bedroom-22570.json"
# with open(json_path, 'r', encoding='utf-8') as f:
#     data = json.load(f)
# floor = data.get("floor", [])
# floor_xyz = floor["xyz"]
# num_points = len(floor_xyz) // 3
# floor_vertices = []
# for i in range(num_points):
#     x = floor_xyz[3 * i]
#     y = floor_xyz[3 * i + 1]
#     z = floor_xyz[3 * i + 2]
#     ## transform pos
#     floor_vertices.append([x, y, z])
# # print("points:", points)
# # min_x, max_x, min_z, max_z = compute_floor_aabb(points)
# # print("min_x, max_x, min_z, max_z:", min_x, max_x, min_z, max_z)
# objs = data.get("furniture")
# free_center = find_free_area_on_floor_random(floor_vertices, objs, candidate_size=0.6, max_attempts=10000, floor_y=1.0)
# if free_center is not None:
#     print("找到空闲区域,中心坐标为：", free_center)
# else:
#     print("未能找到空闲区域")


# 测试示例
# points = [[5.619, 0, 1.7995], [5.27, 0, 1.8765], [5.27, 0, 1.7995], [5.619, 0, 1.8765], [5.268, 0, 1.7995], [4.668, 0, 1.8765], [4.668, 0, 1.7995], [5.268, 0, 1.8765], [4.666, 0, 1.7995], [4.066, 0, 1.8765], [4.066, 0, 1.7995], [4.666, 0, 1.8765], [4.064, 0, 1.7995], [3.61713, 0, 1.8765], [3.61713, 0, 1.7995], [4.064, 0, 1.8765], [3.51513, 0, 1.7995], [3.464, 0, 1.8765], [3.464, 0, 1.7995], [3.51513, 0, 1.8765], [3.462, 0, 1.7995], [2.862, 0, 1.8765], [2.862, 0, 1.7995], [3.462, 0, 1.8765], [2.86, 0, 1.7995], [2.511, 0, 1.8765], [2.511, 0, 1.7995], [2.86, 0, 1.8765], [5.619, 0, 1.4975], [5.27, 0, 1.7975], [5.27, 0, 1.4975], [5.619, 0, 1.7975], [5.268, 0, 1.4975], [4.668, 0, 1.7975], [4.668, 0, 1.4975], [5.268, 0, 1.7975], [4.666, 0, 1.4975], [4.066, 0, 1.7975], [4.066, 0, 1.4975], [4.666, 0, 1.7975], [3.51513, 0, 1.4975], [3.464, 0, 1.7975], [3.464, 0, 1.4975], [3.51513, 0, 1.7975], [4.064, 0, 1.4975], [3.61713, 0, 1.7975], [3.61713, 0, 1.4975], [4.064, 0, 1.7975], [3.462, 0, 1.4975], [2.862, 0, 1.7975], [2.862, 0, 1.4975], [3.462, 0, 1.7975], [2.86, 0, 1.4975], [2.511, 0, 1.7975], [2.511, 0, 1.4975], [2.86, 0, 1.7975], [5.619, 0, 1.1955], [5.27, 0, 1.4955], [5.27, 0, 1.1955], [5.619, 0, 1.4955], [5.268, 0, 1.1955], [4.668, 0, 1.4955], [4.668, 0, 1.1955], [5.268, 0, 1.4955], [4.666, 0, 1.1955], [4.066, 0, 1.4955], [4.066, 0, 1.1955], [4.666, 0, 1.4955], [3.464, 0, 1.1955], [3.51513, 0, 1.26902], [3.464, 0, 1.4955], [4.064, 0, 1.1955], [3.61713, 0, 1.26902], [3.61713, 0, 1.4955], [4.064, 0, 1.4955], [3.51513, 0, 1.4955], [3.462, 0, 1.1955], [2.862, 0, 1.4955], [2.862, 0, 1.1955], [3.462, 0, 1.4955], [2.86, 0, 1.1955], [2.511, 0, 1.4955], [2.511, 0, 1.1955], [2.86, 0, 1.4955], [5.619, 0, 0.8935], [5.27, 0, 1.1935], [5.27, 0, 0.8935], [5.619, 0, 1.1935], [5.268, 0, 0.8935], [4.668, 0, 1.1935], [4.668, 0, 0.8935], [5.268, 0, 1.1935], [4.666, 0, 0.8935], [4.066, 0, 1.1935], [4.066, 0, 0.8935], [4.666, 0, 1.1935], [4.064, 0, 0.8935], [3.464, 0, 1.1935], [3.464, 0, 0.8935], [4.064, 0, 1.1935], [3.462, 0, 0.8935], [2.862, 0, 1.1935], [2.862, 0, 0.8935], [3.462, 0, 1.1935], [2.86, 0, 0.8935], [2.511, 0, 1.1935], [2.511, 0, 0.8935], [2.86, 0, 1.1935], [5.619, 0, 0.5915], [5.27, 0, 0.8915], [5.27, 0, 0.5915], [5.619, 0, 0.8915], [5.268, 0, 0.5915], [4.668, 0, 0.8915], [4.668, 0, 0.5915], [5.268, 0, 0.8915], [4.666, 0, 0.5915], [4.066, 0, 0.8915], [4.066, 0, 0.5915], [4.666, 0, 0.8915], [4.064, 0, 0.5915], [3.464, 0, 0.8915], [3.464, 0, 0.5915], [4.064, 0, 0.8915], [3.462, 0, 0.5915], [2.862, 0, 0.8915], [2.862, 0, 0.5915], [3.462, 0, 0.8915], [2.86, 0, 0.5915], [2.511, 0, 0.8915], [2.511, 0, 0.5915], [2.86, 0, 0.8915], [5.619, 0, 0.2895], [5.27, 0, 0.5895], [5.27, 0, 0.2895], [5.619, 0, 0.5895], [5.268, 0, 0.2895], [4.668, 0, 0.5895], [4.668, 0, 0.2895], [5.268, 0, 0.5895], [4.666, 0, 0.2895], [4.066, 0, 0.5895], [4.066, 0, 0.2895], [4.666, 0, 0.5895], [3.51542, 0, 0.53823], [4.064, 0, 0.5895], [3.464, 0, 0.5895], [3.61742, 0, 0.53876], [3.61872, 0, 0.2895], [4.064, 0, 0.2895], [3.51672, 0, 0.2895], [3.464, 0, 0.2895], [3.462, 0, 0.2895], [2.862, 0, 0.5895], [2.862, 0, 0.2895], [3.462, 0, 0.5895], [2.86, 0, 0.2895], [2.511, 0, 0.5895], [2.511, 0, 0.2895], [2.86, 0, 0.5895], [5.619, 0, -0.0125], [5.27, 0, 0.2875], [5.27, 0, -0.0125], [5.619, 0, 0.2875], [5.268, 0, -0.0125], [4.668, 0, 0.2875], [4.668, 0, -0.0125], [5.268, 0, 0.2875], [4.666, 0, -0.0125], [4.066, 0, 0.2875], [4.066, 0, -0.0125], [4.666, 0, 0.2875], [3.51829, 0, -0.0125], [3.464, 0, 0.2875], [3.464, 0, -0.0125], [3.51672, 0, 0.2875], [3.62029, 0, -0.0125], [4.064, 0, 0.2875], [3.61873, 0, 0.2875], [4.064, 0, -0.0125], [3.462, 0, -0.0125], [2.862, 0, 0.2875], [2.862, 0, -0.0125], [3.462, 0, 0.2875], [2.86, 0, -0.0125], [2.511, 0, 0.2875], [2.511, 0, -0.0125], [2.86, 0, 0.2875], [5.619, 0, -0.0915], [5.27, 0, -0.0145], [5.27, 0, -0.0915], [5.619, 0, -0.0145], [5.268, 0, -0.0915], [4.668, 0, -0.0145], [4.668, 0, -0.0915], [5.268, 0, -0.0145], [4.666, 0, -0.0915], [4.066, 0, -0.0145], [4.066, 0, -0.0915], [4.666, 0, -0.0145], [3.5187, 0, -0.0915], [3.464, 0, -0.0145], [3.464, 0, -0.0915], [3.5183, 0, -0.0145], [3.6207, 0, -0.0915], [4.064, 0, -0.0145], [3.6203, 0, -0.0145], [4.064, 0, -0.0915], [3.462, 0, -0.0915], [2.862, 0, -0.0145], [2.862, 0, -0.0915], [3.462, 0, -0.0145], [2.86, 0, -0.0915], [2.511, 0, -0.0145], [2.511, 0, -0.0915], [2.86, 0, -0.0145]]
# floor_poly = [[v[0], v[2]] for v in points]
# floor_poly = remove_useless_points(floor_poly)
# print("floor_poly:", floor_poly)


def sample_point_around_object(
    env, obj_n,                                    
    robot_radius: float = 0.4,                 # 机器人半径
    clearance_min: float = 0.10,                # 机器人边界与物体边界的最小距离
    clearance_max: float = 0.20,                # 机器人边界与物体边界的最大距离
    max_iter: int = 1000                      # 最大采样次数
):

    rect_points = get_obj_bbox(env, obj_n)

    # 构造 Shapely 多边形对象
    rectangle = pol(rect_points)

    # 设定距离要求
    min_dist = robot_radius + clearance_min
    max_dist = robot_radius + clearance_max

    # 根据矩形的 bounding box（加上一定 margin）采样候选点
    def sample_point_within_bounds(poly, margin=1.0):
        minx, miny, maxx, maxy = poly.bounds
        x = random.uniform(minx - margin, maxx + margin)
        y = random.uniform(miny - margin, maxy + margin)
        return poi(x, y)

    # 随机采样满足条件的点
    candidate = None
    for i in range(max_iter):
        pt = sample_point_within_bounds(rectangle, margin=1.0)
        d = rectangle.distance(pt)
        if d > min_dist and d < max_dist:
            candidate = pt
            break

    assert candidate, "未能采样到满足距离条件的点！"

    d_candidate = rectangle.distance(candidate)
    # print(f"point around {obj_n} is:", (candidate.x, candidate.y))
    # print("该点到矩形的距离：", d_candidate)
    return [candidate.x, candidate.y]