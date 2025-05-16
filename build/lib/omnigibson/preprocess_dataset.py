import json
import os
import numpy as np
import re
import random
from omnigibson.macros import gm
from omnigibson.utils.bbox_utils import remove_duplicate_vertices, remove_useless_points
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import time
import signal


class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException

# 为 SIGALRM 设置信号处理器
signal.signal(signal.SIGALRM, timeout_handler)
# 输入 JSON 文件所在的文件夹路径
input_folder = '/home/pilab/Downloads/3D-FRONT'
# USD 输出的基础文件夹（后续会根据 JSON 内 uid 创建子文件夹）
output_base = '/home/pilab/Siqi/github/OmniGibson-Rearrange/omnigibson/data/3d_front/scenes'
# print(gm.ThreeD_FRONT_DATASET_PATH)
# pp = gm.ThreeD_FRONT_DATASET_PATH + "/data/3d_front/usd_objects"
# print(pp)

# 遍历输入文件夹下所有的 JSON 文件
for filename in os.listdir(input_folder):
    if not filename.endswith(".json"):
        continue

    json_path = os.path.join(input_folder, filename)
    print(f"正在处理文件:{json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 获取 JSON 文件中顶层的 uid 字段,如果不存在则用文件名（去除后缀）作为 uid
    json_uid = data.get("uid", os.path.splitext(filename)[0])

    # 遍历 scene -> room
    scene = data.get("scene", {})
    rooms = scene.get("room", [])
    for room in rooms:

        floor_too_small = False
        is_timeout = False
        furniture_num = 0
        room_instanceid = room.get("instanceid")
        print("instanceid:", room_instanceid)
        # import pdb;pdb.set_trace()
        room_type = room.get("type")
        if not room_instanceid:
            continue
        new_json = {
            "metadata": {},
            "state": {
                "system_registry": {},
                "object_registry": {},
            },
            "objects_info": {
                "init_info": {}
            }
        }
        
       

        for child in room.get("children", []):
            child_instanceid = child.get("instanceid", "")
            
            ref = child.get("ref")
            if not ref:
                print(f"警告:room {room_instanceid} 中 children 缺少 ref 字段,跳过。")
                continue
            

            if "furniture" in child_instanceid:
                # 在 furniture 数组中查找 uid == ref 的条目
                furniture_found = None
                furniture_data = data.get("furniture", [])
                for furniture in furniture_data:
                    if not (furniture.get("valid", False) and furniture["valid"] == True):
                        continue
                    if furniture.get("uid") == ref:
                        furniture_found = furniture
                        pos = child["pos"]
                        ori = child["rot"]
                        scale = child["scale"]
                        category = furniture.get("category", None)
                        bbox = furniture.get("bbox", [1, 1, 1])
                        is_to_rearrange = False
                        is_fixed_base = True
                        
                        scale_ = np.array(scale)
                        bbox_ = np.array(bbox)
                        # handle the exception
                        if bbox_.ndim == 2:
                            bbox_ = bbox_.ravel()
                            # print("bbox_:", bbox_)
                        bbox_scale = scale_ * bbox_
                        # print("bbox_scale:", bbox_scale)
                        # vol = bbox_scale[0] * bbox_scale[1] * bbox_scale[2]
                        if max(bbox_scale[0] * bbox_scale[1], bbox_scale[0] * bbox_scale[2], bbox_scale[1] * bbox_scale[2]) < 1.2:
                            is_to_rearrange = True
                            is_fixed_base = False
                        
                        # if vol < 1.0:
                        #     is_to_rearrange = True
                        #     is_fixed_base = False
                        if category is not None and ("Ceiling" in category or "Lamp" in category):
                            break
                        # delete objects from ceilings
                        if pos[1] > 1.0:
                            break
                        
                        # # transform pos ,as the gravity of the original data is downwards along y-axis
                        # xx = pos[2]
                        # yy = pos[0]
                        # zz = pos[1]
                        # pos_rotate = [xx, yy, zz]
                        # # rotate 90 degree around x-axis
                        # x = ori[0]
                        # y = ori[1]
                        # z = ori[2]
                        # w = ori[3]
                        # ori_rotate = [float(0.70711*(x+w)), float(0.70711*(y-z)), float(0.70711*(y+z)), float(0.70711*(w-x))]

                        object_name = "object_" + re.sub(r'[^a-zA-Z0-9]', '_', ref)
                        # may have the same furniture model in the scene, avoid overwriting
                        modified_object_name = object_name + '_' + child_instanceid
                        modified_object_name = re.sub(r'[^a-zA-Z0-9]', '_', modified_object_name)
                        new_json["state"]["object_registry"][modified_object_name] = {
                            "root_link": {
                                "pos": pos,
                                "ori": ori,
                                "lin_vel": [0.0, 0.0, 0.0],
                                "ang_vel": [0.0, 0.0, 0.0],
                            },
                            "joints": {},
                            "non_kin": {
                                "Temperature": {
                                    "temperature": 23.0
                                }
                            }
                        }

                        jjid = furniture["jid"]
                        new_json["objects_info"]["init_info"][modified_object_name] = {
                            "class_module": "omnigibson.objects.threed_front_object",
                            "class_name": "ThreeD_FRONTObject",
                            "args": {
                                "name": modified_object_name,
                                "prim_path": f"/World/{modified_object_name}", 
                                "usd_path": f"/usd_objects/{jjid}_converted/raw_model.usd",  
                                "scale": scale,
                                "fixed_base": is_fixed_base,
                                "is_to_rearrange": is_to_rearrange,
                                "bddl_object_scope": None
                            }
                        }
                        furniture_num += 1
                        break
                if furniture_found is None:
                    print(f"提示:未在 furniture 数组中找到 uid 为 {ref} 的条目,跳过。")
                    continue

            elif "mesh" in child_instanceid:
                # 在 mesh 数组中查找 uid == ref 的条目
                mesh_found = None
                for mesh_entry in data.get("mesh", []):
                    
                    mesh_type = mesh_entry.get("type")
                    if "Ceiling" in mesh_type or "Cabinet" in mesh_type or "Customized" in mesh_type:
                        continue
                    
                    if mesh_entry.get("uid") == ref:
                        mesh_found = mesh_entry
                        pos = child["pos"]
                        ori = child["rot"]
                        scale = child["scale"]

                        # rotate 90 degree around x-axis
                        # x = ori[0]
                        # y = ori[1]
                        # z = ori[2]
                        # w = ori[3]
                        # ori_rotate = [float(0.70711*(x+w)), float(0.70711*(y-z)), float(0.70711*(y+z)), float(0.70711*(w-x))]

                        mesh_name = "mesh_" + re.sub(r'[^a-zA-Z0-9]', '_', ref)
                        modified_mesh_name = mesh_name + '_' + child_instanceid
                        modified_mesh_name = re.sub(r'[^a-zA-Z0-9]', '_', modified_mesh_name)
                        new_json["state"]["object_registry"][modified_mesh_name] = {
                            "root_link": {
                                "pos": pos,
                                "ori": ori,
                                "lin_vel": [0.0, 0.0, 0.0],
                                "ang_vel": [0.0, 0.0, 0.0],
                            },
                            "joints": {},
                            "non_kin": {
                                "Temperature": {
                                    "temperature": 23.0
                                }
                            }
                        }

                        floor_xyz = None
                        if mesh_type == "Floor":
                            
                            floor_xyz = mesh_entry.get("xyz")
                            
                            num_points = len(floor_xyz) // 3
                            floor_vertices = []
                            for i in range(num_points):
                                x = floor_xyz[3*i]
                                y = floor_xyz[3*i+1]
                                z = floor_xyz[3*i+2]
                                ## transform pos
                                floor_vertices.append([x, y, z])
                            
                            floor_vertices = remove_duplicate_vertices(floor_vertices)
                            # print("floor_vertices:", floor_vertices)
                            # print("len(floor_vertices):", len(floor_vertices))
                            # if len(floor_vertices) > 15:
                            #     floor_too_small = True
                            #     break

                            # if len(floor_vertices) > 15:
                            #     floor_too_small = True
                            #     break
                            # import pdb;pdb.set_trace()
                            floor_poly = [[v[0], v[2]] for v in floor_vertices]
                            # start_time = time.time()
                            signal.setitimer(signal.ITIMER_REAL, 0.01)
                            try:
                                floor_poly = remove_useless_points(floor_poly)
                            except TimeoutException:
                                print("函数执行超时，继续下一次循环")
                                is_timeout = True
                                break  # 超时后跳过本次循环
                            finally:
                                # 取消定时器
                                signal.setitimer(signal.ITIMER_REAL, 0)
                            # floor_poly = remove_useless_points(floor_poly)
                            # end_time = time.time()
                            # floor_poly_time = end_time - start_time
                            # print("time:", floor_poly_time)
                            
                            print("floor_poly:", floor_poly)
                            # import pdb;pdb.set_trace()
                            floor_polygon = Polygon(floor_poly)
                            area = floor_polygon.area
                            print("area:", area)
                            if area < 10:
                                floor_too_small = True
                                break
                            # x_big, y_big = floor_polygon.exterior.xy
                            # fig, ax = plt.subplots(figsize=(8, 8))
                            # ax.plot(x_big, y_big, color='blue', lw=2, label='大多边形')
                            # ax.legend(loc='upper right')
                            # ax.set_title("大多边形及其减去小矩形后的剩余面积")
                            # ax.set_xlabel("X 轴")
                            # ax.set_ylabel("Y 轴")
                            # plt.axis('equal')
                            # plt.savefig("floor.png")
                            # import pdb;pdb.set_trace()

                        new_json["objects_info"]["init_info"][modified_mesh_name] = {
                            "class_module": "omnigibson.objects.threed_front_object",
                            "class_name": "ThreeD_FRONTObject",
                            "args": {
                                "name": modified_mesh_name,
                                "prim_path": f"/World/{modified_mesh_name}", 
                                "usd_path": f"/mesh/{json_uid}/{room_instanceid}/{mesh_name}.usd",  
                                "scale": scale,
                                "fixed_base": True,
                                "is_to_rearrange": False,
                                "mesh_type": mesh_type,
                                "floor_xyz": floor_xyz,
                                "bddl_object_scope": None
                            }
                        }

                        break

                if mesh_found is None:
                    print(f"提示:未在 mesh 数组中找到 uid 为 {ref} 的条目,跳过。")
                    continue
            else:
                print(f"警告:object type {child_instanceid} 既不是furniture,也不是mesh,跳过。")
            if floor_too_small:
                print("room is too small.")
                break
            if is_timeout:
                break
        
        if floor_too_small or furniture_num < 2 or is_timeout:
            print(f"{room_instanceid} 跳过, furniture_num:{furniture_num}")
            continue
        # random get rearrange objects(>4)
        init_info = new_json["objects_info"]["init_info"]
        to_rearrange_keys = [key for key, obj in init_info.items() if obj["args"].get("is_to_rearrange", False)]

        if len(to_rearrange_keys) < 1:  
            print(f"{room_instanceid} 跳过, rearrange furniture_num:{len(to_rearrange_keys)}")
            continue
        if len(to_rearrange_keys) >= 4:
            preserved_keys = set(random.sample(to_rearrange_keys, 4))
            for key in to_rearrange_keys:
                if key not in preserved_keys:
                    init_info[key]["args"]["fixed_base"] = True
                    init_info[key]["args"]["is_to_rearrange"] = False
        
        new_dir_name = json_uid + "_" + room_instanceid
        new_dir_path = os.path.join(output_base, new_dir_name)
        if not os.path.exists(new_dir_path):
            os.makedirs(new_dir_path)
        file_name = json_uid + "_" + f"{room_instanceid}_target.json"
        new_file_path = os.path.join(new_dir_path, file_name)

        with open(new_file_path, 'w') as f:
            json.dump(new_json, f, indent=4)

        print(f"Modified JSON file saved as {new_file_path}")

print("批量转换完成。")
            