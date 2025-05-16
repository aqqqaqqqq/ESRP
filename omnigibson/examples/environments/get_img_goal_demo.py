import os
import json
import torch as th
import numpy as np
import math
import omnigibson as og
from omnigibson.macros import gm
from omnigibson.utils.asset_utils import get_available_3dfront_scenes, get_available_3dfront_target_scenes
from omnigibson.utils.bbox_utils import remove_duplicate_vertices, remove_useless_points
import omnigibson.lazy as lazy
from omnigibson.robots import REGISTERED_ROBOTS
from omnigibson.sensors.vision_sensor import VisionSensor
from omnigibson.utils.ui_utils import choose_from_options
from PIL import Image
from shapely.geometry import Polygon as pol
from PIL import Image
import imagehash
import shutil


def save_img(t, file_path):
    im = Image.fromarray(t.numpy())
    im.save(file_path)

def capture_top_down_image(cam):
    img = cam.get_obs()[0]['rgb']
    print(cam.get_obs()[0]['rgb'].shape)
    return img

def modi_main(random_selection=False, headless=False, short_exec=False, quickstart=False):

    scene_type = "Threed_FRONTScene"
    # Choose the scene model to load
    threed_front_path = gm.ThreeD_FRONT_DATASET_PATH
    scene_path = os.path.join(threed_front_path, "scenes", "8148b1a7-7c15-4b53-9be3-8b5a617ba9d2_Bedroom-29109")

    cfg = {
        "env": {
            "scene_names": scene_names,
            "rearrangement": False,
            "modify_reload_model": True
        },
        "scene": {
            "type": scene_type,
            "scene_model": "8148b1a7-7c15-4b53-9be3-8b5a617ba9d2_Bedroom-29109_target.json",
            "scene_type_path": scene_path
        },
        "render": {
            "viewer_width": 512,
            "viewer_height": 512
        },
    
    }

    # Load the environment
    env = og.Environment(configs=cfg)

    room_model = env.scene_config["scene_model"]
    scene_path =env.scene_config["scene_type_path"]
    scene_model_name = room_model.replace("_target.json", "")
    scene_names.remove(scene_model_name)

    room_path = os.path.join(scene_path, room_model)
    with open(room_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    objs = data["objects_info"]["init_info"]
    floor_xyz = []
    for obj_n, obj_info in objs.items():
        if "mesh" not in obj_n:
            continue
        if obj_info["args"]["floor_xyz"] is not None:
            floor_xyz = obj_info["args"]["floor_xyz"]
            break

    num_points = len(floor_xyz) // 3
    floor_vertices = []
    for i in range(num_points):
        x_floor = floor_xyz[3*i]
        y_floor = floor_xyz[3*i+1]
        z_floor = floor_xyz[3*i+2]
        floor_vertices.append([x_floor, y_floor, z_floor])
    floor_vertices = remove_duplicate_vertices(floor_vertices)
    floor_poly = [[v[0], v[2]] for v in floor_vertices]
    floor_poly = remove_useless_points(floor_poly)

    print("floor_poly:", floor_poly)

    polygon = pol(floor_poly)
    x = polygon.centroid.x
    z = polygon.centroid.y
    
    top_down_position = th.tensor([x, 11.0, z])
    # top_down_orientation = th.tensor([0.0, 0.0, -0.70711, 0.70711])
    top_down_orientation = th.tensor([-0.5, -0.5, -0.5, 0.5])
    # top_down_orientation = th.tensor([0.0, -0.70711, 0.70711, 0.0])
    cam = og.sim.viewer_camera
    cam.set_position_orientation(top_down_position, top_down_orientation)

    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()

    # Run a simple loop and reset periodically
    max_iterations = 1 if not short_exec else 1
    for j in range(max_iterations):
        print("Resetting environment")
        # env.reset()
        for i in range(200):

            if i == 50:
                for obj_n, obj_info in env.scene._init_state.items():
                    pos_list = np.array(obj_info["root_link"]["pos"])
                    if np.isnan(pos_list).any():
                        print("NaN! break")
                        break
                        
                
            env.step(th.empty(0))
            if i == 100:
                img = capture_top_down_image(cam)
                # if img correct, break
                if img is not None:
                    save_dir = scene_path
                    file_path = os.path.join(save_dir, room_model.replace('json', 'png'))
                    save_img(img, file_path)
                    print(f"save img at {file_path}")

    # Always close the environment at the end
    # env.close()
    og.clear()


scene_names = []
        
if __name__ == "__main__":
    import argparse

    # parser = argparse.ArgumentParser(description="Get the top-down image as the img-goal.")
    # parser.add_argument(
    #     "--scene",
    #     type=str,
    #     help="The scene identifier to load"
    # )
    # args = parser.parse_args()
    os.environ["OMNIGIBSON_HEADLESS"] = "1"

    threed_front_path = gm.ThreeD_FRONT_DATASET_PATH
    scenes_dir_path = os.path.join(threed_front_path, "scenes")
    parent_folder = scenes_dir_path

    total = 0
    target_image_path = "122feb6c-450f-4d1b-a02a-25c976b14ba4_Bedroom-9204_target"
    target_hash = imagehash.phash(Image.open(target_image_path))
    
    identical_files = []
    for entry in os.listdir(scenes_dir_path):
        print("deal", total)
        total += 1
        entry_path = os.path.join(scenes_dir_path, entry)
        if os.path.isdir(entry_path):
            scene_names.append(entry)
            for file in os.listdir(entry_path):
                if file.lower().endswith('.png'):
                    file_path = os.path.join(entry_path, file)
                    # try:
                    #     # 打开并计算当前图片的感知哈希值
                    #     with Image.open(file_path) as img:
                    #         current_hash = imagehash.phash(img)
                    #         # 哈希值完全相同则视为相同图片
                    #         if current_hash == target_hash:
                    #             identical_files.append(file_path)
                    #             os.remove(file_path)
                    # except (IOError, SyntaxError):
                    #     # 非图片或无法打开的文件跳过
                    #     continue
                    # os.remove(file_path)
                    scene_names.remove(entry)
                    break
    # print(scene_names)
    print("scene_names:", len(scene_names))
