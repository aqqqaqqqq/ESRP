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
    # print(cam.get_obs()[0]['rgb'].shape)
    return img
    
import numpy as np

def compute_camera_height_from_polygon(sensor, polygon: np.ndarray) -> float:
    """
    计算俯视相机高度，使给定多边形在画面中尽可能充满  renderProductResolution 指定的图像。

    1) 先根据 renderProductResolution 裁剪 cameraAperture（mm）以匹配图像宽高比，
    2) 再把 mm 单位转换到场景单位，最后
    3) 用 pinhole 模型算出高度。

    Args:
        polygon (np.ndarray): 多边形顶点列表，形状 (N,2)，单位为场景单位。

    Returns:
        float: 建议相机高度，单位为场景单位。
    """
    # —————————————————————————————
    # 1. 读属性
    # —————————————————————————————
    # 物理传感器（aperture）宽/高，单位 mm
    ap_w_mm, ap_h_mm = sensor.camera_parameters["cameraAperture"].tolist()
    print("ap_w_mm, ap_h_mm:", ap_w_mm, ap_h_mm)
    # 渲染产物的像素分辨率
    img_w_px, img_h_px = sensor.camera_parameters["renderProductResolution"].tolist()
    print("img_w_px, img_h_px:", img_w_px, img_h_px)
    # 场景单位到米的转换：1 su = metersPerSceneUnit 米
    m_per_su = sensor.camera_parameters["metersPerSceneUnit"]
    # 焦距 mm
    f_mm = sensor.camera_parameters["cameraFocalLength"]
    print("f_mm:", f_mm)

    # —————————————————————————————
    # 2. 按分辨率裁剪 aperture
    # —————————————————————————————
    img_ar    = img_w_px / img_h_px
    sensor_ar = ap_w_mm  / ap_h_mm

    if sensor_ar > img_ar:
        # 传感器比图像更“宽”，裁掉左右 → 用新的 sensor 宽度
        eff_h_mm = ap_h_mm
        eff_w_mm = ap_h_mm * img_ar
    else:
        # 传感器比图像更“高”，裁掉上下 → 用新的 sensor 高度
        eff_w_mm = ap_w_mm
        eff_h_mm = ap_w_mm / img_ar

    # —————————————————————————————
    # 3. 转换到场景单位
    # —————————————————————————————
    mm_to_su   = (1e-3 / m_per_su)
    sensor_w_su = eff_w_mm * mm_to_su
    sensor_h_su = eff_h_mm * mm_to_su
    focal_su    = f_mm    * mm_to_su

    # —————————————————————————————
    # 4. 多边形世界大小
    # —————————————————————————————
    center       = polygon.mean(axis=0)
    offsets      = polygon - center
    width_world  = np.max(np.abs(offsets[:, 0])) * 2
    height_world = np.max(np.abs(offsets[:, 1])) * 2

    # —————————————————————————————
    # 5. 针孔模型算高度
    # —————————————————————————————
    h_x = focal_su * width_world  / sensor_w_su
    h_y = focal_su * height_world / sensor_h_su

    return float(max(h_x, h_y))

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

    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()

    # Run a simple loop and reset periodically
    max_iterations = len(scene_names) if not short_exec else 1
    for j in range(max_iterations):

        print("Resetting environment")
        print("scene_name:", env.scene.scene_model)
        print(f"{j+6294-max_iterations+1}/6294")
        room_model = env.scene_config["scene_model"]
        scene_path =env.scene_config["scene_type_path"]
        scene_model_name = room_model.replace("_target.json", "")
        # scene_names.remove(scene_model_name)

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
        y = 400.0

        cam = og.sim.viewer_camera
        cam.focal_length = 1000.0
        print(cam.camera_parameters["cameraAperture"])
        # y = compute_camera_height(np.array(floor_poly), cam.focal_length, cam.camera_parameters["cameraAperture"][0], cam.camera_parameters["cameraAperture"][1])
        y = compute_camera_height_from_polygon(cam, np.array(floor_poly))
        print("y:", y)
        top_down_position = th.tensor([x + 20, y, z])
        top_down_orientation = th.tensor([-0.5, -0.5, -0.5, 0.5])
        cam.set_position_orientation(top_down_position, top_down_orientation)

        for i in range(2000):
                        
            env.step(th.empty(0))
            if i == 100:
                # import pdb;pdb.set_trace()
                img = capture_top_down_image(cam)
                # if img correct, break
                if img is not None:
                    save_dir = scene_path
                    file_path = os.path.join(save_dir, room_model.replace('json', 'png'))
                    import pdb; pdb.set_trace()
                    save_img(img, file_path)
                    print(f"save img at {file_path}")
                    # break

        scene_names.remove(env.scene.scene_model.replace("_target.json", ""))
        env.scene_names = scene_names
        env.reset()
        # import pdb;pdb.set_trace()

    # Always close the environment at the end
    # env.close()
    og.clear()


scene_names = []
imitation_names = []
        
if __name__ == "__main__":
    os.environ["OMNIGIBSON_HEADLESS"] = "1"

    threed_front_path = gm.ThreeD_FRONT_DATASET_PATH
    scenes_dir_path = os.path.join(threed_front_path, "scenes")
 
    total = 0
    for entry in os.listdir(scenes_dir_path):
        print(f"deal: {total+1}/6294")
        total += 1
        entry_path = os.path.join(scenes_dir_path, entry)
        if os.path.isdir(entry_path):
            scene_names.append(entry)
            
            for file in os.listdir(entry_path):
                if ".png" in file:
                    png_path = os.path.join(entry_path, file)
                    # os.remove(png_path)
                    scene_names.remove(entry)
    print("scene_names:", len(scene_names))
    scene_names= ["0a8d471a-2587-458a-9214-586e003e9cf9_LivingDiningRoom-4017"]
    modi_main()
