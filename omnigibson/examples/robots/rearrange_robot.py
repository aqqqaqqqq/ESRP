import os
import torch as th
import omnigibson as og
from omnigibson.macros import gm
import argparse
import yaml
import matplotlib
import cProfile
from omnigibson.examples.environments.new_env import FastEnv
from omnigibson.utils.bbox_utils import remove_duplicate_vertices, remove_useless_points
from omnigibson.examples.environments.get_camera_picture import compute_camera_height_from_polygon, save_img, capture_top_down_image
import json
from shapely.geometry import Polygon as pol
import numpy as np
from PIL import Image

matplotlib.use('Agg')
profiler = cProfile.Profile() 

# We don't need object states nor transitions rules, so we disable them now, and also enable flatcache for maximum speed
gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_TRANSITION_RULES = True
gm.ENABLE_FLATCACHE = True
gm.RENDER_VIEWER_CAMERA = True

CONTROL_MODES = dict(
    random="Use autonomous random actions (default)",
    teleop="Use keyboard control",
)

# SCENES = dict(
#     Rs_int="Realistic interactive home environment (default)",
#     empty="Empty environment with no objects",
# )

def save_img(t, file_path, file_name):
    if not os.path.exists(file_path):#检查目录是否存在
        os.makedirs(file_path)
    im = Image.fromarray(t.numpy())
    im.save(file_path + '/' + file_name)

def main(random_selection=False, headless=False, short_exec=False, quickstart=False):
    """
    Robot control demo with selection
    Queries the user to select a robot, the controllers, a scene and a type of input (random actions or teleop)
    """
    # Choose scene to load

    config_filename = config_filename = os.path.join(og.example_config_path, f"rearrange.yaml")
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
    
    scene_name = "50519e1e-1355-41f3-b092-6256d2ce205f_LivingRoom-6803"
    config['env']['scene_names'] = [scene_name]

    config["render"]["viewer_width"] = 2048
    config["render"]["viewer_height"] = 2048

    env = og.Environment(configs=config)
    rearrangement_env = FastEnv(env)

    room_model = rearrangement_env.env.scene_config["scene_model"]
    scene_path = rearrangement_env.env.scene_config["scene_type_path"]
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

    polygon = pol(floor_poly)
    x = polygon.centroid.x
    z = polygon.centroid.y
    y = 400.0

    cam = og.sim.viewer_camera
    cam.focal_length = 1000.0
    # y = compute_camera_height(np.array(floor_poly), cam.focal_length, cam.camera_parameters["cameraAperture"][0], cam.camera_parameters["cameraAperture"][1])
    y = compute_camera_height_from_polygon(cam, np.array(floor_poly))
    top_down_position = th.tensor([x + 20, y, z])
    top_down_orientation = th.tensor([-0.5, -0.5, -0.5, 0.5])
    cam.set_position_orientation(top_down_position, top_down_orientation)

    for i in range(10):
        og.sim.render()

    # Other helpful user info
    print("Running demo.")
    print("Press ESC to quit")

    # 循环控制
    max_steps = -1 if not short_exec else 100
    step = 0
    action_empty = th.zeros(13)
    grasping_obj = None
    TAKE_PICTURE = True

    while step != max_steps:
        if TAKE_PICTURE:
            img = capture_top_down_image(cam)
            result_path = " To Do "
            file_path = os.path.join(result_path, scene_name)
            file_name = str(step) + ".png"
            save_img(img, file_path, file_name)

        action = int(input("请输入动作编号 (0-5): "))
        # 执行动作
        obs, reward, terminated, truncated, info = rearrangement_env.step(action)
        pointgoal_rewards = info['reward']['reward_breakdown']['pointgoal']
        # rgb_obs = obs[:-1].resize(2048,2048,6)[:,:,:3]
        # from PIL import Image
        # im = Image.fromarray(rgb_obs.numpy())
        # arrival_rewards = info['reward']['reward_breakdown']['arrival']
        # potential_rewards = info['reward']['reward_breakdown']['potential']
        # grasping_rewards = info['reward']['reward_breakdown']['grasping']
        # living_rewards = info['reward']['reward_breakdown']['living']
        # print(pointgoal_rewards)
        # import pdb; pdb.set_trace()
        step += 1

    # Always shut down the environment cleanly at the end
    og.clear()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Teleoperate a robot in a BEHAVIOR scene.")

    parser.add_argument(
        "--quickstart",
        action="store_true",
        help="Whether the example should be loaded with default settings for a quick start.",
    )
    args = parser.parse_args()
    main(quickstart=args.quickstart)