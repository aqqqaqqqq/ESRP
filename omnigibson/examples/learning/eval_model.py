import os

import gymnasium as gym
import numpy as np
import omnigibson as og
import torch
import yaml
from gymnasium.spaces import Box
from loguru import logger
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from omnigibson.examples.environments.new_env import FastEnv
from omnigibson.macros import gm
from omnigibson.utils.model_utils import LSTMContainingRLModule

from shapely.geometry import Polygon as pol
from omnigibson.examples.environments.get_camera_picture import compute_camera_height_from_polygon, capture_top_down_image
from omnigibson.examples.robots.rearrange_robot import add_external_sensors, save_img


gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_TRANSITION_RULES = True
gm.ENABLE_FLATCACHE = True
gm.RENDER_VIEWER_CAMERA = True

MODEL_PATH = "/home/user/Desktop/wq/try/second/saved_models/4500"
NEW_LOG = "/home/user/Desktop/wq/pictures/second/more_metric.log"
# SCENE_LIST_PATH = "/home/user/Desktop/rl/omnigibson/data/test_all_data.txt"
SCENE_LIST_PATH = "/home/user/Desktop/wq/pictures/second/visualize_scene.txt"
result_path = "/home/user/Desktop/wq/pictures/second"
OBSERVATION_SPACE = Box(low=0, high=255, shape=(98305,), dtype=np.uint8)
ACTION_SPACE = gym.spaces.Discrete(6)

def main():
    spec = RLModuleSpec(
        LSTMContainingRLModule,
        observation_space=OBSERVATION_SPACE,
        action_space=ACTION_SPACE,
    )
    rl_module = spec.build()
    rl_module.restore_from_path(path=MODEL_PATH)

    config_filename = os.path.join(og.example_config_path, "rearrange.yaml")
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    existing_content = ""
    if os.path.exists(NEW_LOG):
        with open(NEW_LOG, "r", encoding="utf-8") as f:
            existing_content = f.read()

    scene_names = []
    with open(SCENE_LIST_PATH, "r", encoding="utf-8") as f:
        for line in f:
            scene_name = line.rstrip("\n")
            if scene_name and scene_name not in existing_content:
                scene_names.append(scene_name)

    threed_front_path = gm.ThreeD_FRONT_DATASET_PATH
    scene_path = os.path.join(threed_front_path, "scenes", "8148b1a7-7c15-4b53-9be3-8b5a617ba9d2_Bedroom-29109")
    config["scene"]["scene_model"] = "8148b1a7-7c15-4b53-9be3-8b5a617ba9d2_Bedroom-29109_target.json"
    config["scene"]["scene_type_path"] = scene_path
    config["env"]["scene_names"] = scene_names

    print(config["env"]["scene_names"])
    print("len:", len(config["env"]["scene_names"]))

    TAKE_PICTURE = True
    if TAKE_PICTURE:
        config = add_external_sensors(config)

    env = og.Environment(configs=config)
    rearrangement_env = FastEnv(env)

    logger.remove()
    logger.add(NEW_LOG, encoding="utf-8")

    while rearrangement_env.env.scene_names:
        obs, _ = rearrangement_env.reset(seed=None)
        initial_state = rl_module.get_initial_state()
        module_input = {
            Columns.STATE_IN: {
                "h": torch.tensor(initial_state["h"]).unsqueeze(0),
                "c": torch.tensor(initial_state["c"]).unsqueeze(0),
            },
            Columns.OBS: obs.unsqueeze(0).unsqueeze(0),
        }

        if TAKE_PICTURE:
            floor_poly = rearrangement_env.env.task.get_floor_poly(rearrangement_env.env)
            polygon = pol(floor_poly)
            x = polygon.centroid.x
            z = polygon.centroid.y

            cam = rearrangement_env.env.external_sensors["top_cam"]
            y = compute_camera_height_from_polygon(cam, np.array(floor_poly))
            top_down_position = torch.tensor([x, y, z])
            top_down_orientation = torch.tensor([-0.5, -0.5, -0.5, 0.5])
            cam.set_position_orientation(top_down_position, top_down_orientation, frame="scene")

        infos = []
        scene_step = 0
        success = False
        scene_name = rearrangement_env.env._scene_name
        min_distance_per_object = {}
        min_distance_step_per_object = {}

        while True:
            inference_result = rl_module.forward_inference(module_input)
            probabilities = inference_result[Columns.ACTION_DIST_INPUTS]
            action = torch.multinomial(torch.softmax(probabilities.view(-1), 0), 1)

            obs, _, terminated, truncated, info = rearrangement_env.step(int(action))
            module_input = {
                Columns.OBS: obs.unsqueeze(0).unsqueeze(0),
                Columns.STATE_IN: inference_result[Columns.STATE_OUT],
            }

            scene_step += 1
            distances = rearrangement_env.env.task.get_robot_rearrange_object_distances(rearrangement_env.env)
            for object_name, distance in distances.items():
                if object_name not in min_distance_per_object or distance < min_distance_per_object[object_name]:
                    min_distance_per_object[object_name] = distance
                    min_distance_step_per_object[object_name] = scene_step
            infos.append(info)

            if TAKE_PICTURE:
                img = capture_top_down_image(cam)
                file_path = os.path.join(result_path, scene_name)
                file_name = str(scene_step) + ".png"
                save_img(img, file_path, file_name)

            if terminated:
                success = True
                print("success!")
                break

            if truncated:
                print("not success")
                break

        last_info = infos[-1]
        init_potential = sum(v["pos"] for v in last_info["reward"]["potential"]["ini"].values())
        final_potential = sum(v["pos"] for v in last_info["reward"]["potential"]["end"].values())

        arrival_num = last_info["reward"]["arrival"]
        assert len(arrival_num) == 1
        arrival_num = next(iter(arrival_num))

        eval_dict = {
            "step": last_info["episode_length"],
            "success": success,
            "scene_name": scene_name,
            "obj_num": len(last_info["reward"]["potential"]["ini"]),
            "arrival_num": arrival_num,
            "init_potential": init_potential,
            "fini_potential": final_potential,
            "min_distance_per_object": min_distance_per_object,
            "min_distance_step_per_object": min_distance_step_per_object,
            "first_grasp_step": last_info["eval_metrics"]["first_grasp_step"],
            "released_before_target": last_info["eval_metrics"]["released_before_target"],
            "grasp_events": last_info["eval_metrics"]["grasp_events"],
            "release_events": last_info["eval_metrics"]["release_events"],
        }
        logger.info(eval_dict)
        rearrangement_env.env.scene_names.remove(scene_name)

    og.clear()


if __name__ == "__main__":
    main()
