"""
Example script demo'ing robot control.

Options for random actions, as well as selection of robot action space
"""

import torch as th

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import gm
from omnigibson.robots import REGISTERED_ROBOTS
from omnigibson.utils.ui_utils import KeyboardRobotController, choose_from_options
from omnigibson.sensors import VisionSensor

CONTROL_MODES = dict(
    random="Use autonomous random actions (default)",
    teleop="Use keyboard control",
)

SCENES = dict(
    Rs_int="Realistic interactive home environment (default)",
    empty="Empty environment with no objects",
)

# Don't use GPU dynamics and use flatcache for performance boost
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True


def choose_controllers(robot, random_selection=False):
    """
    For a given robot, iterates over all components of the robot, and returns the requested controller type for each
    component.

    :param robot: BaseRobot, robot class from which to infer relevant valid controller options
    :param random_selection: bool, if the selection is random (for automatic demo execution). Default False

    :return dict: Mapping from individual robot component (e.g.: base, arm, etc.) to selected controller names
    """
    # Create new dict to store responses from user
    controller_choices = dict()

    # Grab the default controller config so we have the registry of all possible controller options
    default_config = robot._default_controller_config

    # Iterate over all components in robot
    for component, controller_options in default_config.items():
        # Select controller
        options = list(sorted(controller_options.keys()))
        choice = "DifferentialDriveController"

        # Add to user responses
        controller_choices[component] = choice

    return controller_choices


def main(random_selection=False, headless=False, short_exec=False, quickstart=False):
    """
    Robot control demo with selection
    Queries the user to select a robot, the controllers, a scene and a type of input (random actions or teleop)
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    # Choose scene to load
    scene_model = "empty"

    # Choose robot to create
    robot_name = "Turtlebot"

    scene_cfg = dict()
    if scene_model == "empty":
        scene_cfg["type"] = "Scene"
    else:
        scene_cfg["type"] = "InteractiveTraversableScene"
        scene_cfg["scene_model"] = scene_model

    # Add the robot we want to load
    robot0_cfg = dict()
    robot0_cfg["type"] = robot_name
    robot0_cfg["obs_modalities"] = ["seg_semantic"]
    robot0_cfg["action_type"] = "continuous"
    robot0_cfg["action_normalize"] = True

    box1_cfg = dict(
        type="PrimitiveObject",
        name="box1",
        primitive_type="Cube",
        rgba=[1.0, 0, 0, 1.0],
        size=0.2,
        position=[1.0, 0.0, 0.0],
        orientation=[1.0, 0.0, 0.0, 0.0],
    )

    box2_cfg = dict(
        type="PrimitiveObject",
        name="box2",
        primitive_type="Cube",
        rgba=[1.0, 1, 0, 0.0],
        size=0.2,
        position=[0.0, 1.0, 0.0],
        orientation=[1.0, 0.0, 0.0, 0.0],
    )
    box3_cfg = dict(
        type="PrimitiveObject",
        name="box3",
        primitive_type="Cube",
        rgba=[0.0, 1, 0, 1.0],
        size=0.2,
        position=[2.0, 1.0, 0.0],
    )

    # Compile config
    cfg = dict(scene=scene_cfg, robots=[robot0_cfg], objects=[box1_cfg, box2_cfg, box3_cfg])

    # Create the environment
    env = og.Environment(configs=cfg)

    # Choose robot controller to use
    robot = env.robots[0]
    
    controller_choices = choose_controllers(robot=robot, random_selection=random_selection)

    control_mode = "teleop"

    # Update the control mode of the robot
    controller_config = {component: {"name": name} for component, name in controller_choices.items()}
    robot.reload_controllers(controller_config=controller_config)

    # Because the controllers have been updated, we need to update the initial state so the correct controller state
    # is preserved
    env.scene.update_initial_state()

    # Update the simulator's viewer camera's pose so it points towards the robot
    og.sim.viewer_camera.set_position_orientation(
        position=th.tensor([1.46949, -3.97358, 2.21529]),
        orientation=th.tensor([0.56829048, 0.09569975, 0.13571846, 0.80589577]),
    )

    # Reset environment and robot
    env.reset()
    robot.set_position_orientation(position=[0, 0, 0])
    robot.reset()
    
    for sensor in robot.sensors.values():
        if isinstance(sensor, VisionSensor):
            sensor.image_height = 720
            sensor.image_width = 720 
            ### Test
            '''
            from PIL import Image
            import numpy as np
            obs, _ = sensor._get_obs()
            
            obs_array1 = obs['seg_semantic']
            obs_array1 = obs_array1.cpu().numpy()
            #print(obs_array.shape)
            rgb_array1 = (obs_array1 / np.max(obs_array1) * 255).astype(np.uint8)
            #print(obs_array1)
            image1 = Image.fromarray(rgb_array1)
            image1.save("/home/pilab/Desktop/try/OmniGibson-main/omnigibson/examples/robots/4.png")
            '''
            '''
            obs_array2 = obs['seg_instance']
            obs_array2 = obs_array2.cpu().numpy()
            rgb_array2 = (obs_array2 / np.max(obs_array2) * 255).astype(np.uint8)
            image2 = Image.fromarray(rgb_array2)
            image2.save("/home/pilab/Desktop/try/OmniGibson-main/omnigibson/examples/robots/5.png")
            '''
            '''
            #print(obs["depth"].numpy())
            obs_array1 = np.array(obs['depth'].numpy())
            obs_array2 = obs['depth_linear'].numpy()
            rgb_array1 = (obs_array1*255).astype(np.uint8)
            rgb_array2 = (obs_array2*255).astype(np.uint8)
            #print(rgb_array)
            image1 = Image.fromarray(rgb_array1)
            image2 = Image.fromarray(rgb_array2)
            image1.save("/home/pilab/Desktop/try/OmniGibson-main/omnigibson/examples/robots/2.png")
            image2.save("/home/pilab/Desktop/try/OmniGibson-main/omnigibson/examples/robots/3.png")
            '''            
            #print(obs['rgb'].shape)
            #rgb_array = obs['rgb'].numpy()
            #print(rgb_array)
            #image = Image.fromarray(rgb_array, mode='RGBA')
            ###
       

    # Create teleop controller
    action_generator = KeyboardRobotController(robot=robot)

    # Register custom binding to reset the environment
    action_generator.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.R,
        description="Reset the robot",
        callback_fn=lambda: env.reset(),
    )

    # Print out relevant keyboard info if using keyboard teleop
    if control_mode == "teleop":
        action_generator.print_keyboard_teleop_info()

    # Other helpful user info
    print("Running demo.")
    print("Press ESC to quit")

    # Loop control until user quits
    max_steps = -1 if not short_exec else 100
    step = 0

    while step != max_steps:
        action = (
            action_generator.get_random_action() if control_mode == "random" else action_generator.get_teleop_action()
        )
        env.step(action=action)
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
