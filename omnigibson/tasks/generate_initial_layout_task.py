import numpy as np
import json
import os
import random
import math
import torch as th
import omnigibson as og
from omnigibson.utils.ui_utils import create_module_logger
from omnigibson.tasks.task_base import BaseTask
from omnigibson.scenes.scene_base import Scene
from omnigibson.object_states import Pose
from omnigibson.reward_functions.point_goal_reward import PointGoalReward
from omnigibson.termination_conditions.rearrange_goal import RearrangeGoal
from omnigibson.reward_functions.arrival_reward import ArrivalReward
from omnigibson.termination_conditions.timeout import Timeout
from omnigibson.utils.python_utils import classproperty
from omnigibson.utils.sim_utils import land_object_rearrange, test_valid_pose
from omnigibson.utils.bbox_utils import is_candidate_colliding, find_free_area_on_floor_random, remove_duplicate_vertices, remove_useless_points, is_point_in_polygon, sample_point_around_object, visualize_scene, get_obj_bbox
import omnigibson.utils.transform_utils as T
from shapely.geometry import Point as poi
from shapely.geometry import Polygon as pol


# Create module logger
log = create_module_logger(module_name=__name__)

class GenerateInitialLayoutTask(BaseTask):

    def __init__(
            self,
            robot_idn=0,
            floor=0,
            obj_num=6,
            initial_pos=None,
            initial_quat=None,
            termination_config=None,
            reward_config=None,
    ):
        self._robot_idn = robot_idn
        self._floor = floor
        self._initial_pos = initial_pos if initial_pos is None else np.array(initial_pos)
        self._initial_quat = initial_quat if initial_quat is None else np.array(initial_quat)
        self._obj_num = obj_num

        self._randomize_initial_pos = initial_pos is None
        self._randomize_initial_quat = initial_quat is None

        # Create other attributes that will be filled in at runtime
        
        self._current_robot_pos = None
        self.objects_initial_pos_ori = {}
        self.objects_target_pos_ori = {}
        # Run super
        super().__init__(termination_config=termination_config, reward_config=reward_config)     

    def _reset_variables(self, env):
        # Run super first
        super()._reset_variables(env=env)

        # Reset internal variables
        self._current_robot_pos = self._initial_pos
        objects_to_rearrange = self.get_rearrange_objects_names(env)
        #print(objects_to_rearrange)
        self.get_initial_pos_ori(env, objects_to_rearrange[0])

    def _find_free_area_on_floor_random(self, env):
        # sample an area for the robot
        floor = self.get_floor(env)
        floor_xyz = floor.floor_xyz
        num_points = len(floor_xyz) // 3
        floor_vertices = []
        for i in range(num_points):
            x = floor_xyz[3*i]
            y = floor_xyz[3*i+1]
            z = floor_xyz[3*i+2]
            ## transform pos
            floor_vertices.append([x, y, z])
        floor_vertices = remove_duplicate_vertices(floor_vertices)
        floor_poly = [[v[0], v[2]] for v in floor_vertices]
        floor_poly = remove_useless_points(floor_poly)
        print("floor_poly:", floor_poly)
        floor_polygon = pol(np.array(floor_poly))
        objs_name = self.get_all_objects_names(env)
        obstacles_polygons = []
        for obj_name in objs_name:
            obstacles_polygons.append(pol(get_obj_bbox(env, obj_name)))

        result = find_free_area_on_floor_random(env, floor_polygon, obstacles_polygons)
        
        # result = ([-5.4344,  1.0000,  4.4591], [-0.70711, 0, 0, 0.70711])
        assert result , "Could not find a free area to place the robot."
        result = ([result.x, 1.0000, result.y], [-0.70711, 0, 0, 0.70711])
        return th.tensor(result[0]), th.tensor(result[1])

    def _sample_point_around_object(self, env, obj_n):

        point_around_object = sample_point_around_object(env, obj_n)
        
        assert point_around_object is not None, f"cannot sample a point around the {obj_n}"
        x = point_around_object[0]
        z = point_around_object[1]
        
        return x, z

    def sample_no_collision_point_around_object(self, env, obj_n):
        x, z = self._sample_point_around_object(env, obj_n)
        # x = -6.62
        # z = 4.66
        # x = -5.6330
        # z = 4.5266
        step = 0
        
        # Get scene data for visualization
        floor = self.get_floor(env)
        floor_xyz = floor.floor_xyz

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

        save_dir = os.path.join(os.getcwd(), f"sample_point_around_{obj_n}")
        os.makedirs(save_dir, exist_ok=True)
        visualize_scene(env, floor_poly, np.array([x, z]), robot_radius=0.4, save_dir=save_dir, filename=f"{step}.png")
        
        while self.is_robot_collision(env, x, z):
            
            # Sample a new point
            x, z = self._sample_point_around_object(env, obj_n)
            step += 1
            assert step <= 10000, f"cannot sample a no-collision point around the {obj_n}"

            visualize_scene(env, floor_poly, np.array([x, z]), robot_radius=0.4, save_dir=save_dir, filename=f"{step}.png")
    
        return x, z

    def is_robot_collision(self, env, robot_x, robot_z):
        floor = self.get_floor(env)
        floor_xyz = floor.floor_xyz
        num_points = len(floor_xyz) // 3
        floor_vertices = []
        for i in range(num_points):
            x = floor_xyz[3*i]
            y = floor_xyz[3*i+1]
            z = floor_xyz[3*i+2]
            ## transform pos
            floor_vertices.append([x, y, z])
        # 计算地板的 XZ AABB
        floor_vertices = remove_duplicate_vertices(floor_vertices)
        floor_poly = [[v[0], v[2]] for v in floor_vertices]
        # floor_poly = convex_hull(floor_po_collision_detectionly)
        floor_poly = remove_useless_points(floor_poly)
        floor_pol = pol(floor_poly)
        robot_point = poi(robot_x, robot_z)
        robot_radius = 0.4
        if not(floor_pol.contains(robot_point) and robot_point.distance(floor_pol.exterior) > robot_radius):
            # print("Robot is out of the floor.")
            return True
        
        candidate = np.array([robot_x, robot_z])
        # candidate1 = np.array([robot_x-robot_radius, robot_z-robot_radius])
        # candidate2 = np.array([robot_x-robot_radius, robot_z+robot_radius])
        # candidate3 = np.array([robot_x+robot_radius, robot_z+robot_radius])
        # candidate4 = np.array([robot_x+robot_radius, robot_z-robot_radius])
        # if not (is_point_in_polygon(candidate1, floor_poly) and is_point_in_polygon(candidate2, floor_poly) and is_point_in_polygon(candidate3, floor_poly) and is_point_in_polygon(candidate4, floor_poly)):
        #     print("Robot is out of the floor.")
        #     return True

        collision = False
        objs_name = self.get_all_objects_names(env)
        for obj_name in objs_name:
            if is_candidate_colliding(env, candidate, obj_name, robot_radius):
                # print(f"Robot has collision with {obj_name}.")
                collision = True
                break

        return collision

    def _reset_agent(self, env):
        # Reset agent
        env.robots[self._robot_idn].reset()

        initial_pos, initial_quat = self._find_free_area_on_floor_random(env)
        # print("initial_pos:", initial_pos)

        # Land the robot
        land_object_rearrange(env.robots[self._robot_idn], initial_pos, initial_quat, env.initial_pos_z_offset)
        # print("robot_initial:", env.robots[0].get_position_orientation(frame="scene"))
        # Store the sampled values internally
        self._initial_pos = initial_pos
        self._initial_quat = initial_quat

    def _load(self, env):
        # Do nothing here
        pass

    def get_floor(self, env):
        all_objects = env.scene._init_objs
        for obj_name, obj in all_objects.items():
            if obj.mesh_type == "Floor":
                return obj
        print("There is no floor.")
        return None

    def get_rearrange_objects_names(self, env):
        # print(list(env._scene._init_objs.keys()))

        self.objects_to_rearrange = []
        all_objects = env.scene._init_objs
        for obj_name, obj in all_objects.items():
            if obj.is_to_rearrange:
                self.objects_to_rearrange.append(obj_name)
        self._obj_num = len(self.objects_to_rearrange)
        # print("num:", len(self.objects_to_rearrange))
        return self.objects_to_rearrange

    def get_all_objects_names(self, env):
        self.objects = []
        all_objects = env.scene._init_objs
        for obj_name, obj in all_objects.items():
            if "object" in obj_name:
                self.objects.append(obj_name)
        # print("num:", len(self.objects_to_rearrange))
        return self.objects

    def get_object_pos_ori(self, env, object_name):
        object = env.scene._init_objs[object_name]
        pos, ori = object.get_position_orientation(frame = "scene")
        return pos, ori

    def get_initial_pos_ori(self, env, object_name):
        if not self.objects_initial_pos_ori:
            if env.scene.scene_file is not None:
                if isinstance(env.scene.scene_file, str):
                    with open(env.scene.scene_file, "r") as f:
                        data = json.load(f)
                else:
                    data = env.scene.scene_file
                object_data = data['state']['object_registry']
                for name, object_data in object_data.items():
                    root_link = object_data.get('root_link', {})
                    pos = root_link.get('pos')
                    ori = root_link.get('ori')
                    if pos and ori:
                        self.objects_initial_pos_ori[name] = {'pos': pos, 'ori': ori}
            else:
                print("There is no scene_file.")
        # print("object_name:", object_name)
        pos_ori = self.objects_initial_pos_ori.get(object_name)
        assert pos_ori, f"{object_name} dosen't exist or no scene_file."
        return th.Tensor(pos_ori.get('pos')), th.Tensor(pos_ori.get('ori'))

    def get_target_pos_ori(self, env, object_name):
        if not self.objects_target_pos_ori:
            
            if env.scene.scene_file is not None:
                
                target_file_path = env.scene.scene_file.replace("initial", "target")
                with open(target_file_path, "r") as f:
                    data = json.load(f)

                object_data = data['state']['object_registry']
                for name, object_data in object_data.items():
                    root_link = object_data.get('root_link', {})
                    pos = root_link.get('pos')
                    ori = root_link.get('ori')
                    if pos and ori:
                        self.objects_target_pos_ori[name] = {'pos': pos, 'ori': ori}
            else:
                print("There is no scene_file.")
        # print("object_name:", object_name)
        pos_ori = self.objects_target_pos_ori.get(object_name)
        assert pos_ori, f"{object_name} dosen't exist or no scene_file."
        return th.Tensor(pos_ori.get('pos')), th.Tensor(pos_ori.get('ori'))

    def get_robot_pos(self, env):
        """
        Returns:
            3-array: (x,y,z) global current position representing the robot
        """
        return env.robots[0].states[Pose].get_value()[0]

    def calculate(self, env, obj_name):
        return False

    def _create_termination_conditions(self):
        terminations = dict()
        terminations["pointgoal"] = RearrangeGoal(self._obj_num)
        # terminations["timeout"] = Timeout(max_steps=self._termination_config["max_steps"])
        #terminations["max_collision"] = MaxCollision(max_collisions=self._termination_config["max_collisions"])
        #terminations["falling"] = Falling(robot_idn=self._robot_idn, fall_height=self._termination_config["fall_height"])
        return terminations

    def _step_termination(self, env, action, info=None):
        _, info = super()._step_termination(env=env, action=action, info=info)
        info["success"] = False
        done = False
        return done, info

    def _create_reward_functions(self):
        rewards = dict()
        rewards["pointgoal"] = PointGoalReward(
            pointgoal=self._termination_conditions["pointgoal"],
            r_pointgoal=self._reward_config["r_pointgoal"],
        )
        rewards["arrival"] = ArrivalReward(
            obj_num=4,
            first_arrival=self._reward_config["first_arrival"],
            multi_arrival=self._reward_config["multi_arrival"],
        )
        #rewards["collision"] = RobotCollisionReward(r_collision=self._reward_config["r_collision"])
        return rewards

    def _get_obs(self, env):
        # No task-specific obs of any kind
        return dict(), dict()

    def _load_non_low_dim_observation_space(self):
        # No non-low dim observations so we return an empty dict
        return dict()

    @classproperty
    def valid_scene_types(cls):
        # Any scene works
        return {Scene}

    @classproperty
    def default_termination_config(cls):
        return {
            "max_collisions": 500,
            "max_steps": 1e9,
            "fall_height": 0.03,
        }

    @classproperty
    def default_reward_config(cls):
        return {
            "r_pointgoal": 10.0,
            "first_arrival": 4.0,
            "first_leave": -4.0,
            "multi_arrival": 2.0,
            "multi_leave": -2.0,
            "r_collision": 0.1,
        }