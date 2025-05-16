import random
import math
import json
import omnigibson as og
import os

from omnigibson.utils.ui_utils import create_module_logger

from omnigibson.scenes.scene_base import Scene
from omnigibson.tasks.task_base import BaseTask
from omnigibson.utils.python_utils import classproperty
from omnigibson.utils.sim_utils import land_object_rearrange
from omnigibson.utils.bbox_utils import quat_to_rotmat, compute_floor_aabb, aabb_overlap_2d, find_free_area_on_floor_random, remove_duplicate_vertices, remove_useless_points, get_obj_bbox
from omnigibson.object_states import Pose
import torch as th
import omnigibson.utils.transform_utils as T
from omnigibson.objects.primitive_object import PrimitiveObject
from omnigibson.termination_conditions.rearrange_goal import RearrangeGoal
from omnigibson.termination_conditions.falling import Falling
from omnigibson.termination_conditions.max_collision import MaxCollision
from omnigibson.reward_functions.point_goal_reward import PointGoalReward
from omnigibson.reward_functions.arrival_reward import ArrivalReward
from omnigibson.reward_functions.rearrange_potential_reward import RearrangePotentialReward
from omnigibson.reward_functions.grasping_reward import GraspingReward
from omnigibson.reward_functions.robot_collision_reward import RobotCollisionReward
from omnigibson.reward_functions.living_reward import LivingReward

import numpy as np
from omnigibson.termination_conditions.timeout import Timeout

from shapely.geometry import Polygon 
from shapely.geometry import Point 

# Create module logger
log = create_module_logger(module_name=__name__)

class RearrangeDlTask(BaseTask):

    def __init__(
            self,
            robot_idn = 0,
            obj_num = None,
            tolerance=0.5,
            initial_pos=None,
            initial_quat=None,
            termination_config=None,
            reward_config=None,
        ):
        self._robot_idn = robot_idn
        self._obj_num = obj_num
        self._tolerance = tolerance

        self._fetch_if = -1
        self._fetch_name = None
        self._last_name = None

        self._initial_pos = initial_pos if initial_pos is None else np.array(initial_pos)
        self._initial_quat = initial_quat if initial_quat is None else np.array(initial_quat)

        # Create other attributes that will be filled in at runtime
        self._path_length = None
        self._current_robot_pos = None
        self.objects_initial_pos_ori = {}
        self.objects_target_pos_ori = {} 
        self.objects_initial_potential = {}
        self.objects_current_potential = {}
        self.objects_target_polygon = {}
        self.objects_current_polygon = {}

        # Run super
        super().__init__()

    def _load(self, env):
        # no need load
        pass

    def _load_non_low_dim_observation_space(self):
        # No non-low dim observations so we return an empty dict
        return dict()

    def _create_termination_conditions(self):
        terminations = dict()
        terminations["pointgoal"] = RearrangeGoal(self._obj_num)
        terminations["timeout"] = Timeout(max_steps=self._termination_config["max_steps"])
        #terminations["max_collision"] = MaxCollision(max_collisions=self._termination_config["max_collisions"])
        terminations["falling"] = Falling(robot_idn=self._robot_idn, fall_height=self._termination_config["fall_height"])
        return terminations

    def _create_reward_functions(self):
        rewards = dict()
        rewards["pointgoal"] = PointGoalReward(
            pointgoal=self._termination_conditions["pointgoal"],
            r_pointgoal=self._reward_config["r_pointgoal"],
        )
        rewards["arrival"] = ArrivalReward(
            obj_num=self._obj_num,
            first_arrival=self._reward_config["first_arrival"]
        )
        # rewards["collision"] = RobotCollisionReward(r_collision=self._reward_config["r_collision"])
        rewards["potential"] = RearrangePotentialReward(r_potential = self._reward_config["r_potential"])
        rewards["grasping"] = GraspingReward(r_grasping = self._reward_config["r_grasping"])
        rewards["living"] = LivingReward(r_living = self._reward_config["r_living"])
        return rewards

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
        floor_polygon = Polygon(np.array(floor_poly))
        objs_name = self.get_all_objects_names(env)
        obstacles_polygons = []
        for obj_name in objs_name:
            obstacles_polygons.append(Polygon(get_obj_bbox(env, obj_name)))

        result = find_free_area_on_floor_random(env, floor_polygon, obstacles_polygons)
        
        # result = ([-5.4344,  1.0000,  4.4591], [-0.70711, 0, 0, 0.70711])
        assert result , "Could not find a free area to place the robot."
        result = ([result.x, 1.0000, result.y], [-0.70711, 0, 0, 0.70711])
        return th.tensor(result[0]), th.tensor(result[1])
    
    def get_floor(self, env):
        all_objects = env.scene._init_objs
        for obj_name, obj in all_objects.items():
            if obj.mesh_type == "Floor":
                return obj
        print("There is no floor.")
        return None
    
    def get_all_objects_names(self, env):
        self.objects = []
        all_objects = env.scene._init_objs
        for obj_name, obj in all_objects.items():
            if "object" in obj_name:
                self.objects.append(obj_name)
        return self.objects

    def get_rearrange_objects_names(self, env):
        self.objects_to_rearrange = []
        all_objects = env.scene._init_objs
        for obj_name, obj in all_objects.items():
            if obj.is_to_rearrange:
                self.objects_to_rearrange.append(obj_name)
        self._obj_num = len(self.objects_to_rearrange)
        return self.objects_to_rearrange

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
                    if "mesh" in name:
                        continue
                    root_link = object_data.get('root_link', {})
                    pos = root_link.get('pos')
                    ori = root_link.get('ori')
                    ### Test
                    #pos, ori = self.get_object_pos_ori(env, object_name)
                    if pos and ori:
                        self.objects_initial_pos_ori[name] = {'pos': pos, 'ori': ori}
            else:
                print("There is no scene_file.")
        pos_ori = self.objects_initial_pos_ori.get(object_name)
        assert pos_ori, f"{object_name} dosen't exist or no scene_file."
        return th.Tensor(pos_ori.get('pos')), th.Tensor(pos_ori.get('ori'))

    def get_target_pos_ori(self, env, object_name):
        if not self.objects_target_pos_ori:
            if env.scene.scene_file is not None:
                target_file_path = env.scene.scene_file.replace("initial","target")
                with open(target_file_path, "r") as f:
                    all_data = json.load(f)
                object_data = all_data['state']['object_registry']
                for name, data in object_data.items():
                    root_link = data.get('root_link', {})
                    pos = root_link.get('pos')
                    ori = root_link.get('ori')
                    if pos and ori:
                        self.objects_target_pos_ori[name] = {'pos': pos, 'ori': ori}
            else:
                print("There is no scene file.")
        pos_ori = self.objects_target_pos_ori.get(object_name)
        assert pos_ori, f"{object_name} dosen't exist or no scene_file."
        return th.Tensor(pos_ori.get('pos')), th.Tensor(pos_ori.get('ori'))
    
    def get_objects_potential(self, env):
        self.objects_initial_potential = {}
        for object_name in self.objects_to_rearrange:
            i_pos, i_ori = self.get_initial_pos_ori(env, object_name)
            t_pos, t_ori = self.get_target_pos_ori(env, object_name)
            self.objects_initial_potential[object_name] = {
                'pos': T.l2_distance(i_pos, t_pos), 
                'ori': T.l2_distance(i_ori, t_ori)
                }
    
    def _get_obs(self, env):
        # No task-specific obs of any kind
        return dict(), dict()
    
    def get_target_polygon(self, env):
        self.objects_target_polygon = {}
        all_objects = env.scene._init_objs
        for obj_name, obj in all_objects.items():
            if obj.is_to_rearrange:
                self.objects_target_polygon[obj_name] = Polygon(np.array(obj.target_bbox))

        
    def get_current_potential(self, env):
        self.objects_current_potential = {}
        for object_name in self.objects_to_rearrange:
            pos, ori = self.get_object_pos_ori(env, object_name)
            t_pos, t_ori = self.get_target_pos_ori(env, object_name)
            ori = T.get_yaw(ori)
            t_ori = T.get_yaw(t_ori)
            diff = th.abs(ori - t_ori)
            dis_ori = diff if diff <= math.pi else 2 * math.pi - diff
            self.objects_current_potential[object_name] = {
                'pos': T.l2_distance(pos, t_pos), 
                'ori': dis_ori
                }
            
    def get_fetch(self, fetch_if, fetch_name):
        self._fetch_if = fetch_if
        if fetch_if == 1:
            self._fetch_name = fetch_name
            self._last_name = None
        if fetch_if == 0:
            self._last_name = self._fetch_name
            self._fetch_name = None

    def get_current_polygon(self, obstacles):
        self.objects_current_polygon = obstacles
        
    def _reset_variables(self, env):
        # Run super first
        super()._reset_variables(env=env)
        # Reset internal variables
        self._path_length = 0.0
        self._current_robot_pos = self._initial_pos
        objects_to_rearrange = self.get_rearrange_objects_names(env)
        self.get_initial_pos_ori(env, objects_to_rearrange[0])
        self.get_objects_potential(env)
        self.get_target_polygon(env)

        self._fetch_if = -1
        self._fetch_name = None
        self._last_name = None
        
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

    def check_target(self, object_name):
        if object_name is not None:
            target = self.objects_target_polygon[object_name]
            current = self.objects_current_polygon[object_name]
            intersection_area = current.intersection(target).area
            target_area = target.area
            if intersection_area / target_area > self._tolerance:
                return True
        return False
    
    def _step_termination(self, env, action, info=None):
        # Run super first
        done, info = super()._step_termination(env=env, action=action, info=info)

        # Add additional info
        info["path_length"] = self._path_length
        return done, info

    def step(self, env, action):
        self.get_current_potential(env)

        # Run super method first
        reward, done, info = super().step(env=env, action=action)
        #print(reward, done)

        # Update other internal variables
        new_robot_pos, _ = env.robots[self._robot_idn].get_position_orientation(frame = "scene")
        self._path_length += T.l2_distance(self._current_robot_pos, new_robot_pos)
        self._current_robot_pos = new_robot_pos
        self._fetch_if = -1

        return reward, done, info
    
    @classproperty
    def valid_scene_types(cls):
        # Any scene works
        return {Scene}

    @classproperty
    def default_termination_config(cls):
        return {
            "max_collisions": 500,
            "max_steps": 500,
            "fall_height": 0.03,
        }

    @classproperty
    def default_reward_config(cls):
        return {
            "r_pointgoal": 10,
            "first_arrival": 1.0,
            "r_potential": 0.5,
            "r_grasping": 0.01,
            "r_living": 0.0
        }
