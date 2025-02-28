import numpy as np
import os
import collections
import matplotlib.pyplot as plt
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from pyquaternion import Quaternion
from .constants import PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN
from .constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN
from .constants import PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN
from .constants import PUPPET_GRIPPER_POSITION_CLOSE,START_ARM_POSE
from pathlib import Path
import IPython
e = IPython.embed

BOX_POSE = [None] # to be changed from outside

_HERE = Path(__file__).parent

_XML = _HERE / "xml"


"""
Environment for simulated robot bi-manual manipulation, with joint position control
Action space:      [left_arm_qpos (6),             # absolute joint position
                    left_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
                    right_arm_qpos (6),            # absolute joint position
                    right_gripper_positions (1),]  # normalized gripper position (0: close, 1: open)

Observation space: {"qpos": Concat[ left_arm_qpos (6),         # absolute joint position
                                    left_gripper_position (1),  # normalized gripper position (0: close, 1: open)
                                    right_arm_qpos (6),         # absolute joint position
                                    right_gripper_qpos (1)]     # normalized gripper position (0: close, 1: open)
                    "qvel": Concat[ left_arm_qvel (6),         # absolute joint velocity (rad)
                                    left_gripper_velocity (1),  # normalized gripper velocity (pos: opening, neg: closing)
                                    right_arm_qvel (6),         # absolute joint velocity (rad)
                                    right_gripper_qvel (1)]     # normalized gripper velocity (pos: opening, neg: closing)
                    "images": {"main": (480x640x3)}        # h, w, c, dtype='uint8'
"""
def make_env(xml_name,DT,task):
    xml_path = os.path.join(_XML,xml_name)
    physics=mujoco.Physics.from_xml_path(xml_path)  
    return control.Environment(physics,task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)


def sample_box_pose(seed = None):
    x_range = [0.0, 0.2]
    y_range = [-0.1, 0.1]
    z_range = [0.02, 0.02]

    rng = np.random.RandomState(seed)
    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = rng.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_multiple_box_pose(cubes:list,seed = None):
    
    # ranges of bounding box
    x_range = [-0.03, 0.27]
    y_range = [-0.28, 0.28]
    z_range = [0.02, 0.02]

    # get random positions based on seed
    rng = np.random.RandomState(seed)
    ranges = np.vstack([x_range, y_range, z_range])
    
    # set positions for each cube 
    cube_quat = np.array([1, 0, 0, 0])
    cube_positions ={}
    for cube in cubes:
        cube_positions[cube] = rng.uniform(ranges[:, 0], ranges[:, 1])
        cube_positions[cube] = np.concatenate([cube_positions[cube],cube_quat ])
        
    # print(cube_positions)
    return cube_positions

class BimanualViperXTask(base.Task):
    def __init__(self, cubes,task,cam_names,image_dim,seed,random=None):
        '''
        base task class is inherited for all tasks 
        Args: 
            cubes: list of all colored cubes in workspace
            cam_names: list of all cameras you want use
            image_dim = image size to save
            seed: seed for randomness generator
            task: brief task description 
        '''
        
        super().__init__(random=random)
        self.seed = seed
        self.cam_names = cam_names
        self.image_dim = image_dim
        self.cubes = cubes
        self.task = task
    def before_step(self, action, physics):

        left_arm_action,right_arm_action = np.split(action,2)

        normalized_left_gripper_action = left_arm_action[-1]
        normalized_right_gripper_action = right_arm_action[-1]

        left_gripper_action = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(normalized_left_gripper_action)
        right_gripper_action = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(normalized_right_gripper_action)

        env_action = np.concatenate([left_arm_action[:-1], [left_gripper_action], right_arm_action[:-1], [right_gripper_action]])
        super().before_step(env_action, physics)
        return

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        super().initialize_episode(physics)

    def initialize_robots(self, physics):
        # reset joint position
        physics.data.qpos[:16] = START_ARM_POSE

        # test xpos of mocap pos
        #0.18753877 -0.019       0.32524417
        # np.copyto(physics.data.mocap_pos[0], [-0.31718881, -0.019, 0.31525084])
        # np.copyto(physics.data.mocap_quat[0], [1, 0, 0, 0])
        # # right
        # np.copyto(physics.data.mocap_pos[1], [0.31718881, -0.019, 0.31525084])
        # np.copyto(physics.data.mocap_quat[1],  [1, 0, 0, 0])

        np.copyto(physics.data.mocap_pos[0], [-0.18753877, -0.019, 0.32524417])
        np.copyto(physics.data.mocap_quat[0], [1, 0, 0, 0])
        # right
        right_quat = Quaternion([1.0,0.0,0.0,0.0]) * Quaternion(axis=[0,0,1],degrees=180)
        np.copyto(physics.data.mocap_pos[1], [0.18753877, -0.019, 0.32524417])
        np.copyto(physics.data.mocap_quat[1],  right_quat.elements)

        # reset gripper control

        action = np.concatenate([START_ARM_POSE[:6],[PUPPET_GRIPPER_POSITION_CLOSE],
                                 START_ARM_POSE[8:8+6],[PUPPET_GRIPPER_POSITION_CLOSE]])

       
        np.copyto(physics.data.ctrl, action)


    @staticmethod
    def get_qpos(physics):
        qpos_raw = physics.data.qpos.copy()
        left_qpos_raw = qpos_raw[:8]
        right_qpos_raw = qpos_raw[8:16]
        left_arm_qpos = left_qpos_raw[:6]
        right_arm_qpos = right_qpos_raw[:6]
        left_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(left_qpos_raw[6])]
        right_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(right_qpos_raw[6])]
        return np.concatenate([left_arm_qpos, left_gripper_qpos, right_arm_qpos, right_gripper_qpos])

    @staticmethod
    def get_qvel(physics):
        qvel_raw = physics.data.qvel.copy()
        left_qvel_raw = qvel_raw[:8]
        right_qvel_raw = qvel_raw[8:16]
        left_arm_qvel = left_qvel_raw[:6]
        right_arm_qvel = right_qvel_raw[:6]
        left_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(left_qvel_raw[6])]
        right_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(right_qvel_raw[6])]
        return np.concatenate([left_arm_qvel, left_gripper_qvel, right_arm_qvel, right_gripper_qvel])

    @staticmethod
    def get_env_state(physics):
        raise NotImplementedError

    def get_observation(self, physics):
        # note: it is important to do .copy()
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos(physics)
        obs['qvel'] = self.get_qvel(physics)
        obs['env_state'] = self.get_env_state(physics,self.cubes)
        obs['images'] = dict()
        
        for cam in self.cam_names:
            obs['images'][cam] = physics.render(height=self.image_dim[0], width=self.image_dim[1], camera_id=cam)

        # used in scripted policy to obtain starting pose
        obs['mocap_pose_left'] = np.concatenate([physics.data.mocap_pos[0], physics.data.mocap_quat[0]]).copy()
        obs['mocap_pose_right'] = np.concatenate([physics.data.mocap_pos[1], physics.data.mocap_quat[1]]).copy()

        # used when replaying joint trajectory
        ctrl = physics.data.ctrl.copy()
        left_ctrl,right_ctrl = np.split(ctrl,2)

        # left_ctrl = ctrl[:len(ctrl)//2]
        # right_ctrl = ctrl[len(ctrl)//2:]

        obs['gripper_ctrl'] = [left_ctrl[-1],right_ctrl[-1]]
        return obs

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        raise NotImplementedError



class PickPlaceTask(BimanualViperXTask):
    def __init__(self,seed,cam_names,image_dim,random=None):
        super().__init__(random=random,seed=seed,cam_names=cam_names,image_dim=image_dim)
        self.max_reward = 3
    
    def initialize_episode(self, physics):
        self.initialize_robots(physics)
        # cube_pose = [0,0.5,0.05,1,0,0,0]

        cube_pose = sample_box_pose(seed=self.seed)

        box_start_idx = physics.model.name2id('red_box_joint', 'joint')
        np.copyto(physics.data.qpos[box_start_idx : box_start_idx + 7], cube_pose)
        # print(f"randomized cube position to {cube_position}")

        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state ={}
        cubes_qpos_data = physics.data.qpos.copy()[16:]
        # for i,cube in enumerate(cubes):
        #     env_state[cube] = cubes_qpos_data[:(i+1)*7]
        return cubes_qpos_data


    def get_reward(self, physics):
        # return whether left gripper is holding the box
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_right_right_gripper_finger = ("red_box", "right_right_finger_coll") in all_contact_pairs
        touch_right_left_gripper_finger = ("red_box", "right_left_finger_coll") in all_contact_pairs

        touch_table = ("red_box", "table") in all_contact_pairs
        touch_plate = ("red_box","plate") in all_contact_pairs
        reward = 0
        if touch_right_left_gripper_finger and touch_right_right_gripper_finger:
            reward = 1
        if touch_right_left_gripper_finger and touch_right_right_gripper_finger and not touch_table: # lifted
            reward = 2
        if touch_plate:  # cube placed on plate 
            reward = 3
        return reward
    

class MultiBlockPickPlaceTask(BimanualViperXTask):
    def __init__(self, cubes,cam_names,task,seed,image_dim,random=None):
        super().__init__(random=random,task= task,seed=seed,cubes=cubes,cam_names=cam_names,image_dim=image_dim)
        self.max_reward = 3

    def initialize_episode(self, physics):
        self.initialize_robots(physics)
        # cube_pose = [0,0.5,0.05,1,0,0,0]

        cube_pose = sample_multiple_box_pose(seed=self.seed,cubes=self.cubes)
        # print('cube pose',cube_pose)
        id2index = lambda j_id: 16 + (j_id - 16) * 7 # first 16 is robot qpos, 7 is pose dim # hacky

        for cube in self.cubes:
            box_start_id = physics.model.name2id(f'{cube}_box_joint', 'joint')
            # print(f'{cube} id: {box_start_id}')
            box_start_idx = id2index(box_start_id)
            np.copyto(physics.data.qpos[box_start_idx : box_start_idx + 7], cube_pose[cube])
            # print(f"randomized cube position to {cube_pose[cube]}")
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics,cubes):
        
        env_state={}
        cube_poses = physics.data.qpos.copy()[16:]
        for i,cube in enumerate(cubes):
            env_state[cube] = cube_poses[i*7:(i+1)*7]
        return env_state


    def get_reward(self, physics):
        # return whether left gripper is holding the box
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        if "red" in self.task:
            touch_right_right_gripper_finger = ("red_box", "right_right_finger_coll") in all_contact_pairs
            touch_right_left_gripper_finger = ("red_box", "right_left_finger_coll") in all_contact_pairs
            touch_table = ("red_box", "table") in all_contact_pairs
            touch_plate = ("red_box","plate") in all_contact_pairs
        elif "blue" in self.task:
            touch_right_right_gripper_finger = ("blue_box", "right_right_finger_coll") in all_contact_pairs
            touch_right_left_gripper_finger = ("blue_box", "right_left_finger_coll") in all_contact_pairs
            touch_table = ("blue_box", "table") in all_contact_pairs
            touch_plate = ("blue_box","plate") in all_contact_pairs
        elif "yellow" in self.task:
            touch_right_right_gripper_finger = ("yellow_box", "right_right_finger_coll") in all_contact_pairs
            touch_right_left_gripper_finger = ("yellow_box", "right_left_finger_coll") in all_contact_pairs
            touch_table = ("yellow_box", "table") in all_contact_pairs
            touch_plate = ("yellow_box","plate") in all_contact_pairs

        reward = 0
        if touch_right_right_gripper_finger and touch_right_left_gripper_finger:
            reward = 1
        if touch_right_right_gripper_finger and touch_right_left_gripper_finger and not touch_table: # lifted
            reward = 2
        if touch_plate:  # cube placed on plate 
            reward = 3
        return reward