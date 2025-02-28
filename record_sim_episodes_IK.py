import time
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import h5py
import hydra 
from hydra.utils import instantiate,call
from omegaconf import DictConfig
os.environ['HYDRA_FULL_ERROR'] = '1'
from tqdm import tqdm
from aloha2.sim_env import make_env
import pandas as pd
from collections import OrderedDict
from aloha2.ik import _get_qpos
from pathlib import Path

# define relative config path
_HERE = Path(__file__).parent
_CONFIG  = str(_HERE / "configs")


# function to deternmine robot action timestep if defined some robotfrequency
def timesteps_for_robot_action(robot_freq,DT):
    return int(1/(DT*robot_freq))



@hydra.main(version_base=None,config_path=_CONFIG,config_name='pick_place')
def record_episodes(cfg: DictConfig):


    dataset_dir = cfg.create_dataset.dataset_dir
    num_episodes = cfg.create_dataset.num_episodes
    instruction = cfg.create_dataset.instruction
    episode_len = cfg.create_dataset.episode_len
    cam_names = cfg.create_dataset.cam_names

    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir,exist_ok=True)
    

    episode_ends = []
    episode=[]


    #define csv to save text instructions
    csv_file = "instructions.csv"
    csv_path = os.path.join(dataset_dir,csv_file)

    for episode_idx in range(num_episodes):

        print("Episode:",episode_idx)
        
        # inbstantiate scripted policy to generate trajectory 
        policy = instantiate(cfg.scripted_policy)

        # instantiate sim env 
        env = instantiate(cfg.env)

        # initialize episode 
        ts = env.reset()
        episode.append(ts)
        


        robot_action_ts = timesteps_for_robot_action(cfg.create_dataset.robot_freq,DT= cfg.env.DT)
        max_sim_time = episode_len * robot_action_ts
        for step in tqdm(range(max_sim_time),desc='ts'):
            if step%robot_action_ts == 0:
                action = policy(ts)
                ik_action = _get_qpos(physics=env.physics,target_mocap_pose=action)

                ts = env.step(ik_action)
                episode.append(ts)
            else:
                ts = env.step(ik_action)
        

        # create a combined zarr dataset composed of all epidoes 

        episode_ends.append((episode_len-1)*(episode_idx+1))

        #get rewards and max reward for episode
        episode_return = np.sum([ts.reward for ts in episode[1:]])
        episode_max_reward = np.max([ts.reward for ts in episode[1:]])
        print('episode max reward:', episode_max_reward)
        is_success = episode_max_reward == env.task.max_reward
        if is_success:
            print(f"{episode_idx=} Successful, {episode_return=}")
        else:
            print(f"{episode_idx=} Failed")

    
        
        actions = [np.concatenate((ts.observation['mocap_pose_left'],[ts.observation['gripper_ctrl'][0]],
                                   ts.observation['mocap_pose_right'],[ts.observation['gripper_ctrl'][1]])) for ts in episode]
    
        # actions = np.concatenate((ee_pose_traj,gripper_ctrl_traj),axis=1)

        
        data_dict = {
            'observations':{'qpos': [],
                            'qvel': [],
                            'images':{}},
            'action': [],
            'stats': {}
        }
        
        episode = episode[:-1]
        actions = actions[:-1]


        for cam in cam_names:
            data_dict['observations']['images'][cam] = []



        max_timesteps = len(actions)
        while actions:
            action = actions.pop(0)
            ts = episode.pop(0)
            data_dict['observations']['qpos'].append(ts.observation['qpos'])
            data_dict['observations']['qvel'].append(ts.observation['qvel'])
            data_dict['action'].append(action)
            for cam_name in cam_names:
                data_dict['observations']['images'][cam_name].append(ts.observation['images'][cam_name])
    
            # HDF5
        t0 = time.time()




        #TODO get stats
        data_dict['stats']= {'qpos':{'min': np.min(data_dict['observations']['qpos'],axis=0),
                                    'max': np.max(data_dict['observations']['qpos'],axis=0)},

                                    'action': {'min': np.min(data_dict['action'],axis=0),
                                    'max': np.max(data_dict['action'],axis=0)}}

        





        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx+200}')
        with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root = create_datagroups(data_dict=data_dict,root=root)
        print(f'Saving: {time.time() - t0:.1f} secs\n')
    
        # create/ load csv and save data
        csv_data = {'instructions': instruction,
                    'is_success': is_success}

        if os.path.isfile(csv_path):
            # Load existing CSV file
            df = pd.read_csv(csv_path)
            df = pd.concat([df, pd.DataFrame([csv_data])], ignore_index=True)
        else:
            # Create new DataFrame if file doesn't exist
            df = pd.DataFrame([csv_data])

        # Save back to CSV
        df.to_csv(csv_path, index=False)
        
        print(f'Saved hdf5 to {dataset_dir}')

    



# function to recurseively make datagroups based on data dect
def create_datagroups(data_dict,root):
    if isinstance(data_dict,dict):
        for k,v in data_dict.items():
            if isinstance(v,dict):
                k = root.create_group(k)
                create_datagroups(v,k)
            else:
                k = root.create_dataset(k,data=v)

    return root
 


if __name__=='__main__':

    

  record_episodes()
