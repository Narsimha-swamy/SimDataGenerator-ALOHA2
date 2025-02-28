## OVERVIEW
This repository is designed for collecting simulation data for Aloha2, with plans for future integration of Real2Sim teleoperation. Utilize this repo to generate scripted data for tasks such as pick-and-place and multi-block pick-and-place operations. Thanks to the Hydra-based configuration system, you can easily define all task-related parameters in a YAML file. These parameters can be dynamically updated within the Python script or during command-line initialization of the record_sim_episodes script. Currently, the record_episode script captures end-effector poses in XYZ + Quaternions as actions, with the flexibility to switch to storing robot joint angles based on your specific use case.

## Replicate:
1. $ pip install -r requirements.txt
2. $ python record_sim_episodes_IK.py --config-name pick_place

## Custom Dataset Generation
For custom task and dataset generation.
1. Create a new task specific XML file in aloha2/xml dir.
2. Write new task specific trajectory generation policy in aloha2/scripted_policy.py. 
3. Create task with appropriate rewards in aloha2/sim_env.py . 
4. Make a config file that reflects all the changes and use task-related paramters.

## Useful context
total camera views avalaible in mujoco aloha2 scene are overhead_cam, worms_eye_cam, wrist_cam_right, wrist_cam_left, teleoperator_pov, collaborator_pov. These cam views can be defined in the config to collect data from these views. dt is the simulation timestep, robot_freq is the robot frequency, and data is collected from the timesteps robot performs actions. 