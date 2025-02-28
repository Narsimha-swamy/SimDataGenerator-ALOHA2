## OVERVIEW
This repo is for aloha2 simulation data collection and will be later upgraded to include real2sim teleoperation. 

## Replicate:
1. $ pip install -r requirements.txt
2. $ python record_sim_episodes_IK.py --config-name pick_place

## Custom Dataset Generation
For custom task and dataset generation. Write new task specific trajectory generation policy in aloha2/scripted_policy.py. Create task with appropriate rewards in aloha2/sim_env.py . 