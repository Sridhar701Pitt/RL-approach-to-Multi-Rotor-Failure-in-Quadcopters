"""Test script for single agent problems.

This scripts runs the best model found by one of the executions of `singleagent.py`

Example
-------
To run the script, type in a terminal:

    $ python test_singleagent.py --exp ./results/save-<env>-<algo>-<obs>-<act>-<time_date>

"""
import os
import time
from datetime import datetime
import argparse
import re
from turtle import goto
import numpy as np
import gym
import torch
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import A2C
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3 import TD3
from stable_baselines3 import DDPG
from stable_baselines3.common.policies import ActorCriticPolicy as a2cppoMlpPolicy
from stable_baselines3.common.policies import ActorCriticCnnPolicy as a2cppoCnnPolicy
from stable_baselines3.sac.policies import SACPolicy as sacMlpPolicy
from stable_baselines3.sac import CnnPolicy as sacCnnPolicy
from stable_baselines3.td3 import MlpPolicy as td3ddpgMlpPolicy
from stable_baselines3.td3 import CnnPolicy as td3ddpgCnnPolicy
from stable_baselines3.common.evaluation import evaluate_policy

from gym_pybullet_drones.utils.utils import sync
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.single_agent_rl.TakeoffAviary import TakeoffAviary
from gym_pybullet_drones.envs.single_agent_rl.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.single_agent_rl.FlyThruGateAviary import FlyThruGateAviary
from gym_pybullet_drones.envs.single_agent_rl.TuneAviary import TuneAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType

#plot
import matplotlib.pyplot as plt

import shared_constants

if __name__ == "__main__":

    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script using TakeoffAviary')
    parser.add_argument('--exp',                           type=str,            help='The experiment folder written as ./results/save-<env>-<algo>-<obs>-<act>-<time_date>', metavar='')
        # Use predefined goal path
    parser.add_argument('--goalpath', type=bool, default=False, help='set to True if using the predefined goal path')
    ARGS = parser.parse_args()

    #### Load the model from file ##############################
    algo = ARGS.exp.split("-")[2]

    if os.path.isfile(ARGS.exp+'/success_model.zip'):
        path = ARGS.exp+'/success_model.zip'
    elif os.path.isfile(ARGS.exp+'/best_model.zip'):
        path = ARGS.exp+'/best_model.zip'
    else:
        print("[ERROR]: no model under the specified path", ARGS.exp)
    if algo == 'a2c':
        model = A2C.load(path)
    if algo == 'ppo':
        model = PPO.load(path)
    if algo == 'sac':
        model = SAC.load(path)
    if algo == 'td3':
        model = TD3.load(path)
    if algo == 'ddpg':
        model = DDPG.load(path)

    #### Parameters to recreate the environment ################
    env_name = ARGS.exp.split("-")[1]+"-aviary-v0"
    OBS = ObservationType.KIN if ARGS.exp.split("-")[3] == 'kin' else ObservationType.RGB
    if ARGS.exp.split("-")[4] == 'rpm':
        ACT = ActionType.RPM
    elif ARGS.exp.split("-")[4] == 'dyn':
        ACT = ActionType.DYN
    elif ARGS.exp.split("-")[4] == 'pid':
        ACT = ActionType.PID
    elif ARGS.exp.split("-")[4] == 'vel':
        ACT = ActionType.VEL
    elif ARGS.exp.split("-")[4] == 'tun':
        ACT = ActionType.TUN
    elif ARGS.exp.split("-")[4] == 'one_d_rpm':
        ACT = ActionType.ONE_D_RPM
    elif ARGS.exp.split("-")[4] == 'one_d_dyn':
        ACT = ActionType.ONE_D_DYN
    elif ARGS.exp.split("-")[4] == 'one_d_pid':
        ACT = ActionType.ONE_D_PID

    #### Evaluate the model ####################################
    eval_env = gym.make(env_name,
                        aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                        obs=OBS,
                        act=ACT
                        )
    mean_reward, std_reward = evaluate_policy(model,
                                              eval_env,
                                              n_eval_episodes=10
                                              )
    print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")

    save_it = False
    record_it = True
    plot_it = True

    #### Show, record a video, and log the model's performance #
    test_env = gym.make(env_name,
                        gui=True,
                        record=record_it,
                        aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                        obs=OBS,
                        act=ACT,
                        dogoalpath=True if ARGS.goalpath else False
                        )

    #### Create a goal path for test env
    if ARGS.goalpath:
        total_sec = 25
    else:
        total_sec = 15
    
    steps_per_sec = int(test_env.SIM_FREQ/test_env.AGGR_PHY_STEPS)
    total_steps = total_sec*int(test_env.SIM_FREQ/test_env.AGGR_PHY_STEPS)
    if ARGS.goalpath:
        goal_path = []
        # Square path
        steps_2_5 = np.floor(2.5*steps_per_sec)
        steps_5 = np.floor(5*steps_per_sec)
        corner_timestamp = np.floor(steps_per_sec*np.array([0, 2.5, 2.5+5, 2.5+10, 2.5+15, 2.5+20, 25]))
        corners = 2.0*np.array([[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,1.0,0.0],[1.0,1.0,0.0],[1.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]])
        for i in range(total_steps):
            goal_x = np.interp(i,corner_timestamp,corners[:,0])
            goal_y = np.interp(i,corner_timestamp,corners[:,1])
            goal_z = np.interp(i,corner_timestamp,corners[:,2])
            goal_path.append(np.array([goal_x,goal_y,goal_z]))
        #Send goal to script
        test_env.injectGoals(goal_path)


    logger = Logger(logging_freq_hz=int(test_env.SIM_FREQ/test_env.AGGR_PHY_STEPS),
                    num_drones=1
                    )
    obs = test_env.reset()
    print("OBS (on reset): ", obs)
    start = time.time()

    #reward curve
    # print("steps per second: ", int(test_env.SIM_FREQ/test_env.AGGR_PHY_STEPS))
    rewardY = np.zeros(total_steps)
    for i in range(total_steps): # Up to 6''
        action, _states = model.predict(obs,
                                        deterministic=True # OPTIONAL 'deterministic=False'
                                        )
        obs, rewardY[i], done, info = test_env.step(action)

        test_env.render()
        if OBS==ObservationType.KIN:
            logger.log(drone=0,
                       timestamp=i/test_env.SIM_FREQ,
                       state= np.hstack([obs[0:3], np.zeros(4), obs[3:15],  np.resize(action, (4))]),
                       control=np.zeros(12)
                       )
        sync(np.floor(i*test_env.AGGR_PHY_STEPS), start, test_env.TIMESTEP)
        if ARGS.goalpath == False:
            if done: obs = test_env.reset() # OPTIONAL EPISODE HALT
    
    # Get drone path for plotting/reporting purposes
    drone_path = test_env.getDronePath()

    if not ARGS.goalpath:
        goal_points = test_env.getGoalPoints()
    
    test_env.close()
    # logger.save_as_csv("sa") # Optional CSV save
    log_timestamps, log_states, log_controls = logger.getVariables() # For data storage

    if plot_it:
        logger.plot()
        if ARGS.goalpath:
            plt.savefig('states_square.png',bbox_inches='tight')
        else:
            plt.savefig('states_random.png',bbox_inches='tight')
        plt.show()
        


        plt.plot(rewardY)
        if ARGS.goalpath:
            plt.savefig('reward_square.png',bbox_inches='tight')
        else:
            plt.savefig('reward_random.png',bbox_inches='tight')    
        plt.show()

        if ARGS.goalpath:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            drone_path_np = np.stack(drone_path, axis=0)
            goal_path_np = np.stack(goal_path, axis=0)
            ax.plot(drone_path_np[:,0],drone_path_np[:,1],drone_path_np[:,2])
            ax.plot(goal_path_np[:,0],goal_path_np[:,1],goal_path_np[:,2])
            ax.scatter(0.0,0.0,0.0,color='C1')

            plt.title("Drone trajectories (red) along the square path (green)")

            plt.savefig('traj_square.png',bbox_inches='tight')
            plt.show()
        else:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            drone_path_np = np.stack(drone_path, axis=0)
            goal_path_np = np.stack(goal_points, axis=0)
            ax.plot(drone_path_np[:,0],drone_path_np[:,1],drone_path_np[:,2])
            ax.scatter(goal_path_np[:,0],goal_path_np[:,1],goal_path_np[:,2])
            ax.scatter(0.0,0.0,0.0,color='C1')
            for goal_point in goal_path_np:
                ax.plot(np.array([0.0,goal_point[0]]),np.array([0.0,goal_point[1]]),np.array([0.0,goal_point[2]]),color='C2',linestyle='solid',linewidth=1.5)
            
            plt.title("Drone trajectories (red path) to origin (green dot)")
            
            plt.savefig('traj_random.png',bbox_inches='tight')  
            plt.show()

    

    if save_it:

        # Save Data as numpy arrays
        file_name = ARGS.exp.split("/")[-1]
        task_name = "hover_task_XI_5Mt_DFailAdj_sac_gpu_20220418_194504" ## TO be manually inputted
        checkpoint_name = "final" ## Manual input
        best_success = "success" ##Manual input best or success model used

        npz_title = task_name+ '_' + checkpoint_name + '_' + best_success

        print("Saving data to npz....")
        if ARGS.goalpath:
            np.savez(npz_title + "_data_Sq",
                    task_name=npz_title,
                    file_name=file_name,
                    notes="Custom goal (square)", 
                    total_sec=total_sec, 
                    steps_per_sec=steps_per_sec, 
                    reward_curve=rewardY, 
                    drone_path=drone_path_np,
                    goal_path=goal_path_np,
                    log_timestamps=log_timestamps,
                    log_states=log_states,
                    log_controls=log_controls
                    )
        else:
            np.savez(npz_title + "_data_Rd",
                    task_name=npz_title,
                    file_name=file_name,
                    notes="Random goal points", 
                    total_sec=total_sec, 
                    steps_per_sec=steps_per_sec, 
                    reward_curve=rewardY, 
                    drone_path=drone_path_np,
                    goal_points=goal_path_np,
                    log_timestamps=log_timestamps,
                    log_states=log_states,
                    log_controls=log_controls
                    )
        print("Data saved to: ", npz_title)

    # with np.load(ARGS.exp+'/evaluations.npz') as data:
    #     print(data.files)
    #     print(data['timesteps'])
    #     print(data['results'])
    #     print(data['ep_lengths'])
