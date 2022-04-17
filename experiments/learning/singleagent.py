"""Learning script for single agent problems.

Agents are based on `stable_baselines3`'s implementation of A2C, PPO SAC, TD3, DDPG.

Example
-------
To run the script, type in a terminal:

    $ python singleagent.py --env <env> --algo <alg> --obs <ObservationType> --act <ActionType> --cpu <cpu_num>

Notes
-----
Use:

    $ tensorboard --logdir ./results/save-<env>-<algo>-<obs>-<act>-<time-date>/tb/

To check the tensorboard results at:

    http://localhost:6006/

"""
import os
import time
from datetime import datetime
from sys import platform
import argparse
import subprocess
import numpy as np
import gym
import torch
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.cmd_util import make_vec_env # Module cmd_util will be renamed to env_util https://github.com/DLR-RM/stable-baselines3/pull/197
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage
from stable_baselines3.common.utils import set_random_seed
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
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, StopTrainingOnRewardThreshold, EveryNTimesteps, BaseCallback, CallbackList

from gym_pybullet_drones.envs.single_agent_rl.TakeoffAviary import TakeoffAviary
from gym_pybullet_drones.envs.single_agent_rl.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.single_agent_rl.FlyThruGateAviary import FlyThruGateAviary
from gym_pybullet_drones.envs.single_agent_rl.TuneAviary import TuneAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.envs.single_agent_rl.SingleRotorFailure import SingleRotorFailure

import shared_constants

## Custom callback for saving to gcp
class SaveGCPStorageCallback(BaseCallback):
    def __init__(self, foldername, filename, modelDir, verbose=0):
        super(SaveGCPStorageCallback, self).__init__(verbose)
        self.checkpoint_counter = 1
        self.foldername = foldername
        self.filename = filename
        self.modelDir = modelDir
    
    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.

        """
        print("Saving to GCP (callback)")
        gcp_checkpoint_folder = 'checkpoint_' + str(self.checkpoint_counter)

        subprocess.check_call([
        'gsutil', 'cp', '-r', self.filename, os.path.join(self.modelDir,gcp_checkpoint_folder,self.foldername)])

        print("Saved to GCP Storage (callback)")
        print("--------------------------------")
        self.checkpoint_counter = self.checkpoint_counter + 1

        return True
    
    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass

EPISODE_REWARD_THRESHOLD = -0 # Upperbound: rewards are always negative, but non-zero
"""float: Reward threshold to halt the script."""

if __name__ == "__main__":

    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning experiments script')
    parser.add_argument('--env',        default='hover',      type=str,             choices=['takeoff', 'hover', 'flythrugate', 'tune', 'singlerotor'], help='Task (default: hover)', metavar='')
    parser.add_argument('--algo',       default='sac',        type=str,             choices=['a2c', 'ppo', 'sac', 'td3', 'ddpg'],        help='RL agent (default: ppo)', metavar='')
    parser.add_argument('--obs',        default='kin',        type=ObservationType,                                                      help='Observation space (default: kin)', metavar='')
    parser.add_argument('--act',        default='rpm',  type=ActionType,                                                           help='Action space (default: one_d_rpm)', metavar='')
    parser.add_argument('--cpu',        default='1',          type=int,                                                                  help='Number of training environments (default: 1)', metavar='') 
    parser.add_argument('--steps',        default=10000,          type=int,                                                                  help='Number of time steps (default: 10000)', metavar='')       
    # GCP params
    parser.add_argument('--gcp_save_interval',   default=4000,        type=int,       help='Number of timesteps between saves (default: 4000)', metavar='')
    parser.add_argument('--gcp', type=bool, default=False, help='set to True if running on gcp')
    parser.add_argument('--model-dir', default=None, help='The directory to store the model - relevant if gcp is true')
    # Use pretrained model params
    parser.add_argument('--premod', type=bool, default=False, help='set to True if using the pretrained model')
    parser.add_argument('--preloc', type=str, help='The pretrained folder written as pretrained/save-<env>-<algo>-<obs>-<act>-<time_date>', metavar='')
    

    ARGS = parser.parse_args()

    #### Save directory ########################################
    foldername = 'save-'+ARGS.env+'-'+ARGS.algo+'-'+ARGS.obs.value+'-'+ARGS.act.value+'-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
    filename = os.path.dirname(os.path.abspath(__file__))+'/results/'+foldername
    print(filename)
    if not os.path.exists(filename):
        os.makedirs(filename+'/')
    
    #### Print config #################################
    print("=============================")
    print("EXPERIMENT: hover_taskV_5Mt_Sfail_sac_gpu")
    print("-----------------------------")    

    #### Print out current git commit hash #####################
    # Comment for now to work with docker
    # if platform == "linux" or platform == "darwin":
    #     git_commit = subprocess.check_output(["git", "describe", "--tags"]).strip()
    #     with open(filename+'/git_commit.txt', 'w+') as f:
    #         f.write(str(git_commit))

    #### Warning ###############################################
    if ARGS.env == 'tune' and ARGS.act != ActionType.TUN:
        print("\n\n\n[WARNING] TuneAviary is intended for use with ActionType.TUN\n\n\n")
    if ARGS.act == ActionType.ONE_D_RPM or ARGS.act == ActionType.ONE_D_DYN or ARGS.act == ActionType.ONE_D_PID:
        print("\n\n\n[WARNING] Simplified 1D problem for debugging purposes\n\n\n")
    if ARGS.act == ActionType.RPM:
        print("\n\n\n RPM action type\n\n\n")
    #### Errors ################################################
        """
        if not ARGS.env in ['takeoff', 'hover']: 
            print("[ERROR] 1D action space is only compatible with Takeoff and HoverAviary")
            exit()
        """
    if ARGS.act == ActionType.TUN and ARGS.env != 'tune' :
        print("[ERROR] ActionType.TUN is only compatible with TuneAviary")
        exit()
    if ARGS.algo in ['sac', 'td3', 'ddpg'] and ARGS.cpu!=1: 
        print("[ERROR] The selected algorithm does not support multiple environments")
        exit()

    #### Uncomment to debug slurm scripts ######################
    # exit()

    env_name = ARGS.env+"-aviary-v0"
    sa_env_kwargs = dict(aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS, obs=ARGS.obs, act=ARGS.act)
    # train_env = gym.make(env_name, aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS, obs=ARGS.obs, act=ARGS.act) # single environment instead of a vectorized one    
    if env_name == "takeoff-aviary-v0":
        train_env = make_vec_env(TakeoffAviary,
                                 env_kwargs=sa_env_kwargs,
                                 n_envs=ARGS.cpu,
                                 seed=0
                                 )
    if env_name == "hover-aviary-v0":
        train_env = make_vec_env(HoverAviary,
                                 env_kwargs=sa_env_kwargs,
                                 n_envs=ARGS.cpu,
                                 seed=0
                                 )
    if env_name == "singlerotor-aviary-v0":
        train_env = make_vec_env(SingleRotorFailure,
                                 env_kwargs=sa_env_kwargs,
                                 n_envs=ARGS.cpu,
                                 seed=0
                                 )
    if env_name == "flythrugate-aviary-v0":
        train_env = make_vec_env(FlyThruGateAviary,
                                 env_kwargs=sa_env_kwargs,
                                 n_envs=ARGS.cpu,
                                 seed=0
                                 )
    if env_name == "tune-aviary-v0":
        train_env = make_vec_env(TuneAviary,
                                 env_kwargs=sa_env_kwargs,
                                 n_envs=ARGS.cpu,
                                 seed=0
                                 )
    print("[INFO] Action space:", train_env.action_space)
    print("[INFO] Observation space:", train_env.observation_space)
    # check_env(train_env, warn=True, skip_render_check=True)
    
    #### On-policy algorithms ##################################
    onpolicy_kwargs = dict(activation_fn=torch.nn.ReLU,
                           net_arch=[512, 512, dict(vf=[256, 128], pi=[256, 128])]
                           ) # or None
    if ARGS.algo == 'a2c':
        model = A2C(a2cppoMlpPolicy,
                    train_env,
                    policy_kwargs=onpolicy_kwargs,
                    tensorboard_log=filename+'/tb/',
                    verbose=1
                    ) if ARGS.obs == ObservationType.KIN else A2C(a2cppoCnnPolicy,
                                                                  train_env,
                                                                  policy_kwargs=onpolicy_kwargs,
                                                                  tensorboard_log=filename+'/tb/',
                                                                  verbose=1
                                                                  )
    if ARGS.algo == 'ppo':
        model = PPO(a2cppoMlpPolicy,
                    train_env,
                    policy_kwargs=onpolicy_kwargs,
                    tensorboard_log=filename+'/tb/',
                    verbose=1
                    ) if ARGS.obs == ObservationType.KIN else PPO(a2cppoCnnPolicy,
                                                                  train_env,
                                                                  policy_kwargs=onpolicy_kwargs,
                                                                  tensorboard_log=filename+'/tb/',
                                                                  verbose=1
                                                                  )

    #### Off-policy algorithms #################################
    offpolicy_kwargs = dict(activation_fn=torch.nn.ReLU,
                            net_arch=[512, 512, 256, 128]
                            ) # or None # or dict(net_arch=dict(qf=[256, 128, 64, 32], pi=[256, 128, 64, 32]))
    if ARGS.algo == 'sac':
        model = SAC(sacMlpPolicy,
                    train_env,
                    policy_kwargs=offpolicy_kwargs,
                    tensorboard_log=filename+'/tb/',
                    verbose=1
                    ) if ARGS.obs==ObservationType.KIN else SAC(SRFNetwork,
                                                                train_env,
                                                                policy_kwargs=offpolicy_kwargs,
                                                                tensorboard_log=filename+'/tb/',
                                                                verbose=1
                                                                )

        
        #load from a saved model
        if ARGS.premod == True:
            preloc_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), ARGS.preloc)

            if os.path.isfile(preloc_filename+'success_model.zip'):
                preloc_path = preloc_filename+'success_model.zip'
            elif os.path.isfile(preloc_filename+'best_model.zip'):
                preloc_path = preloc_filename+'best_model.zip'
            else:
                print("[ERROR]: no model under the specified path", preloc_filename)
                exit("[ERROR]: Exit: No saved model Found")

            print("[INFO]: Loading saved model from: ", preloc_path)
            model.set_parameters(preloc_path)
            print("[INFO]: SAC Model Loaded")
        else:
            print("[INFO]: Model is being trained from scratch")

    if ARGS.algo == 'td3':
        model = TD3(td3ddpgMlpPolicy,
                    train_env,
                    policy_kwargs=offpolicy_kwargs,
                    tensorboard_log=filename+'/tb/',
                    verbose=1
                    ) if ARGS.obs==ObservationType.KIN else TD3(td3ddpgCnnPolicy,
                                                                train_env,
                                                                policy_kwargs=offpolicy_kwargs,
                                                                tensorboard_log=filename+'/tb/',
                                                                verbose=1
                                                                )
    if ARGS.algo == 'ddpg':
        model = DDPG(td3ddpgMlpPolicy,
                    train_env,
                    policy_kwargs=offpolicy_kwargs,
                    tensorboard_log=filename+'/tb/',
                    verbose=1
                    ) if ARGS.obs==ObservationType.KIN else DDPG(td3ddpgCnnPolicy,
                                                                train_env,
                                                                policy_kwargs=offpolicy_kwargs,
                                                                tensorboard_log=filename+'/tb/',
                                                                verbose=1
                                                                )

    #### Create eveluation environment #########################
    if ARGS.obs == ObservationType.KIN: 
        eval_env = gym.make(env_name,
                            aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                            obs=ARGS.obs,
                            act=ARGS.act
                            )
        print("/n/n OBS (on reset)")
        print(eval_env.reset())
        eval_env.reset
    elif ARGS.obs == ObservationType.RGB:
        if env_name == "takeoff-aviary-v0": 
            eval_env = make_vec_env(TakeoffAviary,
                                    env_kwargs=sa_env_kwargs,
                                    n_envs=1,
                                    seed=0
                                    )
        if env_name == "hover-aviary-v0": 
            eval_env = make_vec_env(HoverAviary,
                                    env_kwargs=sa_env_kwargs,
                                    n_envs=1,
                                    seed=0
                                    )
        if env_name == "singlerotor-aviary-v0":
            train_env = make_vec_env(SingleRotorFailure,
                                     env_kwargs=sa_env_kwargs,
                                     n_envs=ARGS.cpu,
                                     seed=0
                                     )
        if env_name == "flythrugate-aviary-v0": 
            eval_env = make_vec_env(FlyThruGateAviary,
                                    env_kwargs=sa_env_kwargs,
                                    n_envs=1,
                                    seed=0
                                    )
        if env_name == "tune-aviary-v0": 
            eval_env = make_vec_env(TuneAviary,
                                    env_kwargs=sa_env_kwargs,
                                    n_envs=1,
                                    seed=0
                                    )
        eval_env = VecTransposeImage(eval_env)

    #### Train the model #######################################
    
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=EPISODE_REWARD_THRESHOLD,
                                                     verbose=1
                                                     )

    eval_callback = EvalCallback(eval_env,
                                 callback_on_new_best=callback_on_best,
                                 verbose=1,
                                 best_model_save_path=filename+'/',
                                 log_path=filename+'/',
                                 eval_freq=int(2000/ARGS.cpu),
                                 deterministic=True,
                                 render=False
                                 )

    #Save to GCP storage every Ntime steps
    if ARGS.gcp:
        checkpoint_callback = CheckpointCallback(save_freq=1, save_path=filename+'-logs/', name_prefix='rl_model')
        callback_on_gcp_save = SaveGCPStorageCallback(foldername=foldername,filename=filename, modelDir=ARGS.model_dir)        
        event_callback = EveryNTimesteps(n_steps=ARGS.gcp_save_interval, callback=CallbackList([checkpoint_callback, callback_on_gcp_save]))

        model.learn(total_timesteps=ARGS.steps, #int(1e12),
                    callback=CallbackList([eval_callback, event_callback]),
                    log_interval=100,
                    )
    else:
        model.learn(total_timesteps=ARGS.steps, #int(1e12),
                    callback=eval_callback,
                    log_interval=100,
                    )

    #### Save the model ########################################
    model.save(filename+'/success_model.zip')
    print(filename)

    if ARGS.gcp: 
        print("Saving to GCP...")
        subprocess.check_call([
                'gsutil', 'cp', '-r', filename, os.path.join(ARGS.model_dir,foldername)])
        print("saving to GCP... Done")

    #### Print training progression ############################
    with np.load(filename+'/evaluations.npz') as data:
        for j in range(data['timesteps'].shape[0]):
            print(str(data['timesteps'][j])+","+str(data['results'][j][0]))
