"""Script demonstrating the use of `gym_pybullet_drones`' Gym interface.

Class HoverAviary is used as a learning env for the A2C and PPO algorithms.

Example
-------
In a terminal, run as:

    $ python learn.py

Notes
-----
The boolean argument --rllib switches between `stable-baselines3` and `ray[rllib]`.
This is a minimal working example integrating `gym-pybullet-drones` with
reinforcement learning libraries `stable-baselines3` and `ray[rllib]`.
It is not meant as a good/effective learning example.

"""
import time
import argparse
import os
import subprocess
import numpy as np
import ray
import torch
from ray.tune import register_env
from ray.rllib.agents import sac
from os.path import exists

# from SRFNetwork import SRFNetwork
from ray.rllib.models import ModelCatalog

# from envs.SingleRotorFailure.SingleRotorFailureHover import SingleRotorFailureHover, ChangeObs
from gym_pybullet_drones.envs.single_agent_rl.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

MODEL_FILE_NAME = 'torch.model'

if __name__ == "__main__":
    print("hover_learn_gcp_sac | Now running....")
    print(f"Cuda available: {torch.cuda.is_available()}")

    ##Nikhil: python hover_learn_gcp_sac.py --episodes 1

    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script using HoverAviary')
    parser.add_argument('--model-dir', default=None, help='The directory to store the model - relevant if gcp is true')
    parser.add_argument('--episodes', default=5, type=int, help='Number of episodes to train (default: 5)', metavar='')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workes to use (default: 1)', metavar='')
    parser.add_argument('--num_cpus_per_worker', default=1, type=int, help='Number of workes to use (default: 1)',
                        metavar='')
    parser.add_argument('--checkpoint_interval', default=100, type=int,
                        help='Number of episodes between checkpoint saves (default: 100)', metavar='')
    parser.add_argument('--gcp_save_interval', default=1000, type=int,
                        help='Number of episodes between checkpoint saves (default: 100)', metavar='')
    parser.add_argument('--gcp', type=bool, default=False, help='set to True if running on gcp')
    ARGS = parser.parse_args()

    ## Print config
    print("=============================")
    print("EXPERIMENT: hover_10sec_50kiter_bignegative")
    print("-----------------------------")
    env = HoverAviary()
    # print("Observation Space")
    # print(env.printObsSpace())
    # print("-------------------")
    # print("Observations (on reset)")
    print(env.reset())
    env.close()

    #### Train the model #######################################
    ray.shutdown()
    ray.init(ignore_reinit_error=True)
    register_env("hover-aviary-v0", lambda _: HoverAviary())
    config = sac.DEFAULT_CONFIG.copy()  # default config for PPO: https://docs.ray.io/en/latest/rllib-algorithms.html?highlight=DEFAULT_CONFIG#policy-gradients

    # From baseline paper
    # ModelCatalog.register_custom_model("SRFNetwork", SRFNetwork)
    # config["Q_model"]["fcnet_hiddens"] = [256, 256, 256]
    # config["policy_model"]["custom_model"] = "SRFNetwork"
    config["num_workers"] = ARGS.num_workers
    config["num_cpus_per_worker"] = ARGS.num_cpus_per_worker
    config["num_gpus"] = 0.2

    config["framework"] = "torch"
    config["env"] = 'hover-aviary-v0'
    agent = sac.SACTrainer(config)
    print("Config")
    print(config)
    print("--------------------")

    ##Nikhil: Modify this to reflect the location of your checkpoint file
    # checkpoint_location_local = "/root/Python_ws/results/SACTrainer_single-rotor-failure-hover-v0_2022-03-09_00-55-543q78yabn/checkpoint_050000/checkpoint-50000"
    # checkpoint_location_local = '/Users/nikhilbadami/Spring 2022/Deep Reinforcement Learning/checkpoints/final_checkpoint_10s_5kiterations/checkpoint-4501'

    """
    if exists(checkpoint_location_local):
        agent.restore(checkpoint_location_local)
        print("Agent restored from checkpoint.")
    """

    for i in range(ARGS.episodes):
        results = agent.train()
        print("[INFO] {:d}: episode_reward max {:f} min {:f} mean {:f}".format(i,
                                                                               results["episode_reward_max"],
                                                                               results["episode_reward_min"],
                                                                               results["episode_reward_mean"]
                                                                               ))
        if i % ARGS.checkpoint_interval == 0:
            checkpoint_path = agent.save()
            print("Check point path: ", checkpoint_path)

        # save to gcp storage at certain intervals for ease of use
        if i % ARGS.gcp_save_interval == 0:
            if ARGS.gcp and ARGS.model_dir:
                folder_temp = 'backup_' + str(i + 1)
                dir_str = os.path.join('/root/ray_results_backup', folder_temp)
                os.makedirs(dir_str)
                checkpoint_path = agent.save(dir_str)
                tmp_model_folder = os.path.normpath(dir_str)
                subprocess.check_call([
                    'gsutil', 'cp', '-r', tmp_model_folder, os.path.join(ARGS.model_dir, folder_temp)])

                print("Saved to gcp cloud at i = ", i)
                print("--------------------------------")

    checkpoint_path = agent.save()
    print("Final check point path: ", checkpoint_path)
    agent.stop()

    if ARGS.gcp:
        ray.shutdown()
        if ARGS.model_dir:
            tmp_model_folder = os.path.normpath('/root/ray_results/')
            subprocess.check_call([
                'gsutil', 'cp', '-r', tmp_model_folder, ARGS.model_dir])
    else:
        policy = agent.get_policy()
        print(policy)
        ray.shutdown()
        #### Show (and record a video of) the model's performance ##
        env = HoverAviary()
        logger = Logger(logging_freq_hz=int(env.SIM_FREQ / env.AGGR_PHY_STEPS),
                        num_drones=1
                        )
        obs = env.reset()
        """
        logger.log(drone=0,
                   timestamp=0 / env.SIM_FREQ,
                   # state=np.hstack([obs[0:3], np.zeros(4), obs[3:15],  np.resize(action, (4))]),
                   # RED Edits
                   # state=np.hstack([obs[0:3], np.zeros(4), np.zeros(3), obs[7:13], np.resize(action, (4))]),
                   state=np.hstack([obs[0:3], np.zeros(4), np.zeros(3), obs[3:9], obs[9:13]]),
                   # control=np.zeros(12)
                   )
        """
        start = time.time()
        for i in range(30 * env.SIM_FREQ):
            action, _states, _dict = policy.compute_single_action(obs)
            obs, reward, done, info = env.step(action)

            """
            logger.log(drone=0,
                       timestamp=i / env.SIM_FREQ,
                       # state=np.hstack([obs[0:3], np.zeros(4), obs[3:15],  np.resize(action, (4))]),
                       # RED Edits
                       # state=np.hstack([obs[0:3], np.zeros(4), np.zeros(3), obs[7:13], np.resize(action, (4))]),
                       state=np.hstack([obs[0:3], np.zeros(4), env.get_rpy(), obs[3:9], obs[9:13]]),
                       # control=np.zeros(12)
                       )
            """
            if i % env.SIM_FREQ == 0:
                env.render()
                print("rendering")
                print(done)
            sync(i, start, env.TIMESTEP)
            if done:
                obs = env.reset()
        env.close()
        logger.plot()


