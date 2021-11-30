import argparse

import numpy

from simulator import Simulator
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3 import DDPG, PPO, TD3
from utils.callbacks import getBestRewardCallback, logDir


def parse_opt():
    parser = argparse.ArgumentParser(description='Select the train model')
    parser.add_argument('--source', type=str, default='./sample/Icecream.csv', help='an ice-cream source path')
    parser.add_argument('--tool', type=str, default='./sample/Scoop.csv', help='a scooping tool path')
    parser.add_argument('--target', type=float, default=40., help='a target for one scoop')
    parser.add_argument('--time-steps', type=int, default=300000, help='total time steps')
    parser.add_argument('--algorithm', type=str, default='DDPG', help='use algorithm (DDPG, PPO, TD3)')
    parser.add_argument('--verbose', action='store_true', default=False, help='verbose the tracing')
    opt = parser.parse_args()
    return opt


def run(source: str,
        tool: str,
        target: float,
        time_steps: int,
        algorithm: str,
        verbose=False):
    env = Simulator(target, source, tool, verbose=verbose)
    env = Monitor(env, logDir(), allow_early_resets=True)
    callback = getBestRewardCallback()
    # the noise objects for DDPG
    actions = env.action_space.shape[-1]
    param_noise = None
    action_noise = OrnsteinUhlenbeckActionNoise(mean=numpy.zeros(actions), sigma=float(0.5) * numpy.ones(actions))
    # construct the model
    if algorithm == 'DDPG':
        from stable_baselines3.ddpg.policies import MlpPolicy
        model = DDPG(MlpPolicy, env, verbose=1, action_noise=action_noise)
        model.learn(total_timesteps=time_steps, log_interval=100, callback=callback)
    elif algorithm == 'PPO':
        from stable_baselines3.ppo.policies import MlpPolicy
        model = PPO(MlpPolicy, env, verbose=1)
        model.learn(total_timesteps=time_steps, log_interval=100, callback=callback)
    elif algorithm == 'TD3':
        from stable_baselines3.td3.policies import MlpPolicy
        model = TD3(MlpPolicy, env, verbose=1)
        model.learn(total_timesteps=time_steps, log_interval=100, callback=callback)
    else:
        raise Exception(f"Not supported {algorithm} algorithm")


if __name__ == "__main__":
    arg = parse_opt()
    run(**vars(arg))
