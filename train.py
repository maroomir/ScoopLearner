import numpy

from simulator import Simulator
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.ddpg.policies import MlpPolicy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3 import DDPG
from utils.callbacks import getBestRewardCallback, logDir

if __name__ == "__main__":
    target = 40.
    source = './sample/Icecream.csv'
    tool = './sample/Scoop.csv'
    env = Simulator(target, source, tool, verbose=True)
    env = Monitor(env, logDir(), allow_early_resets=True)
    callback = getBestRewardCallback()
    # the noise objects for DDPG
    actions = env.action_space.shape[-1]
    param_noise = None
    action_noise = OrnsteinUhlenbeckActionNoise(mean=numpy.zeros(actions), sigma=float(0.5) * numpy.ones(actions))
    # construct the model
    model = DDPG(MlpPolicy, env, verbose=1, action_noise=action_noise)
    model.learn(total_timesteps=50000, log_interval=100, callback=callback)
