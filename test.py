import re
from glob import glob

import torch.cuda
from stable_baselines3 import DDPG

from utils.callbacks import logDir
from simulator import Simulator

if __name__ == "__main__":
    target = 40.
    source = './sample/Icecream.csv'
    tool = './sample/Scoop.csv'
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    env = Simulator(target, source, tool, verbose=True)
    model_files = sorted(glob(logDir() + '*_best_model.pkl'))
    model_files.sort(key=lambda var: [int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
    best_model = model_files[-1]
    model = DDPG.load(best_model, device=device)
    test_episode = 100
    for i in range(test_episode):
        obs = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            if done:
                env.show()
                '''
                print(
                    "cup={}, step={}, weight={}, reward={}".format(i, info['step_count'], info['total_weight'], reward))
                '''
                break
