import re
import time
from glob import glob

import torch.cuda
from stable_baselines3 import DDPG

from utils.callbacks import logDir
from simulator import Simulator


def model_verify(env_: Simulator,
                 device,
                 thresh_=100,
                 episode_=100):
    bk_verbose = env_.verbose
    env_.verbose = False
    env_.verify = True
    env_.thresh_step = thresh_
    model_files = sorted(glob(logDir() + '*_best_model.pkl'))
    model_files.sort(key=lambda var: [int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
    res_files = []
    for model_path in model_files:
        _model = DDPG.load(model_path, device=device)
        env_.reset(hard=True)
        try:
            for _ in range(episode_):
                _obs = env.reset()
                _done = False
                while not _done:
                    _action, _states = _model.predict(_obs)
                    _obs, _, _done, _ = env.step(_action)
            print(f"{model_path} verified completed")
            res_files.append(model_path)
        except:
            print(f"{model_path} verified failed [Episode={env_.episode_count}]")
    env_.verbose = bk_verbose
    env_.verify = False
    env_.reset(hard=True)
    return res_files


if __name__ == "__main__":
    target = 40.
    source = './sample/Icecream.csv'
    tool = './sample/Scoop.csv'
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    env = Simulator(target, source, tool, verbose=False)
    verified_files = model_verify(env, device, thresh_=50, episode_=50)
    if len(verified_files) == 0:
        print("No models can be tested")
    else:
        print(f"Verified model = {verified_files}")
        best_model = verified_files[-1]
        model = DDPG.load(best_model, device=device)
        test_episode = 100
        time.sleep(5)
        print(f"Test ready to use {best_model}, episode={test_episode}")
        for i in range(test_episode):
            obs = env.reset()
            done = False
            while not done:
                action, states = model.predict(obs)
                obs, reward, done, info = env.step(action)
                if done:
                    env.show()
                    print(
                        "cup={}, step={}, weight={}, reward={}".format(i, info['step_count'], info['total_weight'],
                                                                       reward))
