import argparse
import re
import time
from glob import glob

import torch.cuda
from stable_baselines3 import DDPG, PPO, TD3

from utils.callbacks import logDir
from simulator import Simulator


def model_verify(env_: Simulator,
                 device,
                 algorithm,
                 thresh_=50,
                 episode_=50):
    bk_verbose = env_.verbose
    env_.verbose = False
    env_.verify = True
    env_.thresh_step = thresh_
    model_files = sorted(glob(logDir() + '*_best_model.pkl'))
    model_files.sort(key=lambda var: [int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
    res_files = []
    for model_path in model_files:
        if algorithm == 'DDPG':
            _model = DDPG.load(model_path, device=device)
        elif algorithm == 'PPO':
            _model = PPO.load(model_path, device=device)
        elif algorithm == 'TD3':
            _model = TD3.load(model_path, device=device)
        else:
            raise Exception(f"Not supported {algorithm} algorithm")
        env_.reset(hard=True)
        try:
            for _ in range(episode_):
                _obs = env_.reset()
                _done = False
                while not _done:
                    _action, _states = _model.predict(_obs)
                    _obs, _, _done, _ = env_.step(_action)
            print(f"{model_path} verified completed")
            res_files.append(model_path)
        except:
            print(f"{model_path} verified failed [Episode={env_.episode_count}]")
    env_.verbose = bk_verbose
    env_.verify = False
    env_.reset(hard=True)
    return res_files


def parse_opt():
    parser = argparse.ArgumentParser(description='Select the test model')
    parser.add_argument('--source', type=str, default='./sample/Icecream.csv', help='an ice-cream source path')
    parser.add_argument('--tool', type=str, default='./sample/Scoop.csv', help='a scooping tool path')
    parser.add_argument('--target', type=float, default=40., help='a target for one scoop')
    parser.add_argument('--algorithm', type=str, default='DDPG', help='use algorithm (DDPG, PPO, TD3)')
    parser.add_argument('--verify', action='store_true', default=False, help='skip the model varify')
    parser.add_argument('--plot', action='store_true', default=False, help='skip the drawing ice-cream plot')
    parser.add_argument('--lstm', action='store_true', default=False, help='use lstm module at the statement')
    parser.add_argument('--verbose', action='store_true', default=False, help='verbose the tracing')
    opt = parser.parse_args()
    return opt


def run(source: str,
        tool: str,
        target: float,
        algorithm: str,
        verify=True,
        plot=True,
        lstm=False,
        verbose=False):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    env = Simulator(target, source, tool, lstm=lstm, verbose=verbose)
    if verify:
        verified_files = model_verify(env, device, algorithm)
        if len(verified_files) == 0:
            print("No models can be tested")
            return
        else:
            print(f"Verified model = {verified_files}")
            best_model = verified_files[-1]
    else:
        model_files = sorted(glob(logDir() + '*_best_model.pkl'))
        if len(model_files) == 0:
            print("Any modules remain the path")
            return
        else:
            best_model = model_files[-1]

    if algorithm == 'DDPG':
        model = DDPG.load(best_model, device=device)
    elif algorithm == 'PPO':
        model = PPO.load(best_model, device=device)
    elif algorithm == 'TD3':
        model = TD3.load(best_model, device=device)
    else:
        raise Exception(f"Not supported {algorithm} algorithm")
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
                print(
                    "cup={}, step={}, weight={}, reward={}".format(i, info['step_count'], info['total_weight'], reward)
                )
                if plot:
                    env.show()


if __name__ == "__main__":
    arg = parse_opt()
    run(**vars(arg))
