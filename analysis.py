import os.path
import re
from glob import glob

import matplotlib.pyplot


def parse_log(dir_path, verbose=False):
    assert os.path.exists(dir_path), f"{dir_path} is not exists"
    model_files = sorted(glob(os.path.join(dir_path, '*_best_model.pkl')))
    model_files.sort(key=lambda var: [int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
    log_file = os.path.join(dir_path, 'log.txt')
    with open(log_file, "r") as file:
        logs = file.readlines()
    episodes = [line for line in logs if re.search(r'cup\=[0-9]+', line) is not None]
    steps = [line for line in logs if re.search(r'==> pos', line) is not None]
    replaces = [line for line in logs if re.search(r'Replace container[0-9]+', line) is not None]
    if verbose:
        print(f"Path = {log_file}, Lines = {len(logs)}")
        print(f"Episode = {len(episodes)}")
        print(f"step = {len(steps)}")
        print(f"replace = {len(replaces)}")

    def collapse_step(_logs):
        res = []
        _items: list = None
        for _log in _logs:
            # Add the episode count when 'step=1'
            if re.search(r'step\=1,', _log) is not None:
                if isinstance(_items, list) and len(_items) > 0:
                    res.append(_items.copy())
                _items = []
            _parts = re.findall(r'([-a-z]+\=)([-0-9.\-\+]+)', _log)
            _parts = {k.replace('=', ''): v for k, v in _parts}
            _items.append(_parts)
        # Add the last episode
        res.append(_items.copy())
        return res

    def collapse_episode(_logs):
        res = []
        for _log in _logs:
            _parts = re.findall(r'([-a-z]+\=)([-0-9.\-\+]+)', _log)
            _parts = {k.replace('=', ''): v for k, v in _parts}
            res.append(_parts)
        return res

    steps = collapse_step(steps)
    episodes = collapse_episode(episodes)
    assert len(episodes) == len(steps), f"{dir_path} episode count abnormal"
    return steps, episodes, len(replaces)  # step, episode, replaces


def show_episode_plot(logs, target: float):
    x = range(len(logs))
    y_steps = [int(log['step']) for log in logs]
    y_weights = [float(log['weight']) - target for log in logs]
    total_episodes = len(x)
    max_step = max(y_steps)
    max_weight = max(y_weights)
    # Train step graph
    matplotlib.pyplot.subplot(1, 2, 1)
    matplotlib.pyplot.plot(x, y_steps, '-')
    matplotlib.pyplot.title(f'Train steps (max={max_step})')
    matplotlib.pyplot.xlabel(f'episodes({total_episodes})')
    # Weight graph
    matplotlib.pyplot.subplot(1, 2, 2)
    matplotlib.pyplot.bar(x, y_weights)
    matplotlib.pyplot.title(f'Weights (target={target}, max={max_weight})')
    matplotlib.pyplot.xlabel(f'episodes({total_episodes})')
    matplotlib.pyplot.show()


if __name__ == "__main__":
    step_log, episode_log, containers = parse_log("logs/Logs_211205_PPO", verbose=True)
    show_episode_plot(episode_log, 40.0)
