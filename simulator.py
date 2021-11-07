import math
import random
import time

import matplotlib.pyplot
import numpy
from numpy import ndarray
import gym


class Simulator(gym.Env):
    def __init__(self,
                 target: float,
                 src_path: str,
                 tool_path: str,
                 verbose=False):
        # Construct the source and tool
        self.source = numpy.loadtxt(src_path, delimiter=',', dtype=numpy.float32, encoding='utf-8-sig')
        self.tool = numpy.loadtxt(tool_path, delimiter=',', dtype=numpy.float32, encoding='utf-8-sig')
        assert isinstance(self.source, ndarray) and isinstance(self.tool, ndarray)
        self.pipeline = self.source.copy()
        # Construct the target count and weight
        self.cup_weight = 0
        self.target_weight = target
        # Calculate the expected scoop weight and iteration times
        source_count = math.ceil(self.source.size / self.tool.sum())
        self.target_scoop = self.source.sum() / source_count
        # Bring-up the other parameters
        self.verbose = verbose
        self.cmd_count = self.source.shape[1] - self.tool.shape[1]
        self.step_count = 0
        self.min_step_count = 1000
        self.episode_count = 0
        self.container_count = 0
        self.done = False
        # Construct the spaces to command size [ex> source=10, scoop=4, command = 10-4]
        obs_size = self.source.shape[1]  # Observation Size is same the command size
        action_size = 1  # Action is only ONE Scoop
        # Box is contained "float" because of the normalization issue under the INTEGER (in td3.py module)
        self.observation_space = gym.spaces.Box(low=0, high=self.cmd_count, shape=(obs_size,), dtype=numpy.float32)
        self.action_space = gym.spaces.Box(low=0, high=self.cmd_count, shape=(action_size,), dtype=numpy.float32)

    # Called at the beginning of each episode
    def reset(self):
        assert isinstance(self.source, ndarray)
        if self.verbose:
            print("-------------")
            print(self.episode_count, "cups")
        self.step_count = 0
        self.cup_weight = 0
        self.done = False
        self.episode_count += 1
        return self._obs()

    # Adjust the gravity at the source
    def __gravity(self):
        source_height, source_width = self.pipeline.shape
        assert source_width != 0 and source_height != 0
        lines = None
        for x in range(0, source_width):
            line = self.pipeline[:, x]
            mass_line = line[line > 0]
            empty_line = numpy.zeros(source_height - mass_line.shape[0])
            line = numpy.concatenate([empty_line, mass_line], axis=0).reshape((source_height, 1))
            lines = line if lines is None else numpy.hstack([lines, line])
        assert lines.shape != source_height, source_width
        return lines

    # Replace the source
    def __replace(self):
        if self.verbose:
            print(
                "==> Replace container{}(weight={})-> {}(weight={})".format(self.container_count, self.pipeline.sum(),
                                                                            self.container_count + 1,
                                                                            self.source.sum()))
        self.container_count += 1
        return self.source.copy()

    # Calculate the candidate points
    def inspect(self):
        tool_height, tool_width = self.tool.shape
        source_height, source_width = self.pipeline.shape
        assert source_width != 0 and source_height != 0
        verifiers = list()
        candidates = list()
        for x in range(0, (source_width - tool_width) + 1):
            line = self.pipeline[:, x]
            for y in range(0, (source_height - tool_height) + 1):
                if line[y] != 0:
                    candidates.append((x, y))
                    verifiers.append(True)
                    break
                elif y == source_height - tool_height:
                    candidates.append((x, y))
                    verifiers.append(False)
        return candidates, verifiers

    def _obs(self):
        # Output the remained ice-cream weight
        weights = self.pipeline.sum(axis=0)
        return weights

    def _reward(self,
                scoop,
                epsilon=0.5):
        # Put the ice cream close to target weight as few times as possible
        if self.done:
            if self.min_step_count > self.step_count:
                self.min_step_count = self.step_count
            weight_score = self.target_weight / (self.cup_weight - self.target_weight + epsilon)
            time_score = 0
            # time_score = - self.target_scoop * (self.step_count - self.min_step_count) ** 2
            return weight_score + time_score
        else:
            return abs((self.target_scoop - scoop) * self.step_count) * (-1 if scoop < self.target_scoop else +1)

    def _infos(self):
        return {'total_weight': self.cup_weight, 'step_count': self.step_count}

    def step(self, action: list):
        tool_height, tool_width = self.tool.shape
        source_height, source_width = self.pipeline.shape
        # Get a position from the action command
        assert source_width != 0 and source_height != 0
        pos_candidates, _ = self.inspect()
        x, y = pos_candidates[int(action)]
        assert x + tool_width <= source_width and y + tool_height <= source_height
        # Scoop the ice-cream
        scoop = self.pipeline[y:y + tool_height, x:x + tool_width] * self.tool
        # Measure the weight of scoop
        scoop_weight = scoop.sum()
        self.cup_weight += scoop_weight
        # Remove the ice-cream as the scoop
        self.pipeline[y:y + tool_height, x:x + tool_width] -= scoop
        self.pipeline = self.__gravity()
        self.step_count += 1
        # Stop the episode when the cup weight is over than target
        if self.cup_weight >= self.target_weight:
            self.done = True
            if self.verbose:
                print(
                    "cup={}, step={}, weight={}, reward={}".format(self.episode_count, self.step_count,
                                                                   self.cup_weight, self._reward(scoop_weight))
                )
        else:
            if self.verbose:
                print("==> pos={}, step={}, weight={}, reward={}".format((x, y), self.step_count, scoop_weight,
                                                                         self._reward(scoop_weight)))
        # Replace the next source when the previous ice-cream is empty
        if self.pipeline.sum() < self.target_weight:
            self.pipeline = self.__replace()
        return self._obs(), self._reward(scoop_weight), self.done, self._infos()

    def show(self):
        colors = matplotlib.pyplot.cm.rainbow
        matplotlib.pyplot.figure()
        matplotlib.pyplot.imshow(self.pipeline, interpolation='nearest', cmap=colors)
        matplotlib.pyplot.title('Ice-cream')
        matplotlib.pyplot.show(block=False)
        matplotlib.pyplot.pause(0.5)
        matplotlib.pyplot.close()


# Test the simulator
if __name__ == "__main__":
    target = 40.
    source = './sample/Icecream.csv'
    tool = './sample/Scoop.csv'
    sim = Simulator(target, source, tool, True)
    count = 100
    print("Ready to the simulator")
    time.sleep(2)
    for i in range(count):
        sim.reset()
        done = False
        while not done:
            action = random.randrange(0, sim.cmd_count + 1)
            obs, reward, done, info = sim.step([action])
            if done:
                sim.show()
                '''
                print(
                    "cup={}, step={}, weight={}, reward={}".format(i, info['step_count'], info['total_weight'], reward))
                '''
                break
