import random
import time

import matplotlib.pyplot
import numpy


class Simulator:
    def __init__(self,
                 src_path: str,
                 tool_path: str):
        self.ice_cream = numpy.loadtxt(src_path, delimiter=',', dtype=numpy.float32, encoding='utf-8-sig')
        self.scooper = numpy.loadtxt(tool_path, delimiter=',', dtype=numpy.float32, encoding='utf-8-sig')
        self.weight = 0

    # gravity module
    def __gravity__(self):
        src_h, src_w = self.ice_cream.shape
        assert src_w != 0 and src_h != 0
        lines = None
        for x in range(0, src_w):
            line = self.ice_cream[:, x]
            mass = line[line > 0]
            empty = numpy.zeros(src_h - mass.shape[0])
            line = numpy.concatenate([empty, mass], axis=0).reshape((src_h, 1))
            lines = line if lines is None else numpy.hstack([lines, line])
        assert lines.shape != src_h, src_w
        return lines

    def __call__(self, pos: tuple):
        tool_h, tool_w = self.scooper.shape
        src_h, src_w = self.ice_cream.shape
        assert src_w != 0 and src_h != 0
        x, y = pos
        assert x + tool_w <= src_w and y + tool_h <= src_h
        # Scoop the ice-cream
        scoop = self.ice_cream[y:y + tool_h, x:x + tool_w] * self.scooper
        # Measure the weight of scoop
        self.weight = scoop.sum()
        # Remove as the mass of scoop
        self.ice_cream[y:y + tool_h, x:x + tool_w] -= scoop
        # Adjust the gravity module
        self.ice_cream = self.__gravity__()
        return self.weight

    def inspection(self):
        tool_h, tool_w = self.scooper.shape
        src_h, src_w = self.ice_cream.shape
        assert src_w != 0 and src_h != 0
        candidates = list()
        for x in range(0, (src_w - tool_w) + 1):
            line = self.ice_cream[:, x]
            for y in range(0, (src_h - tool_h) + 1):
                if line[y] != 0:
                    candidates.append((x, y))
                    break
        return candidates

    def show(self):
        colors = matplotlib.pyplot.cm.rainbow
        matplotlib.pyplot.imshow(self.ice_cream, interpolation='nearest', cmap=colors)
        matplotlib.pyplot.title('Ice-cream')
        matplotlib.pyplot.show(block=False)
        matplotlib.pyplot.pause(0.5)
        matplotlib.pyplot.close()


# Test the simulator
if __name__ == "__main__":
    source = './sample/Icecream.csv'
    tool = './sample/Scoop.csv'
    sim = Simulator(source, tool)
    no = 0
    print("Ready to the simulator")
    time.sleep(2)
    while True:
        points = sim.inspection()
        if len(points) == 0:
            break
        pos = random.choice(points)
        sim(pos)
        matplotlib.pyplot.figure()
        sim.show()
        print("Job {}, Weight {}".format(no, sim.weight))
        no += 1
