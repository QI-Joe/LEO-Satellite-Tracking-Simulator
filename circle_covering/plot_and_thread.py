from time import time
from multiprocessing import Process, Queue
from queue import Empty
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
from typing import *
import sys

N = 5
R = 4

class plot_thread:
    def __init__(self, queue, interval):
        self.queue = queue
        self.interval = interval

    def __call__(self):
        try:
            self.fig, self.ax = plt.subplots()
            timer = self.fig.canvas.new_timer(interval=(1000*self.interval))
            timer.add_callback(self.callback)
            timer.start()
            plt.show()
        except KeyboardInterrupt:
            pass
        self.terminate()

    def callback(self):
        while True:
            try:
                xr = self.queue.get_nowait()
                if xr is None:
                    self.terminate()
                    return False
                x, r = xr
                x = np.reshape(x, (-1, 2))
                self.ax.clear()
                self.ax.scatter(x[:, 0], x[:, 1])
                self.ax.add_patch(plt.Circle((0, 0), R, fill=False))
                for i, p in enumerate(x):
                    self.ax.add_patch(plt.Circle(p, r, fill=False))
                    self.ax.annotate(str(i), p)
                # plt.show(block=False)
                self.ax.set_xlim((-1.25*R, 1.25*R))
                self.ax.set_ylim((-1.25*R, 1.25*R))
                self.ax.set_title(f'r = {r:.6f}, R = {R/r:.6f}')
                self.ax.set_aspect('equal')
            except Empty:
                break
        self.fig.canvas.draw()
        return True

    def terminate(self):
        plt.close(self.fig)


class plotting_wrapper:
    def __init__(self, fun):
        self.fun = fun
        self.count = 0
        self.best_value = None
        self.best_points = None
        self.new_best = False
        self.last_plot_time = None

        self.plot_interval = 1  # seconds
        self.plot_queue = Queue()
        self.plotter = plot_thread(self.plot_queue, self.plot_interval)
        self.plot_thread = Process(target=self.plotter)
        self.plot_thread.start()

    def __call__(self, x):
        self.count += 1
        r = self.fun(x)
        self.store_solution(x, r)
        return r

    def store_solution(self, x, r):
        # store best for printing
        if self.best_value is None or r < self.best_value:
            self.best_value = r
            self.best_points = np.reshape(x, (-1, 2))
            self.new_best = True
        # print best periodically
        now = time()
        if self.last_plot_time is None or self.last_plot_time + 1 < now:
            self.last_plot_time = now
            # print(f'{self.count:,d} {self.best:.6f}')
            if self.new_best:
                self.print_best()
            self.new_best = False

    def print_best(self):
        r = self.best_value
        x = self.best_points
        print(f'current best (r = {r:.6f}, R = {R/r:.6f}):')
        for i, p in enumerate(x):
            print(f'{i:2d}, {p[0]:8.5f}, {p[1]:8.5f}')
        self.plot_queue.put((x, r))

    def terminate(self):
        self.plot_queue.put(None)
        self.plot_thread.join()

    def __del__(self):
        self.terminate()


class Wrapper:
    def __init__(self, func) -> None:
        self.func = func
        self.count = 0
        self.best_value = None
        self.best_points = None
        self.new_best = False
        self.last_plot_time = None

    def __call__(self, x, *args: Any, **kwds: Any) -> Any:
        self.count += 1
        r = self.func(x)
        self.store_solution(x, r)
        return r

    def store_solution(self, x, r):
        # store best for printing
        if self.best_value is None or r < self.best_value:
            self.best_value = r
            self.best_points = np.reshape(x, (-1, 2))
            self.new_best = True
        # print best periodically
        now = time()
        if self.last_plot_time is None or self.last_plot_time + 1 < now:
            self.last_plot_time = now
            # print(f'{self.count:,d} {self.best:.6f}')
            if self.new_best:
                self.print_best()
            self.new_best = False

    def print_best(self):
        r = self.best_value
        x = self.best_points
        print(f'current best (r = {r:.6f}, R = {R/r:.6f}):')
        for i, p in enumerate(x):
            print(f'{i:2d}, {p[0]:8.5f}, {p[1]:8.5f}')

    # def __del__(self):
    #     sys.exit(0)
    