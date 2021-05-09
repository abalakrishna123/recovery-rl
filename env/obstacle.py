import numpy as np


class Obstacle:
    '''
        Implementation of 2D rectangular obstacle
    '''
    def __init__(self, boundsx, boundsy, penalty=100):
        self.boundsx = boundsx
        self.boundsy = boundsy
        self.penalty = 1

    def __call__(self, x):
        return (self.boundsx[0] <= x[0] <= self.boundsx[1]
                and self.boundsy[0] <= x[1] <= self.boundsy[1]) * self.penalty


class Obstacle3D:
    '''
        Implementation of 3D box obstacle
    '''
    def __init__(self, boundsx, boundsy, boundsz, penalty=100):
        self.boundsx = boundsx
        self.boundsy = boundsy
        self.boundsz = boundsz
        self.penalty = 1

    def __call__(self, x):
        return (self.boundsx[0] <= x[0] <= self.boundsx[1]
                and self.boundsy[0] <= x[1] <= self.boundsy[1]
                and self.boundsz[0] <= x[2] <= self.boundsz[1]) * self.penalty


class ComplexObstacle(Obstacle):
    '''
        Implementation of 2D obstacle which consists of a composition of
        2D rectangular obstacles
    '''
    def __init__(self, bounds):
        self.obs = []
        for boundsx, boundsy in bounds:
            self.obs.append(Obstacle(boundsx, boundsy))

    def __call__(self, x):
        return np.max([o(x) for o in self.obs])
