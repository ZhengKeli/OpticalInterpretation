import numpy as np

import braandket as bnk


class Cavity(bnk.KetSpace):
    def __init__(self, n, energy=1.0, name=None):
        super().__init__(n, name)
        self.increase = sum(
            np.sqrt(i + 1) * self.eigenstate(i + 1) @ self.eigenstate(i).ct
            for i in range(self.n - 1))
        self.decrease = self.increase.ct

        self.energy = energy * sum(
            i * self.projector(i)
            for i in range(self.n))


class Orbit(bnk.KetSpace):
    def __init__(self, potential=0.0, name=None):
        super().__init__(2, name)
        self.increase = self.eigenstate(1) @ self.eigenstate(0).ct
        self.decrease = self.increase.ct

        self.potential = potential * self.eigenstate(1) @ self.eigenstate(1).ct


class Band(bnk.KetSpace):
    def __init__(self, n, potential=0.0, name=None):
        super().__init__(n, name)
        self.increase = sum(
            np.sqrt(i + 1) * self.eigenstate(i + 1) @ self.eigenstate(i).ct
            for i in range(self.n - 1))
        self.decrease = self.increase.ct

        self.potential = potential * sum(
            i * self.projector(i)
            for i in range(self.n))
