import braandket as bnk
from components import Cavity, Band
from utils import eig_map


class Atom(bnk.KetSpace):
    """ Atom with 2 electron orbits, having 4 possible states. """

    def __init__(self, potential_0=0.0, potential_1=-1.0, potential_2=-4.0, potential_3=-5.0, name=None):
        super().__init__(4, name)

        self.potential = bnk.sum(
            potential_0 * self.projector(0),
            potential_1 * self.projector(1),
            potential_2 * self.projector(2),
            potential_3 * self.projector(3),
        )

    def transition(self, g_01, g_12, g_23, c_01: Cavity, c_12: Cavity, c_23: Cavity, tp: Band):
        return bnk.sum_ct(
            g_01 * c_01.increase @ self.operator(1, 0) @ tp.decrease,
            g_12 * c_12.increase @ self.operator(2, 1),
            g_23 * c_23.increase @ self.operator(3, 2) @ tp.decrease,
        )

    @staticmethod
    def electrons(i):
        return [
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
        ][i]


class ChemicalModel:
    """ Chemical model """

    def __init__(
            self,
            potential_0=0.0, potential_1=-1.0, potential_2=-4.0, potential_3=-5.0,
            g_01=0.02, g_12=0.02, g_23=0.02,
            gamma_01=0.002, gamma_12=0.002, gamma_23=0.002,
            hb=1.0):
        at0 = Atom(potential_0, potential_1, potential_2, potential_3, name='at0')
        at1 = Atom(potential_0, potential_1, potential_2, potential_3, name='at1')
        tp = Band(3, potential=potential_0, name='tp')
        c01 = Cavity(3, energy=(potential_0 - potential_1), name='c01')
        c12 = Cavity(3, energy=(potential_1 - potential_2), name='c12')
        c23 = Cavity(2, energy=(potential_2 - potential_3), name='c23')

        self.at0 = at0
        self.at1 = at1
        self.tp = tp
        self.c01 = c01
        self.c12 = c12
        self.c23 = c23

        hmt_energy = bnk.sum([
            at0.potential,
            at1.potential,
            tp.potential,
            c01.energy,
            c12.energy,
            c23.energy,
        ])

        hmt_interact = bnk.sum(
            at0.transition(g_01, g_12, g_23, c01, c12, c23, tp),
            at1.transition(g_01, g_12, g_23, c01, c12, c23, tp),
        )

        hmt = hmt_energy + hmt_interact

        gamma, deco = zip(
            (gamma_01, c01.decrease),
            (gamma_12, c12.decrease),
            (gamma_23, c23.decrease),
        )

        self.hb = hb
        self.hmt = hmt
        self.gamma = gamma
        self.deco = deco

    def eigenstate(self, at0, at1, tp, c01, c12, c23):
        return bnk.prod(
            self.at0.eigenstate(at0),
            self.at1.eigenstate(at1),
            self.tp.eigenstate(tp),
            self.c01.eigenstate(c01),
            self.c12.eigenstate(c12),
            self.c23.eigenstate(c23),
        )


class PrunedChemicalModel:
    def __init__(self, org: ChemicalModel, initial):
        space = bnk.PrunedKetSpace.from_initial(initial, [org.hmt, org.deco])

        self.org = org
        self.space = space

        self.hb = org.hb
        self.hmt = space.prune(org.hmt)
        self.gamma = org.gamma
        self.deco = space.prune(org.deco)

    def labels(self):
        model = self.org

        labels = []
        for i, psi in enumerate(self.space.org_eigenstates):
            em = eig_map(psi)
            label = f"$" \
                    f"|{em[model.at0]}\\rangle_{{at0}} " \
                    f"|{em[model.at1]}\\rangle_{{at1}} " \
                    f"|{em[model.tp]}\\rangle_{{tp}} " \
                    f"|{em[model.c01]}{em[model.c12]}{em[model.c23]}\\rangle_{{\\omega_{{01}}\\omega_{{12}}\\omega_{{23}}}} " \
                    f"$"
            labels.append(label)
        return tuple(labels)
