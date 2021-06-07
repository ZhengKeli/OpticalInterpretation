import braandket as bnk
from components import Band, Cavity
from model_chemical import Atom
from utils import eig_map


class AtomTransportAnode:
    """ Subsystem of atom, transport level and anode level """

    def __init__(
            self,
            potential_tp=0.0, potential_1=-1.0, potential_2=-4.0, potential_3=-5.0,
            potential_an=-6.0
    ):
        self.at = Atom(potential_tp, potential_1, potential_2, potential_3, name='at')
        self.tp = Band(3, potential=potential_tp, name='tp')
        self.an = Band(3, potential=potential_an, name='an')

        self.hmt_energy = bnk.sum(
            self.at.potential,
            self.tp.potential,
            self.an.potential,
        )

    def hmt_transition(self, g_01, g_12, g_23, g_ta, c01: Cavity, c12: Cavity, c23: Cavity, cta: Cavity):
        return bnk.sum(
            self.at.transition(g_01, g_12, g_23, c01, c12, c23, self.tp),
            g_ta * bnk.sum_ct(cta.increase @ self.tp.decrease @ self.an.increase)
        )

    def eigenstate(self, at_i, tp_i, an_i=None):
        if an_i is None:
            an_i = 2 - tp_i - [0, 1, 1, 2][at_i]
        return bnk.prod(
            self.tp.eigenstate(tp_i),
            self.at.eigenstate(at_i),
            self.an.eigenstate(an_i),
        )


class PrunedAtomTransportAnode:
    """ Pruned subsystem of AtomTransportAnode

    Considering the limitation of memory, need to firstly prune the subsystems, and then construct the big system
    """

    def __init__(self, org: AtomTransportAnode):
        self.org = org
        self.space = bnk.PrunedKetSpace.from_initial(
            org.eigenstate(0, 0, 2),
            bnk.sum_ct(
                org.at.operator(1, 0) @ org.tp.decrease,
                org.at.operator(2, 1),
                org.at.operator(3, 2) @ org.tp.decrease,
                org.tp.decrease @ org.an.increase
            )
        )

        self.hmt_energy = self.space.prune(self.org.hmt_energy)

    def hmt_transition(self, g_01, g_12, g_23, g_ta, c01: Cavity, c12: Cavity, c23: Cavity, cta: Cavity):
        return self.space.prune(self.org.hmt_transition(g_01, g_12, g_23, g_ta, c01, c12, c23, cta))

    def eigenstate(self, at_i, tp_i, an_i=None):
        return self.space.prune(self.org.eigenstate(at_i, tp_i, an_i))


class OpticalModel:
    """ Quantum system of the optical model """

    def __init__(
            self,
            potential_0=0.0, potential_1=-1.0, potential_2=-4.0, potential_3=-5.0,
            g_01=0.02, g_12=0.02, g_23=0.02,
            gamma_01=0.002, gamma_12=0.002, gamma_23=0.002,

            potential_an=-6.0, g_ta=0.02, gamma_ta=0.0,

            hb=1.0):
        part0 = PrunedAtomTransportAnode(
            AtomTransportAnode(potential_0, potential_1, potential_2, potential_3, potential_an))
        part1 = PrunedAtomTransportAnode(
            AtomTransportAnode(potential_0, potential_1, potential_2, potential_3, potential_an))

        c01 = Cavity(3, energy=(potential_0 - potential_1), name='c01')
        c12 = Cavity(3, energy=(potential_1 - potential_2), name='c12')
        c23 = Cavity(2, energy=(potential_2 - potential_3), name='c23')
        cta = Cavity(3, energy=(potential_0 - potential_an), name='cta')

        self.part0 = part0
        self.part1 = part1
        self.c01 = c01
        self.c12 = c12
        self.c23 = c23
        self.cta = cta

        hmt_energy = bnk.sum(
            part0.hmt_energy,
            part1.hmt_energy,
            c01.energy,
            c12.energy,
            c23.energy,
            cta.energy,
        )

        hmt_transition = bnk.sum(
            part0.hmt_transition(g_01, g_12, g_23, g_ta, c01, c12, c23, cta),
            part1.hmt_transition(g_01, g_12, g_23, g_ta, c01, c12, c23, cta),
        )

        hmt = hmt_energy + hmt_transition

        gamma_deco = [
            (gamma_01, c01.decrease),
            (gamma_12, c12.decrease),
            (gamma_23, c23.decrease),
            (gamma_ta, cta.decrease)]
        gamma, deco = zip(*gamma_deco)

        self.hb = hb
        self.hmt = hmt
        self.gamma = gamma
        self.deco = deco

    def eigenstate(self, part0, part1, c01, c12, c23, cta=0):
        return bnk.prod(
            self.part0.eigenstate(*part0),
            self.part1.eigenstate(*part1),
            self.c01.eigenstate(c01),
            self.c12.eigenstate(c12),
            self.c23.eigenstate(c23),
            self.cta.eigenstate(cta),
        )


class PrunedOpticalModel:
    """ Pruned quantum system of the optical model """

    def __init__(self, org: OpticalModel, initial):
        space = bnk.PrunedKetSpace.from_initial(
            initial,
            [org.hmt, org.deco]
        )

        self.org = org
        self.space = space

        self.hb = org.hb
        self.hmt = space.prune(org.hmt)
        self.gamma = org.gamma
        self.deco = space.prune(org.deco)

    @property
    def org_eigenstates(self):
        """ The original eigenstates, consisting of the most primitive parts. """
        org_eigenstates = []
        for org_eigenstate in self.space.org_eigenstates:
            org_eigenstate = self.org.part0.space.restore(org_eigenstate)
            org_eigenstate = self.org.part1.space.restore(org_eigenstate)
            org_eigenstates.append(org_eigenstate)
        return tuple(org_eigenstates)

    def labels(self):
        model = self.org
        part0 = model.part0.org
        part1 = model.part1.org

        labels = []
        for i, psi in enumerate(self.org_eigenstates):
            em = eig_map(psi)
            label = "$" \
                    f"|{em[part0.at]}{em[part0.tp]},{em[part0.an]}\\rangle_{{at_0 tp_0 an_0}}" \
                    f"|{em[part1.at]}{em[part1.tp]},{em[part1.an]}\\rangle_{{at_1 tp_1 an_1}}" \
                    f"|{em[model.c01]}{em[model.c12]}{em[model.c23]},{em[model.cta]}\\rangle_{{\\omega_{{01}} \\omega_{{12}} \\omega_{{23}} \\Omega}}" \
                    "$"
            labels.append(label)
        return tuple(labels)
