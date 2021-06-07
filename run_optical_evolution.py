import matplotlib.pyplot as plt
import numpy as np

import braandket as bnk
from model_optical import OpticalModel, PrunedOpticalModel
from utils import plot_probs_logs

# model configuration
# %%

model = OpticalModel(
    potential_0=0.0, potential_1=-1.0, potential_2=-4.0, potential_3=-5.0,
    g_01=0.02, g_12=0.02, g_23=0.02,
    gamma_01=0.002, gamma_12=0.002, gamma_23=0.002,

    potential_an=-6.0,
    g_ta=0.02,
    # gamma_ta=0.002
    gamma_ta=0.0
)

# model pruning
# %%

psi0 = model.eigenstate(
    (3, 0),  # at, tp, an=2-tp-an
    (0, 0),  # at, tp, an=2-tp-an
    0, 0, 1  # c01, c12, c23, cta=0
    # 1, 1, 1
)

#

model = PrunedOpticalModel(model, psi0)
labels = model.labels()

print()
for i, label in enumerate(labels):
    print(i, label)
print(f"Totally {len(labels)} basic states:")
print(flush=True)

#

t0 = 0.0
psi0 = model.space.prune(psi0)
rho0 = psi0 @ psi0.ct
del psi0

# evolution configuration
# %%

span = 4000
dt = 0.1
dlt = span / 1024


def log_func(t, rho: bnk.QTensor):
    t_logs.append(t)

    probs = np.diag(rho.flatten())
    probs = np.abs(probs)
    probs_logs.append(probs)


# reset initial state
# %%

t = t0
rho = rho0

t_logs = []
probs_logs = []

# evolution computing
# %%

bnk.lindblad_evolve(
    t, rho,
    model.hmt, model.deco, model.gamma, model.hb,
    span, dt,
    dlt=dlt, log_func=log_func,
    reduce=False, method='rk4'
)

# process logs
# %%

t_logs = np.asarray(t_logs)
probs_logs = np.transpose(probs_logs)

#

plt.figure(figsize=(10, 5))

plot_probs_logs(t_logs, probs_logs, labels)

plt.tight_layout()
plt.savefig("./optical.svg")
plt.savefig("./optical.png", dpi=200)
plt.show()
