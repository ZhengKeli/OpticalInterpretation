import matplotlib.pyplot as plt
import numpy as np

import braandket as bnk


def eig_map(psi: bnk.QTensor):
    spaces = psi.spaces
    values = psi[spaces]
    indices = np.unravel_index(np.argmax(values), np.shape(values))
    em = {space: index for space, index in zip(psi.spaces, indices)}
    return em


def plot_probs_logs(t_logs, probs_logs, labels, threshold=0.02):
    final_indices = []
    initial_indices = []
    intermediate_indices = []
    for i, prob_logs in enumerate(probs_logs):
        if prob_logs[-1] > threshold:
            final_indices.append(i)
        elif prob_logs[0] > threshold:
            initial_indices.append(i)
        else:
            intermediate_indices.append(i)

    print("Final states:")
    for i in final_indices:
        label = labels[i]
        prob_logs = probs_logs[i]

        print(i, label)
        plt.plot(t_logs, prob_logs, label=(label + " ◾"))

    print("Initial states:")
    for i in initial_indices:
        label = labels[i]
        prob_logs = probs_logs[i]

        print(i, label)
        plt.plot(t_logs, prob_logs, label=(label + " ▶"))

    for i in intermediate_indices:
        plt.plot(t_logs, probs_logs[i])

    plt.xlim(t_logs[0], t_logs[-1])
    plt.ylim(0, 1)
    plt.xlabel("t")
    plt.ylabel("prob")
    plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.1))
    plt.gca().yaxis.tick_right()
    plt.grid(axis='y')
    plt.legend()
