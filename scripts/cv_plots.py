# imports
import numpy as np
import matplotlib.pyplot as plt

from sources.ECVS import ECVS
from sources.Networks import NN

nn = NN()
model = ECVS()

T = np.linspace(.1,10,100)
_, activations = nn.trained_network(15)
Es, _ = model.calc_energies(activations)
CVs = model.calc_cv(T, Es)

_, activations_ut = nn.untrained_network(15)
Es_ut, _ = model.calc_energies(activations)
CVs_ut = model.calc_cv(Es_ut)

beta = [1/t for t in T]

fig, axs = plt.subplots(1,2)

axs[0].plot(beta, CVs_ut)
axs[0].set_title('Heat Capacity vs Beta Trained')
axs[0].set_xlabel('Beta')
axs[0].set_ylabel('Heat Capacity')

axs[1].plot(beta, CVs)
axs[1].set_title('Heat Capacity vs Beta Untrained')
axs[1].set_xlabel('Beta')
axs[1].set_ylabel('Heat Capacity')

plt.tight_layout()
plt.show()