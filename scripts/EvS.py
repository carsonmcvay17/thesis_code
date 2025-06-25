# imports
import numpy as np
import matplotlib.pyplot as plt

from sources.ECVS import ECVS
from sources.Networks import NN

model = ECVS()
nn = NN()

_, act_train = nn.trained_network(15)
trained_E, counts_trained = model.calc_energies(act_train)
trained_S = model.calc_Entropy(trained_E, counts_trained)

_, act_untrain = nn.untrained_network(15)
untrained_E, counts_untrained = model.calc_energies(act_untrain)
untrained_S = model.calc_Entropy(untrained_E, counts_untrained)

plt.scatter(trained_E, trained_S)
plt.scatter(untrained_E,untrained_S)
plt.xscale('log')
plt.yscale('log')
plt.title("Energy vs Entropy")
plt.legend(['trained network', 'untrained network'])
plt.xlabel('Energy')
plt.ylabel('Entropy')
plt.show()
