# imports
import numpy as np
import math

class ECVS:
    """
    This class contains code to get energy, calculate heat capacity, 
    and calculate entropy
    """

    def calc_energies(self, activations: np.ndarray)->np.ndarray:
        """
        This function calculates the energy of the activations of the NN
        Arguments:
        activations: array of the activations of the NN

        Returns: 
        Es: np.ndarray that stores the values of the energies of the activations
        """
        counts = {}
        for act in activations:
            foo = [int((np.sign(i)+1)/2) for i in act[0]] # discretizes to binary
            str_foo = ''
            for f in foo:
                str_foo += str(f) # binary string
            # counts each unique binary string
            if str_foo in counts:
                counts[str_foo] += 1
            else:
                counts[str_foo] = 1
        
        # counts lists are energies
        Es = []
        num_things = sum(counts.values())

        # computes relative probability of each "energy"
        # lower energy states more likely
        for key in counts:
            ns = counts[key]
            prob = ns/num_things
            Es.append(-np.log(prob))
        return Es
    
    
    def calc_cv(self, T: np.ndarray, Es: np.ndarray)->np.ndarray:
        """
        Calculates heat capacity
        Arguments:
        T: an array of temperatures
        Es: an array of energies

        Returns:
        cvs: array of heat capacities
        """
        beta = [1/t for t in T]
        cvs = []
        
        for b in beta:
            Eground = np.min(Es)
            # calculate p_beta
            top = [math.exp(-b*(x-Eground)) for x in Es]
            pbeta = [n/sum(top) for n in top]
            Es_squared = [x**2 for x in Es]
            res_list = []
            res_list2 = []
            for i in range(0, len(pbeta)):
                res_list.append(pbeta[i] * Es_squared[i])
                res_list2.append(pbeta[i] * Es[i])
            cvs.append(b ** 2 * (sum(res_list)-(sum(res_list2) ** 2))) # if this looks wrong check parentheses
        return cvs
            
