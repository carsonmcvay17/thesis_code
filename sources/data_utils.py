# imports
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from sources.ECVS import ECVS
from sources.Networks import NN

class Utils:
    """
    This is a class that stores all the functions to handle and clean up data
    """

    def normalize_img(self, image: np.ndarray, label: str)->np.ndarray:
        """
        Normalizes images
        Arguments:
        image: Pretty sure it's an array
        label: string? I think. Tbh I'm not checking

        Returns: Normalized image probably an array
        """
        return tf.cast(image, tf.float32) / 255., label
    
    def confidence_intervals(self, T: np.ndarray, num: int, n: int, nn_type:str='trained')-> tuple:
        """
        Calculates the confidence intervals of heat capacity plots
        Args:
        T: array of temperatures
        num: integer number of times to calculate heat capacity
        n: number of neurons
        nn_type: trained or untrained default trained

        Returns:
        left_cis: array of the lowest 5 percent 
        right_cis: array of highest 5 percent
        mean: array of the means
        """
        left_cis = []
        right_cis = []
        mean = []

        # calc cv num times
        cv_counts = []
        model = ECVS()
        nn = NN()
        for i in range(num):
            if nn_type == 'trained':
               _, activations = nn.trained_network(n)
            elif nn_type == 'untrained':
               _, activations == nn.untrained_network(n)
            else:
                print("you can't do that. Pick trained or untrained. Running with trained")
                _, activations = nn.trained_network

            Es, _ = model.calc_energies(activations)
            cv_counts.append(model.calc_cv(T, Es))

        # cv counts is an array of the arrays of cv for each temp
        # cv counts has length num
        # loop over each temperature
        for t in range(len(T)):
            temp_cv = []
            for hc in cv_counts:
                temp_cv.append(hc[t])
            left_cis.append(np.percentile(temp_cv, 5))
            right_cis.append(np.percentile(temp_cv, 95))
            mean.append(np.mean(temp_cv))
        return left_cis, right_cis, mean
    
    
    def remove_small_x(self, Es: np.ndarray, S: np.ndarray, xmin: int)->tuple[np.ndarray, np.ndarray]:
        """
        Don't really remember why I needed to do this, but it's in the code so I'm writing a function
        Gets rid of data points less than xmin
        Args:
        new_E: array of energies
        new_S: array of entropies
        xmin: min x value

        Returns:
        Es: Es with values removed
        S: S with values removed
        """
        new_E = []
        new_S = []
        for i in range(len(S)):
            if S[i]>= xmin:
                new_S.append(S[i])
                new_E.append(Es[i])
        return new_E, new_S
    
    def dataset(self)->tuple:
        """
        Makes the dataset so that you don't have to do it over and over again. Also has all the nice normalizing and such

        Returns:
        ds_train: training dataset
        ds_test: testing dataset
        """
        (ds_train, ds_test), ds_info = tfds.load(
            'mnist',
            split=['train', 'test'],
            shiffle_files=True,
            as_supervised=True,
            with_info=True
        )
        ds_train = ds_train.map(
        self.normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
        ds_train = ds_train.cache()
        ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
        ds_train = ds_train.batch(128)
        ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

        s_test = ds_test.map(
        self.normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
        ds_test = ds_test.batch(128)
        ds_test = ds_test.cache()
        ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

        return ds_train, ds_test
