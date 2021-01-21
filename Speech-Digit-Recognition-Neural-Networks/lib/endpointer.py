

from sklearn.mixture import GaussianMixture
import numpy as np


class Endpointer:
    """
    Endpointer
    Unsupervised voice activity detector
    
    Uses a GMM to learn a two class distribution of RMS energy vectors
    
    """
    Nclasses = 2    # two classes, speech/noise
        
    def __init__(self, train):
        """
        Endpointer(iterable) - Create an endpointer that is trained from an instance of a class
        that iterates to produce features
        :param train:  Numpy N x D matrix of N examples of D dimensional features
        """

        self.gmm = GaussianMixture(self.Nclasses)
        self.gmm.fit(train)

        # Determine which mixture was which  
        self.mixture_labels = dict()

        dim = train.shape[1]   # dimensionality of feature
        for extremumfn, label in [ [np.argmin, "noise"], [np.argmax, "speech"]]:
            if dim > 1:
                # DFT coefficient is the sum of all the other bins and is thus proportional to total energy
                energy = self.gmm.means_[:,0]
                self.mixture_labels[label] = extremumfn(energy)
            else:
                # scalar case: Assume speech is largest mean value
                self.mixture_labels[label] = extremumfn(self.gmm.means_)




    def predict(self, features):
        """
        predict
        :param features: Numpy N x D matrix of N examples of D dimensional features
        :return: binary vector, True for frames classified as speech
        """
        """predict - Extract RMS from file using the same
        framing parameters as the constructor and return vector
        of booleans where True indicates that speech occurred
        """

        # Get mixture membership predictions
        decisions = self.gmm.predict(features)

        # Convert to speech predicate vector
        speech = decisions == self.mixture_labels["speech"]
        
        return speech

    def speech_indices(self, features):
        """
        speech_indices - Return indices predcited to contain speech
        :param features:
        :return:
        """

        # Get predictions
        speech_indicator = self.predict(features)
        # Find where speech is true
        return np.where(speech_indicator == True)[0]

            
    