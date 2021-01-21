import numpy as np
import os
from lib.audioframes import AudioFrames
from lib.dftstream import DFTStream
from lib.endpointer import Endpointer
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical

#Digit Generator for individual data
class DigitGenerator(Sequence):

    def __init__(self, filenames, model_type = 'rnn',cutoff_width = 50, batch_size=32,
                 shuffle=True):
        self.filenames = np.array(filenames)
        self.batch_size = batch_size
        self.idx = np.arange(len(self.filenames))
        self.shuffle = shuffle
        self._epoch = -1
        #Model Type
        self.model_type = model_type
        self.cutoff_width = cutoff_width
        self.adv_ms = 10
        self.len_ms = 20
        self.melN = 48
        
        self.digits = ['z','1','2','3','4','5','6','7','8','9','o']  # List of digits
        self.digitsN = len(self.digits)
        self.digit_dict = dict(map(lambda x: (x[1], x[0]), enumerate(self.digits)))
        self.on_epoch_end()


    def __getitem__(self, batch_idx):
        #indexes of the shuffled batch
        idxs = self.idx[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size]
        #grab retrospective spectrograms from audiostream of file
        audioframes = [AudioFrames(file,self.adv_ms,self.len_ms) for file in self.filenames[idxs]]
        dftstreams = [np.array(list(DFTStream(af,'Mel',self.melN))) for af in audioframes]
        trimmed_dfts = [dft[Endpointer(dft).speech_indices(dft)][0:self.cutoff_width] for dft in dftstreams]
        #This code converts the filename into the correct softmax output
        digits = [self.digit_dict[os.path.split(file)[1][0]] for file in self.filenames[idxs]]
        X = np.zeros((self.batch_size,self.cutoff_width,self.melN))
        #pad and normalize dft
        i = 0
        for dft in trimmed_dfts:
            X[i][0:dft.shape[0]][:][:] = (dft - dft.min())/(dft.max()- dft.min())
            i = i + 1
        y = to_categorical(digits,self.digitsN)
        #CNN requires 4D output
        if(self.model_type == 'cnn'):
            X = X[:,:,:,None]
        return X,y


    @property
    def epoch(self):
        return self._epoch
    def epoch(self,value):
        self._epoch = value


    def __len__(self):
        return len(self.filenames)//self.batch_size


    def on_epoch_end(self):
        self._epoch += 1  # Note next epoch
        if self.shuffle:
            np.random.shuffle(self.idx)  # reshuffle the data

