import numpy as np
import os
from lib.audioframes import AudioFrames
from lib.dftstream import DFTStream
from lib.endpointer import Endpointer
from lib.rmsstream import RMSStream
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical

#This generator take filenames of wav files with multiple files and breaks it into individual digit spectrograms


class MultiDigitGenerator(Sequence):

    def __init__(self, filenames, model_type = 'rnn',cutoff_width = 50,stream = 'rms',evaluate = False, batch_size=32,
                 shuffle=True):
        #All Files Mode if true just takes all files, divides them into however many spectrogram the Endpointer determines
        #Therefore X,y and are going to be different sizes
        #if All Files Mode is false then it discards files that dont divide correctly, not to be used anymore
        self.allFilesMode = evaluate
        self.filenames = list()
 
        if self.allFilesMode:
            self.filenames = filenames
        
        #General AudioFrame and DFTStream Settings
        self.adv_ms = 10
        self.len_ms = 20
        self.melN = 48
        self.speech_idxs = list()
        self.lens = list()


        #determines whether to use Endpointer on DFT or RMS, can change accuracy of idx splits
        self.stream = stream


        '''
This code is meant to filter out invalidly seperated audio files, basically ones where the audio is spoken too fast to properly guess the correct # of digits. This is not used in the final project, however I left it here because this was used in the Final Presentation


It also tosses the file when a digit is less than 15 frames long (that was the min size of any training digit)
Then these files are appended to the list
The idxs of their speech is saved, not just to save computational but because if you calculate it again the result can sometimes differ slightly leading to incompatable array sizes

This is only to make sure that model.evaluate works because they arrays are the same size


        '''
        #split based on DFT
        if not(self.allFilesMode) and (self.stream == 'dft'):
            for file in filenames:
                af = AudioFrames(file,self.adv_ms,self.len_ms)
                dft = np.array(list(DFTStream(af,'Mel',self.melN)))
                idxs = Endpointer(dft).speech_indices(dft)
                file_end = os.path.split(file)[1]
                splits = np.where(np.diff(idxs) > 1)[0]
                valid = True
                last_idx = 0
                for idx in splits:
                    if idx == 0 or (idxs[idx] - idxs[last_idx]) < 15:
                        valid = False
                        break
                    last_idx = idx
                if valid and (idxs[-1] - idxs[last_idx]) < 15:
                    valid = False
                if valid and  (len(splits) + 1 == len(file_end) - 5):
                    self.filenames.append(file)
                    self.speech_idxs.append(idxs)
                    self.lens.append(len(file_end) - 5)
        #split based on RMS
        if not(self.allFilesMode) and (self.stream == 'rms'):
            for file in filenames:
                af = AudioFrames(file,self.adv_ms,self.len_ms)
                rms = np.array(list(RMSStream(af)))
                idxs = Endpointer(rms.reshape(1,-1).T).speech_indices(rms.reshape(1,-1).T)
                file_end = os.path.split(file)[1]
                splits = np.where(np.diff(idxs) > 1)[0]
                valid = True
                last_idx = 0
                for idx in splits:
                    if idx == 0 or (idxs[idx] - idxs[last_idx]) < 15:
                        valid = False
                        break
                    last_idx = idx
                if valid and (idxs[-1] - idxs[last_idx]) < 15:
                    valid = False
                if valid and  (len(splits) + 1 == len(file_end) - 5):
                    self.filenames.append(file)
                    self.speech_idxs.append(idxs)
                    self.lens.append(len(file_end) - 5)
        #convert to np arrays for indexing purposes
        self.speech_idxs = np.array(self.speech_idxs) 
        self.lens = np.array(self.lens)
        self.filenames = np.array(self.filenames)
    
        self.batch_size = batch_size
        self.idx = np.arange(len(self.filenames))
        self.shuffle = shuffle
        self._epoch = -1
        self.model_type = model_type

        #Cutoff Width is Max Length of spectrogram
        self.cutoff_width = cutoff_width

        
        
        self.digits = ['z','1','2','3','4','5','6','7','8','9','o']  # List of digits
        self.digitsN = len(self.digits)
        self.digit_dict = dict(map(lambda x: (x[1], x[0]), enumerate(self.digits)))
        self.on_epoch_end()


    def __getitem__(self, batch_idx):
        idxs = self.idx[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size]
        #grab audioframes and dftstreams, the usual
        audioframes = [AudioFrames(file,self.adv_ms,self.len_ms) for file in self.filenames[idxs]]
        dftstreams = [np.array(list(DFTStream(af,'Mel',self.melN))) for af in audioframes]
        #speech idxs are not calculated yet if All Files Mode is true
        if (self.allFilesMode) and (self.stream == 'dft'):
            speech_idxs = [Endpointer(dft).speech_indices(dft) for dft in dftstreams]
        elif (self.allFilesMode) and (self.stream == 'rms'):
            rmsstreams = [np.array(list(RMSStream(AudioFrames(file,self.adv_ms,self.len_ms)))) for file in self.filenames[idxs]]
            speech_idxs = [Endpointer(rms.reshape(1,-1).T).speech_indices(rms.reshape(1,-1).T) for rms in rmsstreams]
        else:
            speech_idxs = self.speech_idxs[idxs]
        trimmed_dfts = list()
        '''
            Loop through files and split along where there is a gap in speech indices
            if the number of frames of speech in a block is less than 15 then it is assumed to be a part of another digit
            splits is array that tracks index of starts and ends
            add to list of trimmed dfts
        '''
        for dft,speech in zip(dftstreams,speech_idxs):
            splits = np.where(np.diff(speech) > 1)[0]
            last_idx = 0
            for idx in splits:
                if(speech[idx] - speech[last_idx] < 15):
                    continue
                temp_dft = dft[speech[last_idx]:speech[idx]][0:self.cutoff_width]
                trimmed_dfts.append(temp_dft)
                last_idx = idx
            temp_dft = dft[speech[last_idx]:speech[-1]][0:self.cutoff_width]
            trimmed_dfts.append(temp_dft)
        

        #code to get correct file sequence out of filenames
        multidigits = [os.path.split(file)[1][0:-5] for file in self.filenames[idxs]]
        digits = [self.digit_dict[digit] for digits in multidigits for digit in digits]
        X = np.zeros((len(trimmed_dfts),self.cutoff_width,self.melN))
        #normalize and pad spectrogram
        i = 0
        for dft in trimmed_dfts:
            X[i][0:dft.shape[0]][:][:] = (dft - dft.min())/(dft.max()- dft.min())
            i = i + 1
        y = to_categorical(digits,self.digitsN)
        #Take 3D array to 4D array for CNN
        if(self.model_type == 'cnn'):
            X = X[:,:,:,None]
        return X,y


    #This function was for length analysis, its not useful if all files mode is used
    def __getlen__(self, batch_idx):
        z = list()
        idxs = self.idx[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size]
        lens = self.lens[idxs]
        for s in lens:
            for t in np.arange(s):
                z.append(s)
        return np.array(z)
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

