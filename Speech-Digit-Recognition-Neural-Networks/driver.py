import numpy as np
import matplotlib.pyplot as plt
from lib.timit import Timit
import os
import tensorflow as tf
from DigitGenerator import DigitGenerator
from MultiDigitGenerator import MultiDigitGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import datasets, layers, models
import pydot, graphviz

def main():
    #mode variable determines if you run the CNN, or the LSTM
    #'cnn' is CNN, 'rnn' is LSTM as well as 'lstm'
    mode = 'cnn'

    #Directory of audio files, grabs all the subfolders
    #Change directory as needed
    timit = Timit('../tidigits_allaudio/')
    #Combine all the training mens and women filenames into one group
    men_filenames = timit.get_filenames('train','man')
    women_filenames = timit.get_filenames('train','woman')
    training_filenames = list()
    training_filenames.extend(men_filenames)
    training_filenames.extend(women_filenames)
    #Divide files with only one digit from the ones with 2,3,4,5 or 7
    single_digits = [file for file in training_filenames if len(os.path.split(file)[1]) <= 6]
    multiple_digits = [file for file in training_filenames if len(os.path.split(file)[1]) > 6]
    '''
    This code was put in for using a validation generator, it didnt provide any benefit in the final results


    #np.random.shuffle(single_digits)
    #sd_len = len(single_digits)
    #validation_digits = DigitGenerator(single_digits[int(.75 * sd_len):],mode)


    
    '''
    #Call DigitGenerator with filenames and mode
    train_digits = DigitGenerator(single_digits,mode)
    
    #Testing filesnames grouped together
    test_men_filenames = timit.get_filenames('test','man')
    test_women_filenames = timit.get_filenames('test','woman')
    testing_filenames  = list()
    testing_filenames.extend(test_men_filenames)
    testing_filenames.extend(test_women_filenames)
    #Divide testing files
    test_single_digits = [file for file in testing_filenames if len(os.path.split(file)[1]) <= 6]
    test_multiple_digits = [file for file in testing_filenames if len(os.path.split(file)[1]) > 6]
    #Test Digits Generator for evaluation
    test_digits = DigitGenerator(test_single_digits,mode)

    #Large Batch Size so I can run WER calculation on enough data
    #Multi Digit Generator is for outputing trimmed multidigit audio files
    #Evaluation boolean determines whether to filter based on correct # of digit classification, for TF evalution
    multi_test_digits = MultiDigitGenerator(test_multiple_digits,mode,stream = 'rms',evaluate = True)
    #Grab the first big batch to test on trained network
    #X,y = multi_test_digits.__getitem__(0)
    #multi_test_digits2 = MultiDigitGenerator(test_multiple_digits,mode,stream = 'dft')
    
    #model is determined by which mode
    model = models.Sequential()
    if mode == 'cnn':       
        model.add(layers.Conv2D(16, (7, 7), activation='relu', input_shape=(50, 48, 1),kernel_regularizer='l2'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(16, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(11, activation='softmax'))
    if mode == 'rnn' or mode == 'lstm':
        model.add(layers.Masking(input_shape=(50,48)))
        model.add(layers.LSTM(64))
        model.add(layers.Dense(11, activation='softmax'))
    model.compile(optimizer='adam',loss=[WER_Loss],metrics=[WER],run_eagerly=True)
    '''
    leftover code from print model summary and image
    #layer_outputs = [layer.output for layer in model.layers]
    #activation_model = models.Model(inputs = model.input, outputs = layer_outputs) 
    #dot_img_file = '../model_1.png'
    #tf.keras.utils.plot_model(activation_model, to_file=dot_img_file, show_shapes=True)

    '''
    #Fit model using 10 epochs
    model.fit(train_digits,epochs = 4)
    #Evaluate Model
    model.evaluate(test_digits)
    model.evaluate(multi_test_digits)
    


def WER(y_true, y_pred):
    #tensorflow code
    if(len(y_pred.shape) > 1):
        y_true = tf.argmax(y_true,axis = 1).numpy()
        y_pred = tf.argmax(y_pred,axis = 1).numpy()
    levDist = levenshteinDistance(y_true,y_pred)
    return 1.0 - (levDist/len(y_true))

def WER_Loss(y_true, y_pred):
    if(len(y_true) == len(y_pred)):
        return tf.keras.losses.categorical_crossentropy(y_true,y_pred)
    return -1.0

def levenshteinDistance(arr1,arr2):

#Code for the Levenshtein Distance
#Adapted code for strings and used it for arrays
    len1 = len(arr1)
    len2 = len(arr2)
    len_arr = np.zeros((len1+1,len2+1))
    for x1 in range(len1 + 1):
        len_arr[x1][0] = x1
    for x2 in range(len2 + 1):
        len_arr[0][x2] = x2
    a = 0
    b = 0
    c = 0
    
    for x1 in range(1, len1 + 1):
        for x2 in range(1, len2 + 1):
            if (arr1[x1-1] == arr2[x2-1]):
                len_arr[x1,x2] = len_arr[x1 - 1,x2 - 1]
            else:
                a = len_arr[x1,x2 - 1]
                b = len_arr[x1 - 1,x2]
                c = len_arr[x1 - 1,x2 - 1]
                if (a <= b and a <= c):
                    len_arr[x1,x2] = a + 1
                elif (b <= a and b <= c):
                    len_arr[x1,x2] = b + 1
                else:
                    len_arr[x1,x2] = c + 1
    return len_arr[len1,len2]

#Longest Common Subsequence, can be used to gather number of Deletions and Insertions
def lcs(arr1,arr2):
    len1 = len(arr1)
    len2 = len(arr2)
    len_arr = np.zeros((len1+1,len2+1))
    for i in range(len1+1):
        for j in range(len2 + 1):
            if(i == 0 or j == 0):
                len_arr[i,j] = 0
            elif(arr1[i-1] == arr2[j-1]):
                len_arr[i,j] = len_arr[i-1,j-1] + 1
            else:
                len_arr[i,j] = max(len_arr[i-1,j],len_arr[i,j-1])
    return len_arr[len1,len2]

if __name__ == '__main__':
    plt.ion()
    
    main()
