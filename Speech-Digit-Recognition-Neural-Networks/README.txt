This project trains and tests Neural Networks for speech recognition of digits. Then it takes the trained networks and applies them to audio files with strings of digits. In order to run it requires the TIDIGIT's All Audio Corpus. This directory can be downloaded at https://catalog.ldc.upenn.edu/LDC93S10 .

The networks take in an audio spectogram, one network uses a Convolutional Neural Network architecture to classify digits and the other uses an Recurrent Neural Network on each time frame of the spectrogram to classify the digits. The two architectures have similar accuracies on single digit files (97% Testing Accuracy for CNN, 96% Testing Accuracy for RNN). The RNN outperforms the CNN on multidigit files, this is in part due to the fact that the accuracy is limited by how well the data is trimmed, such as when two digits get stuck together. Recurrent Neural Networks that aren't bidirectional would guess the last digit, while a CNN might be unable to correctly process the spectrogram.
To improve the multidigit accuracy, the audio would need to be broken down at the phoneme level. Currently this project uses a untrained Gaussian Mixture Model to determine speech from sound and split into strings of digits. This works well for slower spoken files. The accuracy metric used is Word Error Rate (WER) which uses the Levenshtein Distance to give the accuracy rate for arrays of differing sizes.
Because custom accuracy and loss metrics are use, this requires Tensorflow version 2.0 or greater. The classical categorical cross entropy loss function doesn't work on two different sized arrays. In this code I just return -1 as the minimal difference in information between the two arrays becomes a very difficult problem to solve and is not required s no training is done on multiple digits.

Instructions

1. Open driver.py and edit the directory to the TIDIGIT's AllAudio Directory
2. Choose what Neural Network you want to train and evaluate, edit the mode variable on driver.py 
    'cnn' for using Convolutional Neural Network, 'rnn' for Recurrent Neural Network
3. Run driver.py from terminal
