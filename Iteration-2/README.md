# LESCO-Iteration-2
This is the second iteration over LESCO video classification. This time CNNs and RNNs are used to train models against the data generated at iteration 1

## 1-1D-Convolutional-NeuralNet-LESCO.ipynb

A 1-dimensional convolutional neural network was used to train a simple model: results train (91%) test (56%) : summary - training set overfitting

## 2-SimpleRNN-LESCO.ipynb

A a SimpleRNN recurrent neural network was used instead of the 1D CNN: results train (94%) test (47%) : summary - training set overfitting

## 1-1D-Convolutional-NeuralNet-LESCO.ipynb

SimpleRNN cell was replaced with LSTM cell: results train (97%) test (51%) : summary - training set overfitting

## 1-1D-Convolutional-NeuralNet-LESCO.ipynb

LSTM cell was replaced with GRU cell: results train (99%) test (57%) : summary - training set overfitting

# Summary

Seems that RNNs are the way to go (if we have more data samples)! The training set overfitting tells it requires many more LESCO videos in the training set to improve generalization. This is pretty much the same result I obtained in my doctoral dissertation. 

If we compare Iteration 2 with Iteration 1, we can tell that Manhattan Distance for LESCO classification is our best option for now, when data is scarse.

# Whats Next

Iteration 3 will revisit Iteration 1 for improvements. Some things to try:
- use t-SNE instead of PCA
- use Singular Value Decomposition instead of PCA.
- Test to see if there is an improvement in test set accuracy.
