---
layout: post
comments: true
mathjax: true
title: “Deep learning without going down the rabbit holes too early.”
excerpt: “How to learn deep learning from easy concept to complex idea? How to build insight along the way?”
date: 2017-03-01 14:00:00
---
** This is work in progress **

### What is deep learning?
People may approach deep learning by explaining the neural network in our brain. Historically, this is where deep learning gets its insight.  Nevertheless, once you pass this and realize building a deep learning network is about building a **function estimator**, you will unveil the real potential of deep learning in AI.
 
Let us build a new andriod named Pieter. Our first task is to teach Pieter visual recognition. Can the visual system in our brain be replaced by a big function estimator? Can we read the pixel value of an image, pass it to a function and calculate the chance that it is a school bus, an airplane or a truck etc ...

<div class="imgcap">
<img src="/assets/dl_intro/deep_learner.jpg" style="border:none;">
</div>

Indeed, in our later example of hand writing recognition, we will build a system very similar to the following:
<div class="imgcap">
<img src="/assets/dl_intro/fc.jpg" style="border:none;">
</div>

with the simple equations
$$
f(x)
$$ be:

$$
z_j =  \sum_{j} W_{ij}  x_i + b_j_
$$

$$
f(x) =  max(0, z_{ij})
$$

We will miss a few more pieces, but the fundermental is there. The system will recognize the zip code written in our letter's envelop with reasonable high successful rate.

### Build a Linear regression model

### Learning from mistakes

#### Gradient descent

#### Backpropagation

#### Learning rate

### Non-linearity

### Classifier

#### Logistic regression (Sigmoid)

### Deep learing network (Fully-connected layers)

#### Sigmoid classifier

#### Mean square error 

### Backpropagation

### Issues

### Exploding and vanishing gradient

### Cross entropy cost function

### Softmax classifier

### Log likelihood

### Activation function
#### Sigmoid
#### ReLU
#### tanh

### Network layers

### Implementation

### Mini-batch gradient descent

### Overfit

### Regularization
#### Train/validation accuracy
#### L0, L1, L2 regularization
#### Gradient clipping
#### Dropout

### Weight initialization

### Insanity check
#### Gradient checking
#### Initial loss
#### Without regularization and with small dataset

### Trouble shooting
#### Plotting loss
#### Train/validation accuracy
#### Ratio of weight updates
#### Activation per layer
#### First layer visualization

### Cost function
#### MSE
#### Cross entropy, Negative likelihood
#### Margin loss/hinge loss/SVM
#### L2 Loss vs softmax

### Training parameters
#### Momentum update
#### Adagrad
#### Adam
#### Rate decay

### Data preprocessing
#### Scaling (Mean/normalization)
#### Whitening 

### Batch normalization

### Hyperparameter tuning

#### Cross validation
#### Random search

### CNN

### LSTM

### Backprogation

### Data argumentation

### Model ensembles
























