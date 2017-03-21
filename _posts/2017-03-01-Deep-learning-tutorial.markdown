---
layout: post
comments: true
mathjax: true
title: “Deep learning without going down the rabbit holes too early.”
excerpt: “How to learn deep learning from easy concept to complex idea? How to build insight along the way?”
date: 2017-03-01 14:00:00
---
**This is work in progress**

### What is deep learning?
**Deep learing is about building a function estimator.** Historically, people describe deep learning through the neural network in our brain. Indeed, this is where deep learning gets its insight.  Nevertheless, deep learning has out grown this explaination. Once you realize building a deep learning network is about building a function estimator, you will unveil its real potential in AI.
 
Let us build a new andriod named Pieter. Our first task is to teach Pieter how to recognize visual objects. Can the visual system in our brain be replaced by a big function estimator? Can we pass the pixel values of an image to a function and calculate the chance that it is a school bus, an airplane or a truck etc ...?

<div class="imgcap">
<img src="/assets/dl_intro/deep_learner.jpg" style="border:none;">
</div>

Indeed, in our later example of hand writing recognition, we will build a system very similar to the following:
<div class="imgcap">
<img src="/assets/dl_intro/fc.jpg" style="border:none;">
</div>

For every node, we compute
$$
f(x)
$$ 
:

$$
f(z_j) = \frac{1}{1 + e^{-z_j}}
$$

which, 

$$
z_j = \sum_{i} W_{ij} x_{i} + b_{i}
$$

which
$$
x_{i}
$$ 
represents the pixel value i for the first hidden layer or the output from the previous layer.
These equation looks intimidating. But let us go through one example to illustrate how simple it is. For example, for a grayscale image with just 4 pixels (0.1, 0.3, 0.2, 0.1) and weight (0.3, 0.2, 0.4, 0.3) and bias (-0.8), the output of the first node circled in red will be:

$$
z_j =  0.3*0.1 + 0.2*0.3 + 0.4*0.2 + 0.3*0.1  - 0.8 = -0.6
$$

$$
f(z) =  \frac{1}{1 + e^{-(-0.6)}} = 0.3543
$$

Each node will have its own set of weight (W) and bias (b). From the left most layer, we compute the output of each node and we feed forward the output to the layer on the right. Eventually, the right most layer is the likeliness for each object classification (a school bus, an airplane or a truck). In this exercise, we supply the weight and bias values for each node to our android Pieter. But as the term "deep learning" may imply, by the end of this tutorial, Pieter will manage to learn those parameters by himself. We still miss a few pieces of the puzzle, and we will go deeper into the details later. But the network diagram above and the equations already lay down the fundation of a deep learning network. In fact, this design can recognize the zip code written on a envelop with reasonable high accuracy.

#### XOR
For the skeptics, we will build an exclusive "or" (a xor b) using a similar approach:
<div class="imgcap">
<img src="/assets/dl_intro/xor.jpg" style="border:none;">
</div>
with the same equations:

$$
h_j = \sigma(z) = \frac{1}{1 + e^{-z_j}}
$$

$$
z_j =  \sum_{i} W_i * x_i + b_i
$$

The following is the code implementing which is pretty easy to understand without further explanation.
```python
import numpy as np

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def layer1(a, b):
    w11, w21, b1 = 20, 20, -10
    w12, w22, b2 = -20, -20, 30

    v = w11*a + w21*b + b1
    h11 = sigmoid(v)

    v = w12*a + w22*b + b2
    h12 = sigmoid(v)

    return h11, h12

def layer2(a, b):
    w1, w2, bias = 20, 20, -30

    v = w1*a + w2*b + bias
    return sigmoid(v)

def xor(a, b):
    h11, h12 = layer1(a, b)
    return layer2(h11, h12)

print(" 0 ^ 0 = %.2f" % xor(0, 0))   # 0.00
print(" 0 ^ 1 = %.2f" % xor(0, 1))   # 1.00
print(" 1 ^ 0 = %.2f" % xor(1, 0))   # 1.00
print(" 1 ^ 1 = %.2f" % xor(1, 1))   # 0.00
```
And the XOR output matches with its expected logical value:
```
 0 ^ 0 = 0.00
 0 ^ 1 = 1.00
 1 ^ 0 = 1.00
 1 ^ 1 = 0.00
```
#### Delta function
Back to the basic calculus, we know that functions can be construct with infinite narrow rectangles. Can we use the technique above to construct such narrow shaped rectangles. (a.k.a. delta function)

<div class="imgcap">
<img src="/assets/dl_intro/delta.png" style="border:none;width:50%">
</div>

Here is the code listing using the same set of equations and network layoyt:
```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def layer1(x):
    h11 = sigmoid(1000 * x - 400)
    h12 = sigmoid(1000 * x - 500)
    return h11, h12

def layer2(v1, v2):
    return sigmoid(0.8 * v1 - 0.8 * v2)

def func_estimator(x):
    h11, h12 = layer1(x)
    return layer2(h11, h12)

x = np.arange(0, 3, 0.001)
y = func_estimator(x)

plt.plot(x, y)
plt.show()
```
Which output something similar to a delta function:
<div class="imgcap">
<img src="/assets/dl_intro/delta_func.png" style="border:none;width:60%">
</div>

Discussing XOR or the delta function is not significant in deep learning.  But through this exercise, we demonstrate the potential of building a complex function estimator through a network of simple computation nodes.

### Build a Linear regression model


### Learning from mistakes
**Deep learing is about learning from mistakes.**

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
























