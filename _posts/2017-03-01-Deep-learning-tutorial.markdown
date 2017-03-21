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

with
$$
f(x)
$$ 
be:

$$
f(x) =  max(0, z_{j})
$$

$$
z_j =  \sum_{i} W_{ij}  x_i + b_j which x_i is the value in each pixel.
$$

We will miss a few more pieces, but the fundermental is there. This system can recognize the zip code written in a letter's envelop with reasonable high successful rate.

#### XOR
For the skeptics, we will build an exclusive "or" (XOR) using the following networks:
<div class="imgcap">
<img src="/assets/dl_intro/xor.jpg" style="border:none;width:50%">
</div>
and the corresponding equations:

$$
h_j = \sigma(z) = \frac{1}{1 + e^{-z_j}}
$$

$$
z_j =  \sum_{i} W_i * x_i + b_i
$$

The following is the code listing implementing the network above.
```python
import numpy as np

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def layer1(a, b):
    w11, w21, b1 = 20, 20, -10
    w12, w22, b2 = -20, -20, 30

    v = w11 * a + w21 * b + b1
    h11 = sigmoid(v)

    v = w12 * a + w22 * b + b2
    h12 = sigmoid(v)

    return h11, h12

def layer2(a, b):
    w1, w2, bias = 20, 20, -30

    v = w1 * a + w2 * b + bias
    return sigmoid(v)


def xor(a, b):
    h11, h12 = layer1(a, b)
    return layer2(h11, h12)

print("%.2f" % xor(0, 0))   # 0.00
print("%.2f" % xor(0, 1))   # 1.00
print("%.2f" % xor(1, 0))   # 1.00
print("%.2f" % xor(1, 1))   # 0.00
````
And the XOR output:
```
 0 ^ 0 = 0.00
 0 ^ 1 = 1.00
 1 ^ 0 = 1.00
 1 ^ 1 = 0.00
```
#### Delta function
Back to the basic calculus, we know that functions can decomposed to narrow rectangles. Can we use the technique above to construct such function. (aka delta function)

<div class="imgcap">
<img src="/assets/dl_intro/delta.png" style="border:none;;width:50%">
</div>

Here is the code listing:
```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def f_layer1(x):
    h11 = sigmoid(1000 * x - 400)
    h12 = sigmoid(1000 * x - 500)
    return h11, h12

def f_layer2(v1, v2):
    return sigmoid(0.8 * v1 - 0.8 * v2)

def func_estimator(x):
    h11, h12 = f_layer1(x)
    return f_layer2(h11, h12)

x = np.arange(0, 3, 0.001)
y = func_estimator(x)

plt.plot(x, y)
plt.show()
```
Output:
<div class="imgcap">
<img src="/assets/dl_intro/delta_func.png" style="border:none;">
</div>


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
























