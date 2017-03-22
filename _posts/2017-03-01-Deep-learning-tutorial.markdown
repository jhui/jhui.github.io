---
layout: post
comments: true
mathjax: true
title: “Deep learning without going down the rabbit holes.”
excerpt: “How to learn deep learning from easy concept to complex idea? How to build insight along the way?”
date: 2017-03-01 14:00:00
---
**This is work in progress...**

### What is deep learning (DL)?
**Deep learing is about building a function estimator.** Historically, people describe deep learning (DL) using the neural network in our brain. Indeed, this is where deep learning gets its insight.  Nevertheless, deep learning has out grown this explaination. Once you realize building a deep learning network is about building a function estimator, you will unveil its real potential in AI.
 
Let us build a new andriod named Pieter. Our first task is to teach Pieter how to recognize visual objects. Can the visual system in our brain be replaced by a big function estimator? Can we pass the pixel values of an image to a function and calculate the likeliness that it is a school bus, an airplane or a truck etc ...?

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

with, 

$$
z_j = \sum_{i} W_{ij} x_{i} + b_{i}
$$

which
$$
x_{i}
$$ 
represents the output from the previous layer or the pixel value i for the first hidden layer. 
These equation looks intimidating. But let us go through one example to illustrate how simple it is. For example, with weight W  (0.3, 0.2, 0.4, 0.3), bias b (-0.8) and a grayscale image with just 4 pixels (0.1, 0.3, 0.2, 0.1), the output of the first node circled in red above will be:

$$
z_j =  0.3*0.1 + 0.2*0.3 + 0.4*0.2 + 0.3*0.1  - 0.8 = -0.6
$$

$$
f(z) =  \frac{1}{1 + e^{-(-0.6)}} = 0.3543
$$

Each node will have its own set of weight (W) and bias (b). From the left most layer, we compute the output of each node and we feed forward the output to the next layer. Eventually, the right most layer is the likeliness for each object classification (a school bus, an airplane or a truck). In this exercise, we supply all the weight and bias values to our android Pieter. But as the term "deep learning" may imply, by the end of this tutorial, Pieter will learn those parameters by himself. We still miss a few pieces for the puzzle. But the network diagram and the equations above already lay down the foundation of a deep learning network. In fact, this simple design can recognize the zip code written on a envelop with reasonable high accuracy.

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

The following is the code implementation. It is self-explainatory and a working implementation can help you to identify any mis-conceptions.
```python
import numpy as np

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def layer1(x1, x2):
    w11, w21, b1 = 20, 20, -10
    w12, w22, b2 = -20, -20, 30

    v = w11*x1 + w21*x2 + b1
    h11 = sigmoid(v)

    v = w12*x1 + w22*x2 + b2
    h12 = sigmoid(v)

    return h11, h12

def layer2(x1, x2):
    w1, w2, bias = 20, 20, -30

    v = w1*x1 + w2*x2 + bias
    return sigmoid(v)

def xor(a, b):
    h11, h12 = layer1(a, b)
    return layer2(h11, h12)          # Feed the output of last layer to the next layer

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
Back to the basic calculus, a function can be constructed with infinite narrow rectangles. Can we use the technique above to construct a narrow shaped rectangles. (a.k.a. delta function)

<div class="imgcap">
<img src="/assets/dl_intro/delta.png" style="border:none;width:50%">
</div>

Here is the code using the same set of equations and network layout:
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
Which output something with shape like a delta function:
<div class="imgcap">
<img src="/assets/dl_intro/delta_func.png" style="border:none;width:60%">
</div>

Implement a XOR or a delta function is not important for deep learning (DL). Nevertheless, we demonstrate the possibilities of building a complex function estimator through a network of simple computation nodes. In both cases, we need a network with 2 layers. A network with 3 or 4 layer can push the hand written recognition of numbers to an accuracy of 95%. Naturally, a network with many layers (deeper) can reproduce a much complicated model. For example, Microsoft ResNet for image recognition has 100+ layers.

### Build a Linear regression model
Before teaching Pieter how to learn those parameters, we try to build a simple model first. For example, Pieter wants to expand on his horizon and try to start online dating. He wants to find out the relationship between the number of online dates with the number of years in eductaion and the monthly income.  Pieter starts with a simple linear model as follows:

$$
dates = W_1* \text{years in school} + W_2*\text{monthly income} + bias
$$

He asks 1000 people in each community and collect the information on their income, education and the corresponding number of online dates.  Pieter is interested in finding out how each community values their intellectual vs his humble post-doc salary.  So even this model looks overwhemly simple, it serves its purpose.

**Deep learing is about learning from mistakes.** His stratragy to create a model for each community will be:
1. Take a first guess on W and b.
2. Use the model above to compute the number of dates.
3. With the computed value and the number provided by each sample, he compute the error of his model.
4. Pieter will also compute how a small change in the current value of W and b will impact on the error.
5. With this rate change information regarding W & b on the error, Pieter re-adjust W & b. (**Gradient descent**)
6. Go back to step 2 for N iterations.
7. When it is complete, we get the correct parameter for W and b trainned with the sample data. 
8. We use the final W & b to make a prediction on how Pieter will do in each community.

#### Gradient descent
Step 3-5 is called the gradient descent in DL. First we need to define a function to measure our errors betweent the real life and our model. In DL, we call this error function **cost function**. Mean square error (MSE) is one obvious candidate. 

$$
J(W, b, h, y) = \text{mean square error } (W, b, h, y) = \frac{1}{N} \sum_i (h_i - y_i)^2
$$

where h is what we predict in the mode, y is the value from our data and N is the number of samples. The intution is pretty simple.  We can visualize the cost as below with x being all the possible value of
$$
W_1
$$
and y the possible value of
$$
W_2
$$
between -1 and 1 and z the corresponding cost for (x, y). The solution of out model is where W and b has the lowest cost. i.e. picking the value of W and b such that the cost is the lowest (the blue area).

<div class="imgcap">
<img src="/assets/dl_intro/solution.png" style="border:none;">
</div>

We can first pick a random W and b, visualize dropping a marble at 
$$
(W_1, W_2)
$$
and let the gravity to do its work. Visualize 3D or higher dimensions are hard. It is much easier to study the problem in 2D since the solution usually works in higher dimension.

<div class="imgcap">
<img src="/assets/dl_intro/gd.jpg" style="border:none;">
</div>

The X-axis is the value of 
$$
W_1
$$
and the y axis is its corresponding cost average by N samples.

$$
J(W, b, h, y) = \frac{1}{N} \sum_i (W_1*x_i + - y_i)^2
$$

We are ignoring 
$$
W_2 \text{ and } b.
$$
It is clear that when the gradient at L1 is negative (as shown), we should move W to the right. But by how much? We can plot the value of 
$$
W_2
$$
with 
$$
W_1
$$
set to the x value of L1. We realize at L2, the gradient is smaller. It means the change of 
$$
W_2
$$
has a smaller impact on the cost comparable with
$$
W_1.
$$
It is pretty obvious that the amount of adjustment for each paramter should be proportional to its partial gradient.

$$
\Delta W_i \propto \frac{\partial J}{\partial W_i}
$$

$$
W_i = W_i - \alpha \Delta W_i
$$

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
























