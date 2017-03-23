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

Implement a XOR or a delta function is not important for deep learning (DL). Nevertheless, we demonstrate the possibilities of building a complex function estimator through a network of simple computation nodes. In both cases, we need a network with 2 layers. A network with 3 or 4 layer can push the hand written recognition of numbers to an accuracy of 90+%. Naturally, a network with many layers (deeper) can reproduce a much complicated model. For example, Microsoft ResNet for image recognition has 100+ layers.

### Build a Linear regression model
Before teaching Pieter how to learn those parameters, we try to build a simple model first. For example, Pieter wants to expand on his horizon and try to start online dating. He wants to find out the relationship between the number of online dates with the number of years in eductaion and the monthly income.  Pieter starts with a simple linear model as follows:

$$
dates = W_1* \text{years in school} + W_2*\text{monthly income} + bias
$$

He asks 1000 people in each community and collect the information on their income, education and the corresponding number of online dates.  Pieter is interested in finding out how each community values their intellectual vs his humble post-doc salary.  So even this model looks overwhemly simple, it serves its purpose.

Pieter will define a model with trainable parameters (W & b). He make a guess on the parameters and calculate how a tiny change in those parameters will impact on the error. With this informatin, he make tiny change to those parameter. He keeps continue until the parameters converge to stable numbers.

**Deep learing is about learning from mistakes.** This is the high level steps for Pieter to build the model.
1. Take a first guess on W and b.
2. Use the model above to compute the number of dates.
3. With the computed value and the number of dates provided by each sample, compute the mean square error of the model.
4. Then compute how a small change in the current W and b may impact on the error.
5. Re-adjust W & b according to this error rate change relative to W & b, . (**Gradient descent**)
6. Go back to step 2.
7. After N iterations or when the parameters converge, we stop and have the final value for W & b. 
8. Use the sample from another community to build a model with different W & b.
9. Make a prediction on how Pieter will do in each community with different models.

### Gradient descent
Step 3-5 is called the gradient descent in DL. First we need to define a function to measure our errors between the real life and our model. In DL, we call this error function **cost function** or **loss function**. Mean square error (MSE) is one obvious candidate. 

$$
J(W, b, h, y) = \text{mean square error } (W, b, h, y) = \frac{1}{N} \sum_i (h_i - y_i)^2
$$

where h is what we predict about the number of dates in our model, y is the value from our sample data and N is the number of samples. The intution is pretty simple.  We can visualize the cost as below with x-axis being all the possible value of
$$
W_1
$$
and y-axis the possible value of
$$
W_2
$$
between -1 and 1, and z the corresponding cost J(x, y). The solution of our model is where W and b has the lowest cost. i.e. picking the value of W and b such that the cost is the lowest (the lowest oint in the blue region).Visualize dropping a marble at a random point
$$
(W_1, W_2)
$$
and let the gravity to do its work. 

<div class="imgcap">
<img src="/assets/dl_intro/solution.png" style="border:none;">
</div>

### Learning rate

Thinking in 3D or higher dimensions are hard to impossible. Always think in 2D first.

Consider a point at (L1, L2), we cut through the diagram alone the red and orange and plot those curve in a 2D diagram:
<div class="imgcap">
<img src="/assets/dl_intro/solution_2d.jpg" style="border:none;">
</div>

<div class="imgcap">
<img src="/assets/dl_intro/gd.jpg" style="border:none;">
</div>

The X-axis is the value of 
$$
W
$$
and the y axis is its corresponding average cost for the data samples.

$$
J(W, b, h, y) = \frac{1}{N} \sum_i (W_1*x_i - y_i)^2
$$

Since the gradient at L1 is negative (as shown), we move 
$$
W_1
$$
to the right. But by how much? Let's compare the gradient for L1 and L2. We realize L2 has a smaller gradient. i.e. the change of
$$
W2
$$
has a smaller impact to the change of cost compare to L1. Obviosuly, the greater the impact, the larger adjustment we should make. Therefore, the amount of adjustment for the parameters
$$
(W_1, W_2)
$$
should be proportional to its partial gradient at that point. i.e.

$$
\Delta W_i \propto \frac{\partial J}{\partial W_i} 
$$

$$
\text{ i.e. } \Delta W_1 \propto \frac{\partial J}{\partial W_1} \text{ and } \Delta W_2 \propto \frac{\partial J}{\partial W_2}
$$

$$
\Delta W_i = \alpha \frac{\partial J}{\partial W_i}
$$

$$
W_i = W_i - \Delta W_i
$$

In DL, the varaible 
$$
\alpha
$$
introduce here is called the **learning rate**.  Small learning rate will take a longer time (more iteration) to find the minima. However, as we learn from calculus, the larger the step, the bigger the error in our calculation. In DL, finding the right value of learning rate is sometimes a try and error exercise.  Sometimes we will try values ranging from 1e-7 to 1 in logarithmic scale (1e-7, 5e-7, 1e-6, 5e-6, 1e-5 ...). 

Large learning step may cost w to oscillate with increasing cost:
<div class="imgcap">
<img src="/assets/dl_intro/learning_rate.jpg" style="border:none;">
</div>

We start with w = -6 (x-axis) at L1 , if the gradient is huge, a relatively large learning rate will swing w far to the other side to L2 with even a larger gradient. Eventually, rather than drop down slowly to a minima, w keeps oscalliate and the cost keep increasing. The follow demonstrates how a learning rate of 0.8 may swing the cost upward instead of downward. When loss starts going upward, we need to reduce the learning rate. The following table traces how W change from L1 to L2 and then L3.

<div class="imgcap">
<img src="/assets/dl_intro/lr_flow.png" style="border:none;">
</div>

Here is another illustration that actually could happen.  When we gradudally descent, we land in an area with high gradient that make it bounce way back with high cost. This type of shape will be very hard to reach the minima with the current descent method.
<div class="imgcap">
<img src="/assets/dl_intro/ping.jpg" style="border:none;">
</div>

> Sometimes, we need to be careful about the scale used in plotting the x-axis and y-axis. In the diagram shown above, the gradient does not seem large.  It is because we use a much smaller scale for y-axis than the x-axis (0 to 150 vs -10 to 10).

#### Naive gradient checking
There are many ways to compute the paritial derviative. One naive but important method is using the simple partial derviative definition.

Here is a simple demonstration of finding the derivative of 
$$
x^2 \text{ at } x = 4
$$

$$
\frac{\partial f}{\partial x} = \frac{f(x+\Delta x_i) - f(x-\Delta x_i) } { 2 \Delta x_i} 
$$

```python
def gradient_check(f, x, h=0.00001):
  grad = (f(x+h) - f(x-h)) / (2*h)
  return grad

f = lambda x: x**2
print(gradient_check(f, 4))
```
We don't call this method in the production code. But computing partial derviative can be tedious and therefore we always verify the value we computed with this naive method during the development time.

### Backpropagation
To compute the partial derviatives, 
$$
\frac{\partial J}{\partial W_i}
$$
We can start from each node in the left most layer and compute the gradient using the naive gradient checking, and progagate the result until it reach the right most layer that computing the cost.  Then we move to the next layer and start the process again. For a deep network, this is very inefficient.

To compute the partial gradient efficiently, we perform a foward pass to compute the cost.
<div class="imgcap">
<img src="/assets/dl_intro/fp.jpg" style="border:none;">
</div>

> Always keep track of the shape (dimension) of the data. This is one great tip when you program DL. (N,) means a 1-D array with N elements. (N,1) means 2-D array with N rows each containing 1 element. (N, 3, 4) means a 3D array.

$$
out = W_1* X_1 + W_2*X_2 + b
$$

```python
def forward(x, W, b):
    # x: input sample (N, 2)
    # W: Weight (2,)
    # b: bias float
    # out: (N,)
    out = x.dot(W) + b        # Multiple X with W + b: (N, 2) * (2,) -> (N,)
    return out
```

$$
J = \frac{1}{N} \sum_i (out - y_i)^2
$$

```python
def mean_square_loss(h, y):
    # h: prediction (N,)
    # y: true value (N,)
    N = X.shape[0]            # Find the number of samples
    loss = np.sum(np.square(h - y)) / N   # Compute the mean square error from its true value y
    return loss
```

Then we backprogragate the gradient from the right most layer to the left in one single pass.
<div class="imgcap">
<img src="/assets/dl_intro/bp.jpg" style="border:none;">
</div>

$$
J(out) = \frac{1}{N} \sum_i (out_i - y_i)^2
$$

$$
J(out_i) = \frac{1}{N} (out_i - y_i)^2
$$

$$
\frac{\partial J}{\partial \text{ out}_i} = \frac{2}{N} (out_i - y_i)
$$

```python
def mean_square_loss(h, y):
    # h: prediction (N,)
    # y: true value (N,)
    ...
    dout = 2 * (h-y) / N                  # Compute the partial derviative of J relative to out
    return loss, dout
```

Now we have
$$
\frac{\partial J}{\partial out_i}
$$ 
. We apply the chain rule to compute the gradient at the second right layer.  (Backpropagate the gradient from right to left.)

$$
\frac{\partial J}{\partial W} = \frac{\partial J}{\partial out} \frac{\partial out}{\partial W}  
$$ 

$$
out = W * X + b
$$

$$
\frac{\partial out}{\partial W}  = X
$$ 

$$
\frac{\partial out}{\partial b}  = 1
$$ 

$$
\frac{\partial J}{\partial W} = \frac{\partial J}{\partial out} X
$$ 

$$
\frac{\partial J}{\partial b} = \frac{\partial J}{\partial out}
$$ 

<div class="imgcap">
<img src="/assets/dl_intro/bp3.jpg" style="border:none;">
</div>

<div class="imgcap">
<img src="/assets/dl_intro/bp2.jpg" style="border:none;">
</div>

In DL programing, we often name

$$
\frac{\partial J}{\partial \text{dout}} \text{ as dout}
$$

$$
\frac{\partial \text{ next}}{\partial \text{ current}} \text{ as dcurrent}
$$

Here is the code computing
$$
dW, db
$$

```python
def forward(x, W, b):
    out = x.dot(W) + b
    cache = (x, W, b)
    return out, cache

def backward(dout, cache):
    # dout: dJ/dout (N,)
    # x: input sample (N, 2)
    # W: Weight (2,)
    # b: bias float
    x, W, b = cache
    dW = x.T.dot(dout)            # Transpose x (N, 2) -> (2, N) and multiple with dout. (2, N) * (N, ) -> (2,)
    db = np.sum(dout, axis=0)     # Add all dout (N,) -> scalar
    return dW, db

def compute_loss(X, W, b, y=None):
    h, cache = forward(X, W, b)

    loss, dout = mean_square_loss(h, y)
    dW, db = backward(dout, cache)
    return loss, dW, db
```

Here is the full listing of the code
```python
import numpy as np

iteration = 10000
learning_rate = 1e-6
N = 100

def true_y(education, income):
    """ Find the number of dates corresponding to the years of education and monthly income.
    Instead of collecting the real data, an Oracle provides us a formula to compute the true value.
    """
    dates = 0.8 * education + 0.003 * income + 2
    return dates

def sample(education, income):
    """We generate sample of possible dates from education and income value.
    Instead of collecting the real data, we find the value from the true_y
    and add some noise to make it looks like sample data.
    """
    dates = true_y(education, income)
    dates += dates * 0.01 * np.random.randn(education.shape[0]) # Add some noise
    return dates

def forward(x, W, b):
    # x: input sample (N, 2)
    # W: Weight (2,)
    # b: bias float
    # out: (N,)
    out = x.dot(W) + b        # Multiple X with W + b: (N, 2) * (2,) -> (N,)
    cache = (x, W, b)
    return out, cache

def mean_square_loss(h, y):
    # h: prediction (N,)
    # y: true value (N,)
    N = X.shape[0]            # Find the number of samples
    loss = np.sum(np.square(h - y)) / N   # Compute the mean square error from its true value y
    dout = 2 * (h-y) / N                  # Compute the partial derviative of J relative to out
    return loss, dout

def backward(dout, cache):
    # dout: dJ/dout (N,)
    # x: input sample (N, 2)
    # W: Weight (2,)
    # b: bias float
    x, W, b = cache
    dw = x.T.dot(dout)            # Transpose x (N, 2) -> (2, N) and multiple with dout. (2, N) * (N, ) -> (2,)
    db = np.sum(dout, axis=0)     # Add all dout (N,) -> scalar
    return dw, db

def compute_loss(X, W, b, y=None):
    h, cache = forward(X, W, b)

    if y is None:
        return h

    loss, dout = mean_square_loss(h, y)
    dW, db = backward(dout, cache)
    return loss, dW, db


education = np.random.randint(26, size=N) # (N,) Generate 10 random sample with years of education from 0 to 25.
income = np.random.randint(10000, size=education.shape[0]) # (N,) Generate the corresponding income.

# The number of dates according to the formula of an Oracle.
# In practice, the value come with each sample data.
Y = sample(education, income)    # (N,)
W = np.array([0.1, 0.1])         # (2,)
b = 0

X = np.concatenate((education[:, np.newaxis], income[:, np.newaxis]), axis=1) # (N, 2) N samples with 2 features

for i in range(iteration):
    loss, dW, db = compute_loss(X, W, b, Y)
    W -= learning_rate * dW
    b -= learning_rate * db
    if i%100==0:
        print(f"iteration {i}: loss={loss:.4} W1={W[0]:.4} dW1={dW[0]:.4} W2={W[1]:.4} dW2={dW[1]:.4} b= {b:.4} db = {db:.4}")

print(f"W = {W}")
print(f"b = {b}")
```

#### General principle in backpropagation

We can genealize the above method to multiple layers:
<div class="imgcap">
<img src="/assets/dl_intro/bp1.jpg" style="border:none;">
</div>

Provide with a cost fumction like:

$$
J(out) = \frac{1}{N} \sum_i (out_i - y_i)^2
$$

Compute the dervative:

$$
\frac{\partial J}{\partial \text{ out}} = \frac{2}{N} (out - y)
$$

Progagate from the right layer to the left using the chain rule one layer at a time:

$$
\frac{\partial J}{\partial \text{out}_{k-1}} = \frac{\partial J}{\partial \text{out}_{k}} \frac{\partial \text{out}_k}{\partial \text{out}_{k-1}}  
$$ 

and, compute

$$
\frac{\partial \text{out}_k}{\partial \text{out}_{k-1}}  = \frac{\partial f_{k}}{\partial \text{out}_{k-1}} 
$$

In backprogragation, we may backprogate multiple path back to the same node. To compute the gradient correctly, we need to add both path together:
<div class="imgcap">
<img src="/assets/dl_intro/bp_m1.jpg" style="border:none;">
</div>

$$
\frac{\partial J}{\partial o_3}  = \frac{\partial J}{\partial o_4} \frac{\partial f_4} {\partial o_3} *+ \frac{\partial J}{\partial o_5} \frac{\partial f_4} {\partial o_3} 
$$

Backprogation is tedious and error prone. But most of the time, it is because we lost track of the notations and index.
> For backprogation, try to draw a diagram with the shape information. Name your key variables consistently and put the derivative equation under each node. Expand equations with sub-index for analysis if needed.

<div class="imgcap">
<img src="/assets/dl_intro/bp.jpg" style="border:none;width:80%">
</div>

$$
\frac{\partial J}{\partial \text{ out}_i} = \frac{2}{N} (out_i - y_i)
$$

### Trouble shooting

Many places can go wrong when training a deep network. We will cover more technical topics on how to train a DL network. But here are some simple tips:
* Unit test the forward pass, back propagation and code with a lot of math & vectorization.
* Compare the back progataion result with the naive gradient check.
* Create scenaiors to test the code easier. For example, remove the nose or assign W & b guess to be the same as the true model.
* Create simple cases and verify whether the matrics collected are expected.
* Don't be too aggressive in build up your model at the begining. Trouble shoot multiple issues at a time is particular hard in DL.
* Instead, build up a simple but working model first. 
* Start debugging with 1-2 sample data with a small number of iteration.
* Don't waste time in large dataset and iterations at the beginning. Look for sign that your model beat the random odd of guessing.
* At the early debugging, use non-random data for input and parameters.
* Always keep track of the shape of the data and doucment it in the code.
* Use consistence naming for variable in the forward pass and backpropagation.
* Verify the sample data in your training.
* Keep track of the loss, and when in debugging also the magnitude of the gradient.
* Plot out the loss, accuracy or some runtime data after the training.

I strongly recommend you to think about a linear regression model inerested you and train a simple network now. A lot of issues happened in complex model will show up even in such a simple model. Through this process, you will learn trouble shooting techniques as well as how these training parameters changed during learning. Work with a simple model allows you to trace the data easier and learn better. Most tutorial have already pre-cooked parameters. So they teach you the easier part without letting you to complete the real tough part.

So let Pieter train the system.
```
iteration 0: loss=2.825e+05 W1=0.09882 dW1=1.183e+04 W2=-0.4929 dW2=5.929e+06 b= -8.915e-05 db = 891.5
iteration 200: loss=3.899e+292 W1=-3.741e+140 dW1=4.458e+147 W2=-1.849e+143 dW2=2.203e+150 b= -2.8e+139 db = 3.337e+146
iteration 400: loss=inf W1=-1.39e+284 dW1=1.656e+291 W2=-6.869e+286 dW2=8.184e+293 b= -1.04e+283 db = 1.24e+290
iteration 600: loss=nan W1=nan dW1=nan W2=nan dW2=nan b= nan db = nan
```
The application overflow within 600 iterations! Since the loss and the graident is so high, we can try out whether we have the learning rate too high. We decrease the learning rate and run just a short time to see if any changes.

For learning rate of 1e-8, we do not have the overflow problem but the result is not good. We can try much smaller value and more iteration.
```
iteration 90000: loss=4.3e+01 W1= 0.23 dW1=-1.3e+02 W2=0.0044 dW2= 0.25 b= 0.0045 db = -4.633
W = [ 0.2437896   0.00434705]
b = 0.004981262980767952
```

We are very reluctant to take action without information. But since the application run very fast, we can give a few simple guess. With10000000 iterations and a learning_rate of 1e-10. The application run for a few minutes but we are still not there. Running much longer may improve the result. But it will be better to trace the source of problem.
```
iteration 9990000: loss=3.7e+01 W1= 0.22 dW1=-1.1e+02 W2=0.0043 dW2= 0.19 b= 0.0049 db = -4.593
W = [ 0.22137005  0.00429005]
b = 0.004940551119084607
```
The loss in our first try have similar symptoms with bad learning rate. But it may not be the cause. After some tracing, we find the gradient is very high. Unlike many real DL problems, the model is not a black box to us, we can plot the cost function related with W.

This is a U shape curve which is different from a bowl shape curve that we used for gradient descent explanation. 
<div class="imgcap">
<img src="/assets/dl_intro/ushape.png" style="border:none;width:50%">
</div>

If we change the y-axis closer to the range that we are interested in, we will find the situation much worse.
<div class="imgcap">
<img src="/assets/dl_intro/ushape2.png" style="border:none;width:50%">
</div>

<div class="imgcap">
<img src="/assets/dl_intro/solution.png" style="border:none;width:50%">
</div>

The y-axis is 
$$
W_2
$$
which the cost is much responsive to change comparing with the x-axis
$$
W_1![Gauss2](../../../Desktop/gauss2.png)
$$
. If we look at the linear model 
$$
X_1 
$$ 
represent the years of education which may range from 0 to 30. 
$$
X_2 
$$
is the monthly income from 0 to 10,000. 

Obviously, the different of scale in these 2 features cause a major difference in its gradient. Simply say, we cannot find a single learning rate than can work well with both of them. The solution is pretty simple with a couple line of code change. We re-scale the income value. Here is the output which is close to our true model:
```
iteration 0: loss=518.7 W1=0.1624 dW1=-624.5 W2=0.3585 dW2=-2.585e+03 b= 0.004237 db = -42.37
iteration 10000: loss=0.4414 W1=0.8392 dW1=0.0129 W2=0.3128 dW2=0.004501 b= 0.5781 db = -0.4719
iteration 20000: loss=0.2764 W1=0.8281 dW1=0.009391 W2=0.3089 dW2=0.003277 b= 0.9825 db = -0.3436
iteration 30000: loss=0.1889 W1=0.8201 dW1=0.006837 W2=0.3061 dW2=0.002386 b= 1.277 db = -0.2501
iteration 40000: loss=0.1425 W1=0.8142 dW1=0.004978 W2=0.3041 dW2=0.001737 b= 1.491 db = -0.1821
iteration 50000: loss=0.1179 W1=0.81 dW1=0.003624 W2=0.3026 dW2=0.001265 b= 1.647 db = -0.1326
iteration 60000: loss=0.1049 W1=0.8069 dW1=0.002639 W2=0.3015 dW2=0.0009208 b= 1.761 db = -0.09653
iteration 70000: loss=0.09801 W1=0.8046 dW1=0.001921 W2=0.3007 dW2=0.0006704 b= 1.844 db = -0.07028
iteration 80000: loss=0.09435 W1=0.803 dW1=0.001399 W2=0.3001 dW2=0.0004881 b= 1.904 db = -0.05117
iteration 90000: loss=0.09241 W1=0.8018 dW1=0.001018 W2=0.2997 dW2=0.0003553 b= 1.948 db = -0.03725
W = [ 0.80088848  0.29941657]
b = 1.9795590414763997
```

### Non-linearity

Pieter come back and say our linear model is not adequate enough. He find that the relationship between years of education and dates are closer to:
<div class="imgcap">
<img src="/assets/dl_intro/educ.png" style="border:none;width:50%">
</div>

Can we combine linear functions with multiple layers to form a non-linear function?
$$
f(x) = Wx + b
$$

$$
g(z) = Uz + c
$$

The answer is no.
$$
g(f(x)) = Vx + d
$$

As shown in our first example, we apply a non-linear functional to our output.

$$
f(z_j) = \frac{1}{1 + e^{-z_j}}
$$

After some thoughts, we apply to Pieter's data.
$$
f(x) = max(0, x)
$$

As shown below, we should be able to construct a non-linear function addressing Pieter's requirement.
<div class="imgcap">
<img src="/assets/dl_intro/solution.png" style="border:none;width:50%">
</div>
<div class="imgcap">
<img src="/assets/dl_intro/solution.png" style="border:none;width:50%">
</div>

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
























