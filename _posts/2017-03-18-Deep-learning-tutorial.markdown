---
layout: post
comments: true
mathjax: true
title: “Deep learning without going down the rabbit holes.”
excerpt: “How to learn deep learning from easy concept to complex idea? How to build insight along the way?”
date: 2017-03-18 14:00:00
---
**This is work in progress...**

### What is deep learning (DL)?
**Deep learning is about building a function estimator.** Historically, people explains deep learning (DL) using the neural network. This is where deep learning gets its insight.  Nevertheless, deep learning has outgrown this explanation. Once you realize building a deep learning network is about building a function estimator, you will unveil its real potential in AI.
 
Let us build a new andriod named Pieter. Our first task is to teach Pieter how to recognize visual objects. Can the human visual system be replaced by a big function estimator? Can we pass the pixel values to a function and classify it as a school bus, an airplane or a truck?

<div class="imgcap">
<img src="/assets/dl/deep_learner.jpg" style="border:none;width:70%;">
</div>

Indeed, in our later example of visual recognition, we will build a system very similar to the following:
<div class="imgcap">
<img src="/assets/dl/fc.jpg" style="border:none;width:80%;">
</div>

For every node, we compute:

$$
z_j = \sum_{i} W_{ij} x_{i} + b_{i}
$$

$$
f(z_j) = \frac{1}{1 + e^{-z_j}}
$$

which $$ x_{i} $$ is the input to the node or the pixel values if this is the first layer. 

> Deep learning has many scary looking equations. We will walk through examples to show how it works. Most of them are pretty simple.

For example, with weight W  (0.3, 0.2, 0.4, 0.3), bias b (-0.8) and a grayscale image with just 4 pixels (0.1, 0.3, 0.2, 0.1), the output of the first node circled in red above will be:

$$
z_j =  0.3*0.1 + 0.2*0.3 + 0.4*0.2 + 0.3*0.1  - 0.8 = -0.6
$$

$$
f(z) =  \frac{1}{1 + e^{-(-0.6)}} = 0.3543
$$

Each node has its own weight (W) and bias (b). From the left most layer, we compute the output of each node, and feed it to the next layer. Eventually, the right most layer is the probability for each object classification (a school bus 0.88, an airplane 0.08 or a truck 0.04). In this exercise, we supply all the weight and bias values to our android Pieter. But as the term "deep learning" implies, by the end of this tutorial, Pieter will learn those parameters by himself. We still miss a few pieces for the puzzle. But the network diagram and the equations above lay down the foundation of a deep learning network. In fact, this simple design can recognize the zip code written on a envelop with very high accuracy.

#### XOR
For the skeptics, we will build an exclusive "or" (a xor b) using a simple network like:
<div class="imgcap">
<img src="/assets/dl/xor.jpg" style="border:none;width:40%">
</div>
For each node, we apply the same equations mentioned before:

$$
z_j =  \sum_{i} W_i * x_i + b_{i}
$$

$$
h_j = \sigma(z) = \frac{1}{1 + e^{-z_j}}
$$

The following is our code implementation which is pretty self-explainatory. The purpose is to demonstrate exactly how we do the weight multiplication and apply the sigmoid function. In this program, we use Numpy which is a package for scientific computing with Python. It provides many mathematic operations and array manipulations that we need.

> We provide coding to help audience to verify their understanding. Nevertheless, a full understand of the code is not needed or suggested.

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
And the XOR output matches with its expected logical values:
```
 0 ^ 0 = 0.00
 0 ^ 1 = 1.00
 1 ^ 0 = 1.00
 1 ^ 1 = 0.00
```
#### Delta function
Back to the basic calculus, a function can be constructed with infinite narrow rectangles (a.k.a. delta function). If we can construct such rectangles with a network, we can built on top of it to build any functions.

<div class="imgcap">
<img src="/assets/dl/delta.png" style="border:none;width:50%">
</div>

Here is the code using the same set of equations and network layout before:
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
<img src="/assets/dl/delta_func.png" style="border:none;width:50%">
</div>

Implement a XOR or a delta function is not important for deep learning (DL). Nevertheless, we demonstrate the possibilities of building a complex function estimator through a network of simple computation nodes. A 3-layer network can implement a hand written recognition system for numbers with an accuracy of 95+%. The deeper a network the more complex model that we can build. For example, Microsoft ResNet (2015) for visual recognition has 151 layers. In many modern models, there are 10 million tunable parameters. For many AI problems, the model needed to solve the problem is very complex. In automous driving, we can model a policy (turn, accelerate or brake) to approximate what a human will do for what they see in front of them. This policy is too hard to model it analytically. Alterantive, with enough training data, we train a deep learning network with accuracy of a regular driver.

<div class="imgcap">
<img src="/assets/dl/drive.jpg" style="border:none;width:80%">
</div>

> Autonomus driving involves many aspects of AI. DL provides a model estimator that cannot be done analytically.

### Build a Linear regression model
**DL solves problem by learning from data.** We will demonstrate how Pieter learns the model parameters $$W$$ by processing training data. For example, Pieter wants to expand on his horizon and start online dating. He wants to find out the relationship between the number of online dates with the years of eductaion and the monthly income.  Pieter starts with a simple linear model as follows:

$$
\text {number of dates} = W_1* \text{years in school} + W_2*\text{monthly income} + bias
$$

He asks 1000 people in different communities and collects the information on their income, education and the corresponding number of online dates.  Pieter interests in knowing how each community values the intellectual vs his humble post-doc salary.  So even this model looks overwhemly simple, it serves its purpose. So the task for Pieter is to find the parameter values W and b in this model with training data collected by him.

This is the high-level steps:
1. Take the first guess on W and b.
2. Use the model to predict the number of dates for each sample in the training dataset.
3. Compute the mean square error between the computed value and the true value in the dataset.
4. Compute how much the error will change when we change W and b.
5. Re-adjust W & b according to this error rate change. (**Gradient descent**)
6. Back to step 2 for N iterations.
7. Use the last value of W & b for our model.

We build a model for each community, and use these models to predict how well Pieter may do in each community.

> In our model, we predict the number of dates for people with certain income and years of education. The corresponding values (the number of dates) in the training dataset are called the **true values or true labels**.

### Gradient descent
**Deep learning is about learning how much it cost.** Step 2-5 is called the gradient descent in DL. We define a function to measure our errors between our model and the true values. In DL, this error function is called **cost function** or **loss function**. Mean square error (MSE) is one obvious candidate.

$$
\text{mean square error} = J(h, y, W, b) = \frac{1}{N} \sum_i (h_i - y_i)^2
$$

where $$ h_i $$ is the model prediction and $$ y_i $$ is the true value for sample $$ i $$. We sum over all the samples and take the average.
We can visualize the cost below with x-axis being $$ W_1 $$ and y-axis being $$ W_2 $$ and z-axis being the cost J. The solution of our model is to find $$ W_1 $$ and $$ W_2 $$ where the cost is lowest. We can visualize this as dropping a marble at a random point $$ (W_1, W_2) $$ and let the gravity to do its work. 

<div class="imgcap">
<img src="/assets/dl/solution2.png" style="border:none;">
</div>

> Optimize a deep network means find all the $$ W $$, $$ b $$ and other tunable parameters to minimize cost.

### Learning rate

Thinking in 3D or high dimensions is hard to impossible. Try to think DL problems in 2D first. Consider a point at (L1, L2), we cut through the diagram alone the blue and orange line, and plot those curves in a 2D diagram:
<div class="imgcap">
<img src="/assets/dl/solution_2d.jpg" style="border:none;">
</div>

<div class="imgcap">
<img src="/assets/dl/gd.jpg" style="border:none;">
</div>

The x-axis is $$ W $$ and the y-axis is the cost. Since we are holding $$ W_{2} $$ as a constant, we can ignore it in the equation below to simplify the discussion.

$$
J(W, b, h, y) = \frac{1}{N} \sum_i (W_1*x_i - y_i)^2
$$


Since the gradient at L1 is negative, we move $$ W_1 $$ to the right to find the lowest point. But by how much? L2 has a smaller gradient than L1. i.e. changing $$ W2 $$ has a smaller impact on cost compare to L1. Obviously, we update a parameter proportional to its impact. Therefore, adjustment for $$ (W_1, W_2) $$ is proportional to its partial gradient at that point. i.e.

$$
\Delta W_i \propto \frac{\partial J}{\partial W_i} 
$$

$$
\text{ i.e. } \Delta W_1 \propto \frac{\partial J}{\partial W_1} \text{ and } \Delta W_2 \propto \frac{\partial J}{\partial W_2}
$$

Add a ratio value, the adjustments to $$W$$ are:

$$
\Delta W_i = \alpha \frac{\partial J}{\partial W_i}
$$

$$
W_i = W_i - \Delta W_i
$$

The variable $$ \alpha $$ is called the **learning rate**.  Small learning rate takes a longer time (more iteration) to locate the minima. However, as we learn in calculus, a larger step results in a larger error in the calculation. In DL, finding the right value for the learning rate is a try and error exercise.  We usually try values ranging from 1e-7 to 1 in logarithmic scale (1e-7, 5e-7, 1e-6, 5e-6, 1e-5 ...) but this depends on the problem you are solving. 

A large learning step may have other serious problems. It costs $$w$$ to oscillate with increasing cost:
<div class="imgcap">
<img src="/assets/dl/learning_rate.jpg" style="border:none;">
</div>

We start with w = -6 (x-axis) at L1. If the gradient is huge, a learning rate larger than certain value will swing $$w$$ too far to the other side (say L2) with even a larger gradient. Eventually, rather than dropping down slowly to a minima, $$w$$ oscillates and the cost increases. When loss keeps going upward, we need to reduce the learning rate. The follow demonstrates how a learning rate of 0.8 with a steep gradient swings the cost upward instead of downward. The table traces how the oscillation of W causes the cost go upwards from L1 to L2 and then L3.

<div class="imgcap">
<img src="/assets/dl/lr_flow.png" style="border:none;">
</div>

> We need to be careful about the scale used for the x-axis and y-axis. In the diagram above, the gradient does not look steep.  It is because we have a much smaller scale for y-axis than the x-axis (0 to 150 vs -10 to 10).

Here is another illustration of some real problems.  When we gradudally descent, we may land in an area with steep gradient which the $$W$$ will bounce back. This type of shape is very hard to find the minima with a constant learning rate. Advance methods to address this problem will be discussed later.

<div class="imgcap">
<img src="/assets/dl/ping.jpg" style="border:none;">
</div>

This example is real but dramatical. But in a lesser extend, instead of settle down at the bottom, || W || oscillates around the minima slightly. If we drop a ball in Grand Canyon, we expect it to land in the bottom. In DL, this is harder.

#### Naive gradient checking
There are many ways to compute a paritial derviative. One naive but important method is using the simple partial derviative definition.

$$
\frac{\partial f}{\partial x} = \frac{f(x+\Delta x_i) - f(x-\Delta x_i) } { 2 \Delta x_i} 
$$

Here is a simple code demonstration of finding the derivative of 
$$
x^2 \text{ at } x = 4
$$

```python
def gradient_check(f, x, h=0.00001):
  grad = (f(x+h) - f(x-h)) / (2*h)
  return grad

f = lambda x: x**2
print(gradient_check(f, 4))
```
We never call this method in the production code. But computing partial derviative can be tedious and error prone. We use this method to verify a partial derviative implementation during the development time.

#### Mini-batch gradient descent

When computing the cost function, we add all the errors for the processed training data. We can process all the training data at once but this can take too much time for just one update. On the contrray, we can perform stochastic gradient descent which make one W update per training sample. Nevertheless, the gradient descent will follow a zipzap pattern rather than following the curve of the cost function. This can be a problem if you land in a steep gradient area which the parameters may bounce to area with high cost. The training takes longer, and it might zipzag around the minima rather down converge to the minima. 

<div class="imgcap">
<img src="/assets/dl/solution3.png" style="border:none;">
</div>

A good compromise is to process a batch of N samples at a time. This can be a tunable hyper-parameter but usually not very critical and may start with 64 subject to the memory consumptions.

$$
J = \frac{1}{N} \sum_i (W_1*x_i - y_i)^2
$$

### Backpropagation
To compute the partial derviatives, $$ \frac{\partial J}{\partial W_i} $$, we can start from each node in the left most layer and propagate the gradient until it reach the right most layer.  Then we move to the next layer and start the process again. For a deep network, this is very inefficient. To compute the partial gradient efficiently, we perform a foward pass and a backprogagation. 

#### Forward pass
First, we compute the cost in a forward pass:
<div class="imgcap">
<img src="/assets/dl/fp.jpg" style="border:none;width:80%">
</div>

> Keep track of the naming of your input & output, its shape (dimension) and the equations. This is one great tip when you program DL. (N,) means a 1-D array with N elements. (N,1) means 2-D array with N rows each containing 1 element. (N, 3, 4) means a 3D array.

The method "forward" computes the equation below:

$$
out = W_1* X_1 + W_2*X_{2} + b
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

To compute the mean square loss:

$$
J = \frac{1}{N} \sum_i (out - y_{i})^2
$$

```python
def mean_square_loss(h, y):
    # h: prediction (N,)
    # y: true value (N,)
    N = X.shape[0]            # Find the number of samples
    loss = np.sum(np.square(h - y)) / N   # Compute the mean square error from its true value y
    return loss
```

#### Backpropagation pass
Then we backprogragate the gradient from the right most layer to the left in one single pass.
<div class="imgcap">
<img src="/assets/dl/bp.jpg" style="border:none;width:80%">
</div>

Compute the first paritial derivative $$ \frac{\partial J}{\partial out_i} $$ from the right most layer.

$$
J = \frac{1}{N} \sum_i (out_i - y_i)^2
$$

$$
J_i = \frac{1}{N} (out_i - y_i)^2
$$

$$
\frac{\partial f}{\partial out_i} = \frac{2}{N} (out_i - y_i)
$$

We add a line of code in the mean square loss to compute $$ \frac{\partial J}{\partial out_{i}} $$

```python
def mean_square_loss(h, y):
    # h: prediction (N,)
    # y: true value (N,)
    ...
    dout = 2 * (h-y) / N                  # Compute the partial derviative of J relative to out
    return loss, dout
```

In DL programing, we often name our backpropagation derivative as:

$$
\frac{\partial \text{ J}}{\partial \text{ out}} \text{ as dout}
$$

$$
\frac{\partial \text{ f}}{\partial \text{ var}} \text{ as dvar}
$$

Now we have
$$
\frac{\partial J}{\partial out_i}
$$ 
. We apply the chain rule to backpropagate the gradient one more layer to the left to compute $$ \frac{\partial J}{\partial W} \text{ as } dW $$ and $$ \frac{\partial J}{\partial b} \text{ as } db $$.

<div class="imgcap">
<img src="/assets/dl/bp3.jpg" style="border:none;width:60%">
</div>

<div class="imgcap">
<img src="/assets/dl/bp2.jpg" style="border:none;width:60%">
</div>

Apply chain rule:

$$
J = \frac{1}{N} \sum_i (out_i - y_{i})^2
$$

$$
\frac{\partial J}{\partial W} = \frac{\partial J}{\partial out} \frac{\partial out}{\partial W}  
$$ 

$$
\frac{\partial J}{\partial b} = \frac{\partial J}{\partial out} \frac{\partial out}{\partial b}  
$$ 

With the equation $$ out $$, we take the partial derviative.

$$
out = W * X + b
$$

$$
\frac{\partial out}{\partial W}  = X
$$ 

$$
\frac{\partial out}{\partial b}  = 1
$$ 

Compute the derivative with the chain rule:

$$
\frac{\partial J}{\partial W} = \frac{\partial J}{\partial out} X
$$ 

$$
\frac{\partial J}{\partial b} = \frac{\partial J}{\partial out}
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

Here is the full listing of the code:
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

### General principle in backpropagation

Machine learning library provides pre-built layers with feed forward and backpropagation. However many DL class assignments spend un-proportional amount of time in backpropagation. With vectorization and some ad hoc functions, the process is error prone but not necessary hard. Let's summaries the step above again with some tips.

> Draw the forward pass and backpropagation pass with clear notication of variables that we used in the program.

> Add functions, derivatives and the shape for easy reference.

<div class="imgcap">
<img src="/assets/dl/fp.jpg" style="border:none;width:75%">
</div>

<div class="imgcap">
<img src="/assets/dl/bp.jpg" style="border:none;width:75%">
</div>

Perform a forwad pass to calculate the output and the cost:

$$
out = W_1 \cdot  X_{1}  + W_2 \cdot X_{2} + b
$$

$$
J = \frac{1}{N} \sum_i (out - y_{i})^2
$$

Find the partial derviative of the cost:

$$
\frac{\partial J}{\partial out} = \frac{2}{N} (out - y)
$$

For every layer, compute the function derviate:

$$
out = W * X + b
$$

$$
\frac{\partial out}{\partial W}  = X
$$ 

$$
\frac{\partial out}{\partial b}  = 1
$$ 

Find the total gradient with the chain rule from right to left:

$$
\frac{\partial J}{\partial l_{k-1}} = \frac{\partial J}{\partial l_{k}} \frac{\partial l_k}{\partial l_{k-1}}  
$$ 

<div class="imgcap">
<img src="/assets/dl/chain.png" style="border:none;width:75%">
</div>

$$
\text{dl1} \equiv \frac{\partial J}{\partial l_{1}} = \frac{\partial J}{\partial out} \frac{\partial out}{\partial l_{1}}  = \frac{\partial J}{\partial out} \frac{\partial f_{2}}{\partial l_{1}}  
$$ 

$$
\text{dl1} = \text{dout} \cdot \frac{\partial f_{2}}{\partial l_{1}}  
$$ 

Similary,

$$
\text{dW} = \text{dl1} \cdot \frac{\partial f_{1}}{\partial W}  
$$ 

> Put the function derviative in the diagram make this step easy.

> Expanding the equation with index sometimes will help to figure out the backpropagation step.

#### More on backpropagation

In backprogragation, we may backprogate multiple path back to the same node. To compute the gradient correctly, we need to add both path together:
<div class="imgcap">
<img src="/assets/dl/bp_m1.jpg" style="border:none;width:50%">
</div>

$$
\frac{\partial J}{\partial o_3}  = \frac{\partial J}{\partial o_4} \frac{\partial f_4} {\partial o_3} + \frac{\partial J}{\partial o_5} \frac{\partial f_4} {\partial o_{3}} 
$$

$$
\text{do3}  = \text{do4} \frac{\partial f_4} {\partial o_3} + \text{do5} \frac{\partial f_4} {\partial o_{3}} 
$$

### Testing the model

I strongly recommend you to think about a linear regression problem that interested you, and train a simple network now. A lot of issues happened in a complex model will show up even in such a simple model. With a complex model, you treat it as a black box and many actions are purely random guesses. Work with a model designed by yourself, you can create better sceaniors to test your theories and develop a better insight on DL. Most tutorial have already pre-cooked parameters. So they teach you the easier part without having you to struggle on the hard part.

So let Pieter train the system.
```
iteration 0: loss=2.825e+05 W1=0.09882 dW1=1.183e+04 W2=-0.4929 dW2=5.929e+06 b= -8.915e-05 db = 891.5
iteration 200: loss=3.899e+292 W1=-3.741e+140 dW1=4.458e+147 W2=-1.849e+143 dW2=2.203e+150 b= -2.8e+139 db = 3.337e+146
iteration 400: loss=inf W1=-1.39e+284 dW1=1.656e+291 W2=-6.869e+286 dW2=8.184e+293 b= -1.04e+283 db = 1.24e+290
iteration 600: loss=nan W1=nan dW1=nan W2=nan dW2=nan b= nan db = nan
```
The application overflow within 600 iterations! Since the loss and the graident is so high, we test out whether the learning rate is too high. We decrease the learning rate. With learning rate of 1e-8, we do not have the overflow problem but the loss is high.
```
iteration 90000: loss=4.3e+01 W1= 0.23 dW1=-1.3e+02 W2=0.0044 dW2= 0.25 b= 0.0045 db = -4.633
W = [ 0.2437896   0.00434705]
b = 0.004981262980767952
```

We are very reluctant to take actions without information. But since the application runs very fast, we can give a few simple guess. With10,000,000 iterations and a learning_rate of 1e-10, the loss is still very high. It will be better to trace the source of problem now.

```
iteration 9990000: loss=3.7e+01 W1= 0.22 dW1=-1.1e+02 W2=0.0043 dW2= 0.19 b= 0.0049 db = -4.593
W = [ 0.22137005  0.00429005]
b = 0.004940551119084607
```

> Tracing gradient is another powerful tool in DL debugging.

Even the loss in our first try shows similar symptom as a bad learning rate, we suspect this is not the root cause. After some tracing, we find the gradient is very high. We plot the cost function relative to W to illustrate the real issue.

This is a U shape curve which is different from a bowl shape curve that we use in the gradient descent explanation. 
<div class="imgcap">
<img src="/assets/dl/ushape.png" style="border:none;width:50%">
</div>

<div class="imgcap">
<img src="/assets/dl/solution.png" style="border:none;width:50%">
</div>

The Y-axis is $$ W_2 $$ (monthly income) and the X-axis is $$ W_1$$ (year of education). Cost response more aggressive with $$ W_2 $$ than $$ W_{1} $$. Monthly income ranges from 0 to 10,000 while year of education range from 0 to 30. Obviously, the different scale in these 2 features cause a major difference in its gradient. Because of the different scale, we cannot have a single learning rate than can work well for both of them. The solution is pretty simple with a couple line of code change. We re-scale the income value.

```python
def true_y(education, income):
    dates = 0.8 * education + 0.3 * income + 2
    return dates

...

income = np.random.randint(10, size=education.shape[0]) # (N,) Generate the corresponding income.
```

 Here is the output which is close to our true model:
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

Pieter come back and realize our linear model is not adequate. Pieter claims the relationship between years of education and dates are not exactly linear. We should give more rewards for people holding advance degree.
<div class="imgcap">
<img src="/assets/dl/educ.png" style="border:none;width:50%">
</div>

Can we combine 2 linear functions with multiple layers to form a non-linear function?

$$
f(x) = Wx + b
$$

$$
g(z) = Uz + c
$$

The answer is no.

$$
g(f(x)) = U(Wx+b) + c = Vx + d
$$

After some thoughts, we apply the following to the output of a computation node.

$$
f(x) = max(0, x)
$$

With this function, in theory, we can construct the non-linear relations that Pieter wants.
<div class="imgcap">
<img src="/assets/dl/l1.png" style="border:none;width:80%">
</div>
Adding both output:
<div class="imgcap">
<img src="/assets/dl/l2.png" style="border:none;width:80%">
</div>

Add a non-linear function after a linear equation can enrich the complexity of our model. These methods are usually call **activation function**. Common functions are tanh and ReLU.
 
#### Sigmoid
Sigmoid is one of the earliest function used in deep networks. Neverthless, as an activation function, its importance is gradually replaced by other functions like ReLU. Currently, sigmoid function is more popular as a gating function in LSTM/GRU (an "on/off" gate) to selectively remember or forget information. Discssion of sigmoid function as an activation function is more of a showcase of explaining why network can be hard to train.

<div class="imgcap">
<img src="/assets/dl/sigmoid.png" style="border:none;width:50%">
</div>

#### ReLU
ReLU is one of the most popular activation function. Its popularity arries because it works better with gradient descent. It performs better than the sigmoid function because the sigmoid node is saturated easier and work less efficient with gradient descent in the saturated area.

<div class="imgcap">
<img src="/assets/dl/relu.png" style="border:none;width:50%">
</div>

#### tanh
Popular for output prefer to be within [-1, 1]
<div class="imgcap">
<img src="/assets/dl/tanh.png" style="border:none;width:50%">
</div>

As mentioned before, your want the gradient descent to follow the curve of the cost function but not in some crazy zipzap pattern. For sigmoid function, the output is always positive.  According to the formular below, the backprogate gradient for W will be subject to the sign of $$  \frac {\partial J}{\partial l_{k+1}} $$ since $$X$$ is always positive after the sigmoid layer. In practice, it forces all $$ W $$ in this layer to move in the same direction and therefore the gradient descent will follow a zipzap pattern. 

$$
\frac {\partial l_{k}}{\partial W} = X \cdot \frac {\partial J}{\partial l_{k+1}} 
$$

SInce tanh has positive and negative output and therefore prefered over sigmoid.

Implement these functions with python and numpy.
```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def relu(x):
  return np.maximum(0, x)

def tanh(x):
  return np.tanh(x)

x = np.arange(-20, 20, 0.001)

y = sigmoid(x)
plt.axvline(x=0, color="0.8")
plt.plot(x, y)
plt.show()

y = relu(x)
plt.axvline(x=0, color="0.8")
plt.axhline(y=0, color="0.8")
plt.plot(x, y)
plt.show()

y = tanh(x)
plt.plot(x, y)
plt.axhline(y=0, color="0.8")
plt.show()

```

### Feed forward and backpropagation for sigmoid and ReLU

The derivative for ReLU function:

$$
f(x)=\text{max}(0,x)
$$

$$
\begin{equation} 
f'(x)=
    \begin{cases}
      1, & \text{if}\ x>0 \\
      0, & \text{otherwise}
    \end{cases}
\end{equation}
$$

The derivative for sigmoid:

$$
\sigma (x) = \frac{1}{1+e^{-x}}
$$

With some calculus:

$$
\frac{d\sigma (x)}{d(x)} = \sigma (x)\cdot (1-\sigma(x))
$$

```python
def relu_forward(x):
    cache = x
    out = np.maximum(0, x)
    return out, cache

def relu_backward(dout, cache):
    out = cache
    dh = dout
    dh[out < 0] = 0
    return dh

def sigmoid_forward(x):
    out = 1 / (1 + np.exp(-x))
    return out, out

def sigmoid_backward(dout, cache):
    dh = cache * (1-cache)
    return dh * dout
```
### Fully connected network

Let's apply all our knowledge so far to build a fully connected network as follows:
<div class="imgcap">
<img src="/assets/dl/fc_net.png" style="border:none;width:40%">
</div>

For each nodes in the hidden layer (except the last layer), we apply:

$$
z_j = \sum_{i} W_{ij} x_{i} + b_{i}
$$

$$
h(z_j)=\text{max}(0,z_{j})
$$

For the layer before the output, we apply only the linear equation but not the ReLU equation.

Here is the code performaing the forward pass with 4 hidden layers:
```python
z1, cache_z1 = affine_forward(X, W[0], b[0])
h1, cache_relu1 = relu_forward(z1)

z2, cache_z2 = affine_forward(h1, W[1], b[1])
h2, cache_relu2 = relu_forward(z2)

z3, cache_z3 = affine_forward(h2, W[2], b[2])
h3, cache_relu3 = relu_forward(z3)

z4, cache_z4 = affine_forward(h3, W[3], b[3])
```

Here is the backpropagation:
```python
dz4, dW[3], db[3] = affine_backward(dout, cache_z4)

dh3 = relu_backward(dz4, cache_relu3)
dz3, dW[2], db[2] = affine_backward(dh3, cache_z3)

dh2 = relu_backward(dz3, cache_relu2)
dz2, dW[1], db[1] = affine_backward(dh2, cache_z2)

dh1 = relu_backward(dz2, cache_relu1)
_, dW[0], db[0] = affine_backward(dh1, cache_z1)
```

For those interested in details, we list some of the code we change/add:
```python
iteration = 100000
learning_rate = 1e-4
N = 100
...
def affine_forward(x, W, b):
    # x: input sample (N, D)
    # W: Weight (2, k)
    # b: bias (k,)
    # out: (N, k)
    out = x.dot(W) + b           # (N, D) * (D, k) -> (N, k)
    cache = (x, W, b)
    return out, cache

def affine_backward(dout, cache):
    # dout: dJ/dout (N, K)
    # x: input sample (N, D)
    # W: Weight (D, K)
    # b: bias (K, )
    x, W, b = cache

    if len(dout.shape)==1:
        dout = dout[:, np.newaxis]
    if len(W.shape) == 1:
        W = W[:, np.newaxis]

    W = W.T
    dx = dout.dot(W)

    dW = x.T.dot(dout)
    db = np.sum(dout, axis=0)

    return dx, dW, db

def relu_forward(x):
    cache = x
    out = np.maximum(0, x)
    return out, cache

def relu_backward(dout, cache):
    out = cache
    dh = dout
    dh[out < 0] = 0
    return dh

def mean_square_loss(h, y):
    # h: prediction (N,)
    # y: true value (N,)
    N = X.shape[0]                        # Find the number of samples
    h = h.reshape(-1)
    loss = np.sum(np.square(h - y)) / N   # Compute the mean square error from its true value y
    nh = h.reshape(-1)
    dout = 2 * (nh-y) / N                  # Compute the partial derivative of J relative to out
    return loss, dout

def compute_loss(X, W, b, y=None, use_relu=True):
    z1, cache_z1 = affine_forward(X, W[0], b[0])
    h1, cache_relu1 = relu_forward(z1)

    z2, cache_z2 = affine_forward(h1, W[1], b[1])
    h2, cache_relu2 = relu_forward(z2)

    z3, cache_z3 = affine_forward(h2, W[2], b[2])
    h3, cache_relu3 = relu_forward(z3)

    z4, cache_z4 = affine_forward(h3, W[3], b[3])

    if y is None:
        return z4, None, None

    dW = [None] * 4
    db = [None] * 4
    loss, dout = mean_square_loss(z4, y)

    dz4, dW[3], db[3] = affine_backward(dout, cache_z4)

    dh3 = relu_backward(dz4, cache_relu3)
    dz3, dW[2], db[2] = affine_backward(dh3, cache_z3)

    dh2 = relu_backward(dz3, cache_relu2)
    dz2, dW[1], db[1] = affine_backward(dh2, cache_z2)

    dh1 = relu_backward(dz2, cache_relu1)
    _, dW[0], db[0] = affine_backward(dh1, cache_z1)

    return loss, dW, db
...

W = [None] * 4
b = [None] * 4

W[0] = np.array([[0.7, 0.05, 0.0, 0.2, 0.2], [0.3, 0.01, 0.01, 0.3, 0.2]])  # (2, K)
b[0] = np.array([0.8, 0.2, 1.0, 0.2, 0.1])

W[1] = np.array([[0.7, 0.05, 0.0, 0.2, 0.1], [0.3, 0.01, 0.01, 0.3, 0.1], [0.3, 0.01, 0.01, 0.3, 0.1], [0.07, 0.04, 0.01, 0.1, 0.1], [0.1, 0.1, 0.01, 0.02, 0.03]])  # (K, K)
b[1] = np.array([0.8, 0.2, 1.0, 0.2, 0.1])

W[2] = np.array([[0.7, 0.05, 0.0, 0.2, 0.1], [0.3, 0.01, 0.01, 0.3, 0.1], [0.3, 0.01, 0.01, 0.3, 0.1], [0.07, 0.04, 0.01, 0.1, 0.1], [0.1, 0.1, 0.01, 0.02, 0.03]])  # (K, K)
b[2] = np.array([0.8, 0.2, 1.0, 0.2, 0.1])

W[3] = np.array([[0.8], [0.5], [0.05], [0.05], [0.1]])                 # (K, 1)
b[3] = np.array([0.2])

X = np.concatenate((education[:, np.newaxis], income[:, np.newaxis]), axis=1) # (N, 2) N samples with 2 features

for i in range(iteration):
    loss, dW, db = compute_loss(X, W, b, Y)
    for j, (cdW, cdb) in enumerate(zip(dW, db)):
        W[j] -= learning_rate * cdW
        b[j] -= learning_rate * cdb
    if i%20000==0:
        print(f"iteration {i}: loss={loss:.4}")
```

We also generate some testing data to measure how well our model can predict.
```python
TN = 100
test_education = np.full(TN, 22)
test_income = np.random.randint(TN, size=test_education.shape[0])
test_income = np.sort(test_income)

true_model_Y = true_y(test_education, test_income)
true_sample_Y = sample(test_education, test_income, verbose=False)
X = np.concatenate((test_education[:, np.newaxis], test_income[:, np.newaxis]), axis=1)

out, _, _ = compute_loss(X, W, b)
loss_model, _ = mean_square_loss(out, true_model_Y)
loss_sample, _ = mean_square_loss(out, true_sample_Y)

print(f"testing: loss (compare with Oracle)={loss_model:.6}")
print(f"testing: loss (compare with sample)={loss_sample:.4}")
```

We plot the result with our predicted value from our computed model vs the one from the true model. (The model from the Oracle.) In this first model, we have 2 hidden layers. When making prediction, we fix the number of years in education to 22, and plot how the number of dates varied with income.The orange line is what we may predicted from our model, the blue dot is from the Oracle, and orange dot is when we add some noise to the Oracle model for the training dataset. The data match pretty well with each other.
<div class="imgcap">
<img src="/assets/dl/fc_2l.png" style="border:none;width:60%">
</div>

> Congratulations! We just solve a problem using deep learning!

In real life, instead of 2 inputs (length of education and monthly income) to the network, there may be a couple dozens **features** (input). For more complex problems, we add more layers to the fully connected network (FC) to enrich the model. For object recognition in a small image (about a thousand pixels), we convert each pixel into a feature and feed it to a FC. Nevertheless, if we want to push the accuracy up for larger image or more complex visual problems, we add convolution layers in front of the FC. Nevertheless, learning the FC covers the critical techniques that are common to other types of networks.

> In machine learning, we call the input "feature". In visual problems, we may visualize what activates a node. We call it what feature a network is extracting.

Now we increase the hidden layer from 2 to 4. We find that our prediction accuracy drops. And it takes more time to train and more tuning. When we plot it in 3D with variable income and eduction, we realized some part of the 2D plain is bended instead of flat.
<div class="imgcap">
<img src="/assets/dl/fc_4l.png" style="border:none;width:60%">
</div>

<div class="imgcap">
<img src="/assets/dl/fc_4l2.png" style="border:none;width:60%">
</div>

When we create the training dataset, we add some noise to our true value (# of dates). When we use a more complex model, it also increase its capability to model the noise signal. When we have a large dataset, the effect of noise should cancel out. However, if the training dataset is not big enough and the model is complex, the accuracies in making prediction actually suffer compare with a simplier model. In this exercise, when we increase the model complexity, the model get harder to train and optimize, and the accuracy drops. In genearl, we should always starts with some simple model and increase its complexity later. It is hard to tell whether we need more tuning/iterations or it simpliy not working if we jump into a complex model too soon.

```python
def sample(education, income, verbose=True):
    dates = true_y(education, income)
    noise =  dates * 0.15 * np.random.randn(education.shape[0]) # Add some noise to our true model.
    return dates + noise
```

We also replace our ReLU function with a sigmoid function and plot the same diagram:

<div class="imgcap">
<img src="/assets/dl/fc_si1.png" style="border:none;width:60%">
</div>
<div class="imgcap">
<img src="/assets/dl/fc_si2.png" style="border:none;width:60%">
</div>

Below is one of the model generated. Because we start with random guess of $$ W $$, we end up with quite different models for each run.
```
Layer 1:
      [[ 1.10727659,  0.22189273,  0.13302861,  0.2646622 ,  0.2835898 ],
       [ 0.23522207,  0.01791731, -0.01386124,  0.28925567,  0.187561  ]]
	   
Layer 2:
 	[[ 0.9450821 ,  0.14869831,  0.07685842,  0.23896402,  0.15320876],
       [ 0.33076781,  0.02230716,  0.01925127,  0.30486342,  0.10669098],
       [ 0.18483084, -0.03456052, -0.01830806,  0.28216702,  0.07498301],
       [ 0.11560201,  0.05810744,  0.021574  ,  0.10670155,  0.11009798],
       [ 0.17446553,  0.12954657,  0.03042245,  0.03142454,  0.04630127]]), 
	   
Layer 3:	   
	[[ 0.79405847,  0.10679984,  0.00465651,  0.20686431,  0.11202472],
       [ 0.31141474,  0.01717969,  0.00995529,  0.30057041,  0.10141655],
       [ 0.13030365, -0.09887915,  0.0265004 ,  0.29536237,  0.07935725],
       [ 0.07790114,  0.04409276,  0.01333717,  0.10145275,  0.10112565],
       [ 0.12152267,  0.11339623,  0.00993313,  0.02115832,  0.03268988]]), 
	   
Layer 4:	  
	[[ 0.67123192],
       [ 0.48754364],
       [-0.2018187 ],
       [-0.03501616],
       [ 0.07363663]])]
```


[Part 2 of the deep learning is here](https://jhui.github.io/2017/03/17/Deep-learning-tutorial-2/)



















