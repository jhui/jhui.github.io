---
layout: post
comments: true
mathjax: true
title: “Deep learning without going down the rabbit holes.”
excerpt: “How to learn deep learning from easy concept to complex idea? How to build insight along the way?”
date: 2017-03-01 14:00:00
---
**This is work in progress... The content needs major editing.**

### What is deep learning (DL)?
**Deep learing is about building a function estimator.** Historically, people explains deep learning (DL) using the neural network in our brain. Indeed, this is where deep learning gets its insight.  Nevertheless, deep learning has out grown this explaination. Once you realize building a deep learning network is about building a function estimator, you will unveil its real potential in AI.
 
Let us build a new andriod named Pieter. Our first task is to teach Pieter how to recognize visual objects. Can the visual system in our brain be replaced by a big function estimator? Can we pass the pixel values of an image to a function and calculate the probability of seeing a school bus, an airplane or a truck etc ...?

<div class="imgcap">
<img src="/assets/dl/deep_learner.jpg" style="border:none;width:70%;">
</div>

Indeed, in our later example of image recognition, we will build a system very similar to the following:
<div class="imgcap">
<img src="/assets/dl/fc.jpg" style="border:none;width:80%;">
</div>

For every node, we compute:

$$
f(z_j) = \frac{1}{1 + e^{-z_j}}
$$

with, 

$$
z_j = \sum_{i} W_{ij} x_{i} + b_{i}
$$

which $$ x_{i} $$ is the input to the node or simply the pixel value if this is the first layer. These equation looks intimidating. But let us go through one example to illustrate how simple it is. For example, with weight W  (0.3, 0.2, 0.4, 0.3), bias b (-0.8) and a grayscale image with just 4 pixels (0.1, 0.3, 0.2, 0.1), the output of the first node circled in red above will be:

$$
z_j =  0.3*0.1 + 0.2*0.3 + 0.4*0.2 + 0.3*0.1  - 0.8 = -0.6
$$

$$
f(z) =  \frac{1}{1 + e^{-(-0.6)}} = 0.3543
$$

Each node will have its own set of weight (W) and bias (b). From the left most layer, we compute the output of each node and we feed forward the output to the next layer. Eventually, the right most layer is the probability for each object classification (a school bus, an airplane or a truck). In this exercise, we supply all the weight and bias values to our android Pieter. But as the term "deep learning" may imply, by the end of this tutorial, Pieter will learn those parameters by himself. We still miss a few pieces for the puzzle. But the network diagram and the equations above already lay down the foundation of a deep learning network. In fact, this simple design can recognize the zip code written on a envelop with reasonable high accuracy.

#### XOR
For the skeptics, we will build an exclusive "or" (a xor b) using a simple network like:
<div class="imgcap">
<img src="/assets/dl/xor.jpg" style="border:none;width:40%">
</div>
For each node, we apply the same equations mentioned before:

$$
h_j = \sigma(z) = \frac{1}{1 + e^{-z_j}}
$$

$$
z_j =  \sum_{i} W_i * x_i + b_{i}
$$

The following is our code implementation. It is pretty self-explainatory. The core purpose is to demonstrate how we perform the weight multiplication and apply the sigmoid function.

> We provide coding to help audience to understand the concept deeper and to verify their understanding. Nevertheless, a full understand of the code is not needed or suggested.

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

Implement a XOR or a delta function is not important for deep learning (DL). Nevertheless, we demonstrate the possibilities of building a complex function estimator through a network of simple computation nodes. A network with 3 layers can implement a hand written recognition system for numbers with an accuracy of 95+%. The deeper a network the more complex model that we can build. For example, Microsoft ResNet (2015) for image recognition has 151 layers. For many AI problems, the model needed to solve the problem is very complex. In automous driving, we can model a policy (turn, accelerate or brake) to approximate what a human will do for what they see in front of them. This policy is too hard to model it analytically. Alterantive, with enough training data, we may train a deep learning network with high enough accuracies as a regular driver.

<div class="imgcap">
<img src="/assets/dl/drive.jpg" style="border:none;width:50%">
</div>

> Autonomus driving involves many aspect of AI. DL provides a model estimator that cannot be done analytically.

### Build a Linear regression model
We will build a model to demonstrate how Pieter can learn the parameters of a model by processing training data. For example, Pieter wants to expand on his horizon and start online dating. He wants to find out the relationship between the number of online dates with the number of years of eductaion and the monthly income.  Pieter starts with a simple linear model as follows:

$$
\text {number of dates} = W_1* \text{years in school} + W_2*\text{monthly income} + bias
$$

He asks 1000 people in each community and collects the information on their income, education and the corresponding number of online dates.  Pieter is interested in knowing how each community values the intellectual vs his humble post-doc salary.  So even this model looks overwhemly simple, it serves its purpose. So the task for Pieter is to find the parameter values for W and b in this model with training data collected by him.

This is the high level steps for Pieter to train a model.
1. Take a first guess on W and b.
2. Use the model above to compute the number of dates for each sample in the training dataset.
3. Compute the mean square error between the computed value and the true value in the dataset.
4. Compute how much the error may change when we change W and b.
5. Re-adjust W & b according to this error rate change. (**Gradient descent**)
6. Repeat step 2 for N iterations.
7. Use the last value of W & b for our model.

We can build a model with different W & b for each community, and use these models to predict how well Pieter may do in each community.

> In our model, we predict the number of dates for people with certain income and year of education. The corresponding values (the number of dates) that we found for each sample in the dataset are called the **true values**.

### Gradient descent
**Deep learing is about learning how much it cost.** Step 2-5 is called the gradient descent in DL. First we define a function to measure our errors between our model and the true values. In DL, we call this error function **cost function** or **loss function**. Mean square error (MSE) is one obvious candidate for our model.

$$
\text{mean square error} = J(h, y, W, b) = \frac{1}{N} \sum_i (h_i - y_i)^2
$$

where $$ h_i $$ is the model prediction and $$ y_i $$ is the true value. We sum over all the samples and take the average.
We can visualize the cost below with x-axis being $$ W_1 $$ and y-axis being $$ W_2 $$ and z being the cost J(x, y). The solution of our model is to find $$ W_1 $$ and $$ W_2 $$ where the cost is lowest. Visualize dropping a marble at a random point $$ (W_1, W_2) $$ and let the gravity to do its work. 

<div class="imgcap">
<img src="/assets/dl/solution.png" style="border:none;">
</div>

### Learning rate

Thinking in 3D or high dimensions are hard to impossible. Try to think DL problems in 2D first.

Consider a point at (L1, L2), we cut through the diagram alone the red and orange and plot those curve in a 2D diagram:
<div class="imgcap">
<img src="/assets/dl/solution_2d.jpg" style="border:none;">
</div>

<div class="imgcap">
<img src="/assets/dl/gd.jpg" style="border:none;">
</div>

The X-axis is the value of $$ W $$ and the Y-axis is its corresponding average cost. (Since we are holding the other input as constant, we will ignore it for now.)

$$
J(W, b, h, y) = \frac{1}{N} \sum_i (W_1*x_i - y_i)^2
$$


Since the gradient at L1 is negative (as shown), we move $$ W_1 $$ to the right to find the lowest point. But by how much? Let's compare the gradient for L1 and L2. We realize L2 has a smaller gradient. i.e. the change of $$ W2 $$ has a smaller impact to the change of cost compare to L1. Obviosuly, we want to change parameter proportional to how fast it can drop the cost. Therefore, the amount of adjustment for the parameters $$ (W_1, W_2) $$ should be proportional to its partial gradient at that point. i.e.

$$
\Delta W_i \propto \frac{\partial J}{\partial W_i} 
$$

$$
\text{ i.e. } \Delta W_1 \propto \frac{\partial J}{\partial W_1} \text{ and } \Delta W_2 \propto \frac{\partial J}{\partial W_2}
$$

So the adjustments we want to make are:

$$
\Delta W_i = \alpha \frac{\partial J}{\partial W_i}
$$

$$
W_i = W_i - \Delta W_i
$$

In DL, the varaible $$ \alpha $$ is called the **learning rate**.  Small learning rate will take a longer time (more iteration) to find the minima. However, as we learn from calculus, the larger the step, the bigger the error in our calculation. In DL, finding the right value of learning rate is sometimes a try and error exercise.  Sometimes we will try values ranging from 1e-7 to 1 in logarithmic scale (1e-7, 5e-7, 1e-6, 5e-6, 1e-5 ...). 

Large learning step may have other serious problem. It costs w to oscillate with increasing cost:
<div class="imgcap">
<img src="/assets/dl/learning_rate.jpg" style="border:none;">
</div>

We start with w = -6 (x-axis) at L1. If the gradient is huge, certain learning rate larger than some value will swing w too far to the other side (say L2) with even a larger gradient. Eventually, rather than drop down slowly to a minima, w keeps oscalliate and the cost keep sincreasing. When loss starts going upward during training, we need to reduce the learning rate. The follow demonstrates how a learning rate of 0.8 with a steep gradient may swing the cost upward instead of downward. The table traces how the oscillation of W causes the cost go upwards from L1 to L2 and then L3.

<div class="imgcap">
<img src="/assets/dl/lr_flow.png" style="border:none;">
</div>

> Sometimes, we need to be careful about the scale used in plotting the x-axis and y-axis. In the diagram shown above, the gradient does not seem large.  It is because we use a much smaller scale for y-axis than the x-axis (0 to 150 vs -10 to 10).

Here is another illustration for some real problems.  When we gradudally descent, we may land in an area with steep gradient which the W will bounce back. This type of shape is very hard to reach the minima with a constant learning rate. Advance methods to address this problem will be discussed later.

<div class="imgcap">
<img src="/assets/dl/ping.jpg" style="border:none;">
</div>

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

When computing the cost function, we add all the errors for the processed training data. We can process all the training data at once but this can take too much time for just one update. On the contrray, we can perform stochastic gradient descent which make one W update per training sample. Nevertheless, the gradient descent will follow a zigzag pattern rather than following the contour of the cost function. For steep gradient area, this can be a problem. Also it may takes longer to reach the minima. A good compromise is to process a batch of N samples at a time. This can be a tunable hyper-parameter but usually not very critical and may start with 64 subject to the memory consumptions.

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

Add a non-linear function after a linear equation can enrich the complexity of our model. These methods are usually call **activation function**. Common functions are sigmoid, tanh and ReLU.
 
#### Sigmoid
Sigmoid is one of the earliest function used in deep networks. Neverthless, as an activation function, its importance is gradually replaced by other functions like ReLU. Currently, sigmoid function is more popular as a gating function in LSTM/GRU (an "on/off" gate) to selectively remember or forget information.  

<div class="imgcap">
<img src="/assets/dl/sigmoid.png" style="border:none;width:50%">
</div>

#### ReLU
ReLU is one of the most popular activation function. Its popularity arries because it works better with gradient descent. It takes less time to train with better accuracies than sigmoid.

<div class="imgcap">
<img src="/assets/dl/relu.png" style="border:none;width:50%">
</div>

#### tanh
Popular for output prefer to be within [-1, 1]
<div class="imgcap">
<img src="/assets/dl/tanh.png" style="border:none;width:50%">
</div>

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

In real life, instead of 2 inputs (length of education and monthly income) to the network, there may be a couple dozens **features** (input). For more complex problems, we add more layers to the fully connected network (FC) to enrich the model. For object recognition in small image (about a thousand pixels), we convert each pixel into a feature and feed it to a FC. Nevertheless, if we want to push the accuracy up for larger image or more complex visual problems, we add convolution layers in front of the FC. Nevertheless, learning the FC covers the critical tecniques that is common to other type of networks.

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

When we model a simple model with 4 layers of computation nodes. We end up with many possible solutions with different $$ W $$. Should we prefer one solution over the other? Should we prefer a solution with smaller values in $$ W $$ over the other? Are part of the network just cancel out the effect of the other part?

### Overfit
This lead us to a very important topic in DL.  We know that when we increase the complexity of our model, we risk the chance of modeling the noise into a model. If we do not have enough sample data to cancel out each other, we make bad predictions. But even with no noise in our data, a complex model can still make mistakes even when we train it well and long enough.

Let's walk through another example. We start with training samples with input values range from 0 to 20, how will you connect the dots or how you will create a equation to model the data below.
<div class="imgcap">
<img src="/assets/dl/d1.png" style="border:none;width:70%">
</div>

One possiblity is
$$
y = x
$$ which is simple and just miss 2 on the left and 2 on the right of the line.
<div class="imgcap">
<img src="/assets/dl/d2.png" style="border:none;width:70%">
</div>

But when we show it to Pieter which has much higher computation capability than us, he models it as:

$$
y = 1.9  \cdot 10^{-7}  x^9 - 1.6 \cdot 10^{-5} x^8 + 5.6 \cdot 10^{-4} x^7 - 0.01 x^6  + 0.11 x^5 - 0.63 x^4 + 1.9  x^3 - 2.19  x^2 + 0.9 x - 0.0082
$$

Which does not miss a single point in the sample.
<div class="imgcap">
<img src="/assets/dl/d3.png" style="border:none;width:70%">
</div>

Which model is correct? The answer is "don't know". Some people may think the first one is simplier and simple explanation deserves more credits. But if you show it to a stock broker, they will say the second curve is more real if it is the market closing price of a stock. 

Instead, we should ask whether our model is too "custom taylor" for the sample data so it makes bad prediction. The second curve fit the sample data100% but will make some bad predictions if the true model is a straight line.

#### Validation
**Machine learning is about making prediction.** A model that has 100% accuracy in training can be a bad model in making prediction. For that, we often split our testing data into 3 parts: 80% for training, 10% for validation and 10% for testing. During training, we use the training dataset to build models with different network designs and hyper parameters like learning rate. We run those models with the validation dataset and pick the model with the highest accuracy. This strategy works if the validation dataset are close to what we want to predict. Otherwise, the picked model can make bad predictions. As the last safeguard, we use the 10% testing data for a final insanity check. This testing data is for one last verification but not for model selection. If your testing result is dramatically difference from the validaton result, we need to randomize the data more, or to collect more data.

#### Visualization 
We can train a model to create a boundary to separate the blue dots from the white dots below. Complex model produces far more sophiscated boundary shape than a low complexity model. In the circled area, if we miss the 2 left white dot samples in our training, a complex model may create an odd shape boundary just to include this white dot. A low complexity model can only produce a smoothier surface which by chance may be more desireable. Complex model may also more vulernable to outliners which a simple model may just ignore like the white dot in the green circle.

<div class="imgcap">
<img src="/assets/dl/of.png" style="border:none;width:60%">
</div>

Recall from Pieter's equation, our sample data can be model nicely with the following equations:

$$
y = 1.9  \cdot 10^{-7}  x^9 - 1.6 \cdot 10^{-5} x^8 + 5.6 \cdot 10^{-4} x^7 - 0.01 x^6  + 0.11 x^5 - 0.63 x^4 + 1.9  x^3 - 2.19  x^2 + 0.9 x - 0.0082
$$

In fact there are infinite answers using different polynomal orders $$ x^k \cdots $$

Comparing with the linear model $$ y = x $$, we realize that the 
$$ || coefficient || $$ 
for Pieter equation is higher. If we have a model using high polynormal order, it will be much harder to train because we are dealing with a far bigger search space for the parameters. In additional, some of the search space will have very steep gradient.

Let us create a polynomal model with order 5 to fit our sample data, and see how the training model make predictions.

$$
y = c_5 x^5 + c_4 x^4 + c_3 x^3 + c_2 x^2 + c_1 x + c_{0}
$$

We need far more iterations to train this model and result in less accuracy than a model with order 3.
<div class="imgcap">
<img src="/assets/dl/p1.png" style="border:none;width:60%">
</div>

But why don't we focus on making a model with the right complexity. In real life problems, a complex model is the only way to push accuracy to an acceptable level. But yet overfitting in some region is un-avoidable. One solution is to add more sample data such that it is much harder to overfit. Here, double the sample data produces a model closer to a straight line. 
<div class="imgcap">
<img src="/assets/dl/p2.png" style="border:none;width:60%">
</div>

### Regularization

As we observe before, there are many solutions to the problem but in order to have a very close fit, the coefficient in our training parameters tends to have larger magnitude. 

$$
||c|| = \sqrt{(c_5^2 + c_3^2 + c_3^2 + c_2^2 + c_1^2 + c_{0}^2)}
$$

For example, if we set $$ c_{2} $$ ... $$ c_{5} $$ to 0 or very close to 0, we have a very simple straight line model. To encourage our training not to be too agressive to overfit the training data, we add a penalty in our cost function to penalize large magnitude.

$$
J = \text{mean square error} + \lambda \cdot ||W||
$$

Techniques to discourage overfitting is called **regularization**. Here we introduce another hyper parameter called regularization factor $$ \lambda $$ to penalize overfitting.

In this example, we use a L2 norm (**L2 regularization**)
$$ ||W|| $$
 as the penality. 

After many try and error, we pick $$ \lambda $$ to be 1. With the regularization, our model make better prediction with the same number of iterations and data point as in our first try.

<div class="imgcap">
<img src="/assets/dl/p3.png" style="border:none;width:60%">
</div>

Like other hyper parameter for training, the process is try and error. In fact we use a relative high $$ \lambda $$
in this problem because there are not too many trainable parameters in our model. In most real life problems, $$ \lambda $$ is lower because we are dealing with much higher number of trainable parameters.

There is another interesting observation during training. The loss may jump up sharply and drop to previous value after a few thousand iterations. 
```
Iteration 87000 [2.5431744485195127]
Iteration 88000 [2.525734745522529]
Iteration 89000 [223.88938197268865]
Iteration 90000 [195.08231216279583]
Iteration 91000 [3.0582387198108449]
Iteration 92000 [2.4587727305339286]
```

If we look into the equation, we realize the gradient

$$
\frac{\partial y}{\partial c_i}   = i  x^{i-1}
$$

can be very steep which can suffer from the learning rate problem discussed before. The cost here escalate very high which takes many more iterations to undo. For example, from iteration 10,000 to 11,000, the coefficient for $$ x^5 $$
only change from -0.000038 to -0.000021 but the cost jump from 82 to 34,312.

```
Iteration 10000 [82.128486144319155, 
 array([[  1.66841311e+00],
       [  5.15883024e-01],
       [  1.05449372e-01],
       [ -1.15910560e-01],
       [  1.32065116e-02],
       [ -3.83863265e-04]])]
Iteration11000 [34312.355686493174, 
  array([[  1.82722611e+00],
       [  5.83582807e-01],
       [  1.28332499e-01],
       [ -1.11263342e-01],
       [  1.24705708e-02],
       [ -2.05131433e-04]])]
```

When we build our model, we try out a polynomial model with order of 9. Even after a long training, the model still makes very poor prediction. We decide to start with a model with order of 3 and increase it gradually: another example to demonstrate why we should start with simple model first. At 7, we find the model is so hard to train to produce good quality model. The following is what a 7-layer model predicts:
<div class="imgcap">
<img src="/assets/dl/p4.png" style="border:none;width:60%">
</div>

### Diminishing and exploding gradient

From our previous example, we demonstrate how important to trace the gradient at different layer to trouble shoot problem. In our online dating model, we log 
$$ || gradient || $$ 
for each layers.

```python
iteration 0: loss=45.6
layer 0: gradient = 226.1446016395799
layer 1: gradient = 566.6340440894377
layer 2: gradient = 371.4818585197662
layer 3: gradient = 371.7283667292019
iteration 10000: loss=12.28
layer 0: gradient = 39.087735791986816
layer 1: gradient = 70.66776450168192
layer 2: gradient = 40.95339598248693
layer 3: gradient = 49.27868977928858
...
iteration 90000: loss=11.78
layer 0: gradient = 8.695315741501654
layer 1: gradient = 13.149909360278247
layer 2: gradient = 9.97983678446837
layer 3: gradient = 7.053793667949491
```

There are a couple things that we need to monitor. Are the magnitude too high or too small? If the magnitude is too high at later stage of training, the gradient descent is having problem to find the minima. Some parameters may be oscillating. For example, when we have the scaling problem with the features (year of education and monthly income), the gradient is so huge that the model learns nothing.
```
iteration 0: ... dW1=1.183e+04 dW2=5.929e+06 ...
iteration 200: ... dW1=4.458e+147 dW2=2.203e+150 ...
iteration 400: ... dW1=1.656e+291 dW2=8.184e+293 ...
iteration 600: ... dW1=nan dW2=nan ...
```

 If gradient is small, changes to the parameters are small. In the following log, the gradient diminishing from the right layer (layer 6) to the left layer (layer 0). We should expect layer 0 learn much slower than layer 6.
```
iteration 0: loss=553.5
layer 0: gradient = 2.337481559834108e-05
layer 1: gradient = 0.00010808796151264163
layer 2: gradient = 0.0012733936924033608
layer 3: gradient = 0.01758514040640722
layer 4: gradient = 0.20165907211476816
layer 5: gradient = 3.3937365923146308
layer 6: gradient = 49.335409914253
iteration 1000: loss=170.4
layer 0: gradient = 0.0005143399278199742
layer 1: gradient = 0.0031069449720360883
layer 2: gradient = 0.03744160389724748
layer 3: gradient = 0.7458109132993136
layer 4: gradient = 5.552521662655173
layer 5: gradient = 16.857110777922465
layer 6: gradient = 37.77102597043024
iteration 2000: loss=75.93
layer 0: gradient = 4.881626633589997e-05
layer 1: gradient = 0.0015526594728625706
layer 2: gradient = 0.01648262093048127
layer 3: gradient = 0.35776408953278077
layer 4: gradient = 1.6930852548061421
layer 5: gradient = 4.064949014764085
layer 6: gradient = 12.7578637206897
```

A network with many deep layers can suffer from this gradient diminishing problem. We need to come back to backpropagation to understand why this happens?

<div class="imgcap">
<img src="/assets/dl/chain.png" style="border:none;width:60%">
</div>

The gradient descent is computed as:

$$
\frac{\partial J}{\partial l_{1}} = \frac{\partial J}{\partial l_{2}} \frac{\partial l_{2}}{\partial l_{1}}  = \frac{\partial J}{\partial l_{3}} \frac{\partial l_{3}}{\partial l_{2}}  \frac{\partial l_{2}}{\partial l_{1}} 
$$ 

$$
\frac{\partial J}{\partial l_{1}} = \frac{\partial J}{\partial l_{10}} \frac{\partial l_{10}}{\partial l_{9}} \cdots  \frac{\partial l_{2}}{\partial l_{1}} 
$$ 

As indicated, the gradient descent is not only depend on the loss $$ \frac{\partial J}{\partial l} $$ but also on the gradients $$ \frac{\partial l_{k+1}}{\partial l_{k}} $$. Let's look at a sigmoid activation function below. If $$ x $$ is higher than 5 or smaller than -5, the gradient is close to 0. Hence, in those region, the node learns slowly with gradient descent regardless of the loss.

<div class="imgcap">
<img src="/assets/dl/sigmoid2.png" style="border:none;width:80%">
</div>

We can visualize the derivative of the sigmoid function behaves like a gate to the loss signal. If the input is > 5 or <-5, the derviative is small and it blocks most of the loss signal to propagage backward. So nodes on its left sides learn less. 

In additon, the chain rule in the gradient descent has a multiplication effect. If we multiple numbers smaller than one, it diminishes quickly. On the contrary, if we multiple numbers greater than one, it explodes. 

$$ 
0.1 \cdot 0.1 \cdot 0.1 \cdot 0.1 \cdot 0.1 = 0.00001 
$$

$$ 
5 \cdot 5 \cdot 5 \cdot 5 \cdot 5 = 3125
$$

So if the network design and the initial parameters have some symmetry that make the nodes behave similarly,  the gradient may diminish quickly or explode. However, we cannot say with certainity on when and how it may happen because we still lack full understanding between the maths of gradient descent and a complex model. Nevertheless, emperical data for deep network indicates it is a problem for deep network.

Microsoft Resnet (2015) has 152 layers. A lot of natural language process (NLP) problems are vulnerable to diminishing and exploding gradients. How can they address the issue? This is the network design for Resnet. Instead of one long chain of nodes, a mechanism is build to bypass a layer to make learning faster. (Source Kaiming He, Xiangyu Zhang ... etc)
<div class="imgcap">
<img src="/assets/dl/resnet.png" style="border:none;width:40%">
</div>

<div class="imgcap">
<img src="/assets/dl/resnet2.png" style="border:none;width:40%">
</div>

As always in DL, an idea often looks complicated in a diagram or a equation. In LSTM, the state of a cell is updated by

$$
C_t = gate_{forget} \cdot C_{t-1} + gate_{input} \cdot \tilde{C}
$$

Bypassing a layer can visualize as feeding the input to the output directly. For $$ C_t $$ to be the same as $$ C_{t-1} $$, we can have $$ gate_ {forget} $$ to be 1 while $$ gate_{input} $$ to be 0. So one way to addressing the diminishing gradient problem is to design a different function for the node.

### Classification

A very important part of deep learning is classification. We have mentioned face detection and object recognition before. These are classification problems asking the question: what is this? For example, for Pieter to safely walk in the street, he needs to learn what is a traffic light, is the pedestrian faceing him or not. Other non-visual problems include how can we classify an email as a spam or not, approve or disapprove a loan etc...

<div class="imgcap">
<img src="/assets/dl/street2.jpg" style="border:none;width:40%">
</div>

Like solving regression problem using DL, we use a deep network to compute a value. In classification, we call this value **a score**. We apply a classifier to convert this score to a probability. To train the network, the training dataset will provide the answers to the classification (school bus, truck, airplane) which we call **true label**.

#### Logistic function (sigmoid function)

A score compute by a network can have any value. We need a classifier to squash it to a probabilty value between 0 and 1. For a "yes" or "no" type of prediction (the email is/is not a spamm, the drug test is positive or negative), we can apply a logistic function (sigmoid function) to the score value. If the output probability is lower than 0.5, we predict "no", otherwise we predict "yes".

$$
p = \sigma(score) = \frac{1}{1 + e^{-score}}
$$

<div class="imgcap">
<img src="/assets/dl/sigmoid.png" style="border:none;width:40%">
</div>


#### Softmax classifier

For many classification problem, we categorize an input to one of the many classes. For example, we can classify an image to one of the 100 possbile image classes, like school bus, airplance, truck, ... etc. We use softmax classifier to compute K probabilites, one per class for an input image. (The combined probabilities remains 1.)

<div class="imgcap">
<img src="/assets/dl/deep_learner2.jpg" style="border:none;width:70%;">
</div>

For a network predicting K classes, the network generates K scores (one for each class.) The probability for class $$ i $$ will be.

$$
p_i =  \frac{e^{z_i}}{\sum e^{z_c}} 
$$

For example, the school bus image above may have a score of (3.2, 0.8, 0) for the class school bus, truck and airplane. The probability for the correponding class is

$$
p_{\text{bus}} =  \frac{e^{3.2}}{ e^{3.2} + e^{0.8} + e^0} = 0.88
$$

$$
p_{\text{truck}} =  \frac{e^{0.2}}{ e^{3.2} + e^{0.8} + e^0} = 0.08
$$

$$
p_{\text{airplane}} =  \frac{e^0}{ e^{3.2} + e^{0.8} + e^0} = 0.04
$$

```python
def softmax(z):
	z -= np.max(z)
    return np.exp(z) / np.sum(np.exp(z))

a = np.array([3.2, 0.8, 0])   # [ 0.88379809  0.08017635  0.03602556]
print(softmax(a))
```

To avoid adding large expotential value that cause numerical stability problem, we subract the the inputs by its max. Adding or subtract a number from the input produce the same probabilities in softmax. 

$$
softmax(z) = \frac{e^{z_i -C}}{\sum e^{z_c  - C}} =  \frac{e^{-C} e^{z_i}}{e^{-C} \sum e^{z_c}} = \frac{e^{z_i}}{\sum e^{z_c}}
$$

```python
z -= np.max(z)
```

We apply softmax on a score to compute the probabilities.

$$
p = softmax(score) = \frac{e^{z_i}}{\sum e^{z_c}}
$$

**logits** are defined as (to measure odd.)

$$
logits = \log(\frac{p}{1-p})
$$

Combine the definition of the softmax and logits, it is easier to see the score is the logit. That is why many literature or API use the term logit for score when softmax is used. However, there are more than 1 function that map scores to probabiities and meet the definition of logits. Sigmoid function is one of them.

#### SVM classifier

Linear SVM classifer applies a linear classifier to map input to K scores, one per class. The class having the highest score will be the class prediction. To train the network, SVM loss is used. We will discuss the SVM in the cost function section. Its main purpose is to create a boundary to separate class with the largest possible margin.

$$
\hat{y} = W x + b
$$

<div class="imgcap">
<img src="/assets/dl/SVM.png" style="border:none;width:40%">
</div>

### Entropy

With a probablistic model, we want a cost function that works with probability predictions. We need to take a break to the information theory on entropy first. Since entropy is heavily used in machine learning, it may worth the time.

Say we have a string "abcc", "a" and "b" occurs 25% of time and "c" with 50%. Entropy defines the minimum amount of bits to represent the string. For most frequent character, we use fewer bits to represent it.

Entropy:

$$
H(y) = \sum_i y_i \log \frac{1}{y_i} = -\sum_i y_i \log y_{i}
$$

The string "abcc" needs 1.5 bit per character. Here is our encoding scheme: 0 represents 'c', 01 for 'a' and 10 for 'b' and the average bit to represent the string is $$ 0.25 \cdot 2 \cdot 2 + 1 \cdot 0.5 = 1.5 $$.
```python
b = -( 0.25 * math.log2(0.25) + 0.25 * math.log2(0.25) + 0.5 * math.log2(0.5) )   # 1.5 bit
```

Entropy is also a measure of randomness (disorder). A fair dice, comparing with a biased dice, has more randomness with even distribution of outcomes. A biased dice is more predictable and therefore less entropy. In entropy, randomness means more information since it requires more bits to represent the information. It needs more time to descibe the details inside a messy room.

```python
b = -(1.0 * math.log2(1.0))                           # 0
b = -(0.5 * math.log2(0.5) + 0.5 * math.log2(0.5)  )  # 1 bit: 0 for head 1 for tail.
```

#### Cross entropy

$$
H(y, \hat{y}) = \sum_i y_i \log \frac{1}{\hat{y}_i} = -\sum_i y_i \log \hat{y}_i
$$

Cross entropy is the amount of bits to encode y but use the $$ \hat{y} $$ distribution to compute the encode scheme.The cross entropy is always higher than entropy until both are the same. You need more bits to encode the information if you use a less optimized scheme. In a probabilitic model, we make predictions with probabilities. This acts as a distribution of what we may predict $$ hat{y} $$. (88% chance a school bus, 8% chance a truck and 4% chance an airplane.) 

#### KL Divergence:

$$
\mbox{KL}(y~||~\hat{y}) = \sum_i y_i \log \frac{1}{\hat{y}_i} - \sum_i y_i \log \frac{1}{y_i} = \sum_i y_i \log \frac{y_i}{\hat{y}_i}
$$

KL divergence is simply cross entropy - entropy. The extra bits need to encode the information. In machine learning, KL Divergenence estimates the difference between 2 distributions. Since $$ y $$ is the true labels and the labels does not change, we can consider it as a constant. Therefore finding a model to minimze KL divergence is the same as minimze the cross entropy. We therefore due with Cross entropy most of the time in this discussion.

### Maximum likelihood estimation (MLE)

What is our objective in training a model? Our objective is to tune our trainable parameters so that the likelihood of our model is maximized. Likelihood measures the probability of making a prediction the same as the true label given the input $$ x $$ and $$ W $$. In plain terms, we want to train the parameters $$ W $$ such that its prediction for the training data is as close to the labels.

The likelihood is define as the probabilty of making a prediction to be the same as the true labels give the input $$X$$ and $$W$$. If we can find the $$ W $$ to maximize the likelihood for all training data, we find our model. In probability, we write it as

$$
p(y | x, W) =  \prod_i p(y_{i} |  x_{i}, W)
$$

For each sample $$i$$,

$$
p(y_{i} |  x_{i}, W) = \hat{y_i}
$$

which $$ y_{1} = (1, 0, 0) $$ in our example (100% for school bus and 0% chance otherwise), and $$ \hat{y_{1}} $$ = $$ (0.88, 0.08, 0.04)$$.

#### Neagtive log-likelihood (NLL)

We want to take the log of the MLE because we can treat the product terms as additions which is easier to handle. Log is a monotonic increase function, and therefore, Finding $$ x $$ to maximize $$ f(x) $$ is the same as find $$x$$ to maximize $$ \log(f(x))$$.

Since probability is between 0 and 1 and its log is negative, we take a negative sign. 

$$
- \log {p(y_{i} |  x_{i}, W)} = - \log{ \hat{y_{i}}} = nll
$$

We call the terms above the negative log-likihood. Hence maximize the "maximum likelihood estimation" (MLE) is the same as minimizing the negative log-likihood. **To train a network, we find $$W $$ to minimizing the negative log-likelihood**.

#### Logistic loss

As an exercise to demonstrate NLL, we can dervive the logistic loss (or even the mean square error) from NLL.

In logistic regression, we compute the probability by

$$
{p(y_{i} |  x_{i}, W)} = \sigma(z) = \frac{1}{1 + e^{-z_i}}
$$

Apply NLL,

$$
\text{nnl} = - \log {p(y_{i} |  x_{i}, W)} = - \log{ \frac{1}{1 + e^{-z_j}} } = - \log{1} + \log (1 + e^{-z_j}) 
$$

This becomes the logistic loss:

$$
\text{nnl} = \sum\limits_{i}    \log (1 + e^{- z}) 
$$

$$
\text{nnl} = \sum\limits_{i} \log (1 + e^{- y_i W^T x_{i}}) 
$$

### Cost function

#### Cross entropy cost function

Cross entropy measure the difference between 2 distribution. (aka probability distribution.) Is that appropriate as a cost function for a network predicts a probability value.

$$
H(y, \hat{y}) = \sum_i y_i \log \frac{1}{\hat{y}_i} = -\sum_i y_i \log \hat{y}_i
$$

Apply NLL to find the cost function for a classification problem:

$$
p(y | x, W) =  \prod_n p(y_{i} |  x{i}, W)
$$

$$
\text{nll} = - \log {(p(y | W, x)} = - \sum_n \log {p(y_{i} | W, x_{i})}
$$

$$
\text{nll} =  - \sum_n \log {\hat{y_{i}}}
$$

Since $$ y_{i} = (0, \cdots,1, \dots, 0, 0) $$ We can put back $$ y_{i} $$ which becomes the cross entropy.

$$
\text{nll} = - \sum_n \sum_i y_{i} \log {\hat{y_{i}}} 
$$

$$
\text{negative log likelihood for probabilitic models} = \text{cross entropy} 
$$

Because $$ y^{i} = (0, \cdots,1, \cdots, 0) $$, we can see the cross entropy cost

$$
H(y, \hat{y}) = \sum_i y_i \log \frac{1}{\hat{y}_i} = -\sum_i y_i \log \hat{y}_i
$$

is usually written as (in classification):

$$
\text{nll} =  - \sum_n \log {\hat{y_{i}}}
$$

#### SVM loss (also called Hinge loss or Max margin loss)

$$
J = \sum_{j\neq y_i} \max(0, score_j - score_{y_i} + 1)
$$

If the margin between the score of a class and that of the true label is greater than -1, we add that to the cost.  For example, a score of (8, 14, 9.5, 10) with the last entry being the true label.

$$
J = max(0, 8 - 10 + 1) + max(0, 14 - 10 + 1) + max(0, 9.5 - 10 + 1)
$$

$$
J = 0 + 5 + 0.5 = 5.5
$$

For SVM, the cost function creates a boundary with the maximum margin to separate classes.

<div class="imgcap">
<img src="/assets/dl/SVM.png" style="border:none;width:40%">
</div>

#### Mean square error (MSE)

$$
\text{mean square error} = \frac{1}{N} \sum_i (h_i - y_i)^2
$$

We demonstrate the use of MSE on regression problems. We can use this in classification. But instead, we use cross entropy loss. Classification problem use a classifier to squash values to a probability between 0 and 1. The mapping is not linear. For a sigmod classfier, a large range of value (less than -5 or greater than 5) is squeezed to 0 or 1. As shown before, those areas have close to 0 partial derviative. Based on the chain rule in the back propagation 

$$
\frac{\partial J}{\partial score} = \frac{\partial J}{\partial out} \frac{\partial out}{\partial score}
$$

The loss signal is hard to propage backward in those region regardless of loss. However, there is a way to solve this issue. The partial derviative of the sigmod function on that score can be small but we can make 
$$ 
\frac{\partial J}{\partial out} 
$$ 
very large if the prediction is bad. The sigmod function squashes values by expontentially. We need a cost function that punish bad prediction in the same scale to counter that. Squaring the error does not make it. Cross entropy works on logarithmic scale which punish bad prediction expotentially. That is why cross entropy cost function trains better than MSE on the classification problems.


### Deep learing network (Fully-connected layers)




### Network layers


### Regularization

We apply regularization to overcome overfit. The idea is to train a model that can make generalized predictions. Force the model not to memorize the small bits of an individual sample that is not part of the genearlized features. Part of the overfit problem is that the model is too powerful. We can reduce the complexity of the model by reducing the number of features: remove features that can be direct or indirect derived by others. Reduce the layers of the network or switch to another design that explore better on the locality of the information. For example, CNN for images to explore spatial locality and LSTM for NPL. Overfit can also be overcomed by adding more training data.

#### L0, L1, L2 regularization

Large W tends to overfit, and large W seems just to cancel the effect of each other. L2 regularization add the norm to the regularization cost of a cost function.

$$
J = \text{data cost} + \lambda \cdot ||W||
$$

$$
||W||_2 = \sum \sqrt{(W_{11}^2 +W_{12}^2 + \cdots + W_{nn}^2)}
$$

There are different function to compute the regularization cost of tunable parameters:

L0 regularization

$$
|| W ||_0 = \sum \begin{cases} 1, & \mbox{if } w \neq 0 \\ 0, & \mbox{\text{otherwise}} \end{cases}
$$

L1 regularization

$$
|| W ||_1 = \sum |w|
$$

L0, L1 and L2 regularization penalize on $$ W $$ but on different extends. L2 put the highest attentions (or penalty) on the large parameter and L0 pay attention on non-zero prameter.  L0 penalizes non-zero parameter which force more parameters to be 0, i.e. increase the sparsity of the parameters. L2 focuses on large parameter and therefore have more non-zero parameters. L1 regularization also rewards sparsity. Sparsity of parameters effectively reduces the features in making prediction. Sometimes it is used as feature selections. L2 is more popular but some problem domain could prefer higher sparsity.


#### Gradient clipping
#### Dropout

### Weight initialization

### Insanity check
#### Gradient checking
#### Initial loss
#### Without regularization and with small dataset

### Trouble shooting

Many places can go wrong when training a deep network. Here are some simple tips:
* Unit test the forward pass and back propagation code.
	* At the begining, test with non-random data.
* Compare the back progataion result with the naive gradient check.
* Always start with a simple network that works. 
	* Push accuracy up should not be the first priority. 
	* Handle multiple battle front in a complext network is not the way to go. Issues grow expotentially in DL.
* Create simple sceairos to verify the network:
	* Test with small training data with few iterations. Verify if we can beat a random guessing.
	* Verify if loss drops and/or accuracies increase during training.
	* Drop regularization - training accuracies should go up.	
	* Validate with training data.
* Keep track of the loss, and when in debugging also the magnitude of the gradient for each layer.
* Do not waste time on large dataset with long iterations during early development.
* Verify how trainable parameters are initialized.	
* Always keep track of the shape of the data and doucment it in the code.
* Display and verify some training samples and the predictions.
* Plot out the loss, accuracy or some runtime data after the training.

### Trouble shooting
#### Plotting loss
#### Train/validation accuracy
#### Ratio of weight updates
#### Activation per layer
#### First layer visualization


### Training parameters
#### Momentum update
#### Adagrad
#### Adam
#### Rate decay

### Feature Scaling (normalization)

As we found out before, we want the feature input to the network to be scaled correctly (normalized). If the features do not have the proper scale, it will be much harder for the gradient descent to work. The training parameters may oscaillate.

<div class="imgcap">
<img src="/assets/dl/gauss_s.jpg" style="border:none;width:40%">
</div>

For example, with 2 input features, we want the shape to be as close to a circle as possible.
<div class="imgcap">
<img src="/assets/dl/gauss_shape.jpg" style="border:none;">
</div>

We normalize the features in the dataset to have zero mean and unit variance. 

$$
z = \frac{x - \mu}{\sigma}
$$

For image, we normalize every pixels independently. We compute a mean and variance at each pixel location for the whole training dataset. Therefore, for an image with NxN pixels, we use NxN means and variances to normalize the image.

$$
z_{ij} = \frac{x_{ij} - \mu_{ij}}{\sigma{ij}}
$$

In practice, we do not read all the trainning data at once to compute the mean or variance. For example, we compute a running mean during the training. Here is the formula for the running mean:

$$
\mu_{n} = \mu_{n-1}  + k \cdot (x_{i}-\mu_{n-1})
$$

which $$k$$ is a small constant.

#### Whitening

> This is an advance topic. We will cover the topic briefly only.

In machine learning, we prefer features to be un-related. For example, in a dating application, a person may prefer a tall person but not a thin person. However, weight and heigth is co-related. A taller person is heavier than a shorter person in average. Re-scaling these features independently can only tell whether a person is thinner than average in the population, but not whether the person is thin. A taller person is thinner if both have the same weight. Weigth increases with height:
<div class="imgcap">
<img src="/assets/dl/gauss.jpg" style="border:none;">
</div>

A network learns faster if features are un-related. We express the co-relations between a feature $$x_i$$ and $$ x_{j} $$ in terms of a covariance matrix below:

$$
\sum = \begin{bmatrix}
    E[(x_{1} - \mu_{1})(x_{1} - \mu_{1})] & E[(x_{1} - \mu_{1})(x_{2} - \mu_{2})] & \dots  & E[(x_{1} - \mu_{1})(x_{n} - \mu_{n})] \\
    E[(x_{2} - \mu_{2})(x_{1} - \mu_{1})] & E[(x_{2} - \mu_{2})(x_{2} - \mu_{2})] & \dots  & E[(x_{2} - \mu_{2})(x_{n} - \mu_{n})] \\
    \vdots & \vdots & \ddots & \vdots \\
    E[(x_{n} - \mu_{n})(x_{1} - \mu_{1})] & E[(x_{n} - \mu_{n})(x_{2} - \mu_{2})] & \dots  & E[(x_{n} - \mu_{n})(x_{n} - \mu_{n})]
\end{bmatrix}
$$

Which $$ E $$ is the expected value.

Consider 2 data samples : (10, 20) and (32, 52). 
The mean of $$ x_1 $$ will be $$ \mu_1 = \frac {10+32}{2} = 21 $$ and $$ \mu_2 = 36 $$

The expected value of the first element in the second row will be:

$$
E[(x_{2} - \mu_{2})(x_{1} - \mu_{1})] = \frac {(20 - 36)(10 - 21) + (52 - 36)(32 - 21)} {2}
$$

From the covariance matrix $$ \sum $$, we can a matrix $$W$$ to convert the input $$ X $$ to $$ Y = W X $$. (We will skip how to find $$ W $$ here.) The purpose of whitening is to change the feature distribtion from the left to the right one.

<div class="imgcap">
<img src="/assets/dl/gaussf.jpg" style="border:none;width:50%">
</div>

### Batch normalization

### Hyperparameter tuning

#### Random search

### CNN

### LSTM

### Data argumentation

### Model ensembles

























