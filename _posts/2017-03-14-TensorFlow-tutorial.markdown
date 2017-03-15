---
layout: post
comments: true
mathjax: true
title: “TensorFlow overview”
excerpt: “TensorFlow is a very powerful platform for Machine Learning. I will go over some of the basic in this TensorFlow tutorials.”
date: 2017-03-14 14:00:00
---
### Basic
TensorFlow is an open source software library for machine learning developed by Google. This tutorial is designed to teach the basic concepts and how to use it.
#### First TensorFlow program
TensorFlow represents computations by linking op nodes into graphs. TensorFlow programs are structured into a construction phase and an execution phase. The following program:
1. Constructs a computation graph for a matrix multiplication. 
2. Open a TensorFlow session and compute the matrix multiplication by execute the computation graph.

```python
import tensorflow as tf

# Construct 2 op nodes (m1, m2) representing 2 matrix.
m1 = tf.constant([[3, 5]])
m2 = tf.constant([[2],[4]])

product = tf.matmul(m1, m2)    # A matrix multiplication op node

with tf.Session() as sess:     # Open a TensorFlow session to execute the graph. 
    result = sess.run(product) # Compute the result for “product”
    print(result)              # 3*2+5*4: [[26]]
```
The above program hardwire the matrix as a constant. We will implement a new linear equation that feed the graph with input data on execution.

```python
import tensorflow as tf
import numpy as np

W = tf.constant([[3, 5]])

# Allow data to be supplied later during execution.
x = tf.placeholder(tf.int32, shape=(2, 1))
b = tf.placeholder(tf.int32)

# A linear model y = Wx + b
product = tf.matmul(W, x) + b

with tf.Session() as sess:
    # Feed data into the place holder (x & b) before execution.
    result = sess.run(product, feed_dict={x: np.array([[2],[4]]), b:1})
    print(result)              # 3*2+5*4+1 = [[27]]
```

#### Train a linear model
Let’s do a simple linear regression with a linear model below.

$$
y = Wx + b
$$

We will supply the model with training data (x, y) and later compute the corresponding model parameter W & b.
```python
import tensorflow as tf

### Define a computational graph
# Parameters for a linear model y = Wx + b
W = tf.Variable([0.1], tf.float32)
b = tf.Variable([0.0], tf.float32)

# Placeholder for input and prediction
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# Define a linear model y = Wx + b
model = W * x + b

# Define a cost function (Mean square error - MSE)
loss = tf.reduce_sum(tf.square(model - y))

# Optimizer with a 0.01 learning rate
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

### Running the computational graph (Fitting)
# Training data
x_train = [1.0, 2.0, 3.0, 4.0]
y_train = [1.5, 3.5, 5.5, 7.5]

# Retrieve the variable initializer op and initialize variable W & b.
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        sess.run(train, {x:x_train, y:y_train})
        if i%100==0:
            l_cost = sess.run(loss, {x:x_train, y:y_train})
            print(f"i: {i} cost: {l_cost}")

    # Evaluate training accuracy
    l_W, l_b, l_cost  = sess.run([W, b, loss], {x:x_train, y:y_train})
    print(f"W: {l_W} b: {l_b} cost: {l_cost}")
    # W: [ 1.99999797] b: [-0.49999401] cost: 2.2751578399038408e-11
```
In this program, we define the training parameters (W & b) in the computational graph as follows:
```python
W = tf.Variable([0.1], tf.float32)
b = tf.Variable([0.0], tf.float32)
```
We define the Mean Square Error (MSE) cost function:
```python
loss = tf.reduce_sum(tf.square(model - y))
```
We define a gradient descent optimizer and trainer to find an optimal solution that can fit our training data with the minimum loss.
```python
# Optimizer with a 0.01 learning rate
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
```
Before any execution, we need to initialize all the parameters:
```python
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
```
We train our data with 1000 iterations.  For every 100 iteration, we compute the loss and print it out.
```python
for i in range(1000):
   sess.run(train, {x:x_train, y:y_train})
   if i%100==0:
        l_cost = sess.run(loss, {x:x_train, y:y_train})
        print(f"i: {i} cost: {l_cost}")
```
Once 1000 iterations are done, we print out W, b and the loss:
```python
# Evaluate training accuracy
l_W, l_b, l_cost  = sess.run([W, b, loss], {x:x_train, y:y_train})
print(f"W: {l_W} b: {l_b} cost: {l_cost}")
# W: [ 1.99999797] b: [-0.49999401] cost: 2.2751578399038408e-11
```
Here we model our training data as:
$$
y = 2x - 0.5
$$

### Linear Regressor
TensorFlow comes with many prebuilt models. The following code replace the last program with a prebuilt Linear Regressor.

```python
import tensorflow as tf

import numpy as np

features = [tf.contrib.layers.real_valued_column("x", dimension=1)]
estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)

x = np.array([1., 2., 3., 4.])
y = np.array([1.5, 3.5, 5.5, 7.5])
input_fn = tf.contrib.learn.io.numpy_input_fn({"x":x}, y, batch_size=4, num_epochs=1000)

estimator.fit(input_fn=input_fn, steps=1000)
result = estimator.evaluate(input_fn=input_fn)
print(f"loss = {result['loss']}")

for name in estimator.get_variable_names():
    print(f'{name} = {estimator.get_variable_value(name)}')

# loss = 0.013192394748330116
# linear/x/weight = [[ 1.90707111]]
# linear/bias_weight = [-0.21857721]
```

### Custom model
As shown above, the Linear Regressor has a larger error than the last program. Sometimes, we can plug in our own model as following:

```python
import numpy as np
import tensorflow as tf

def model(features, labels, mode):

  W = tf.get_variable("W", [1], dtype=tf.float64)
  b = tf.get_variable("b", [1], dtype=tf.float64)
  y = W*features['x'] + b

  loss = tf.reduce_sum(tf.square(y - labels))
  global_step = tf.train.get_global_step()
  optimizer = tf.train.GradientDescentOptimizer(0.01)
  train = tf.group(optimizer.minimize(loss), tf.assign_add(global_step, 1))
  return tf.contrib.learn.ModelFnOps(
      mode=mode, predictions=y,
      loss=loss,
      train_op=train)

estimator = tf.contrib.learn.Estimator(model_fn=model)
x = np.array([1., 2., 3., 4.])
y = np.array([1.5, 3.5, 5.5, 7.5])
input_fn = tf.contrib.learn.io.numpy_input_fn({"x": x}, y, 4, num_epochs=1000)

estimator.fit(input_fn=input_fn, steps=1000)
print(estimator.evaluate(input_fn=input_fn, steps=10))

for name in estimator.get_variable_names():
    print(f'{name} = {estimator.get_variable_value(name)}')

# {'loss': 6.7292158e-11, 'global_step': 1000}
# W = [ 1.99999637]
# b = [-0.4999892]
```
Plugin a new model to the estimator:
```python
def model(features, labels, mode):

  W = tf.get_variable("W", [1], dtype=tf.float64)
  b = tf.get_variable("b", [1], dtype=tf.float64)
  y = W*features['x'] + b

  loss = tf.reduce_sum(tf.square(y - labels))
  global_step = tf.train.get_global_step()
  optimizer = tf.train.GradientDescentOptimizer(0.01)
  train = tf.group(optimizer.minimize(loss), tf.assign_add(global_step, 1))
  return tf.contrib.learn.ModelFnOps(
      mode=mode, predictions=y,
      loss=loss,
      train_op=train)

estimator = tf.contrib.learn.Estimator(model_fn=model)
```

#### Solving MNist
<div class="imgcap">
<img src=“/assets/tensorflow_basic/mnist.png” style="border:none; width:100%;">
</div>




 