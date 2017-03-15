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
<img src="/assets/tensorflow_basic/mnist.png" style="border:none; width:40%;">
</div>

The MNIST dataset contains handwritten digits with examples shown as above. It has a training set of 60,000 examples, and a test set of 10,000 examples. The following python file from Tensorflow [mnist_softmax.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_softmax.py) train a linear classifier for MNist digit recognition. The following model reaches an accuracy of **92%**.

```python
"""A very simple MNIST classifier.
See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])
  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  y = tf.matmul(x, W) + b

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  # Train
  for _ in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

# 0.9241
```

Read training, validation and testing dataset into “mnist”.
```python
from tensorflow.examples.tutorials.mnist import input_data

def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
```

Each image is 28x28 = 784. We use a linear classifier to classify the handwritten image from either 0 to 9.

```python
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x, W) + b
```

We use cross entropy as the cost functions:
```python
  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # However, the method approach may be numerical unstable.
  #
  # Therefore we replace it with an equivalent stable version.
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
```

#### Solving MNist with deep learning

Now we replace this model using deep learning techniques. This new model achieves an accuracy of **98%**!

```python
import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np

FLAGS = None


def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  ### Building a model
  # Create a fully connected network with 2 hidden layers
  # Initialize the weight with a normal distribution.
  x = tf.placeholder(tf.float32, [None, 784])
  W1 = tf.Variable(tf.truncated_normal([784, 256], stddev=np.sqrt(2.0 / 784)))
  b1 = tf.Variable(tf.zeros([256]))
  W2 = tf.Variable(tf.truncated_normal([256, 100], stddev=np.sqrt(2.0 / 256)))
  b2 = tf.Variable(tf.zeros([100]))
  W3 = tf.Variable(tf.truncated_normal([100, 10], stddev=np.sqrt(2.0 / 100)))
  b3 = tf.Variable(tf.zeros([10]))

  # 2 hidden layers using relu (z = max(0, x)) as an activation function.
  h1 = tf.nn.relu(tf.matmul(x, W1) + b1)
  h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)
  y = tf.matmul(h2, W3) + b3

  # Define loss and optimizer
  labels = tf.placeholder(tf.float32, [None, 10])

  # Use a cross entropy cost fuction with a L2 regularization.
  lmbda = tf.placeholder(tf.float32)
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=y) +
         lmbda * (tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3)))
  train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy)

  init = tf.global_variables_initializer()
  with tf.Session() as sess:
      sess.run(init)
      # Train
      for _ in range(10000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, labels: batch_ys, lmbda:5e-5})

      # Test trained model
      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(labels, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                          labels: mnist.test.labels}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

# 0.9785
```

We create a model with 2 fully connected hidden layers with a linear classifier. We also use relu function as the activation function for our hidden layers.

$$
a = max(0, z)
$$

```python
### Building a model
# Create a fully connected network with 2 hidden layers
# Initialize the weight with a normal distribution.
x = tf.placeholder(tf.float32, [None, 784])
  W1 = tf.Variable(tf.truncated_normal([784, 256], stddev=np.sqrt(2.0 / 784)))
  b1 = tf.Variable(tf.zeros([256]))
  W2 = tf.Variable(tf.truncated_normal([256, 100], stddev=np.sqrt(2.0 / 256)))
  b2 = tf.Variable(tf.zeros([100]))
  W3 = tf.Variable(tf.truncated_normal([100, 10], stddev=np.sqrt(2.0 / 100)))
  b3 = tf.Variable(tf.zeros([10]))

# 2 hidden layers using relu (z = max(0, x)) as an activation function.
h1 = tf.nn.relu(tf.matmul(x, W1) + b1)
h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)
y = tf.matmul(h2, W3) + b3
```


We initializes the weight with a normal distribution with standard deviation inverse proportional to the input size.
```python
W1 = tf.Variable(tf.truncated_normal([784, 256], stddev=np.sqrt(2.0 / 784)))
W2 = tf.Variable(tf.truncated_normal([256, 100], stddev=np.sqrt(2.0 / 256)))
W3 = tf.Variable(tf.truncated_normal([100, 10], stddev=np.sqrt(2.0 / 100)))
```


We add a L2 regularization to the cross entropy lost with regularization factor set to 5e-5.  We also adopted an improved version of gradient descent (Adam optimizer).
```python
  # Use a cross entropy cost fuction with a L2 regularization.
  lmbda = tf.placeholder(tf.float32)
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=y) +
         lmbda * (tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3)))
  train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy)
```
```python
sess.run(train_step, feed_dict={x: batch_xs, labels: batch_ys, lmbda:5e-5})
```


 