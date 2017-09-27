---
layout: post
comments: true
mathjax: true
priority: 800
title: “TensorFlow Basic”
excerpt: “TensorFlow is a very powerful platform for Machine Learning. This tutorial goes over some of the basic of TensorFlow.”
date: 2017-03-14 14:00:00
---

### Basic
TensorFlow is an open source software library for machine learning developed by Google. This tutorial is designed to teach the basic concepts and how to use it.

#### First TensorFlow program
TensorFlow represents computations by linking op nodes into graphs. TensorFlow programs are structured into a construction phase and an execution phase. The following program:
1. Constructs a computation graph for a matrix multiplication. 
2. Open a TensorFlow session and compute the matrix multiplication by executing the computation graph.

```python
import tensorflow as tf

# Construct 2 op nodes (m1, m2) representing 2 matrix.
m1 = tf.constant([[3, 5]])     # (1, 2)
m2 = tf.constant([[2],[4]])    # (2, 1)

product = tf.matmul(m1, m2)    # A matrix multiplication op node

with tf.Session() as sess:     # Open a TensorFlow session to execute the graph. 
    result = sess.run(product) # Compute the result for “product”
    print(result)              # 3*2+5*4: [[26]]
```
_sess.run_ and _tensor.eval()_will return a NumPy array containing the result of the computation.

The above program hardwires the matrix as a constant. We will implement a new linear equation that feeds the graph with input data on execution.

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

> When we construct a graph (tf.constant, tf.get_variable, tf.matmul), we are just building a computation graph. No computation is actually perforned until we run it inside a session (sess.run).

Common tensor types in TensorFlow are:

* tf.Variable
* tf.Constant
* tf.Placeholder
* tf.SparseTensor

#### Train a linear model
Let’s do a simple linear regression with a linear model below.

$$
y = Wx + b
$$

We will supply the model with training data (x, y) and later compute the corresponding model parameter W & b.
```python
import tensorflow as tf

### Define a model: a computational graph
# Parameters for a linear model y = Wx + b
W = tf.get_variable("W", initializer=tf.constant([0.1]))
b = tf.get_variable("b", initializer=tf.constant([0.0]))

# Placeholder for input and prediction
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# Define a linear model y = Wx + b
model = W * x + b

### Define a cost function, an optimizer and a trainer
# Define a cost function (Mean square error - MSE)
loss = tf.reduce_sum(tf.square(model - y))

# Optimizer with a 0.01 learning rate
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

### Training (Fitting)
# Training data
x_train = [1.0, 2.0, 3.0, 4.0]
y_train = [1.5, 3.5, 5.5, 7.5]

with tf.Session() as sess:
    # Retrieve the variable initializer op and initialize variable W & b.
    sess.run(session.run(tf.global_variables_initializer()))
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
A typical TensorFlow program contains:
* Define a model
* Define a loss function and a trainer
* Training (fitting)

#### Model
Define the linear model y = Wx + b. 
```python
### Define a model: a computational graph
# Parameters for a linear model y = Wx + b
W = tf.get_variable("W", initializer=tf.constant([0.1]))
b = tf.get_variable("b", initializer=tf.constant([0.0]))

# Placeholder for input and prediction
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# Define a linear model y = Wx + b
model = W * x + b
```

We define both $$W$$ and $$b$$ as variables initialized as 0.1 and 0 respectively. Variables are trainable and can act as the parameters of a model.
```python
W = tf.get_variable("W", initializer=tf.constant([0.1]))
b = tf.get_variable("b", initializer=tf.constant([0.0]))
```

The shape of a tensor is the dimension of a tensor. For example, a 5x5x3 matrix is a Rank 3 (3-dimensional) tensor with shape (5, 5, 3). By default, the data type (dtype) of a tensor is tf.float32. Here we initialize a tensor with shape (5, 5, 3) of int32 type with 0. 
```python
int_v = tf.get_variable("int_variable", [5, 5, 3], dtype=tf.int32, 
  initializer=tf.zeros_initializer)
```  

#### Lost function and optimizer & trainer
Define the Mean Square Error (MSE) cost function:
```python
loss = tf.reduce_sum(tf.square(model - y))
```
We define a gradient descent optimizer and trainer to find an optimal solution that can fit our training data with the minimum loss.
```python
# Optimizer with a 0.01 learning rate
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
```

#### Training (fitting)
Before any execution, we need to initialize all the parameters:
```python
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
```
We train our data with 1000 iterations.  For every 100 iterations, we compute the loss and print it out.
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
  
### Estimator

TensorFlow provides a higher level Estimator API with pre-built model to train and predict data.

It compose of the following steps:

#### Define the feature columns

```python
x_feature = tf.feature_column.numeric_column('x')
...
```

```python
n_room = tf.feature_column.numeric_column('n_rooms')
sqfeet = tf.feature_column.numeric_column('square_feet',
                    normalizer_fn='lambda a: a - global_size')
```

#### Dataset importing functions for training, validation and prediction

```python
# Training
train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x = {"x": np.array([1., 2., 3., 4.])},      # Input features
      y = np.array([1.5, 3.5, 5.5, 7.5]),         # Output
      batch_size=2,
      num_epochs=None,
      shuffle=True)

# Testing
test_input_fn = tf.estimator.inputs.numpy_input_fn(
      x = {"x": np.array([5., 6., 7.])},
      y = np.array([9.5, 11.5, 13.5]),
      num_epochs=1,
      shuffle=False)

# Prediction
samples = np.array([8., 9.])
predict_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": samples},
      num_epochs=1,
      shuffle=False)
```

#### Create a pre-built estimator 

```python
regressor = tf.estimator.LinearRegressor(
    feature_columns=[x_feature],
    model_dir='./output'
)
```

**model_dir** stores the model data including the statistic information into the specific directory which can be viewed from the TensorBoard latter.

```sh
tensorboard --logdir=output
```

<div class="imgcap">
<img src="/assets/tensorflow/esf.png" style="border:none;">
</div>

#### Training, validation and testing

```python
regressor.train(input_fn=train_input_fn, steps=2500)
average_loss = regressor.evaluate(input_fn=test_input_fn)["average_loss"]
predictions = list(regressor.predict(input_fn=predict_input_fn))
```

#### LinearRegressor
TensorFlow comes with many prebuilt models. The following code replaces the last program with a prebuilt Linear Regressor. It constructs a linear regressor as an estimator and we will create an input function to pre-process and feed data into the models. 

Here is the full program:
```python
import tensorflow as tf

import numpy as np

# Create a linear regressorw with 1 feature "x".
x_feature = tf.feature_column.numeric_column('x')

regressor = tf.estimator.LinearRegressor(
    feature_columns=[x_feature],
    model_dir='./output'
)

# Training
train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x = {"x": np.array([1., 2., 3., 4.])},      # Input features
      y = np.array([1.5, 3.5, 5.5, 7.5]),         # Output
      batch_size=2,
      num_epochs=None,
      shuffle=True)

regressor.train(input_fn=train_input_fn, steps=2500)

# Testing
test_input_fn = tf.estimator.inputs.numpy_input_fn(
      x = {"x": np.array([5., 6., 7.])},
      y = np.array([9.5, 11.5, 13.5]),
      num_epochs=1,
      shuffle=False)

average_loss = regressor.evaluate(input_fn=test_input_fn)["average_loss"]
print(f"Average loss: {average_loss:.4f}")

# Prediction
samples = np.array([8., 9.])
predict_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": samples},
      num_epochs=1,
      shuffle=False)

predictions = list(regressor.predict(input_fn=predict_input_fn))
for input, p in zip(samples, predictions):
    v  = p["predictions"][0]
    print(f"{input} -> {v:.4f}")

# Average loss: 0.0002
# 8.0 -> 15.4773
# 9.0 -> 17.4729
```

#### DNNClassifier

DNNClassifier is another pre-built estimator. We build a DNNClassifier with 3 hidden layers to classify the iris samples into 3 subclasses. We load 150 samples and split it into 120 training data and 30 testing data.

The 4 features used as the model input: (image from wiki)
<div class="imgcap">
<img src="/assets/tensorflow/iris.png" style="border:none;width:60%">
</div>

Here is another Estimator for the classification
```python
import os
import urllib

import numpy as np
import tensorflow as tf

# Data sets
IRIS_TRAINING = "iris_training.csv"
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST = "iris_test.csv"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

def main():
  # Download dataset
  if not os.path.exists(IRIS_TRAINING):
    raw = urllib.urlopen(IRIS_TRAINING_URL).read()
    with open(IRIS_TRAINING, "w") as f:
      f.write(raw)

  if not os.path.exists(IRIS_TEST):
    raw = urllib.urlopen(IRIS_TEST_URL).read()
    with open(IRIS_TEST, "w") as f:
      f.write(raw)

  # Load datasets.
  training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=IRIS_TRAINING,
      target_dtype=np.int,
      features_dtype=np.float32)
  test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=IRIS_TEST,
      target_dtype=np.int,
      features_dtype=np.float32)

  # Specify that all features have real-value data
  feature_columns = [tf.feature_column.numeric_column("x", shape=[4])]

  # Build 3 layer DNN with 10, 20, 10 units respectively.
  classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                          hidden_units=[10, 20, 10],
                                          n_classes=3,
                                          model_dir="/tmp/iris_model")
  # Define the training inputs
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": np.array(training_set.data)},
      y=np.array(training_set.target),
      num_epochs=None,
      shuffle=True)

  # Train model.
  classifier.train(input_fn=train_input_fn, steps=2000)

  # Define the test inputs
  test_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": np.array(test_set.data)},
      y=np.array(test_set.target),
      num_epochs=1,
      shuffle=False)

  # Evaluate accuracy.
  accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]

  print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

  # Classify two new flower samples.
  new_samples = np.array(
      [[6.4, 3.2, 4.5, 1.5],
       [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)
  predict_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": new_samples},
      num_epochs=1,
      shuffle=False)

  predictions = list(classifier.predict(input_fn=predict_input_fn))
  predicted_classes = [p["classes"] for p in predictions]

  print("New Samples Predictions:    {}\n".format(predicted_classes))

if __name__ == "__main__":
    main()
```

### Custom model for Estimator

Implement a model_fn to create a custom model used in an estimator.
```python
import numpy as np
import tensorflow as tf

def model_fn(features, labels, mode):
  # Model
  W = tf.get_variable("W", [1], dtype=tf.float64)
  b = tf.get_variable("b", [1], dtype=tf.float64)
  y = W * features['x'] + b

  loss = tf.reduce_sum(tf.square(y - labels))


  global_step = tf.train.get_global_step()
  optimizer = tf.train.GradientDescentOptimizer(0.01)
  train = tf.group(optimizer.minimize(loss),
                   tf.assign_add(global_step, 1))

  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=y,
      loss=loss,
      train_op=train)

estimator = tf.estimator.Estimator(model_fn=model_fn)

x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])

x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])

input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)

eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False)

# Training
estimator.train(input_fn=input_fn, steps=1000)


train_metrics = estimator.evaluate(input_fn=train_input_fn)
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)

print("train metrics: %r"% train_metrics)
print("eval metrics: %r"% eval_metrics)
```


### Solving MNist

<div class="imgcap">
<img src="/assets/tensorflow/mnist.png" style="border:none; width:40%;">
</div>

The MNIST dataset contains handwritten digits with examples shown as above. It has a training set of 60,000 examples and a test set of 10,000 examples. The following python file from TensorFlow [mnist_softmax.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_softmax.py) train a linear classifier for MNist digit recognition. The following model reaches an accuracy of **92%**.

```python
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

  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  # Train
  for _ in range(1000):
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
W = tf.get_variable("W", [784, 10], initializer=tf.zeros_initializer)
b = tf.get_variable("b", [10], initializer=tf.zeros_initializer)
y = tf.matmul(x, W) + b
```

We use cross-entropy as the cost functions:
```python
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
```

### Solving MNist with a fully connected networking

Now we replace the model using deep learning techniques. This example contains 2 hidden fully connected layers. The new model achieves an accuracy of **98%**.

<div class="imgcap">
<img src="/assets/tensorflow/fc.png" style="border:none; width:80%;">
</div>

For each hidden layer:
```python
with tf.name_scope('hidden1'):   # Create a name scope for hidden layer 1
    weights = tf.Variable(       # Create a variable for weights initialized with truncated normal distribution
        tf.truncated_normal([IMAGE_PIXELS, hidden1_units],
                            stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),
                            name='weights')
    biases = tf.Variable(tf.zeros([hidden1_units]), name='biases') # Create avaraible for the biases
    hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)      # Matrix multiplication followed by a RELU
```

Cost function using cross entropy.
```python
def loss(logits, labels):
  labels = tf.to_int64(labels)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='xentropy')
  return tf.reduce_mean(cross_entropy, name='xentropy_mean')
```

Training and evaluation:
```
def training(loss, learning_rate):
  # Add a scalar summary for the snapshot loss.
  tf.summary.scalar('loss', loss)

  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  global_step = tf.Variable(0, name='global_step', trainable=False)
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op

def evaluation(logits, labels):
  correct = tf.nn.in_top_k(logits, labels, 1)
  return tf.reduce_sum(tf.cast(correct, tf.int32))
```

The full code for defining the model:
```python
import math
import tensorflow as tf

NUM_CLASSES = 10
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

def inference(images, hidden1_units, hidden2_units):
  # Hidden 1
  with tf.name_scope('hidden1'):
    weights = tf.Variable(
        tf.truncated_normal([IMAGE_PIXELS, hidden1_units],
                            stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),
                            name='weights')
    biases = tf.Variable(tf.zeros([hidden1_units]),
                         name='biases')
    hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
  # Hidden 2
  with tf.name_scope('hidden2'):
    weights = tf.Variable(
        tf.truncated_normal([hidden1_units, hidden2_units],
                            stddev=1.0 / math.sqrt(float(hidden1_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([hidden2_units]),
                         name='biases')
    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
  # Linear
  with tf.name_scope('softmax_linear'):
    weights = tf.Variable(
        tf.truncated_normal([hidden2_units, NUM_CLASSES],
                            stddev=1.0 / math.sqrt(float(hidden2_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                         name='biases')
    logits = tf.matmul(hidden2, weights) + biases
  return logits

def loss(logits, labels):
  labels = tf.to_int64(labels)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='xentropy')
  return tf.reduce_mean(cross_entropy, name='xentropy_mean')

def training(loss, learning_rate):
  # Add a scalar summary for the snapshot loss.
  tf.summary.scalar('loss', loss)

  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  global_step = tf.Variable(0, name='global_step', trainable=False)
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op

def evaluation(logits, labels):
  correct = tf.nn.in_top_k(logits, labels, 1)
  return tf.reduce_sum(tf.cast(correct, tf.int32))
```

Here is the main program:
```python
import argparse
import os
import sys
import time

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist

# Basic model parameters as external flags.
FLAGS = None

def placeholder_inputs(batch_size):
  images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, mnist.IMAGE_PIXELS))
  labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
  return images_placeholder, labels_placeholder

def fill_feed_dict(data_set, images_pl, labels_pl):
  images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size, FLAGS.fake_data)
  feed_dict = {images_pl: images_feed, labels_pl: labels_feed,}
  return feed_dict

def do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_set):
  true_count = 0
  steps_per_epoch = data_set.num_examples // FLAGS.batch_size
  num_examples = steps_per_epoch * FLAGS.batch_size
  for step in range(steps_per_epoch):
    feed_dict = fill_feed_dict(data_set, images_placeholder, labels_placeholder)
    true_count += sess.run(eval_correct, feed_dict=feed_dict)
  precision = float(true_count) / num_examples
  print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' % (num_examples, true_count, precision))

def run_training():
  data_sets = input_data.read_data_sets(FLAGS.input_data_dir, FLAGS.fake_data)

  with tf.Graph().as_default():
    images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size)

    logits = mnist.inference(images_placeholder, FLAGS.hidden1, FLAGS.hidden2)
    loss = mnist.loss(logits, labels_placeholder)

    train_op = mnist.training(loss, FLAGS.learning_rate)
    eval_correct = mnist.evaluation(logits, labels_placeholder)

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    summary = tf.summary.merge_all()

    sess = tf.Session()
	
    summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

    sess.run(init)

    for step in range(FLAGS.max_steps):
      start_time = time.time()

      feed_dict = fill_feed_dict(data_sets.train, images_placeholder, labels_placeholder)

      _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

      duration = time.time() - start_time

      if step % 100 == 0:
        print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
        summary_str = sess.run(summary, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()

      if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
        saver.save(sess, checkpoint_file, global_step=step)

        print('Training Data Eval:')
        do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_sets.train)
				
        print('Validation Data Eval:')
        do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_sets.validation)
				
        print('Test Data Eval:')
        do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_sets.test)


def main(_):
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  run_training()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--learning_rate', type=float, default=0.01, help='Initial learning rate.')
  parser.add_argument('--max_steps', type=int, default=2000, help='Number of steps to run trainer.')
  parser.add_argument('--hidden1', type=int, default=128, help='Number of units in hidden layer 1.')
  parser.add_argument('--hidden2', type=int, default=32, help='Number of units in hidden layer 2.')
  parser.add_argument('--batch_size', type=int, default=100, help='Batch size.  Must divide evenly into the dataset sizes.')
  parser.add_argument('--input_data_dir', type=str, default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),  'tensorflow/mnist/input_data'), help='Directory to put the input data.')
  parser.add_argument('--log_dir', type=str, default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'), 'tensorflow/mnist/logs/fully_connected_feed'), help='Directory to put the log data.')
  parser.add_argument('--fake_data', default=False, help='If true, uses fake data for unit testing.', action='store_true')

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
```

Collecting data for TensorBoard
In mnist.py:
```python
def training(loss, learning_rate):
  ...
  tf.summary.scalar('loss', loss)
  ...
```

```python
def run_training():
    ...
    loss = mnist.loss(logits, labels_placeholder)
    train_op = mnist.training(loss, FLAGS.learning_rate)
	
    summary = tf.summary.merge_all()
    sess = tf.Session()
    summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

    ...
    for step in range(FLAGS.max_steps):
      ...
      _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
      ...

      if step % 100 == 0:
        summary_str = sess.run(summary, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()  
```

Further accuracy improvement can be achieved by:
* Increase the number of iterations.
* Change to a CNN architect.
* Replace the regularization with more advanced methods like batch normalization or dropout.
* Fine tuning of the learning rate in the Adam optimizer and the lambda in the L2 regularization.

In next section, we will cover the CNN and dropout implementation.

### MNist with a Convolution network (CNN)

To push the accuracy higher, we will create a model with 2 CNN layers followed by 2 hidden fully connected (FC) layers and the final linear classifier. We also apply:
* a 5x5 filter for both CNN layers.
* a 2x2 max pooling max(z11, z12, z21, z22) for both CNN layers.
* Use RELU max(0, z) for both CNN and FC layer.
* Use dropout for regularization.
* Use cross entropy cost function with Adam optimizer.

The following code reaches an accuracy of **99.4%** with little parameter tuning.

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

  ### Building a model with 2 Convolution layers
  ### followed by 2 fully connected hidden layers and a linear classification layer.
  x = tf.placeholder(tf.float32, [None, 784])

  # Parameters for the 2 convolution layer
  with tf.variable_scope("CNN"):
      cnn_W1 = tf.get_variable("W1", [5, 5, 1, 32], initializer=tf.truncated_normal_initializer(stddev=0.1))
      cnn_b1 = tf.get_variable("b1", [32], initializer=tf.constant_initializer(0.1))
      cnn_W2 = tf.get_variable("W2", [5, 5, 32, 64], initializer=tf.truncated_normal_initializer(stddev=0.1))
      cnn_b2 = tf.get_variable("b2", [64], initializer=tf.constant_initializer(0.1))

  # Parameters for 2 hidden layers with dropout and the linear classification layer.
  # 3136 = 7 * 7 * 64
  W1 = tf.get_variable("W1", [3136, 1000], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(2.0 / 3136)))
  b1 = tf.get_variable("b1", [1000], initializer=tf.zeros_initializer)
  W2 = tf.get_variable("W2", [1000, 100], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(2.0 / 1000)))
  b2 = tf.get_variable("b2", [100], initializer=tf.zeros_initializer)
  W3 = tf.get_variable("W3", [100, 10], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(2.0 / 100)))
  b3 = tf.get_variable("b3", [10], initializer=tf.zeros_initializer)

  keep_prob = tf.placeholder(tf.float32)

  # First CNN with RELU and max pooling.
  x_image = tf.reshape(x, [-1, 28, 28, 1])
  cnn1 = tf.nn.conv2d(x_image, cnn_W1, strides=[1, 1, 1, 1], padding='SAME')
  z1 = tf.nn.relu(cnn1 + cnn_b1)
  h1 = tf.nn.max_pool(z1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

  # Second CNN
  cnn2 = tf.nn.conv2d(h1, cnn_W2, strides=[1, 1, 1, 1], padding='SAME')
  z2 = tf.nn.relu(cnn2 + cnn_b2)
  h2 = tf.nn.max_pool(z2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

  # First FC layer with dropout.
  h2_flat = tf.reshape(h2, [-1, 3136])
  h_fc1 = tf.nn.relu(tf.matmul(h2_flat, W1) + b1)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Second FC
  h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W2) + b2)
  h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

  # Linear classification.
  y = tf.matmul(h_fc2_drop, W3) + b3

  # True label
  labels = tf.placeholder(tf.float32, [None, 10])

  # Cost function & optimizer
  # Use cross entropy with the Adam gradient descent optimizer.
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=y) )
  train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy)

  init = tf.global_variables_initializer()
  with tf.Session() as sess:
      sess.run(init)
      # Train
      for i in range(10001):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, labels: batch_ys, keep_prob:0.5})
        if i%50==0:
          # Test trained model
          correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(labels, 1))
          accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
          result = sess.run(accuracy, feed_dict={x: mnist.test.images,
                                              labels: mnist.test.labels,
                                              keep_prob:1.0})
          print(f"Iteration {i}: accuracy = {result}")

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

# Iteration 10000: accuracy = 0.9943000078201294
```

Define the convolution layer with a 5x5 filter using RELU activation following by a 2x2 max pool:
```python
with tf.variable_scope("CNN"):
    cnn_W1 = tf.get_variable("W1", [5, 5, 1, 32], initializer=tf.truncated_normal_initializer(stddev=0.1))
    cnn_b1 = tf.get_variable("b1", [32], initializer=tf.constant_initializer(0.1))
    cnn_W2 = tf.get_variable("W2", [5, 5, 32, 64], initializer=tf.truncated_normal_initializer(stddev=0.1))
    cnn_b2 = tf.get_variable("b2", [64], initializer=tf.constant_initializer(0.1))
```

We can add scope to a variable by _ tf.variable_scope_. Here, _cnn_W1_ will have the name 'CNN/W1:0'.

```python
# First CNN with RELU and max pooling.
x_image = tf.reshape(x, [-1, 28, 28, 1])
cnn1 = tf.nn.conv2d(x_image, cnn_W1, strides=[1, 1, 1, 1], padding='SAME')
z1 = tf.nn.relu(cnn1 + cnn_b1)
h1 = tf.nn.max_pool(z1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
```
We flatten the 2D features into a 1D array for the fully connected layer. We apply dropout for the regularization.
```python
# First FC layer with dropout.
h2_flat = tf.reshape(h2, [-1, 3136])
h_fc1 = tf.nn.relu(tf.matmul(h2_flat, W1) + b1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
```

Further possible accuracy improvement:
* Apply ensemble learning.
* Use a smaller filter like 3x3.
* Add batch normalization.
* Whitening of the input image.
* Further tuning of the learning rate and dropout parameter.

Here is another CNN implementation example from the TensorFlow distribution:
```python
import argparse
import sys
import tempfile

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def deepnn(x):
  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 28, 28, 1])

  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1)

  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2)

  with tf.name_scope('fc1'):
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  return y_conv, keep_prob


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def main(_):
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  x = tf.placeholder(tf.float32, [None, 784])

  y_ = tf.placeholder(tf.float32, [None, 10])

  y_conv, keep_prob = deepnn(x)

  with tf.name_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                            logits=y_conv)
  cross_entropy = tf.reduce_mean(cross_entropy)

  with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)

  graph_location = tempfile.mkdtemp()
  print('Saving graph to: %s' % graph_location)
  train_writer = tf.summary.FileWriter(graph_location)
  train_writer.add_graph(tf.get_default_graph())

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
      batch = mnist.train.next_batch(50)
      if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0})
        print('step %d, training accuracy %g' % (i, train_accuracy))
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
```

### Reshape Numpy
Find the shape of a Numpy array and reshape it.
```python
import tensorflow as tf
import numpy as np

### ndarray shape
x = np.array([[2, 3], [4, 5], [6, 7]])
print(x.shape)          # (3, 2)

x = x.reshape((2, 3))
print(x.shape)          # (2, 3)

x = x.reshape((-1))
print(x.shape)          # (6,)

x = x.reshape((6, -1))
print(x.shape)          # (6, 1)

x = x.reshape((-1, 6))
print(x.shape)          # (1, 6)
```

### Reshape TensorFlow

Find the shape of a tensor and reshape it
```python
import tensorflow as tf
import numpy as np

### Tensor
W = tf.get_variable("W", [4, 5], initializer=tf.random_uniform_initializer(-1, 1))

print(W.get_shape())    # Get the shape of W (4, 5)

W = tf.reshape(W, [10, 2])
print(W.get_shape())    # (10, 2)

W = tf.reshape(W, [-1])
print(W.get_shape())    # (20,)

W = tf.reshape(W, [5, -1])
print(W.get_shape())    # (5, 4)
```

tf.unique(x) returns a 1D tensor contains all unique elements. The shape is dynamic which depends on "x" and need to evaluate at runtime:
```python
import tensorflow as tf
import numpy as np

c = tf.constant([1, 2, 3, 1])
y, _ = tf.unique(c)     # y only contains the unique elements.

print(y.get_shape())    # (?,) This is a dynamic shape. Only know in runtime

y_shape = tf.shape(y)   # Define an op to get the dynamic shape.

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(y_shape))   # [3] contains 3 unique elements
```

### Initialize variables

Initialize variables with constant:
```python
import tensorflow as tf
import numpy as np

v1 =  tf.get_variable("v1", [5, 5, 3])   # A tensor with shape (5, 5, 3) filled with random values

v2 = tf.get_variable("v2", shape=(), initializer=tf.zeros_initializer())

v3 = tf.get_variable("v3", initializer=tf.constant(2))    # 2, float32 scalar
v4 = tf.get_variable("v4", initializer=tf.constant([2]))  # [2]
v5 = tf.get_variable("v5", initializer=tf.constant([[2, 3], [4, 5]]))  # [[2, 3], [4, 5]]

v6 = tf.get_variable("v6", initializer=tf.constant(2.0), dtype=tf.float64, trainable=True)
```

Note: when we use _tf.constant_ in _tf.get_variable_, we do not need to specify the tensor shape.

Fill with 0, 1 or specific values.
```python
v1 = tf.get_variable("v1", [3, 2], initializer=tf.zeros_initializer)
v2 = tf.get_variable("v2", [3, 2], initializer=tf.ones_initializer)

# [[ 1.  2.], [ 3.  4.], [ 5.  6.]]
v3 = tf.get_variable("v3", [3, 2], initializer=tf.constant_initializer([1, 2, 3, 4, 5, 6])) 

# [[ 1.  2.], [ 2.  2.], [ 2.  2.]]
v4 = tf.get_variable("v4", [3, 2], initializer=tf.constant_initializer([1, 2])) 
```


Randomized the value of variables:
```python
import tensorflow as tf
import numpy as np

W = tf.get_variable("W", [784, 256], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(2.0 / 784)))
Z = tf.get_variable("z", [4, 5], initializer=tf.random_uniform_initializer(-1, 1)) 
```

### Slicing

```python
subdata = data[:, 3]
subdata = data[:, 0:10]
```

### Utilities function
Concat and split
```python
import tensorflow as tf

t1 = [[1, 2], [3, 4]]
t2 = [[5, 6], [7, 8]]
tf.concat([t1, t2], 0) # [[1, 2], [3, 4], [5, 6], [7, 8]]
tf.concat([t1, t2], 1) # [[1, 2, 5, 6], [3, 4, 7, 8]]

value = tf.get_variable("value", [4, 10], initializer=tf.zeros_initializer)

s1, s2, s3 = tf.split(value, [2, 3, 5], 1)
# s1 shape(4, 2)
# s2 shape(4, 3)
# s3 shape(4, 5)

# Split 'value' into 2 tensors along dimension 1
s0, s1= tf.split(value, num_or_size_splits=2, axis=1)  # s0 shape(4, 5)

```
Generate a one-hot vector
```python
import tensorflow as tf

# Generate a one hot array using indexes
indexes = tf.get_variable("indexes", initializer=tf.constant([2, 0, -1, 0]))

target = tf.one_hot(indexes, 3, 2, 0)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(target))
# [[0 0 2]
# [2 0 0]
# [0 0 0]
# [2 0 0]]	
```

### Casting
```python
s0 = tf.cast(s0, tf.int32)
s0 = tf.to_int64(s0)
```

### Training using gradient
During training, we may interest in the gradients for each variable. For example, from the gradients, we may tell how well the gradient descent is working for the deep network. To expose the gradient, replace the following code:
```python
optimizer = tf.train.GradientDescentOptimizer(0.01)
optimizer = optimizer.minimize(loss)
```
With:
```python
global_step = tf.Variable(0)

optimizer = tf.train.GradientDescentOptimizer(0.01)
gradients, v = zip(*optimizer.compute_gradients(loss))
optimizer = optimizer.apply_gradients(zip(gradients, v), global_step=global_step)
```

### Download and reading CSV file
```
import tempfile

import tensorflow as tf
import urllib.request
import numpy as np

FLAGS = None

tf.logging.set_verbosity(tf.logging.INFO)

def maybe_download(train_data):
  if train_data:
    train_file_name = train_data
  else:
    train_file = tempfile.NamedTemporaryFile(delete=False)
    urllib.request.urlretrieve(
        "http://download.tensorflow.org/data/abalone_train.csv",
        train_file.name)
    train_file_name = train_file.name
    train_file.close()
    print("Training data is downloaded to %s" % train_file_name)
  return train_file_name


training_local_file = ""
training_local_file = maybe_download(training_local_file)

training_set = tf.contrib.learn.datasets.base.load_csv_without_header(
  filename=training_local_file, target_dtype=np.int, features_dtype=np.float64)

print(f"data shape = {training_set.data.shape}")      # (3320, 7)
print(f"label shape = {training_set.target.shape}")   # (3320,)
```

### Evaluate & print a tensor

A quick way to evaluate a Tensor in particular for debugging.
```
m1 = tf.constant([[3, 5]])
m2 = tf.constant([[2],[4]])
product = tf.matmul(m1, m2)   

with tf.Session() as sess:     
    v = product.eval()    
    t = tf.Print(v, [v])  # tf.Print return the first parameter
    result = t + 1  # v will be printed only if t is accessed
    result.eval()
```

### InteractiveSession
TensorFlow provides another way to execute a computational graph using *tf.InteractiveSession* which is more convenient for an ipython environment.
```python
import tensorflow as tf
import numpy as np

sess = tf.InteractiveSession()

m1 = tf.get_variable("m1", initializer=tf.constant([[3, 5]]))
m2 = tf.placeholder(tf.int32, shape=(2, 1))
product = tf.matmul(m1, m2)   

m1.initializer.run()   # Run the initialization op (and what it depends)

v1 = m1.eval()    # Evaluate a tensor
p = product.eval(feed_dict={m2: np.array([[1], [2]])}) # with feed

print(f"{v1}, {p}")

# Close the Session when we're done.
sess.close()
```

