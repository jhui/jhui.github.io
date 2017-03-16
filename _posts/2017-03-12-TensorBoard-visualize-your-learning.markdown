---
layout: post
comments: true
title: “TensorBoard - Visualize your learning.”
excerpt: “TensorBoard make your machine learning visualization easy.”
date: 2017-03-12 12:00:00
---
### TensorBoard

TensorBoard is a browser based application that help you to visualize your training parameters (like weights & biases), metrics (like loss), hyper parameters or any statistics. For example, we plot the histogram distribution of the weight for the first fully connected layer every 20 iterations.

<div class="imgcap">
<img src="/assets/tensorboard/tb_hist.png" style="border:none; width:100%;">
</div>

#### Namespace
To create some data hierarchy structure when we view the data like:
<div class="imgcap">
<img src="/assets/tensorboard/tb_scalar_summary.png" style="border:none; width:70%;">
</div>
We use namespace
```python
with tf.name_scope('CNN1'):
    with tf.name_scope('W'):
        mean = tf.reduce_mean(W)
        tf.summary.scalar('mean', mean)
	stddev = tf.sqrt(tf.reduce_mean(tf.square(W - mean)))
	tf.summary.scalar('stddev', stddev)
        tf.summary.histogram('histogram', var)
```
Which create the following data hierarchy and can be browsed with the TensorBoard later:
<div class="imgcap">
<img src="/assets/tensorboard/name_space.png" style="border:none; width:70%;">
</div>


#### Implement TensorBoard
To add & view data summaries to the TensorBoard. We need to:
1. Define all the summary information to be logged.
2. Add summary information to the writer to flush it out to a log file.
3. View the data in the TensorBoard.

### Define summary information
Number can be added to the TensorBoard with *tf_summary.scalar* and array with *tf.summary.histogram*
```python
with tf.name_scope('CNN1'):
    with tf.name_scope('W'):
        mean = tf.reduce_mean(W)
        tf.summary.scalar('mean', mean)
        tf.summary.histogram('histogram', var)
```

Here is the summary of both scalar and histogram summary in the TensorBoard.
For data logged with *tf_summary.scalar*
<div class="imgcap">
<img src="/assets/tensorboard/tb_scalar_summary.png" style="border:none; width:100%;">
</div>

<div class="imgcap">
<img src="/assets/tensorboard/tb_scalar.png" style="border:none; width:100%;">
</div>

For summary logged with *tf.summary.histogram*
<div class="imgcap">
<img src="/assets/tensorboard/tb_hist_summary.png" style="border:none; width:100%;">
</div>

<div class="imgcap">
<img src="/assets/tensorboard/tb_hist.png" style="border:none; width:100%;">
</div>

#### Example
Here is a more complicated example in which we try to summarize the information of the weight in the first fully connected layer.
```python
h1, _ = affine_layer(x, 'layer1', [784, 256], keep_prob)
```
```python
def affine_layer(x, name, shape, keep_prob, act_fn=tf.nn.relu):
    with tf.name_scope(name):
        with tf.name_scope('weights'):
            W = tf.Variable(tf.truncated_normal(shape, stddev=np.sqrt(2.0 / shape[0])))
            variable_summaries(W)
```
```python
def variable_summaries(var):
  """Attach mean/max/min/sd & histogram for TensorBoard visualization."""
  with tf.name_scope('summaries'):
    # Find the mean of the variable say W.
    mean = tf.reduce_mean(var)
    # Log the mean as scalar
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    # Log var as a histogram
    tf.summary.histogram('histogram', var)
```
### Add summary information to a writer
After we define what summary information to be logged, we merge all the summary data into one single operation node with *tf.summary.merge_all()*. We create a summary writer with *tf.summary.FileWriter*, and then write and flush out the information to the log file every 20 iterations:
```python
def main(_):
  ...
  train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy)

  # Merge all summary inforation.
  summary = tf.summary.merge_all()

  init = tf.global_variables_initializer()
  with tf.Session() as sess:
      # Create a writer for the summary data.
      summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
      sess.run(init)
      for step in range(100):
        ...
        sess.run(train_step, feed_dict={x: batch_xs, labels: batch_ys, lmbda:5e-5, keep_prob:0.5})
        if step % 20 == 0:
          # Flush the summary data out for every 20 iterations.
          summary_str = sess.run(summary, feed_dict={x: batch_xs, labels: batch_ys, lmbda:5e-5, keep_prob:0.5})
          summary_writer.add_summary(summary_str, step)
          summary_writer.flush()

        ...
if __name__ == '__main__':
  ...
  # Define the location of the log file used by TensorBoard
  parser.add_argument('--log_dir', type=str, default='/tmp/tensorflow/mnist/log',
                      help='Directory for log')
  ...

```

### View the TensorBoard
Open the terminal and run the tensorboard command with you log file location.
```
$ tensorboard --logdir=/tmp/tensorflow/mnist/log
Starting TensorBoard b'41' on port 6006
(You can navigate to http://192.134.44.11:6006)
```

### Full program listing
```python
import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np

FLAGS = None

def variable_summaries(var):
  """Attach mean/max/min/sd & histogram for TensorBoard visualization."""
  with tf.name_scope('summaries'):
    # Find the mean of the variable say W.
    mean = tf.reduce_mean(var)
    # Log the mean as scalar
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    # Log var as a histogram
    tf.summary.histogram('histogram', var)

def affine_layer(x, name, shape, keep_prob, act_fn=tf.nn.relu):
    with tf.name_scope(name):
        with tf.name_scope('weights'):
            W = tf.Variable(tf.truncated_normal(shape, stddev=np.sqrt(2.0 / shape[0])))
            variable_summaries(W)
        with tf.name_scope('biases'):
            b = tf.Variable(tf.zeros([shape[1]]))
            variable_summaries(b)
        with tf.name_scope('z'):
            z = tf.matmul(x, W) + b
            tf.summary.histogram('summaries/histogram', z)

        h = act_fn(tf.matmul(x, W) + b)
        with tf.name_scope('out/summaries'):
            tf.summary.histogram('histogram', h)

        with tf.name_scope('dropout/summaries'):
            dropped = tf.nn.dropout(h, keep_prob)
            tf.summary.histogram('histogram', dropped)

        return dropped, W

def main(_):
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  with tf.name_scope('dropout'):
      keep_prob = tf.placeholder(tf.float32)
      tf.summary.scalar('dropoout_probability', keep_prob)

  x = tf.placeholder(tf.float32, [None, 784])

  image = tf.reshape(x[:1], [-1, 28, 28, 1])
  tf.summary.image("image", image)

  h1, _ = affine_layer(x, 'layer1', [784, 256], keep_prob)
  h2, _ = affine_layer(h1, 'layer2', [256, 100], keep_prob)
  y, W3 = affine_layer(h2, 'output', [100, 10], keep_prob=1.0, act_fn=tf.identity)

  labels = tf.placeholder(tf.float32, [None, 10])

  lmbda = tf.placeholder(tf.float32)
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=y) +
         lmbda * (tf.nn.l2_loss(W3)))

  tf.summary.scalar('loss', cross_entropy)

  train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy)

  summary = tf.summary.merge_all()

  init = tf.global_variables_initializer()
  with tf.Session() as sess:
      summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
      sess.run(init)
      for step in range(100):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, labels: batch_ys, lmbda:5e-5, keep_prob:0.5})
        if step % 20 == 0:
          # Update the events file.
          summary_str = sess.run(summary, feed_dict={x: batch_xs, labels: batch_ys, lmbda:5e-5, keep_prob:0.5})
          summary_writer.add_summary(summary_str, step)
          summary_writer.flush()

      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(labels, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                          labels: mnist.test.labels, keep_prob:0.5}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  parser.add_argument('--log_dir', type=str, default='/tmp/tensorflow/mnist/log',
                      help='Directory for log')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

# 0.9816
```

### TensorBoard images & embedding
TensorFlow can also plot many different kinds of information including images and word embedding.
To add image summary:
```python
image = tf.reshape(x[:1], [-1, 28, 28, 1])
tf.summary.image("image", image)
```
<div class="imgcap">
<img src="/assets/tensorboard/tb_image.png" style="border:none; width:100%;">
</div>
<div class="imgcap">
<img src="/assets/tensorboard/tb_embedding.png" style="border:none; width:100%;">
</div>

### TensorBoard graph & distribution

TensorFlow automatically plot the computational graph and can be viewed under the graph tab.
<div class="imgcap">
<img src="/assets/tensorboard/tb_graph.png" style="border:none; width:100%;">
</div>
All histogram can also view as distribution.
<div class="imgcap">
<img src="/assets/tensorboard/tb_dist.png" style="border:none; width:100%;">
</div>

