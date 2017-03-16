---
layout: post
comments: true
title: “TensorBoard - Visualize your learning.”
excerpt: “TensorBoard make your machine learning visualization easy.”
date: 2017-03-12 12:00:00
---
### TensorBoard

Use TensorBoard to visualize the trainable parameters, runtime status, and training metric including loss. We need to:
1. Define all the summary information to be logged.
2. Add summary information to the writer to flush it out to a log file.
3. View the data in the TensorBoard.

### Define summary information

We can pass a TensorFlow variable like W or b to *variable_summaries* below to log scalar/histogram information using *tf.summary.scalar* and *tf.summary.histogram*.
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
``
We call affine_layer which calls variable_summaries.
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
Here is the output of all the scalars in the TensorBoard.
<div class="imgcap">
<img src="/assets/tensorboard/tb_scalar_summary.png" style="border:none; width:100%;">
</div>

<div class="imgcap">
<img src="/assets/tensorboard/tb_scalar.png" style="border:none; width:100%;">
</div>

Here is the output of all the histograms in the TensorBoard.
<div class="imgcap">
<img src="/assets/tensorboard/tb_hist_summary.png" style="border:none; width:100%;">
</div>

<div class="imgcap">
<img src="/assets/tensorboard/tb_hist.png" style="border:none; width:100%;">
</div>


