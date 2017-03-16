---
layout: post
comments: true
title: “TensorBoard - Visualize your learning.”
excerpt: “TensorBoard make your machine learning visualization easy.”
date: 2017-03-12 12:00:00
---
### TensorBoard

Create a layer with the name 'layer1'
```python
h1, _ = affine_layer(x, 'layer1', [784, 256], keep_prob)
```
Call varaible_summaries to create a summary with name scope 'layer1/weigths'
```python
def affine_layer(x, name, shape, keep_prob, act_fn=tf.nn.relu):
    with tf.name_scope(name):
        with tf.name_scope('weights'):
            W = tf.Variable(tf.truncated_normal(shape, stddev=np.sqrt(2.0 / shape[0])))
            variable_summaries(W)
```
Call *tf.summary.scalar* to record a scalar value, or *tf.summary.histogram* for non-scalar value.
```python
def variable_summaries(var):
  """Attach mean/max/min/sd & histogram for TensorBoard visualization."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)
```
Here is the output of all the scalars in the TensorBoard.
<div class="imgcap">
<img src="/assets/tensorboard/tb_scalar_summary.png" style="border:none; width:50%;">
</div>

<div class="imgcap">
<img src="/assets/tensorboard/tb_scalar.png" style="border:none; width:50%;">
</div>

Here is the output of all the histograms in the TensorBoard.
<div class="imgcap">
<img src="/assets/tensorboard/tb_hist_summary.png" style="border:none; width:50%;">
</div>

<div class="imgcap">
<img src="/assets/tensorboard/tb_hist.png" style="border:none; width:50%;">
</div>


