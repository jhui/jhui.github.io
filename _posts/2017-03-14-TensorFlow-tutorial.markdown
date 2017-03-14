---
layout: post
comments: true
title: “TensorFlow overview”
excerpt: “TensorFlow is a very powerful platform for Machine Learning. I will go over some of the basic in this TensorFlow tutorials.”
date: 2017-03-14 14:00:00
---
### Basic
TensorFlow is an open source software library for machine learning developed by Google. This tutorial is designed to teach the basic concepts and how to use it.
#### First TensorFlow program
TensorFlow represents computations by linking op nodes into graphs. TensorFlow programs are structured into a construction phase and an execution phase. The following program:
1. Constructs a computation graph for a matrix multiplication. 
2. Open a TensorFlow session and execute the computation graph.

```python
import tensorflow as tf
# Construct 2 op nodes (m1, m2) representing 2 matrix.
m1 = tf.constant([[3, 5]])
m2 = tf.constant([[2],[4]])
product = tf.matmul(m1, m2) # A matrix multiplication op node: 3*2+5*4
with tf.Session() as sess: # Open a TensorFlow session to execute the graph. 
result = sess.run(product) # Compute the result for “product”
print(result) # [[26]]
```
The above program hardwire the matrix as a constant. We will implement a new linear equation that feed the graph with data on execution.
$$
y = Wx + b
$$
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
    # Feed data into the place holder before execution.
 result = sess.run(product, feed_dict={x: np.array([[2],[4]]), b:1})
    print(result)              # 3*2+5*4+1 = [[27]]
```