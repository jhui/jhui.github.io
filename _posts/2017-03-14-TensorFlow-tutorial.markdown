---
layout: post
comments: true
title:  “TensorFlow overview”
excerpt: “TensorFlow is a very powerful platform for Machine Learning.  I will go over some of the basic in this TensorFlow tutorials.”
date:   2017-03-14 14:00:00
---

### Basic
TensorFlow is an open source software library for machine learning developed by Google. This tutorial is designed to teach the basic concepts and the usage.

#### First TensorFlow program

TensorFlow represent computations with op nodes and graphs. TensorFlow programs are structured into a construction phase and an execution phase. The following program:

1. Constructs a computation graph for 2 matrix multiplication.  
2. Open a TensorFlow session and execute the computation graph.


```python
import tensorflow as tf

# Construct the m1 & m2 op node.
m1 = tf.constant([[3, 5]])
m2 = tf.constant([[2],[4]])

product = tf.matmul(m1, m2)    # A matrix multiplication op node: 3*2+5*4

with tf.Session() as sess:     # Open a TensorFlow session to execute the graph. 
    result = sess.run(product)
    print(result)              # [[26]]
```