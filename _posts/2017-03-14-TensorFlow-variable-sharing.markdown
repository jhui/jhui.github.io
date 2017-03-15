---
layout: post
comments: true
title: “TensorFlow variables, variable sharing and scoping.”
excerpt: “Explain the Tensor variables, name sharing & scoping without the confusion.”
date: 2017-03-14 12:00:00
---
### Variables

When you train a model, we use variables to store training parameters like weight and bias, hyper parameters like learning rate, or state information like global step. Variable needs to be initialized explicitly. 

```python
import tensorflow as tf

# Define variables and its initializer
weights = tf.Variable(tf.random_normal([784, 100], stddev=0.1), name="W")
biases = tf.Variable(tf.zeros([100]), name="b")

counter = tf.Variable(0, name="counter")

# Op that assign a value to a variable
increment = tf.assign(counter , counter + 1)

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
  sess.run(init_op)
  
  # Retrieve the value of a variable
  b = sess.run(biases)
  print(b)
```

#### Save and restore variable

Variables can be saved to a disk during and after training and be restored for prediction or analyze.
```python
### Save and restore variables
counter = tf.Variable(0, name="counter")

increment = tf.assign(counter , counter + 1)

# Saver
saver = tf.train.Saver()

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
  sess.run(init_op)
  for _ in range(10):
      sess.run(increment)

  # Save the variables to disk.
  save_path = saver.save(sess, "/tmp/model.ckpt")

  # Restore
  saver.restore(sess, "/tmp/model.ckpt")

  count = sess.run(counter)
  print(count)
```
Save and restore:
```python
  # Save the variables to disk.
  save_path = saver.save(sess, "/tmp/model.ckpt")

  # Restore
  saver.restore(sess, "/tmp/model.ckpt")
```
To save a subset of variables only.
```python
saver = tf.train.Saver({"my_counter": counter})
```