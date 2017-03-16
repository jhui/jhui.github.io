---
layout: post
comments: true
title: “TensorFlow variables, variable sharing and scoping.”
excerpt: “Explain the Tensor variables, name sharing & scoping without the confusion.”
date: 2017-03-14 12:00:00
---
### Variables

When you train a model, we use variables to store training parameters like weight and bias, hyper parameters like learning rate, or state information like global step. The following program:
* Define variables and the initializers.
* Create op to update variable.
* Explicitly initialize the variables. (Always required)
* Retrieve a variable value.

```python
import tensorflow as tf

### Using variables
# Define variables and its initializer
weights = tf.Variable(tf.random_normal([784, 100], stddev=0.1), name="W")
biases = tf.Variable(tf.zeros([100]), name="b")

counter = tf.Variable(0, name="counter")

# Add an Op to increment a counter
increment = tf.assign(counter , counter + 1)

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
  # Execute the init_op to initialize all variables
  sess.run(init_op)

  # Retrieve the value of a variable
  b = sess.run(biases)
  print(b)
```

#### Save and restore variable

Variables can be saved to a disk during and after training and be restored for prediction or analyze later.
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

### Variable sharing
Before looking into variable sharing, we first describe how *tf.Varaible* works. *tf.Variable* always create a new variable even given the same name. 
```python
# tf.Variable always create new variable even given the same name.
v1 = tf.Variable(10, name="name1")
v2 = tf.Variable(10, name="name1")
assert(v1 is not v2)
print(v1.name)  # name1:0
print(v2.name)  # name1_1:0
```
So calling the affine method twice below, we create 2 sets of W and b, i.e. 2 affine layers with their own set of W & b.
```python
def affine(x, shape):
    W = tf.Variable(tf.truncated_normal(shape))
    b = tf.Variable(tf.zeros([shape[1]]))

    model = tf.nn.relu(tf.matmul(x, W) + b)
    return model

x = tf.placeholder(tf.float32, [None, 784])
with tf.variable_scope("n1"):
    n1 = affine(x, [784, 500])

with tf.variable_scope("n1"):
    n2 = affine(x, [784, 500])
```

Sometimes, in a complex model, we want to share a common layer or parameters. How can we have a affine method similar to the code above but share the same W & b.
```python 
def affine_reuseable(x, shape):
    W = tf.get_variable("W", shape,
                    initializer=tf.random_normal_initializer())
    b = tf.get_variable("b", [shape[1]],
                    initializer=tf.constant_initializer(0.0))
    model = tf.nn.relu(tf.matmul(x, W) + b)
    return model

nx = tf.placeholder(tf.float32, [None, 784])
with tf.variable_scope("n2"):
    nn1 = affine_reuseable(x, [784, 500])

with tf.variable_scope("n2", reuse=True):
    nn2 = affine_reuseable(x, [784, 500])
```
If a variable with the give "scope/name" exists, *tf.get_variable* returns that variable instead of creating one.
```python
W = tf.get_variable("W", shape,
                    initializer=tf.random_normal_initializer())
```
So for the second affine_reuseable call, *tf.get_variable* reuse the old variable instead of creating a new one.
```python
with tf.variable_scope("nl1", reuse=True):
    nl2 = affine_reuseable(x, [784, 500])
```

#### Reuse
However, TensorFlow wants the developer to be self-aware whether the variable exists or now.
Both scenario below will throw an exception when calling *tf.get_variable*:
*  if the reuse flag is None and the variable already exists
*  if the reuse flag is True and the variable does not exists

Do **NOT** do this
```python
with tf.variable_scope("foo"):
    v = tf.get_variable("v", [1])
    v1 = tf.get_variable("v", [1])
    # Raises ValueError("... v already exists ...").
    
with tf.variable_scope("foo", reuse=True):
    v = tf.get_variable("v", [1])
    # Raises ValueError("... v does not exists ...").
```
Instead set the reuse flag probably
```python
with tf.variable_scope("foo"):
    v = tf.get_variable("v2", [1])
with tf.variable_scope("foo", reuse=True):
    v1 = tf.get_variable("v2", [1])
assert v1 == v

with tf.variable_scope("foo") as scope:
    v = tf.get_variable("v3", [1])
    scope.reuse_variables()
    v1 = tf.get_variable("v3", [1])
assert v1 == v
```



