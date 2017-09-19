---
layout: post
comments: true
priority: 830
title: “TensorFlow variables, saving/restore”
excerpt: “TensorFlow variables, saving/restore”
date: 2017-03-08 12:00:00
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

####Save a Checkpoint

Variables can be saved to a disk during training. It can be reloaded to continue the training or to make inferences.

```python
# Create some variables 
v1 = tf.get_variable("v1", shape=[3], initializer = tf.zeros_initializer)
v2 = tf.get_variable("v2", shape=[5], initializer = tf.zeros_initializer)

# Create the op
inc_v1 = v1.assign(v1+1)
dec_v2 = v2.assign(v2-1)
init_op = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

with tf.Session() as sess:
  sess.run(init_op)
  inc_v1.op.run()
  dec_v2.op.run()

  # Save the variables to disk.
  save_path = saver.save(sess, "/tmp/model.ckpt")
```

Restore:

```python
# Create some variables. 
# We do not need to provide initializer or init_op if it is restored from a checkpoint.
v1 = tf.get_variable("v1", shape=[3])
v2 = tf.get_variable("v2", shape=[5])

saver = tf.train.Saver()

with tf.Session() as sess:
  # Restore variables from disk.
  saver.restore(sess, "/tmp/model.ckpt")

  # Check the values of the variables
  print("v1 : %s" % v1.eval())
  print("v2 : %s" % v2.eval())
```

To save a subset of variables only.

```python
v1 = tf.get_variable("v1", [3], initializer = tf.zeros_initializer)
v2 = tf.get_variable("v2", [5], initializer = tf.zeros_initializer)

# Save only v2
saver = tf.train.Saver({"v2": v2})

with tf.Session() as sess:
  # Initialize v1 since the saver will not.
  v1.initializer.run()
  saver.restore(sess, "/tmp/model.ckpt")
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
If a variable with the give "scope/name" exists, *tf.get_variable* returns the existing variable instead of creating one.
```python
W = tf.get_variable("W", shape, initializer=tf.random_normal_initializer())
```
So for the second affine_reuseable call below, *tf.get_variable* reuses the W & b variables created before.
```python
with tf.variable_scope("n2", reuse=True):
    nn2 = affine_reuseable(x, [784, 500])
```

#### Reuse
However, TensorFlow wants the developer to be self-aware of whether the variable exists or not. Developers need to have the correct setting for the "reuse" flag before calling *tf.get_variable*. Both scenarios below will throw an exception when calling *tf.get_variable*:
*  if the reuse flag is None (default) and the variable already exists.
*  if the reuse flag is True and the variable does not exist.

Do **NOT** do this
```python
with tf.variable_scope("foo"):
    v = tf.get_variable("v", [1])
    v1 = tf.get_variable("v")
    # Raises ValueError("... v already exists ...").
    
with tf.variable_scope("foo", reuse=True):
    v = tf.get_variable("v")
    # Raises ValueError("... v does not exists ...").
```
Instead set the reuse flag probably.
```python
with tf.variable_scope("foo"):
    v = tf.get_variable("v2", [1]) # Create a new variable.

with tf.variable_scope("foo", reuse=True):
    v1 = tf.get_variable("v2")  # reuse/share the variable "foo/v2".
assert v1 == v

with tf.variable_scope("foo") as scope:
    v = tf.get_variable("v3", [1])
    scope.reuse_variables()
    v1 = tf.get_variable("v3")
assert v1 == v
```

#### Nested scope
```python
with tf.variable_scope("foo"):
    with tf.variable_scope("bar"):
        v = tf.get_variable("v", [1])
        assert v.name == "foo/bar/v:0"
``` 
### Caveat of variable sharing
Most developers are familiar with *tf.name_scope* and *tf.Variables* methods. However, these APIs are NOT for shared variables.  For example, *tf.get_variable* below does not pick up the name scope created from *tf.name_scope*.
```python
with tf.name_scope("foo1"):
    v1 = tf.get_variable("v", [1])
    v2 = tf.Variable(1, name="v2")

with tf.variable_scope("foo2"):
    v3 = tf.get_variable("v", [1])
    v4 = tf.Variable(1, name="v2")

print(v1.name)  # v:0 (Unexpected!)
print(v2.name)  # foo1/v2:0
print(v3.name)  # foo2/v:0  
print(v4.name)  # foo2/v2:0
```

The best way to avoid nasty issues with shared variables are
* Do **NOT** use *tf.name_scope* and *tf.Variables* with shareable variables. 
* Always use *tf.variable_scope* to define the scope of a shared variable.
* Use *tf.get_varaible* to create or retrieve a shared variable.

```python
with tf.variable_scope("foo"):
    v = tf.get_variable("v2", [1])    # Create a new variable

with tf.variable_scope("foo", reuse=True):
    v1 = tf.get_variable("v2")        # Reuse a variable created before.
```


