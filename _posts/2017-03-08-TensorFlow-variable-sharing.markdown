---
layout: post
comments: true
priority: 810
mathjax: true
title: “TensorFlow variables, saving/restore”
excerpt: “TensorFlow variables, saving/restore”
date: 2017-03-08 12:00:00
---
### Variables

```python
# Rank 0 tensor (scalar)
fruit = tf.Variable("Orange", tf.string)
quantity = tf.Variable(2, tf.int16)
price = tf.Variable(3.23, tf.float32)

# Rank 1 tensor
strings = tf.Variable(["Fruit", "orange"], tf.string)
prices  = tf.Variable([3.23, 4.02], tf.float64)

# Rank 2 tensor
answers = tf.Variable([[False, True],[False, False]], tf.bool)
```

When you train a model, we use variables to store training parameters like weight and bias, hyper parameters like learning rate, or state information like global step. 

However, the best way to create a variable is using _tf.get_variable_. It allows deep net to share parameters.
```python
import tensorflow as tf
import numpy as np

v1 = tf.get_variable("v1", [5, 5, 3])   # A tensor with shape (5, 5, 3) filled with random values
v2 = tf.get_variable("v2", initializer=tf.constant(2))    # 2, float32 scalar
v3 = tf.get_variable("v3", initializer=tf.constant([[2, 3], [4, 5]]))  # [[2, 3], [4, 5]]

v4 = tf.get_variable("v1", [3, 2], initializer=tf.zeros_initializer)
v5 = tf.get_variable("v2", [3, 2], initializer=tf.ones_initializer)

# [[ 1.  2.], [ 3.  4.], [ 5.  6.]]
v6 = tf.get_variable("v3", [3, 2], initializer=tf.constant_initializer([1, 2, 3, 4, 5, 6])) 

W = tf.get_variable("W", [784, 256], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(2.0 / 784)))
Z = tf.get_variable("z", [4, 5], initializer=tf.random_uniform_initializer(-1, 1)) 
```

The following program:
* Define variables and the initializers.
* Create op to update variable.
* Explicitly initialize the variables. (Always required)
* Retrieve a variable value.

```python
import tensorflow as tf

### Using variables
# Define variables and its initializer
weights = tf.get_variable("W", [784, 256], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(2.0 / 784)))
biases = tf.get_variable("z", [256], initializer=tf.zeros_initializer) 

counter = tf.get_variable("counter", initializer=tf.constant(0)) 

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

### Save a Checkpoint

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

### Restore a checkpoint

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

### Load a model and saving checkpoints regularly

This is the sample code in loading the model at the beginning and saves it occasionally during training.
```python
import tensorflow as tf
import os

def loadmodel(session, saver, checkpoint_dir):
    session.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(session, os.path.join(checkpoint_dir, ckpt_name))
        return True
    else:
        return False


def save(session, saver, checkpoint_dir, step):
    dir = os.path.join(checkpoint_dir, "model")
    saver.save(session, dir, global_step=step)


with tf.Session() as session:
    saver = tf.train.Saver()
    ...
    loadmodel(session, saver, "./checkpoint")
    ...
    for i in range(10000):
	    ...
        if (i % 1000 == 0):
           save(session, saver, "./checkpoint", i)
```

### Trainable/Non-trainable parameters

In transfer learning, we may load a model from a checkpoint but freeze some of the layers during training by setting "trainable=False". 
```
freezed_W = tf.get_variable('CNN_W!', [5, 5, 3, 32], trainable=False,
                            initializer=tf.truncated_normal_initializer(stddev=0.02))
...
loadmodel(session, saver, "./checkpoint")							
```

In some problems, we may have multiple deep nets to be trained together. To have two different optimizers with different cost functions for different trainable parameters.
```
import tensorflow as tf

def scope_variables(name):
    with tf.variable_scope(name):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 
                       scope=tf.get_variable_scope().name)

# Model parameters for the discriminator network
with tf.variable_scope("discriminator"):
   v1 = tf.get_variable("v1", [3], initializer=tf.zeros_initializer)
   ...
   
# Model parameters for the generator network
with tf.variable_scope("generator"):
   v2 = tf.get_variable("v2", [2], initializer=tf.zeros_initializer)
   ...

# Get all the trainable parameters for the discriminator   
discriminator_variables = scope_variables("discriminator")

# Get all the trainable parameters for the generator 
generator_variables = scope_variables("generator")

# 2 optimizers each for different networks
train_discriminator = discriminator_optimizer.minimize(d_loss, 
                              var_list=discriminator_variables)
train_generator = generator_optimizer.minimize(g_loss, 
                              var_list=generator_variables)
```
								   

### Scoping

We can use scoping such that we can create 2 different layers that have their own parameters from the same method. For example,  _cnn1_ and _cnn2_ have their own $$ w $$ and $$ b $$.

```
import tensorflow as tf

def conv2d(input, output_dim, filter_h=5, filter_w=5, stride_h=2, stride_w=2, stddev=0.02):
    w = tf.get_variable('w', [filter_h, filter_w, input.get_shape()[-1], output_dim],
                        initializer=tf.truncated_normal_initializer(stddev=stddev))
    conv = tf.nn.conv2d(input, w, strides=[1, stride_h, stride_w, 1], padding='SAME')
    biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
    conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

    return conv

input1 = tf.random_normal([1,10,10,32])
input2 = tf.random_normal([1,20,20,32])

with tf.variable_scope("conv1"):
    cnn1 = conv2d(input1, 16)

with tf.variable_scope("conv2"):
    cnn1 = conv2d(input2, 16)
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

If an operation named "name1" exist, the TensorFlow append "_1", "_2" etc.. to the name to make it unique.

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
However, TensorFlow wants the developer to be self-aware of whether the variable exists or not. Developers need to have the correct setting for the "reuse" flag before calling *tf.get_variable*. Both scenarios below will throw an **exception** when calling *tf.get_variable*:
*  if the reuse flag is False or None (default) and the variable already exists.
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

We can reuse _scope_ instead of supplying the scope name again:

```python
with tf.variable_scope("model") as scope:
  output1 = my_image_filter(input1)
with tf.variable_scope(scope, reuse=True):  # Can use scope instead of "model"
  output2 = my_image_filter(input2)
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

### Assignment

```python
v = tf.get_variable("v", shape=(), initializer=tf.zeros_initializer())

v1 = v.assign_add(1)  # 1.0
v.assign(v1)          # 1.0

with tf.Session() as session:
    tf.global_variables_initializer().run()
    value, value1 = session.run([v, v1])
    print(value, value1)

# 1.0 1.0
```

