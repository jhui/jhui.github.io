---
layout: post
comments: true
mathjax: true
priority: 840
title: “TensorFlow performance, GPU and advance topics”
excerpt: “Cover TensorFlow advance topics including performance, GPU and other advance topics.”
date: 2017-03-07 14:00:00
---
### Performance
*  Building from source with compiler optimizations for the target hardware CPU and GPU. Install latest CUDA and cuDNN libraries. 
* Use queue in reading data. Do not use feed data.
* Put the data preprocessing on the CPU.
* Reading many small files are not efficient. Pre-process the data and create a few large one with TFRecord.
* When using tf.contrib.layers.batch_norm, set the attribute fused=True.
* Consider quantizing the Neural network for inference in particular for mobile devices.

### Supervisor

Supervisor allows a long running training to be recovered after a crash.  Checkpoints are constantly made.  When the process dies, it can reload trained parameters from the checkpoint appoint startup. The code related to the supervisor is shown as below:
```python
# For monitor purpose, we need to setup the global_step variable.
global_step = tf.Variable(0, name='global_step', trainable=False)

...
# Define at least one summary to collect.
tf.summary.scalar('loss', loss)

...
train = optimizer.minimize(loss, global_step=global_step)

...

summary_op = tf.summary.merge_all()
...

# Set up a supervisor and use that to start a session.
sv = tf.train.Supervisor(logdir="/tmp/tensorflow/supervisor", summary_op=None)
with sv.managed_session() as sess:
    sess.run(init)
    for step in range(10000):
        # Listen to the supervisor if it want us to stop.
        if sv.should_stop():
            break
        sess.run(train, {x:x_train, y:y_train})

	# For large model, we need to run the summary section manually once a while.
        if step % 1000 == 0:
            _, summ = sess.run([train, summary_op], {x:x_train, y:y_train})
            sv.summary_computed(sess, summ)
        else:
            sess.run(train, {x: x_train, y: y_train})
```

The full source code to train a linear model:
```python
import tensorflow as tf

W = tf.Variable([0.1], tf.float32)
b = tf.Variable([0.0], tf.float32)
global_step = tf.Variable(0, name='global_step', trainable=False)

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# Define a linear model y = Wx + b
model = W * x + b

loss = tf.reduce_sum(tf.square(model - y))
tf.summary.scalar('loss', loss)

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss, global_step=global_step)

x_train = [1.0, 2.0, 3.0, 4.0]
y_train = [1.5, 3.5, 5.5, 7.5]

summary_op = tf.summary.merge_all()

init = tf.global_variables_initializer()

sv = tf.train.Supervisor(logdir="/tmp/tensorflow/supervisor", summary_op=None)
with sv.managed_session() as sess:
    sess.run(init)
    for step in range(1000):
        if sv.should_stop():
            break
        sess.run(train, {x:x_train, y:y_train})

        if step % 100 == 0:
            _, summ = sess.run([train, summary_op], {x:x_train, y:y_train})
            sv.summary_computed(sess, summ)
        else:
            sess.run(train, {x: x_train, y: y_train})

    l_W, l_b, l_cost  = sess.run([W, b, loss], {x:x_train, y:y_train})
    print(f"W: {l_W} b: {l_b} cost: {l_cost}")
    # W: [ 1.99999797] b: [-0.49999401] cost: 2.2751578399038408e-11
```

### Pre-train data
To load pre-train data into a model. The following show the changes added.
```python
logdir = "/tmp/tensorflow/supervisor"

def load_pretrain(sess):
    pre_train_saver.restore(sess, "logdir")

...
# Load the trainable parameters W and b
pre_train_saver = tf.train.Saver([W, b])

...
# Call load_retain when session starts
sv = tf.train.Supervisor(logdir=logdir, summary_op=None, init_fn=load_retrain)
...
```
Here is the full code for the linear model.
```python
import tensorflow as tf

logdir = "/tmp/tensorflow/supervisor"

def load_pretrain(sess):
    pre_train_saver.restore(sess, "logdir")

W = tf.Variable([0.1], tf.float32)
b = tf.Variable([0.0], tf.float32)
global_step = tf.Variable(0, name='global_step', trainable=False)

pre_train_saver = tf.train.Saver([W, b])

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# Define a linear model y = Wx + b
model = W * x + b

loss = tf.reduce_sum(tf.square(model - y))
tf.summary.scalar('loss', loss)

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss, global_step=global_step)

x_train = [1.0, 2.0, 3.0, 4.0]
y_train = [1.5, 3.5, 5.5, 7.5]

summary_op = tf.summary.merge_all()

init = tf.global_variables_initializer()

sv = tf.train.Supervisor(logdir=logdir, summary_op=None, init_fn=load_pretrain)
with sv.managed_session() as sess:
    sess.run(init)
    for step in range(1000):
        if sv.should_stop():
            break
        sess.run(train, {x:x_train, y:y_train})

        if step % 100 == 0:
            _, summ = sess.run([train, summary_op], {x:x_train, y:y_train})
            sv.summary_computed(sess, summ)
        else:
            sess.run(train, {x: x_train, y: y_train})

    l_W, l_b, l_cost  = sess.run([W, b, loss], {x:x_train, y:y_train})
    print(f"W: {l_W} b: {l_b} cost: {l_cost}")
    # W: [ 1.99999797] b: [-0.49999401] cost: 2.2751578399038408e-11
```

### GPU

To determine where your computation node is running on (CPU/GPU)?

```python
import tensorflow as tf

# Construct 2 op nodes (m1, m2) representing 2 matrix.
m1 = tf.constant([[3, 5]])
m2 = tf.constant([[2],[4]])

product = tf.matmul(m1, m2)    # A matrix multiplication op node

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(sess.run(product))

sess.close()

# MatMul: (MatMul): /job:localhost/replica:0/task:0/cpu:0
# Const_1: (Const): /job:localhost/replica:0/task:0/cpu:0
# Const: (Const): /job:localhost/replica:0/task:0/cpu:0
```

To run m1, m2 and product op node on the specific device _CPU 0_.
```python
import tensorflow as tf

# Construct 2 op nodes (m1, m2) representing 2 matrix.
with tf.device('/cpu:0'):
    m1 = tf.constant([[3, 5]])
    m2 = tf.constant([[2],[4]])

    product = tf.matmul(m1, m2)    # A matrix multiplication op node

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(sess.run(product))

sess.close()
```

Run operations on different device:
```python
with tf.device('/cpu:0'):
  # Pinned to the CPU.
  img = tf.decode_jpeg(tf.read_file("img.jpg"))

with tf.device('/gpu:0'):
  result = tf.matmul(weights, img)
```
  
Using multiple GPUs
```python
for d in ['/gpu:1', '/gpu:2']:
  with tf.device(d):
     ...
	 
with tf.device('/cpu:0'):
     ...
```

Soft placement: If GPU 2 does not exist, allow_soft_placement=True will place it onto an alternative device to run the operation.  Otherwise, the operation will throw an exception if GPU 2 does not exist.
```python
with tf.device('/gpu:2'):
	...
sess = tf.Session(config=tf.ConfigProto(
      allow_soft_placement=True, log_device_placement=True))
```

### Operations and tensors

TensorFlow API constructs new tf.Operation (node) and tf.Tensor (edge) objects and add them to a tf.Graph instance. 

* _tf.constant(10.0)_ adds a tf.Operation to the default graph that produces the value 10.0, and returns a tf.Tensor that represents the value of the constant. 
* _tf.matmul(a, b) creates a tf.Operation that multiplies the values of tf.Tensor objects $$a$$ and $$b$$ and returns a tf.Tensor for the multiplication result.
* v = tf.Variable(0) creates a tf.Operation that store a writeable tensor value that persists between tf.Session.run calls. The tf.Variable object wraps this operation, and can be used as a tensor to read the current value.  
* tf.train.Optimizer.minimize will add operations and tensors that calculate gradients and return a tf.Operation that apply those gradient changes to a set of variables.

TensorFlow will create a new tf.Tensor each time when a tensor-like object (numpy.ndarray or list) is passed as parameters. It will run out of memory if the object is used multiple times in constructing nodes. To avoid this, call tf.convert_to_tensor on the tensor-like object once and use the returned tf.Tensor instead.


### Meta-data of session

To collect meta-data of a session:
```python
import tensorflow as tf

y = tf.matmul([[15.0, -3.0], [3.0, 4.0]], tf.random_uniform([2, 2]))

with tf.Session() as sess:
  # Define options for the `sess.run()` call.
  options = tf.RunOptions()
  options.output_partition_graphs = True
  options.trace_level = tf.RunOptions.FULL_TRACE

  # Define a container for the returned metadata.
  metadata = tf.RunMetadata()

  sess.run(y, options=options, run_metadata=metadata)

  # Print the subgraphs that executed on each device.
  print(metadata.partition_graphs)

  # Print the timings of each operation that executed.
  print(metadata.step_stats)
```
