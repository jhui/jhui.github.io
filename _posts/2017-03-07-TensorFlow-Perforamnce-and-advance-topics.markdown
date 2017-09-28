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

Run constant op m1, m2 on the specific device _CPU 0_.
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
# Creates a graph.
c = []
for d in ['/gpu:2', '/gpu:3']:
  with tf.device(d):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2])
    c.append(tf.matmul(a, b))
	
with tf.device('/cpu:0'):
  sum = tf.add_n(c)

# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

# Runs the op.
print(sess.run(sum))
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

