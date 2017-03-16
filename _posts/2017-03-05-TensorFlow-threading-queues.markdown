---
layout: post
comments: true
mathjax: true
title: “TensorFlow - threading and queues”
excerpt: “This is an advance TensorFlow topic discussing how to control threads and queues.”
date: 2017-03-05 14:00:00
---
### Threading
TensorFlow use a coordinator *tf.train.Coordinator* for thread coordination.
```python
import tensorflow as tf
import threading
import time
import random

# Thread body: loop until the coordinator request it to stop.
def loop(coord):
  i = 0
  # Check if the coordinate request me to stop.
  while not coord.should_stop():
    i += 1
    print(f"{threading.get_ident()} say {i}")
    time.sleep(random.randrange(4))
    if i == 4:
      # Request the coord to stop all threads.
      coord.request_stop()

# Main thread: create a coordinator.
coord = tf.train.Coordinator()

# Create 4 threads
threads = [threading.Thread(target=loop, args=(coord,)) for i in range(4)]

# Start the threads and wait for all of them to stop.
for t in threads:
  t.start()

coord.join(threads)
```

Create a coordinator 
```python
coord = tf.train.Coordinator()
```

* Create 4 threads and define its target and the coordinator.  
* Start all 4 threads which invoke the target method. 
* Wait until all threads are completed.

```python
# Create 4 threads
threads = [threading.Thread(target=loop, args=(coord,)) for i in range(4)]

# Start the threads and wait for all of them to stop.
for t in threads:
  t.start()

coord.join(threads)

```

* For each thread, exit when it was told by the coordinator *coord.should_stop*
* Print out a hello message and sleep for a second.
* Whenever a thread has done it 4 times, it will request the coordinator to stop all threads.

```python
def loop(coord):
  i = 0
  # Check if the coordinate request me to stop.
  while not coord.should_stop():
    i += 1
    print(f"{threading.get_ident()} say {i}")
    time.sleep(random.randrange(4))
    if i == 4:
      # Request the coord to stop all threads.
      coord.request_stop()
```

### Queue
TensorFlow allows multiple Threads to write to a queue while another thread dequeue the data to fit into the model. The general steps include:
* Define a queue.
* Define op node(s) to enqueue data.
* Define an op node to dequeue data.
* Create a QueueRunner which start multiple enqueue later.
* Create a TensorFlow session and an coordinator.
	* Have the QueueRunner to launch multiple enqueue node threads.
	* Run the dequeue op
        * Wait until the coordinator to call it quit.

```python
import numpy as np
import tensorflow as tf
import time

NUM_THREADS = 2
N_SAMPLES = 5

x = np.random.randn(N_SAMPLES, 4) + 1         # shape (5, 4)
y = np.random.randint(0, 2, size=N_SAMPLES)   # shape (5, )
x2 = np.zeros((N_SAMPLES, 4))

# Define a FIFOQueue which each queue entry has 2 elements of length 4 and 1 respectively
queue = tf.FIFOQueue(capacity=50, dtypes=[tf.float32, tf.int32], shapes=[[4], []])

# Create an enqueue op to enqueue the [x, y]
enqueue_op1 = queue.enqueue_many([x, y])
enqueue_op2 = queue.enqueue_many([x2, y])

# Create an dequeue op
data_sample, label_sample = queue.dequeue()

# QueueRunner: create a number of threads to enqueue tensors in the queue.
# qr = tf.train.QueueRunner(queue, [enqueue_op1] * NUM_THREADS)
qr = tf.train.QueueRunner(queue, [enqueue_op1, enqueue_op2])

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    # Launch the queue runner threads.
    enqueue_threads = qr.create_threads(sess, coord=coord, start=True)
    for step in range(20):
        if coord.should_stop():
            break
        one_data, one_label = sess.run([data_sample, label_sample])
        print(f"x = {one_data} y = {one_label}")
    coord.request_stop()
    coord.join(enqueue_threads)

# x = [ 0.90066725 -2.47472358  1.4626869   0.93552333] y = 0
# x = [ 3.27642441  0.59251779  2.4254427   0.99563134] y = 0
# x = [-0.36993721  1.10983336  0.07864232  0.78808331] y = 1
# x = [-1.34663463  0.57584733 -0.45564255 -0.27264795] y = 1
# x = [ 1.41686928  0.31506935  0.8132937   1.0751847 ] y = 0
# x = [ 0.  0.  0.  0.] y = 0
# x = [ 0.  0.  0.  0.] y = 0
# ...
```

#### Define a queue.
Each row in this queue has 2 entries. One with 4 elements and the other with 1.
```python
queue = tf.FIFOQueue(capacity=50, dtypes=[tf.float32, tf.int32], shapes=[[4], []])
```

#### Define enqueue op node(s)
```python
enqueue_op = queue.enqueue_many([x, y])
```
Or we can define multiple nodes
```
enqueue_op1 = queue.enqueue_many([x, y])
enqueue_op2 = queue.enqueue_many([x2, y])
```

#### Define a dequeue node
Since each row is defined to have 2 entries, the dequeue returns 2 items.
```python
data_sample, label_sample = queue.dequeue()
```

#### Create a QueueRunner 
Define a QueueRunner which will later start NUM_THREADS of enqueue_op1
```python
qr = tf.train.QueueRunner(queue, [enqueue_op1] * NUM_THREADS)
```
Or create your own list of nodes to start:
```python
qr = tf.train.QueueRunner(queue, [enqueue_op1, enqueue_op2])
```
#### Dequeue data
Within an TensorFlow session and a coordinator
* Have the QueueRunner to start all the threads defined earlier.
* We dequeue one row at a time.

```python
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    # Launch the queue runner threads.
    enqueue_threads = qr.create_threads(sess, coord=coord, start=True)
    for step in range(20):
        if coord.should_stop():
            break
        one_data, one_label = sess.run([data_sample, label_sample])
        print(f"x = {one_data} y = {one_label}")
    coord.request_stop()
    coord.join(enqueue_threads)
```
