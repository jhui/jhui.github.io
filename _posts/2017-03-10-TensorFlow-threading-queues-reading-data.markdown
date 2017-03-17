---
layout: post
comments: true
mathjax: true
title: “TensorFlow - Reading data, threading & queues.”
excerpt: “How threading & queue work? We later focus on how to read data into the TensorFlow.”
date: 2017-03-08 14:00:00
---

Before we learn how to read data into a TensorFlow programing through a pipeline, some basic understanding of the TensorFlow thread and queues are helpful but not required. 

### Threading
TensorFlow use a coordinator *tf.train.Coordinator* for thread coordination. The code below:
* Start 4 threads
	* Each thread print out a hello message and then sleep for a random amount of time.
        * When the thread wake up from sleep, it print another message.
* Whenever a thread print out 4 messages already, it will ask the coordinator to request all other threads to stop.
 
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

```python
# Create 4 threads
threads = [threading.Thread(target=loop, args=(coord,)) for i in range(4)]

# Start the threads and wait for all of them to stop.
for t in threads:
  t.start()

coord.join(threads)
```

* Create 4 threads and define its target and the coordinator.  
* Start all 4 threads which invoke the target method. 
* Wait until all threads are completed.


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

* For each thread, exit when it was told by the coordinator *coord.should_stop*.
* Print out a hello message and sleep for some random of time.
* Whenever a thread has print the message 4 times, it will ask the coordinator to request all other threads to stop.

### Queue
Data reading and preparation may take time. TensorFlow provide a queueing mechanism to allow multiple threads to enqueue data onto the same queue. The general steps are:
* Define a queue.
* Define op node(s) to enqueue data.
* Define an op node to dequeue data.
* Create a QueueRunner which start multiple enqueue later.
* Create a TensorFlow session and an coordinator.
	* Have the QueueRunner to launch multiple enqueue node threads.
	* Run the dequeue op
	* Wait until the coordinator to call it quit.


The following code illustrates the steps above.  However, we will simplify the logic by having the data stored in an array.
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
Define a queue which each row contains 2 entries: one with 4 elements and the other with 1.
```python
queue = tf.FIFOQueue(capacity=50, dtypes=[tf.float32, tf.int32], shapes=[[4], []])
```
Besides a FIFO queue, TensorFlow provides other queue like Priority Queue *tf.PriorityQueue* and Random shuffle queue *tf.RandomShuffleQueue*.

#### Define enqueue op node(s)
Create an op node which will put the (x, y) pair to the queue.
```python
enqueue_op = queue.enqueue_many([x, y])
```
Or we can define multiple nodes to enqueue messages.
```
enqueue_op1 = queue.enqueue_many([x, y])
enqueue_op2 = queue.enqueue_many([x2, y])
```

#### Define a dequeue node
Define a dequeue node. Since each row is defined to have 2 entries, the dequeue returns 2 items.
```python
data_sample, label_sample = queue.dequeue()
```

#### Create a QueueRunner 
Define a QueueRunner which will later start NUM_THREADS of enqueue_op1
```python
qr = tf.train.QueueRunner(queue, [enqueue_op1] * NUM_THREADS)
```
Or create a list of op that will run in parallel:
```python
qr = tf.train.QueueRunner(queue, [enqueue_op1, enqueue_op2])
```

#### Dequeue data
Within an TensorFlow session and a coordinator.
* Have the QueueRunner to start all the enqueue threads defined in the earlier steps.
* Use *sess.run* to dequeue one row at a time from the dequeue op node.

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

#### Handle exception gracefully
Make sure exceptions are handled when executing multiple threads.
```python
try:
    for step in range(100):
        if coord.should_stop():
            break
        sess.run(train_op)
except Exception, e:
    coord.request_stop(e)
finally:
    // ok to request stop twice.
    coord.request_stop()
    coord.join(threads)
```

### Reading data
There are 2 major methods to read data into a TensorFlow program:
* Reading from files: an input pipeline reads the data from files at the beginning of a TensorFlow graph.
* Feeding with feed_dict below:
```python
x = tf.placeholder(tf.float32, [None, 784])
...
sess.run(train_step, feed_dict={x: batch_xs, labels: batch_ys, lmbda:5e-5})
```

### Reading from files
A typical pipeline has the following stages:

* Create a queue for the data filenames.
* Create a Reader for the file format.
* A decoder for a record read by the reader
* Optional data preprocessing.
* Use a queue to hold training data.

```python
import tensorflow as tf

# Create a queue just for the filenames which leads to running multiple threads of reader.
filename_queue = tf.train.string_input_producer(["iris_training.csv", "iris_training2.csv", "iris_training3.csv"])

# Define the reader with input from the filename queue.
reader = tf.TextLineReader(skip_header_lines=1)
key, value = reader.read(filename_queue)

# Default values, in case of empty columns. Also specifies the type of the decoded result.
record_defaults = [[1.0], [1.0], [1.0], [1.0], [1.0]]
# Decode each line into CSV data.
col1, col2, col3, col4, col5 = tf.decode_csv(value, record_defaults=record_defaults)

features = tf.stack([col1, col2, col3, col4])
labels = tf.stack([col5])

with tf.Session() as sess:
 coord = tf.train.Coordinator()
 # Start all QueueRunners added into the graph.
 threads = tf.train.start_queue_runners(coord=coord)
 for _ in range(200):
     # Read one line of data at a time
     # d_features, d_label = sess.run([features, col5])
     # print(f"{d_features} {d_label}")

     min_after_dequeue = 10
     batch_size = 2
     capacity = min_after_dequeue + 3 * batch_size
     # Use shuffle_batch_join for more than 1 reader
     # Use shuffle_batch for 1 reader but possibly more than 1 thread.
     example_batch, label_batch = tf.train.shuffle_batch_join(
          [[features, labels]], batch_size=batch_size, capacity=capacity,
          min_after_dequeue=min_after_dequeue)
     # example_batch, label_batch = tf.train.shuffle_batch(
     #     [features, labels], batch_size=batch_size, capacity=capacity,
     #     min_after_dequeue=min_after_dequeue)
     # example_batch : shape(2, 4)
     # label_batch : shape(2, 1)
```

*string_input_producer* creates a FIFO queue for holding the filenames until the reader op needs them. This queue only holds the filename, not the data itself.
```python
filename_queue = tf.train.string_input_producer(["iris_training.csv", "iris_training2.csv", "iris_training3.csv"])
```

*string_input_producer* has options for shuffling and setting a maximum number of epochs. QueueRunner adds the whole list of filenames to the queue once for each epoch. Optional shuffling within an epoch (shuffle=True) provides a uniform sampling of files.

#### 
```python
reader = tf.TextLineReader(skip_header_lines=1)
key, value = reader.read(filename_queue)
```
We define a file reader for the file format and pass the filename queue to the reader's read method.

```python
record_defaults = [[1.0], [1.0], [1.0], [1.0], [1.0]]
col1, col2, col3, col4, col5 = tf.decode_csv(value, record_defaults=record_defaults)
```
We select a csv decoder to decode each line into 5 column values. Other reader and decoders are available:
* *tf.FixedLengthRecordReader* with *tf.decode_raw* for fixed size binary data.
*  TFRecords which is the standard TensorFlow file format.

```python
with tf.Session() as sess:
 coord = tf.train.Coordinator()
 # Start all QueueRunners added into the graph.
 threads = tf.train.start_queue_runners(coord=coord)
 for _ in range(200):
     # Read one line of data at a time
     d_features, d_label = sess.run([features, col5])
     print(f"{d_features} {d_label}")
```

#### Use another queue to batch data

```python
with tf.Session() as sess:
 coord = tf.train.Coordinator()
 # Start all QueueRunners added into the graph.
 threads = tf.train.start_queue_runners(coord=coord)
 for _ in range(200):

     min_after_dequeue = 10
     batch_size = 2
     capacity = min_after_dequeue + 3 * batch_size
     example_batch, label_batch = tf.train.shuffle_batch(
          [features, labels], batch_size=batch_size, capacity=capacity,
          min_after_dequeue=min_after_dequeue)
     # example_batch : shape(2, 4)
     # label_batch : shape(2, 1)

```

At the end of the pipeline we can use another queue to batch together data. *tf.train.shuffle_batch* randomize the data in its queue for consumptions. If more parallelism is needed, use *tf.train.shuffle_batch_join* to deploy multiple reader instances to enqueue the data queue.

The following animation (from TensorFlow) indicates how data is read from the data pipeline.

<div class="imgcap">
<img src="/assets/read_data/pipeline.png" style="border:none;">
</div>

 