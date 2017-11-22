---
layout: post
comments: true
mathjax: true
priority: 830
title: “TensorFlow - Importing data”
excerpt: “How to read data into the TensorFlow?”
date: 2017-11-21 14:00:00
---tf.data


### Basic

Data can be feed into TensorFlow using iterator.
```python
import tensorflow as tf

dataset = tf.data.Dataset.range(10)

iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
    for i in range(10):
      value = sess.run(next_element)
      print(f"{value} ", end=" ")    # 1 2 3 ... 10	   
```

The datatype and the shape of the dataset can be retrieved by:
```python
print(dataset.output_types)   # <dtype: 'int64'>
print(dataset.output_shapes)  # () - scalar
```

#### Out of range

An iterator can run out of values. Handling iterator's out of range:
```python
import tensorflow as tf

dataset = tf.data.Dataset.range(3)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

result = tf.add(next_element, next_element)

with tf.Session() as sess:
    print(sess.run(result))  # "0"
    print(sess.run(result))  # "2"
    print(sess.run(result))  # "4"
    print(sess.run(result))  
    try:
      sess.run(result)
    except tf.errors.OutOfRangeError:
      print("End of dataset")  # "End of dataset"
```
	  
If we want the iterator to keep repeat the data, we can call _repeat_ so the iterator will repeat itself at the end.	  
```python
dataset = tf.data.Dataset.range(3)
dataset = dataset.repeat()
```
	  
### Create an iterator

#### One-shot

As demonstrated before:
```
dataset = tf.data.Dataset.range(10)

iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
    for i in range(10):
      value = sess.run(next_element)
```
Note: The max range is pre-determined when building the iterator.

#### initializable iterator

In the example below, we allow the max range of the iterator to be supplied at runtime using a placeholder.
```python
import tensorflow as tf

max_value = tf.placeholder(tf.int64, shape=[])
dataset = tf.data.Dataset.range(max_value)    # Take a placeholder to create a dataset
iterator = dataset.make_initializable_iterator()      # Create an initializable iterator
next_element = iterator.get_next()

with tf.Session() as sess:
    # Initialize an iterator over a dataset with 10 elements using placeholder.
    sess.run(iterator.initializer, feed_dict={max_value: 10}) 

    for i in range(10):
        value = sess.run(next_element)
        print(f"{value} ", end=" ")    # 1 2 3 ... 10
```

	  
#### reinitializable iterator

We can create an iterator for different datasets. For example, in training, we use the training dataset for the iterator and the validation dataset for the validation. For reinitializable iterator, both dataset must have the same datatype and shape.

```
import tensorflow as tf


training_dataset = tf.data.Dataset.range(100).map(
    lambda x: x + tf.random_uniform([], -10, 10, tf.int64))
validation_dataset = tf.data.Dataset.range(50)

# Build an iterator that can take different datasets with the same type and shape
iterator = tf.Iterator.from_structure(training_dataset.output_types, training_dataset.output_shapes)
next_element = iterator.get_next()

# Get 2 init op for 2 different dataset
training_init_op = iterator.make_initializer(training_dataset)
validation_init_op = iterator.make_initializer(validation_dataset)

with tf.Session() as sess:
    for _ in range(20):
      sess.run(training_init_op)
      for _ in range(100):
        sess.run(next_element)

      sess.run(validation_init_op)
      for _ in range(50):
        sess.run(next_element)
```
 
#### Feedable iterator

In reinitializable iterator, we reinitialize the iterator evertime when we switch the dataset. In Feedable iterator, the dataset is supplied in the feed_dict in _tf.Session.run_ without the reinitialization.

```python
import tensorflow as tf


# Create 2 dataset witht the same datatype and shape
training_dataset = tf.data.Dataset.range(300).map(lambda x: x + tf.random_uniform([], -10, 10, tf.int64))
validation_dataset = tf.data.Dataset.range(50)

# Create a feedable iterator that use a placeholder to switch between dataset
handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(
    handle, training_dataset.output_types, training_dataset.output_shapes)
next_element = iterator.get_next()

# Create 2 iterators
training_iterator = training_dataset.make_one_shot_iterator()
validation_iterator = validation_dataset.make_initializable_iterator()

with tf.Session() as sess:
    # Return handles that can be feed as the iterator in sess.run
    training_handle = sess.run(training_iterator.string_handle())
    validation_handle = sess.run(validation_iterator.string_handle())

    for _ in range(3):
        for _ in range(100):
            sess.run(next_element, feed_dict={handle: training_handle})

        sess.run(validation_iterator.initializer)
        for _ in range(50):
            sess.run(next_element, feed_dict={handle: validation_handle})
``` 

### Dataset shape

Dataset with Rank 2 (2-D) tensors
```python
import tensorflow as tf

dataset = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))

iterator = dataset.make_initializable_iterator()      # Create an initializable iterator
next_element = iterator.get_next()

with tf.Session() as sess:
    sess.run(iterator.initializer)

    for i in range(4):
      value = sess.run(next_element)
      print(f"{value} ")        # Print out an array with 10 random numbers
```

With shape ((), (100,))
```python
dataset = tf.data.Dataset.from_tensor_slices(
   (tf.random_uniform([4]),              # (tf.float32, tf.int32)
    tf.random_uniform([4, 100], maxval=100, dtype=tf.int32))) # ((), (100,))
```

Zip 2 dataset
```python
dataset = tf.data.Dataset.zip((dataset1, dataset2))
print(dataset3.output_types)  # ==> (tf.float32, (tf.float32, tf.int32))
print(dataset3.output_shapes)  # ==> "(10, ((), (100,)))
```

Data can be read as
```
next1, (next21, next22) = iterator.get_next()
```

Giving labels:
```python
dataset = tf.data.Dataset.from_tensor_slices(
   {"a": tf.random_uniform([4]),
    "b": tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)})
print(dataset.output_types)  # ==> "{'a': tf.float32, 'b': tf.int32}"
print(dataset.output_shapes)  # ==> "{'a': (), 'b': (100,)}"
```

### Consuming Numpy array as data

Saving data in Numpy
```python
import numpy as np
import tensorflow as tf

dt = np.dtype([('features', float, (2,)),
                ('label', int)])

x = np.zeros((2,), dtype=dt)
x[0]['features'] = [3.0, 2.5]
x[0]['label'] = 2

x[1]['features'] = [1.4, 2.1]
x[1]['label'] = 1

np.save('in.npy', x)
```	

Reading Numpy data as TensorFlow dataset.
```python
data = np.load('in.npy')

features = data["features"]
label = data["label"]

features_placeholder = tf.placeholder(features.dtype, features.shape)
labels_placeholder = tf.placeholder(label.dtype, label.shape)

dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
iterator = dataset.make_initializable_iterator()

with tf.Session() as sess:
    sess.run(iterator.initializer, feed_dict={features_placeholder: features, labels_placeholder: label})
```	

### Consuming TFRecord

To create a dataset from TFRecord and have the iteration keep repeating.
```python
filenames = get_filenames()   # Array of filename pathes as string
dataset = tf.data.TFRecordDataset(filenames).repeat()
```

Create the operators to parse the dataset.
```python
def parser(serialized_example):
    """Parses a single tf.Example into image and label tensors."""
	...
    return image, label
	
dataset = dataset.map(
        parser, num_threads=batch_size, output_buffer_size=2 * batch_size)

# Batch it up.
dataset = dataset.batch(batch_size)
```

Create the iterator operators:
```python
iterator = dataset.make_one_shot_iterator()
image_batch, label_batch = iterator.get_next()
```

The full source:
```
import tensorflow as tf
import os

HEIGHT = 32
WIDTH = 32
DEPTH = 3

NUM_PER_EPOCH = 50000

def parser(serialized_example):
    """Parses a single tf.Example into image and label tensors."""
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })
    image = tf.decode_raw(features['image'], tf.uint8)
    image.set_shape([DEPTH * HEIGHT * WIDTH])

    # Reshape from [depth * height * width] to [depth, height, width].
    image = tf.cast(
        tf.transpose(tf.reshape(image, [DEPTH, HEIGHT, WIDTH]), [1, 2, 0]),
        tf.float32)
    label = tf.cast(features['label'], tf.int32)

    return image, label


def make_batch(batch_size):
    """Read the images and labels from 'filenames'."""
    filenames = [os.path.join(".", 'f1.tfrecords'), os.path.join(".", 'f2.tfrecords')]

    # Repeat infinitely.
    dataset = tf.data.TFRecordDataset(filenames).repeat()

    # Parse records.
    dataset = dataset.map(
        parser, num_threads=batch_size, output_buffer_size=2 * batch_size)

    # Potentially shuffle records.
    min_queue_examples = int(NUM_PER_EPOCH * 0.4)
    dataset = dataset.shuffle(buffer_size=min_queue_examples + 3 * batch_size)

    # Batch it up.
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    image_batch, label_batch = iterator.get_next()

    return image_batch, label_batch
```	

it maintains a fixed-size buffer and chooses the next element randomly from the buffer.
```python
dataset = dataset.shuffle(buffer_size=10000)
```

#### Parsing

Parsing each tfrecords
```python
def _parse(example_proto):
  features = {"image": tf.FixedLenFeature((), tf.string, default_value=""),
              "label": tf.FixedLenFeature((), tf.int32, default_value=0)}
  parsed_features = tf.parse_single_example(example_proto, features)
  return parsed_features["image"], parsed_features["label"]

filenames = [os.path.join(".", 'f1.tfrecords'), os.path.join(".", 'f2.tfrecords')]
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(_parse)
```

Parsing images
```python
def _parse(filename, label):
  """ Reading and resize image"""
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_image(image_string)
  image_resized = tf.image.resize_images(image_decoded, [28, 28])
  return image_resized, label

# A vector of filenames.
filenames = tf.constant([os.path.join(".", 'f1.jpg'), os.path.join(".", 'f2.jpg')])
labels = tf.constant([2, 5])

dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.map(_parse)
```

Using TF library is preferred over external libraries for performance reason. Nevertheless, if calling external libraries are needed, use tf.py_func.
```python
import cv2

def _read_py_function(filename, label):
  image_decoded = cv2.imread(image_string, cv2.IMREAD_GRAYSCALE)
  return image_decoded, label

def _resize_function(image_decoded, label):
  image_decoded.set_shape([None, None, None])
  image_resized = tf.image.resize_images(image_decoded, [28, 28])
  return image_resized, label

filenames = tf.constant([os.path.join(".", 'f1.jpg'), os.path.join(".", 'f2.jpg')])
labels = tf.constant([2, 5])

dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.map(
    lambda filename, label: tf.py_func(
        _read_py_function, [filename, label], [tf.uint8, label.dtype]))
dataset = dataset.map(_resize_function)
```

#### Writing tfrecords

Example code to write data into tfrecords
```python 
import tensorflow as tf
from PIL import Image
import numpy as np
import os

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

filename = os.path.join(".", 'f1.tfrecords')

writer = tf.python_io.TFRecordWriter(filename)

img = np.array(Image.open(os.path.join(".", 'f1.jpg')))

height = img.shape[0]
width = img.shape[1]


img_raw = img.tostring()

example = tf.train.Example(features=tf.train.Features(feature={
    'height': _int64_feature(height),
    'width': _int64_feature(width),
    'image_raw': _bytes_feature(img_raw)}))

writer.write(example.SerializeToString())

writer.close()
```

### Reading from text lines

Create a text line dataset
```python
filenames = [os.path.join(".", 'f1.txt'), os.path.join(".", 'f2.txt')]
dataset = tf.data.TextLineDataset(filenames)
```

Filter out first line and comments
```python
filenames = [os.path.join(".", 'f1.txt'), os.path.join(".", 'f2.txt')]

dataset = tf.data.Dataset.from_tensor_slices(filenames)

dataset = dataset.flat_map(
    lambda filename: (
        tf.data.TextLineDataset(filename)
        .skip(1)       # Skip first line
        .filter(lambda line: tf.not_equal(tf.substr(line, 0, 1), "#")))) # Skip comment line
```

### Batching

To create a mini-batch:
```python
dataset = tf.data.Dataset.range(100)
batched_dataset = dataset.batch(4)

iterator = batched_dataset.make_one_shot_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
    print(sess.run(next_element))  # [0 1 2 3]
```	

Use padded_batch for padding batches.
```python
dataset = tf.data.Dataset.range(13)

# For x=0 -> [0], x=2 -> [2, 2], x=3 -> [3, 3, 3]
dataset = dataset.map(lambda x: tf.fill([tf.cast(x, tf.int32)], x))

# Create a mini-batch of size 4. Pad 0 if needed.
dataset = dataset.padded_batch(4, padded_shapes=[None])

iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
    print(sess.run(next_element))  # ==> [[0, 0, 0], [1, 0, 0], [2, 2, 0], [3, 3, 3]]
    print(sess.run(next_element))  # ==> [[4, 4, 4, 4, 0, 0, 0],
                                   #      [5, 5, 5, 5, 5, 0, 0],
                                   #      [6, 6, 6, 6, 6, 6, 0],
                                   #      [7, 7, 7, 7, 7, 7, 7]]
```							   

To run 10 epochs:

```python
for _ in range(10):
  sess.run(iterator.initializer)
  while True:
    try:
      sess.run(next_element)
    except tf.errors.OutOfRangeError:
      break
```	  

### MonitoredTrainingSession

 MonitoredTrainingSession uses OutOfRangeError to signal that training has completed. It is recommended to use make_one_shot_iterator with it.
```python
iterator = dataset.make_one_shot_iterator()
...

with tf.train.MonitoredTrainingSession(...) as sess:
  while not sess.should_stop():
    sess.run(training_op)
```	

### Estimator

Use make_one_shot_iterator with the Estimator.
```
import tensorflow as tf
from PIL import Image
import numpy as np
import os

def dataset_input_fn():
  filenames = ["./file1.tfrecord", "./file2.tfrecord"]
  dataset = tf.data.TFRecordDataset(filenames)

  def parser(record):
    keys_to_features = {
        "image_data": tf.FixedLenFeature((), tf.string, default_value=""),
        "date_time": tf.FixedLenFeature((), tf.int64, default_value=""),
        "label": tf.FixedLenFeature((), tf.int64,
                                    default_value=tf.zeros([], dtype=tf.int64)),
    }
    parsed = tf.parse_single_example(record, keys_to_features)

    image = tf.decode_jpeg(parsed["image_data"])
    image = tf.reshape(image, [299, 299, 1])
    label = tf.cast(parsed["label"], tf.int32)

    return {"image_data": image, "date_time": parsed["date_time"]}, label

  dataset = dataset.map(parser)
  dataset = dataset.shuffle(buffer_size=10000)
  dataset = dataset.batch(32)
  dataset = dataset.repeat(10)
  iterator = dataset.make_one_shot_iterator()

  features, labels = iterator.get_next()
  return features, labels
```  