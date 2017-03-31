---
layout: post
comments: true
mathjax: true
title: “Convolution neural networks (CNN) tutorial”
excerpt: “Convolutional networks explore features by discover its spatial information. This tutorial will build CNN networks for visual recognition.”
date: 2017-03-01 12:00:00
---
**This is work in progress... The content needs major editing.**

### Overview
In a fully connected networks, all nodes in a layer is fully connected to all the nodes in the previous layer. This produces a complex model to explore all possible connections among nodes. But the complexity pays a high price of how easy to train the network and how deep the network can be. For spatial data like image, this complexity provides no additional benefits since most features are localized.

<div class="imgcap">
<img src="/assets/cnn/ppl.jpg" style="border:none;width:30%">
</div>

For face detection, the area of interested are all localized. Convolution neural networks apply small size filter to explore the images.The number of trainable parameters are significantly smaller and therefore allow CNN to use many filters to extract interested features. 

### Filters
Filters are frequently apply to images for different purposes. Human visual system applies edge detection filters to recognize object.

<div class="imgcap">
<img src="/assets/cnn/edge.png" style="border:none;">
</div>

For example, to blur an image, we can apply a filter with patch size 3x3 over every pixels in the image:
<div class="imgcap">
<img src="/assets/cnn/filter_b.png" style="border:none;">
</div>

To apply the filter to an image, we move the fiter 1 pixel at a time from left to right and top to bottom until we process every pixels.
<div class="imgcap">
<img src="/assets/cnn/stride.png" style="border:none;width:50%">
</div>

#### Stride and padding
However, we may encounter some problem on the edge. For example, on the top left corner, a filter may cover beyond the edge of an image. For a filter with patch size 3x3, we may ignore the edge and geneate an output with width and height reduce by 2 pixels. Otherwise, we can pack extra 0 or replicate the edge of the origina image. All these settings are possible and configurable as "padding" in a CNN. 
<div class="imgcap">
<img src="/assets/cnn/padding.png" style="border:none;width:50%">
</div>

For a CNN, sometimes we do not move the filter only by 1 pixel. If we move the filter 2 pixels to the right, we call the "X stride" equal to 2.
<div class="imgcap">
<img src="/assets/cnn/stride2.png" style="border:none;width:50%">
</div>

Notice that both padding and stride may change the spatial dimension of the output. A stride of 2 in X direction will reduce X-dimension by 2. Without padding, the output shrink by N pixels:

$$
N = \frac {\text{filter patch fsize} - 1} {2}
$$

### Convolution neural network (CNN)
A convolution neural network composes of convolution layers, polling layers and fully connected layers(FC). 

<div class="imgcap">
<img src="/assets/cnn/conv_layer.png" style="border:none;width:70%">
</div>

When we process the image, we apply filters which each geneate an output that we call **feature map**. If k features map are created, we have feature maps with depth k.

<div class="imgcap">
<img src="/assets/cnn/filter_m.png" style="border:none;width:70%">
</div>

#### Pooling

To reduce the spatial dimension of a feature map, we apply maximum pool. A 2x2 maximum pool replace a 2x2 area by its maximum. After apply a 2x2 pool, we reduce the spatial dimension for the example below from 4x4 to 2x2.
<div class="imgcap">
<img src="/assets/cnn/pooling.png" style="border:none;width:50%">
</div>

Here, we construct a CNN using convolution and pooling:
<div class="imgcap">
<img src="/assets/cnn/conv_layer2.png" style="border:none;width:50%">
</div>

Pooling is often used with a convolution layer. Therefore, we often consider it as part of the convolution layer rather than a separate layer. Other pooling like average pooling can be applied. However, for image classification, maximum pooling is more common.

### Multiple convolution layers

Like deep learning, the depth of the network increases the complexity of a model. A CNN network usually composes of many convolution layers. 
<div class="imgcap">
<img src="/assets/cnn/convolution_b1.png" style="border:none;width:70%">
</div>

The CNN above composes of 3 convolution layer. We start with a 32x32 pixel image with 3 channels (RGB). We apply a 3x4 filters and a 2x2 max pooling which covert the image to 16x16x4 feature maps.  The following table walks through the filter and layer shape at each layer:
<div class="imgcap">
<img src="/assets/cnn/cnn_chanl.png" style="border:none">
</div>

### Fully connected (FC) layers
After using convolution layers to extract the spatial features of an image, we apply fully connected layers for the final classification. First we flatten the output of the convolution layers. For example, if the final features maps have a dimension of 4x4x512, we will flaten it to an array of 4096 elements. We apply 2 more hidden layers here before we perform the final classification. The techniques needed are no difference from a FC network in deep learning.

<div class="imgcap">
<img src="/assets/cnn/convolution_b2.png" style="border:none;width:50%">
</div>

### Convolutional pyramid

For each convolution layer, we reduce the spatial dimension while increase the depth of the feature maps. Because of the shape, we call this a convolutional pyramid.

<div class="imgcap">
<img src="/assets/cnn/cnn3d.png" style="border:none;">
</div>

Here, we reduce the spatial dimension of each convolution layer through pooling or soemtimes apply a filter stride size > 1.
<div class="imgcap">
<img src="/assets/cnn/cnn3d4.png" style="border:none;width:50%">
</div>

The depth of the feature map can be increased by applying more filters.
<div class="imgcap">
<img src="/assets/cnn/cnn3d2.png" style="border:none;">
</div>

The core thinking of CNN is to apply small filters to explore spatial feature. The spatial dimension will gradually decrease as we go deep into the network. On the other hand, the depth of the feature maps will increase. It will eventually reach a stage that spatial locality is less important and we can apply a FC network for final analysis.

#### Google inceptions

In our previous discussion, the convolution filter in each layer is of the same patch size say 3x3. To increase the depth of the feature maps, we can apply more filters of the same patch size. However, in GoogleNet, it applies a different approach to increase the depth. GoogleNet use different filter patch size with different pooling to create feature maps of the same spatial dimension. Because they are of the same spatial dimension, all the features maps from the same layer can concatentate together to form one single feature maps.

Here we have filters with patch size 3x3 and 1x1 . The first set of filters generate 8 features map while the second one generate 2. We can concantentate them to form maps of depth 10. The inception idea is to increase the depth of the feature map by concantentate feature maps using different patch size of convolution filters and pooling. 
<div class="imgcap">
<img src="/assets/cnn/inception.png" style="border:none;width:60%">
</div>

#### Non-linearity and optimization
Inceptions can be consider as one way to introduce non-linearity into the system. In many CNN, we can apply similar layers we learned from deep learning after the convolution filters. This includes batch normalization and/or ReLU before the pooling for each convolution layer.

#### Fully connected network

After exploring the spatial relationship, we flatten the convolution layer output and connect it to a fully connected network:

<div class="imgcap">
<img src="/assets/cnn/cnn3d5.png" style="border:none;width:70%">
</div>

<div class="imgcap">
<img src="/assets/cnn/cnn3d6.png" style="border:none;width:70%">
</div>

### Tensor code
We will implement coding for a CNN to classify hand writing for digits (0 to 9).

> We will use TensorFlow to implement a CNN. Nevertheless, the puropose is for those curious audience that want details. Full understand of the coding is not needed or suggested even the code is pretty self explainable.

#### Construct a CNN
In the code below, we construct a CNN with 2 convolution layer followed by 2 FC layer and then 1 classifier. Here is where we construct our CNN network.
```python
# Model.
def model(data):
    # First convolution layer with stride = 1 and pad the edge to make the output size the same.
    # Apply ReLU and a maximum 2x2 pool
    conv1 = tf.nn.conv2d(data, cnn1_W, [1, 1, 1, 1], padding='SAME')
    hidden1 = tf.nn.relu(conv1 + cnn1_b)
    pool1 = tf.nn.max_pool(hidden1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    # Second convolution layer
    conv2 = tf.nn.conv2d(pool1, cnn2_W, [1, 1, 1, 1], padding='SAME')
    hidden2 = tf.nn.relu(conv2 + cnn2_b)
    pool2 = tf.nn.max_pool(hidden2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
	
    # Flattern the convolution output
    shape = pool2.get_shape().as_list()
    reshape = tf.reshape(pool2, [shape[0], shape[1] * shape[2] * shape[3]])

    # 2 FC hidden layers
    fc1 = tf.nn.relu(tf.matmul(reshape, fc1_W) + fc1_b)
    fc2 = tf.nn.relu(tf.matmul(fc1, fc2_W) + fc2_b)

    # Return the result of the classifier
    return tf.matmul(fc2, classifier_W) + classifier_b
```

For the convolution filter, we want the output spatial dimension to be the same as the input and therefore we assign padding = "same".  We also have stride = 1. After the filter, we apply the standard ReLU and a 2x2 maximum pool.
```python
conv1 = tf.nn.conv2d(data, cnn1_W, [1, 1, 1, 1], padding='SAME')
hidden1 = tf.nn.relu(conv1 + cnn1_b)
pool1 = tf.nn.max_pool(hidden1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
```

After the second convolution layer, we flatten the layer for the FC layer.
```python
shape = pool2.get_shape().as_list()
reshape = tf.reshape(pool2, [shape[0], shape[1] * shape[2] * shape[3]])
```

We applied 2 FC layers before return the result from the classifier.
```python
fc1 = tf.nn.relu(tf.matmul(reshape, fc1_W) + fc1_b)
fc2 = tf.nn.relu(tf.matmul(fc1, fc2_W) + fc2_b)

# Return the result of the classifier
return tf.matmul(fc2, classifier_W) + classifier_b
```

#### CNN configuration:

Here is our model configuration. We have 2 Convolution layers both using a filter with patch size 5x5 and generate feature maps with depth 16. The first FC output 256 values while the second output 64.
```python
batch_size = 16
patch_size = 5
depth = 16
num_hidden1 = 256
num_hidden2 = 64
```

Here is where we define the trainable parameters for CNN layer 1 and 2. For example, the shape of the weight in cnn1 is 5x5x3x16. It applies 5x5 filter patch for RGB channels which output feature maps with depth 16.
```python
# CNN layer 1 with filter (num_channels, depth) (3, 16)
cnn1_W = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev=0.1))
cnn1_b = tf.Variable(tf.zeros([depth]))

# CNN layer 2 with filter (depth, depth) (16, 16)
cnn2_W = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1))
cnn2_b = tf.Variable(tf.constant(1.0, shape=[depth]))
```

Here is the FC trainable parameters that is not much difference from other deep learning network using FC.
```python
 # Compute the output size of the CNN2 as a 1D array.
size = image_size // 4 * image_size // 4 * depth

# FC1 (size, num_hidden1) (size, 256)
fc1_W = tf.Variable(tf.truncated_normal([size, num_hidden1], stddev=np.sqrt(2.0 / size)))
fc1_b = tf.Variable(tf.constant(1.0, shape=[num_hidden1]))

# FC2 (num_hidden1, num_hidden2) (size, 64)
fc2_W = tf.Variable(tf.truncated_normal([num_hidden1, num_hidden2], stddev=np.sqrt(2.0 / (num_hidden1))))
fc2_b = tf.Variable(tf.constant(1.0, shape=[num_hidden2]))

# Classifier (num_hidden2, num_labels) (64, 10)
classifier_W = tf.Variable(tf.truncated_normal([num_hidden2, num_labels], stddev=np.sqrt(2.0 / (num_hidden2))))
classifier_b = tf.Variable(tf.constant(1.0, shape=[num_labels]))
```

Here is the full code for completness. Nevertheless, the code requires the datafile 'notMNIST.pickle' to run which is not provided here.
```pyhton
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pickle
import os


def set_working_dir():
    tmp_dir = 'tmp'
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    os.chdir(tmp_dir)
    print("Change working directory to", os.getcwd())


set_working_dir()

pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save  # hint to help gc free up memory
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

image_size = 28
num_labels = 10
num_channels = 1 # grayscale


def reformat(dataset, labels):
  dataset = dataset.reshape(
    (-1, image_size, image_size, num_channels)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

batch_size = 16
patch_size = 5
depth = 16
num_hidden1 = 256
num_hidden2 = 64

graph = tf.Graph()

with graph.as_default():
    # Define the training dataset and lables
    tf_train_dataset = tf.placeholder(
        tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))

    # Validation/test dataset
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # CNN layer 1 with filter (num_channels, depth) (3, 16)
    cnn1_W = tf.Variable(tf.truncated_normal(
        [patch_size, patch_size, num_channels, depth], stddev=0.1))
    cnn1_b = tf.Variable(tf.zeros([depth]))

    # CNN layer 2 with filter (depth, depth) (16, 16)
    cnn2_W = tf.Variable(tf.truncated_normal(
        [patch_size, patch_size, depth, depth], stddev=0.1))
    cnn2_b = tf.Variable(tf.constant(1.0, shape=[depth]))

    # Compute the output size of the CNN2 as a 1D array.
    size = image_size // 4 * image_size // 4 * depth

    # FC1 (size, num_hidden1) (size, 256)
    fc1_W = tf.Variable(tf.truncated_normal(
        [size, num_hidden1], stddev=np.sqrt(2.0 / size)))
    fc1_b = tf.Variable(tf.constant(1.0, shape=[num_hidden1]))

    # FC2 (num_hidden1, num_hidden2) (size, 64)
    fc2_W = tf.Variable(tf.truncated_normal(
        [num_hidden1, num_hidden2], stddev=np.sqrt(2.0 / (num_hidden1))))
    fc2_b = tf.Variable(tf.constant(1.0, shape=[num_hidden2]))

    # Classifier (num_hidden2, num_labels) (64, 10)
    classifier_W = tf.Variable(tf.truncated_normal(
        [num_hidden2, num_labels], stddev=np.sqrt(2.0 / (num_hidden2))))
    classifier_b = tf.Variable(tf.constant(1.0, shape=[num_labels]))

    # Model.
    def model(data):
        # First convolution layer with stride = 1 and pad the edge to make the output size the same.
        # Apply ReLU and a maximum 2x2 pool
        conv1 = tf.nn.conv2d(data, cnn1_W, [1, 1, 1, 1], padding='SAME')
        hidden1 = tf.nn.relu(conv1 + cnn1_b)
        pool1 = tf.nn.max_pool(hidden1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

        # Second convolution layer
        conv2 = tf.nn.conv2d(pool1, cnn2_W, [1, 1, 1, 1], padding='SAME')
        hidden2 = tf.nn.relu(conv2 + cnn2_b)
        pool2 = tf.nn.max_pool(hidden2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

        # Flattern the convolution output
        shape = pool2.get_shape().as_list()
        reshape = tf.reshape(pool2, [shape[0], shape[1] * shape[2] * shape[3]])

        # 2 FC hidden layers
        fc1 = tf.nn.relu(tf.matmul(reshape, fc1_W) + fc1_b)
        fc2 = tf.nn.relu(tf.matmul(fc1, fc2_W) + fc2_b)

        # Return the result of the classifier
        return tf.matmul(fc2, classifier_W) + classifier_b

    # Training computation.
    logits = model(tf_train_dataset, True)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

    # Optimizer.
    optimizer = tf.train.AdamOptimizer(0.0005).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
    test_prediction = tf.nn.softmax(model(tf_test_dataset))

num_steps = 20001

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print('Initialized')
  for step in range(num_steps):
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print('Minibatch loss at step %d: %f' % (step, l))
      print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
      print('Validation accuracy: %.1f%%' % accuracy(
        valid_prediction.eval(), valid_labels))
  print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))

  # Test accuracy: 95.2%
```

### Credits
For the TensorFlow coding, we start with the CNN class assignment 4 from the Google deep learning class on Udacity. We implement a CNN design with additional code to complete the assignment.

