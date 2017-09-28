---
layout: post
comments: true
mathjax: true
priority: 805
title: “TensorFlow Estimator”
excerpt: “TensorFlow Estimator”
date: 2017-03-14 14:00:00
---

### Estimator

TensorFlow provides a higher level Estimator API with pre-built model to train and predict data.

It compose of the following steps:

#### Define the feature columns

```python
x_feature = tf.feature_column.numeric_column('x')
...
```

```python
n_room = tf.feature_column.numeric_column('n_rooms')
sqfeet = tf.feature_column.numeric_column('square_feet',
                    normalizer_fn='lambda a: a - global_size')
```

#### Dataset importing functions for training, validation and prediction

```python
# Training
train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x = {"x": np.array([1., 2., 3., 4.])},      # Input features
      y = np.array([1.5, 3.5, 5.5, 7.5]),         # Output
      batch_size=2,
      num_epochs=None,
      shuffle=True)

# Testing
test_input_fn = tf.estimator.inputs.numpy_input_fn(
      x = {"x": np.array([5., 6., 7.])},
      y = np.array([9.5, 11.5, 13.5]),
      num_epochs=1,
      shuffle=False)

# Prediction
samples = np.array([8., 9.])
predict_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": samples},
      num_epochs=1,
      shuffle=False)
```

#### Create a pre-built estimator 

```python
regressor = tf.estimator.LinearRegressor(
    feature_columns=[x_feature],
    model_dir='./output'
)
```

**model_dir** stores the model data including the statistic information into the specific directory which can be viewed from the TensorBoard latter.

```sh
tensorboard --logdir=output
```

<div class="imgcap">
<img src="/assets/tensorflow/esf.png" style="border:none;">
</div>

#### Training, validation and testing

```python
regressor.train(input_fn=train_input_fn, steps=2500)
average_loss = regressor.evaluate(input_fn=test_input_fn)["average_loss"]
predictions = list(regressor.predict(input_fn=predict_input_fn))
```

#### LinearRegressor
TensorFlow comes with many prebuilt models. The following code replaces the last program with a prebuilt Linear Regressor. It constructs a linear regressor as an estimator and we will create an input function to pre-process and feed data into the models. 

Here is the full program:
```python
import tensorflow as tf

import numpy as np

# Create a linear regressorw with 1 feature "x".
x_feature = tf.feature_column.numeric_column('x')

regressor = tf.estimator.LinearRegressor(
    feature_columns=[x_feature],
    model_dir='./output'
)

# Training
train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x = {"x": np.array([1., 2., 3., 4.])},      # Input features
      y = np.array([1.5, 3.5, 5.5, 7.5]),         # Output
      batch_size=2,
      num_epochs=None,
      shuffle=True)

regressor.train(input_fn=train_input_fn, steps=2500)

# Testing
test_input_fn = tf.estimator.inputs.numpy_input_fn(
      x = {"x": np.array([5., 6., 7.])},
      y = np.array([9.5, 11.5, 13.5]),
      num_epochs=1,
      shuffle=False)

average_loss = regressor.evaluate(input_fn=test_input_fn)["average_loss"]
print(f"Average loss: {average_loss:.4f}")

# Prediction
samples = np.array([8., 9.])
predict_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": samples},
      num_epochs=1,
      shuffle=False)

predictions = list(regressor.predict(input_fn=predict_input_fn))
for input, p in zip(samples, predictions):
    v  = p["predictions"][0]
    print(f"{input} -> {v:.4f}")

# Average loss: 0.0002
# 8.0 -> 15.4773
# 9.0 -> 17.4729
```

#### DNNClassifier

DNNClassifier is another pre-built estimator. We build a DNNClassifier with 3 hidden layers to classify the iris samples into 3 subclasses. We load 150 samples and split it into 120 training data and 30 testing data.

The 4 features used as the model input: (image from wiki)
<div class="imgcap">
<img src="/assets/tensorflow/iris.png" style="border:none;width:60%">
</div>

Here is another Estimator for the classification
```python
import os
import urllib

import numpy as np
import tensorflow as tf

# Data sets
IRIS_TRAINING = "iris_training.csv"
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST = "iris_test.csv"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

def main():
  # Download dataset
  if not os.path.exists(IRIS_TRAINING):
    raw = urllib.urlopen(IRIS_TRAINING_URL).read()
    with open(IRIS_TRAINING, "w") as f:
      f.write(raw)

  if not os.path.exists(IRIS_TEST):
    raw = urllib.urlopen(IRIS_TEST_URL).read()
    with open(IRIS_TEST, "w") as f:
      f.write(raw)

  # Load datasets.
  training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=IRIS_TRAINING,
      target_dtype=np.int,
      features_dtype=np.float32)
  test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=IRIS_TEST,
      target_dtype=np.int,
      features_dtype=np.float32)

  # Specify that all features have real-value data
  feature_columns = [tf.feature_column.numeric_column("x", shape=[4])]

  # Build 3 layer DNN with 10, 20, 10 units respectively.
  classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                          hidden_units=[10, 20, 10],
                                          n_classes=3,
                                          model_dir="/tmp/iris_model")
  # Define the training inputs
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": np.array(training_set.data)},
      y=np.array(training_set.target),
      num_epochs=None,
      shuffle=True)

  # Train model.
  classifier.train(input_fn=train_input_fn, steps=2000)

  # Define the test inputs
  test_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": np.array(test_set.data)},
      y=np.array(test_set.target),
      num_epochs=1,
      shuffle=False)

  # Evaluate accuracy.
  accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]

  print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

  # Classify two new flower samples.
  new_samples = np.array(
      [[6.4, 3.2, 4.5, 1.5],
       [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)
  predict_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": new_samples},
      num_epochs=1,
      shuffle=False)

  predictions = list(classifier.predict(input_fn=predict_input_fn))
  predicted_classes = [p["classes"] for p in predictions]

  print("New Samples Predictions:    {}\n".format(predicted_classes))

if __name__ == "__main__":
    main()
```

### Custom model for Estimator

Implement a model_fn to create a custom model used in an estimator.
```python
import numpy as np
import tensorflow as tf

def model_fn(features, labels, mode):
  # Model
  W = tf.get_variable("W", [1], dtype=tf.float64)
  b = tf.get_variable("b", [1], dtype=tf.float64)
  y = W * features['x'] + b

  loss = tf.reduce_sum(tf.square(y - labels))


  global_step = tf.train.get_global_step()
  optimizer = tf.train.GradientDescentOptimizer(0.01)
  train = tf.group(optimizer.minimize(loss),
                   tf.assign_add(global_step, 1))

  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=y,
      loss=loss,
      train_op=train)

estimator = tf.estimator.Estimator(model_fn=model_fn)

x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])

x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])

input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)

eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False)

# Training
estimator.train(input_fn=input_fn, steps=1000)


train_metrics = estimator.evaluate(input_fn=train_input_fn)
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)

print("train metrics: %r"% train_metrics)
print("eval metrics: %r"% eval_metrics)
```

### Estimator using Layer module and Loggin hook

Set the logging level
```python
tf.logging.set_verbosity(tf.logging.INFO)
```

Build a model using the layer module with convolution, ReLU, max pooling, FC layer and dropout.
```python
input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[5, 5], 
              padding="same", activation=tf.nn.relu)
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], 
              padding="same", activation=tf.nn.relu)
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

dropout = tf.layers.dropout(inputs=dense, rate=0.4, 
              training=mode == tf.estimator.ModeKeys.TRAIN)

logits = tf.layers.dense(inputs=dropout, units=10)
```

Computing the cost
```python
def cnn_model_fn(features, labels, mode):
  ...
  
  onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
  loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
```

For training:
```python
def cnn_model_fn(features, labels, mode):
  ...
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

def main(unused_argv):
  mnist = tf.contrib.learn.datasets.load_dataset("mnist")

  train_data = mnist.train.images
  train_labels = np.asarray(mnist.train.labels, dtype=np.int32)

  # Create an estimator 
  mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, 
      model_dir="/tmp/mnist_convnet_model")

  # Create a TensorHook to monitor prob and softmax during training
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": train_data}, 
        y=train_labels, batch_size=100, num_epochs=None, shuffle=True)
  mnist_classifier.train(input_fn=train_input_fn, 
        steps=20000, hooks=[logging_hook])
```

For validation:
```python
def cnn_model_fn(features, labels, mode):
  ...
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
           eval_metric_ops=eval_metric_ops)

def main(unused_argv):
  ...
  eval_data = mnist.test.images
  eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

  mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, 
      model_dir="/tmp/mnist_convnet_model")

  ...
  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": eval_data}, 
        y=eval_labels, num_epochs=1, shuffle=False)
  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)
```  

For making prediction:
```python
# Return an estimator for prediction
predictions = {
   "classes": tf.argmax(input=logits, axis=1),
   "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
}
if mode == tf.estimator.ModeKeys.PREDICT:
  return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
```


Here is the full source code

```python
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

  conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

  dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  logits = tf.layers.dense(inputs=dropout, units=10)

  predictions = {
      "classes": tf.argmax(input=logits, axis=1),
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
  loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
           eval_metric_ops=eval_metric_ops)


def main(unused_argv):
  mnist = tf.contrib.learn.datasets.load_dataset("mnist")

  train_data = mnist.train.images
  train_labels = np.asarray(mnist.train.labels, dtype=np.int32)

  eval_data = mnist.test.images
  eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

  mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, 
      model_dir="/tmp/mnist_convnet_model")

  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": train_data}, 
        y=train_labels, batch_size=100, num_epochs=None, shuffle=True)
  mnist_classifier.train(input_fn=train_input_fn, 
        steps=20000, hooks=[logging_hook])

  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": eval_data}, 
        y=eval_labels, num_epochs=1, shuffle=False)
  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)


if __name__ == "__main__":
  tf.app.run()
```  