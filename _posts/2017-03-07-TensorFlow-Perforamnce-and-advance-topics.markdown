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

#### Input pipeline optimization

Simplify the model to its most simplest form. If there are no performance gain per iterations, the application bottleneck is in the input pipeline in reading and preprocess the data.

If GPU utilization is below 80%, the application may be input pipeline bounded. 
```sh
watch -n 2 nvidia-smi
```

Verify the CPU utilization as well as the file I/O for bottleneck.

#### Use CPU for data preprocessing

Use CPU for data preprocessing so we can focus training on GPU. Estimator already put data processing on the CPU.

```python
with tf.device('/cpu:0'):
  data = load__images()
```

#### Dataset API

Use feed_dict to feed data for placeholder is not optimal.
```
sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
```

Use TensorFlow Dataset API for feed data to the deep net. Dataset API utilizes C++ multi-threading and has a much lower overhead than Python code.

#### Data Format

NHWC (N, Height, width, channel) is the TensorFlow default and NCHW is the optimal format to use for NVIDIA cuDNN. If TensorFlow is compiled with the Intel MKL optimizations, many operations will be optimized and support NCHW. Otherwise, some operations are not supported on CPU when using NCHW.  On GPU, NCHW is faster. But on CPU, NHWC is sometimes faster.

With GPU supports:
```
bn = tf.contrib.layers.batch_norm(
          input_layer, fused=True, data_format='NCHW'
          scope=scope)
```
		  
#### Fused batch norm

Fused Ops combine operations into a single kernel for improved performance. Batch normalization is expensive and therefore should take advantage of fused=True. (12-30% speedup)

```python
bn = tf.layers.batch_normalization(
    input_layer, fused=True, data_format='NCHW')
```	
 
#### Avoid large amount of small files

To reduce File I/O overhead, we can preprocess many small data files into larger (~100MB) TFRecord files. 

```python
import argparse
import cPickle
import os

import tarfile
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

CIFAR_FILENAME = 'cifar-10-python.tar.gz'
CIFAR_DOWNLOAD_URL = 'https://www.cs.toronto.edu/~kriz/' + CIFAR_FILENAME
CIFAR_LOCAL_FOLDER = 'cifar-10-batches-py'


def download_and_extract(data_dir):
  # download CIFAR-10 if not already downloaded.
  tf.contrib.learn.datasets.base.maybe_download(CIFAR_FILENAME, data_dir,
                                                CIFAR_DOWNLOAD_URL)
  tarfile.open(os.path.join(data_dir, CIFAR_FILENAME),
               'r:gz').extractall(data_dir)


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value)]))


def _get_file_names():
  """Returns the file names expected to exist in the input_dir."""
  file_names = {}
  file_names['train'] = ['data_batch_%d' % i for i in xrange(1, 5)]
  file_names['validation'] = ['data_batch_5']
  file_names['eval'] = ['test_batch']
  return file_names


def read_pickle_from_file(filename):
  with tf.gfile.Open(filename, 'r') as f:
    data_dict = cPickle.load(f)
  return data_dict


def convert_to_tfrecord(input_files, output_file):
  """Converts a file to TFRecords."""
  print('Generating %s' % output_file)
  with tf.python_io.TFRecordWriter(output_file) as record_writer:
    for input_file in input_files:
      data_dict = read_pickle_from_file(input_file)
      data = data_dict['data']
      labels = data_dict['labels']
      num_entries_in_batch = len(labels)
      for i in range(num_entries_in_batch):
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'image': _bytes_feature(data[i].tobytes()),
                'label': _int64_feature(labels[i])
            }))
        record_writer.write(example.SerializeToString())


def main(data_dir):
  print('Download from {} and extract.'.format(CIFAR_DOWNLOAD_URL))
  download_and_extract(data_dir)
  file_names = _get_file_names()
  input_dir = os.path.join(data_dir, CIFAR_LOCAL_FOLDER)
  for mode, files in file_names.items():
    input_files = [os.path.join(input_dir, f) for f in files]
    output_file = os.path.join(data_dir, mode + '.tfrecords')
    try:
      os.remove(output_file)
    except OSError:
      pass
    # Convert to tf.train.Example and write the to TFRecords.
    convert_to_tfrecord(input_files, output_file)
  print('Done!')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data-dir',
      type=str,
      default='',
      help='Directory to download and extract CIFAR-10 to.')

  args = parser.parse_args()
  main(args.data_dir)
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

### GPU Optimization

Data parallelism involves making multiple copies of the model (towers), and place one tower on each of the GPUs. Each tower is trained on a different mini-batch of data and then updates trainable variables that shared between the towers. The variable placement and gradient update are important to scale this solution.

* Tesla K80: If the GPUs are on the same PCI Express and are able to communicate using NVIDIA GPUDirect Peer to Peer, we place the variables equally across the GPUs. Otherwise, we place the variables on the CPU.

* Titan X, P100: For models like ResNet and InceptionV3, placing variables on the CPU. But for models with a lot of variables like AlexNet and VGG, using GPUs with NCCL is better.

The coding is still evolving. We refer reader to the original [tutorial](https://www.tensorflow.org/performance/performance_guide#optimizing_for_gpu) for the latest development.

#### Variable Distribution and Gradient Aggregation

* parameter_server where each replica of the training model reads the variables from a parameter server and updates the variable independently. When each model needs the variables, they are copied over through the standard implicit copies added by the TensorFlow runtime. 

Source TensorFlow
<div class="imgcap">
<img src="/assets/tensorflow/vv1.png" style="border:none;width:100%;">
</div>

* replicated places a copy of each training variable on each GPU. Gradients are accumulated across all GPUs, and the aggregated total is applied to each GPU's copy.

* distributed_replicated places a copy of the training parameters on each GPU along with a master copy on the parameter servers.  Gradients are accumulated across all GPUs on each server and the per-server aggregated gradients are applied to the master copy. After all workers are done, each worker updates its copy from the master.

Source TensorFlow
<div class="imgcap">
<img src="/assets/tensorflow/dvv1.png" style="border:none;width:100%;">
</div>

### CPU optimization

#### Build from source

The installed package are not built or optimized for your local CPU. If the application is not GPU bounded, we may want to build the source from scratch to take advantage of the CPU optimization.

#### Intel MKL-DNN

Set the inter_op_parallelism_threads equal to the number of physical CPUs and 

```
KMP_BLOCKTIME=0
KMP_AFFINITY=granularity=fine,verbose,compact,1,0
```

```
os.environ["KMP_BLOCKTIME"] = str(FLAGS.kmp_blocktime)
os.environ["KMP_SETTINGS"] = str(FLAGS.kmp_settings)
os.environ["KMP_AFFINITY"]= FLAGS.kmp_affinity
if FLAGS.num_intra_threads > 0:
  os.environ["OMP_NUM_THREADS"]= str(FLAGS.num_intra_threads
```  

### Quantization

Convert the GoogleNet into 8-bit precision for inference.
```sh
curl http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz -o /tmp/inceptionv3.tgz
tar xzf /tmp/inceptionv3.tgz -C /tmp/
bazel build tensorflow/tools/graph_transforms:transform_graph
bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
  --in_graph=/tmp/classify_image_graph_def.pb \
  --outputs="softmax" --out_graph=/tmp/quantized_graph.pb \
  --transforms='add_default_attributes strip_unused_nodes(type=float, shape="1,299,299,3")
    remove_nodes(op=Identity, op=CheckNumerics) fold_constants(ignore_errors=true)
    fold_batch_norms fold_old_batch_norms quantize_weights quantize_nodes
    strip_unused_nodes sort_by_execution_order'
```

Running the new quantized model:
```sh
# Note: You need to add the dependencies of the quantization operation to the
#       cc_binary in the BUILD file of the label_image program:
#
#     //tensorflow/contrib/quantization:cc_ops
#     //tensorflow/contrib/quantization/kernels:quantized_ops

bazel build tensorflow/examples/label_image:label_image
bazel-bin/tensorflow/examples/label_image/label_image \
--image=<input-image> \
--graph=/tmp/quantized_graph.pb \
--labels=/tmp/imagenet_synset_to_human_label_map.txt \
--input_width=299 \
--input_height=299 \
--input_mean=128 \
--input_std=128 \
--input_layer="Mul:0" \
--output_layer="softmax:0"
```
 
 
### Operations and tensors

TensorFlow API constructs new tf.Operation (node) and tf.Tensor (edge) objects and add them to a tf.Graph instance. 

* _tf.constant(10.0)_ adds a tf.Operation to the default graph that produces the value 10.0, and returns a tf.Tensor that represents the value of the constant. 
* _tf.matmul(a, b) creates a tf.Operation that multiplies the values of tf.Tensor objects $$a$$ and $$b$$ and returns a tf.Tensor for the multiplication result.
* v = tf.Variable(0) creates a tf.Operation that store a writeable tensor value that persists between tf.Session.run calls. The tf.Variable object wraps this operation, and can be used as a tensor to read the current value.  
* tf.train.Optimizer.minimize will add operations and tensors that calculate gradients and return a tf.Operation that apply those gradient changes to a set of variables.

TensorFlow will create a new tf.Tensor each time when a tensor-like object (numpy.ndarray or list) is passed as parameters. It will run out of memory if the object is used multiple times in constructing nodes. To avoid this, call tf.convert_to_tensor on the tensor-like object once and use the returned tf.Tensor instead.

