---
layout: post
comments: true
mathjax: true
priority: 1250
title: “PyTorch - nn modules common APIs”
excerpt: “PyTorch - nn modules common APIs”
date: 2018-02-09 14:00:00
---

The _nn_ modules in PyTorch provides us a higher level API to build and train deep network.

This summarizes some important APIs for the neural networks. The official documentation is located [here](http://pytorch.org/docs/master/nn.html). This is not a full listing of APIs. It is just a glimpse of what the _torch.nn_ and _torch.nn.functional_ is providing.
 
### Convolution layers

#### nn.Conv1d

```python
m = nn.Conv1d(16, 33, 3, stride=2)
input = Variable(torch.randn(20, 16, 50))
output = m(input)
```

#### nn.Conv2d

```python
m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
input = Variable(torch.randn(20, 16, 50, 100))
output = m(input)
```

#### nn.Conv3d

```python
m = nn.Conv3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0))
input = Variable(torch.randn(20, 16, 10, 50, 100))
output = m(input)
```

### Pooling layers

#### torch.nn.MaxPool2d

```python
m = nn.MaxPool2d((3, 2), stride=(2, 1))
input = Variable(torch.randn(20, 16, 50, 32))
output = m(input)
```

#### torch.nn.MaxUnpool2d

```python
pool = nn.MaxPool2d(2, stride=2, return_indices=True)
unpool = nn.MaxUnpool2d(2, stride=2)
input = Variable(torch.Tensor([[[[ 1,  2,  3,  4],
                                 [ 5,  6,  7,  8],
                                 [ 9, 10, 11, 12],
                                 [13, 14, 15, 16]]]]))
output, indices = pool(input)
unpool(output, indices)
# Variable containing:
# (0 ,0 ,.,.) =
#   0   0   0   0
#   0   6   0   8
#   0   0   0   0
#   0  14   0  16
# [torch.FloatTensor of size 1x1x4x4]
```

#### nn.AvgPool2d

```python
m = nn.AvgPool2d((3, 2), stride=(2, 1))
input = Variable(torch.randn(20, 16, 50, 32))
output = m(input)
```

### Padding layers

#### nn.ReplicationPad2d

```python
m = nn.ReplicationPad2d(3)
input = Variable(torch.randn(16, 3, 320, 480))
output = m(input)
```

#### nn.ZeroPad2d

```python
m = nn.ZeroPad2d(3)
input = Variable(torch.randn(16, 3, 320, 480))
output = m(input)
```

### Non-linear activation

#### ReLU

```python
m = nn.ReLU()
input = Variable(torch.randn(2))
print(input)
print(m(input))
```

#### Leaky ReLU

```python
m = nn.LeakyReLU(0.1)
input = Variable(torch.randn(2))
print(input)
print(m(input))
```

#### Sigmoid

```python
m = nn.Sigmoid()
input = Variable(torch.randn(2))
print(input)
print(m(input))
```

#### Softplus

```python
m = nn.Softplus()
input = Variable(torch.randn(2))
print(input)
print(m(input))
```

#### Softmax

```python
m = nn.Softmax()
input = Variable(torch.randn(2, 3))
print(input)
print(m(input))
```

### Normalization layers

#### nn.BatchNorm1d

```python
m = nn.BatchNorm1d(100)
# Without Learnable Parameters
m = nn.BatchNorm1d(100, affine=False)
input = autograd.Variable(torch.randn(20, 100))
output = m(input)
```

#### nn.BatchNorm2d

```python
# With Learnable Parameters
m = nn.BatchNorm2d(100)
# Without Learnable Parameters
m = nn.BatchNorm2d(100, affine=False)
input = autograd.Variable(torch.randn(20, 100, 35, 45))
output = m(input)
```

### Recurrent layers

#### nn.RNN

```python
rnn = nn.RNN(10, 20, 2)
input = Variable(torch.randn(5, 3, 10))
h0 = Variable(torch.randn(2, 3, 20))
output, hn = rnn(input, h0)
```

#### nn.LSTM

```python
rnn = nn.LSTM(10, 20, 2)
input = Variable(torch.randn(5, 3, 10))
h0 = Variable(torch.randn(2, 3, 20))
c0 = Variable(torch.randn(2, 3, 20))
output, hn = rnn(input, (h0, c0))
```

#### nn.GRU

```python
rnn = nn.GRU(10, 20, 2)
input = Variable(torch.randn(5, 3, 10))
h0 = Variable(torch.randn(2, 3, 20))
output, hn = rnn(input, h0)
```

#### nn.RNNCell

```python
rnn = nn.RNNCell(10, 20)
input = Variable(torch.randn(6, 3, 10))
hx = Variable(torch.randn(3, 20))
output = []
for i in range(6):
    hx = rnn(input[i], hx)
    output.append(hx)
```

#### nn.LSTMCell

```python
rnn = nn.LSTMCell(10, 20)
input = Variable(torch.randn(6, 3, 10))
hx = Variable(torch.randn(3, 20))
cx = Variable(torch.randn(3, 20))
output = []
for i in range(6):
    hx, cx = rnn(input[i], (hx, cx))
    output.append(hx)
```

#### nn.GRUCell

```python
rnn = nn.GRUCell(10, 20)
input = Variable(torch.randn(6, 3, 10))
hx = Variable(torch.randn(3, 20))
output = []
for i in range(6):
    hx = rnn(input[i], hx)
    output.append(hx)
```	

### Linear Layer

#### nn.Linear

```python
m = nn.Linear(20, 30)
input = Variable(torch.randn(128, 20))
output = m(input)
print(output.size())
```

### Dropout

#### nn.Dropout

```python
m = nn.Dropout(p=0.2)
input = autograd.Variable(torch.randn(20, 16))
output = m(input)
```

### Sparse layers

#### nn.Embedding

```python
# an Embedding module containing 10 tensors of size 3
embedding = nn.Embedding(10, 3)
# a batch of 2 samples of 4 indices each
input = Variable(torch.LongTensor([[1,2,4,5],[4,3,2,9]]))
embedding(input)

# Variable containing:
# (0 ,.,.) =
# -1.0822  1.2522  0.2434
#  0.8393 -0.6062 -0.3348
#  0.6597  0.0350  0.0837
#  0.5521  0.9447  0.0498
#
# (1 ,.,.) =
#  0.6597  0.0350  0.0837
# -0.1527  0.0877  0.4260
#  0.8393 -0.6062 -0.3348
# -0.8738 -0.9054  0.4281
# [torch.FloatTensor of size 2x4x3]
```

### Distance function

#### nn.CosineSimilarity
```python
input1 = Variable(torch.randn(100, 128))
input2 = Variable(torch.randn(100, 128))
cos = nn.CosineSimilarity(dim=1, eps=1e-6)
output = cos(input1, input2)
print(output)
```

### Loss function

#### nn.L1Loss

```python
loss = nn.L1Loss()
input = autograd.Variable(torch.randn(3, 5), requires_grad=True)
target = autograd.Variable(torch.randn(3, 5))
output = loss(input, target)
output.backward()
```

#### nn.MSELoss

```python
loss = nn.MSELoss()
input = Variable(torch.randn(3, 5), requires_grad=True)
target = Variable(torch.randn(3, 5))
output = loss(input, target)
output.backward()
```

#### nn.CrossEntropyLoss

```python
loss = nn.CrossEntropyLoss()
input = autograd.Variable(torch.randn(3, 5), requires_grad=True)
target = autograd.Variable(torch.LongTensor(3).random_(5))
output = loss(input, target)
output.backward()
```

#### nn.NLLLoss

```python
m = nn.LogSoftmax()
loss = nn.NLLLoss()
# input is of size N x C = 3 x 5
input = Variable(torch.randn(3, 5), requires_grad=True)
# each element in target has to have 0 <= value < C
target = autograd.Variable(torch.LongTensor([1, 0, 4]))
output = loss(m(input), target)
output.backward()
```

#### nn.KLDivLoss

```python
nn.KLDivLoss(size_average=False)
```

### Vision layers

#### nn.PixelShuffle

```python
ps = nn.PixelShuffle(3)
input = autograd.Variable(torch.Tensor(1, 9, 4, 4))
output = ps(input)
print(output.size())
```

#### nn.Upsample

```python
m = nn.Upsample(scale_factor=2, mode='bilinear')
```

#### nn.UpsamplingBilinear2d

```python
m = nn.UpsamplingBilinear2d(scale_factor=2)
```

### Utilities

#### Clip gradient

```python
torch.nn.utils.clip_grad_norm(...)
```

### torch.nn.init

#### nn.init.uniform

```python
w = torch.Tensor(3, 5)
nn.init.uniform(w)
```

#### nn.init.normal

```python
w = torch.Tensor(3, 5)
nn.init.normal(w)
```

#### nn.init.constant

```python
w = torch.Tensor(3, 5)
nn.init.constant(w, 0.3)
```

#### nn.init.xavier_uniform

```python
w = torch.Tensor(3, 5)
nn.init.xavier_uniform(w, gain=nn.init.calculate_gain('relu'))
```

#### nn.init.xavier_normal

```python
w = torch.Tensor(3, 5)
nn.init.xavier_normal(w)
```

#### nn.init.kaiming_normal

```python
w = torch.Tensor(3, 5)
nn.init.kaiming_normal(w, mode='fan_out')
```

### Summary for torch.nn.init
```
torch.nn.init
=============

.. currentmodule:: torch.nn.init
.. autofunction:: calculate_gain
.. autofunction:: uniform
.. autofunction:: normal
.. autofunction:: constant
.. autofunction:: eye
.. autofunction:: dirac
.. autofunction:: xavier_uniform
.. autofunction:: xavier_normal
.. autofunction:: kaiming_uniform
.. autofunction:: kaiming_normal
.. autofunction:: orthogonal
```

### Summary of torch.nn

```
torch.nn
===================================
Containers
----------------------------------
:hidden:`Module`
~~~~~~~~~~~~~~~~
:hidden:`Sequential`
~~~~~~~~~~~~~~~~~~~~
:hidden:`ModuleList`
~~~~~~~~~~~~~~~~~~~~
:hidden:`ParameterList`
~~~~~~~~~~~~~~~~~~~~~~~

Convolution Layers
----------------------------------
:hidden:`Conv1d`
~~~~~~~~~~~~~~~~
:hidden:`Conv2d`
~~~~~~~~~~~~~~~~
:hidden:`Conv3d`
~~~~~~~~~~~~~~~~
:hidden:`ConvTranspose1d`
~~~~~~~~~~~~~~~~~~~~~~~~~
:hidden:`ConvTranspose2d`
~~~~~~~~~~~~~~~~~~~~~~~~~
:hidden:`ConvTranspose3d`
~~~~~~~~~~~~~~~~~~~~~~~~~

Pooling Layers
----------------------------------
:hidden:`MaxPool1d`
~~~~~~~~~~~~~~~~~~~
:hidden:`MaxPool2d`
~~~~~~~~~~~~~~~~~~~
:hidden:`MaxPool3d`
~~~~~~~~~~~~~~~~~~~
:hidden:`MaxUnpool1d`
~~~~~~~~~~~~~~~~~~~~~
:hidden:`MaxUnpool2d`
~~~~~~~~~~~~~~~~~~~~~
:hidden:`MaxUnpool3d`
~~~~~~~~~~~~~~~~~~~~~
:hidden:`AvgPool1d`
~~~~~~~~~~~~~~~~~~~
:hidden:`AvgPool2d`
~~~~~~~~~~~~~~~~~~~
:hidden:`AvgPool3d`
~~~~~~~~~~~~~~~~~~~
:hidden:`FractionalMaxPool2d`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:hidden:`LPPool2d`
~~~~~~~~~~~~~~~~~~
:hidden:`AdaptiveMaxPool1d`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:hidden:`AdaptiveMaxPool2d`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:hidden:`AdaptiveMaxPool3d`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:hidden:`AdaptiveAvgPool1d`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:hidden:`AdaptiveAvgPool2d`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:hidden:`AdaptiveAvgPool3d`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Padding Layers
--------------

:hidden:`ReflectionPad2d`
~~~~~~~~~~~~~~~~~~~~~~~~~
:hidden:`ReplicationPad2d`
~~~~~~~~~~~~~~~~~~~~~~~~~~
:hidden:`ReplicationPad3d`
~~~~~~~~~~~~~~~~~~~~~~~~~~
:hidden:`ZeroPad2d`
~~~~~~~~~~~~~~~~~~~
:hidden:`ConstantPad2d`
~~~~~~~~~~~~~~~~~~~~~~~


Non-linear Activations
----------------------------------
:hidden:`ReLU`
~~~~~~~~~~~~~~
:hidden:`ReLU6`
~~~~~~~~~~~~~~~
:hidden:`ELU`
~~~~~~~~~~~~~
:hidden:`SELU`
~~~~~~~~~~~~~~
:hidden:`PReLU`
~~~~~~~~~~~~~~~
:hidden:`LeakyReLU`
~~~~~~~~~~~~~~~~~~~
:hidden:`Threshold`
~~~~~~~~~~~~~~~~~~~
:hidden:`Hardtanh`
~~~~~~~~~~~~~~~~~~
:hidden:`Sigmoid`
~~~~~~~~~~~~~~~~~
:hidden:`Tanh`
~~~~~~~~~~~~~~
:hidden:`LogSigmoid`
~~~~~~~~~~~~~~~~~~~~
:hidden:`Softplus`
~~~~~~~~~~~~~~~~~~
:hidden:`Softshrink`
~~~~~~~~~~~~~~~~~~~~
:hidden:`Softsign`
~~~~~~~~~~~~~~~~~~
:hidden:`Tanhshrink`
~~~~~~~~~~~~~~~~~~~~
:hidden:`Softmin`
~~~~~~~~~~~~~~~~~
:hidden:`Softmax`
~~~~~~~~~~~~~~~~~
:hidden:`Softmax2d`
~~~~~~~~~~~~~~~~~~~
:hidden:`LogSoftmax`
~~~~~~~~~~~~~~~~~~~~



Normalization layers
----------------------------------
:hidden:`BatchNorm1d`
~~~~~~~~~~~~~~~~~~~~~
:hidden:`BatchNorm2d`
~~~~~~~~~~~~~~~~~~~~~
:hidden:`BatchNorm3d`
~~~~~~~~~~~~~~~~~~~~~
:hidden:`InstanceNorm1d`
~~~~~~~~~~~~~~~~~~~~~~~~
:hidden:`InstanceNorm2d`
~~~~~~~~~~~~~~~~~~~~~~~~
:hidden:`InstanceNorm3d`
~~~~~~~~~~~~~~~~~~~~~~~~
:hidden:`LocalResponseNorm`
~~~~~~~~~~~~~~~~~~~~~~~~

Recurrent layers
----------------------------------
:hidden:`RNN`
~~~~~~~~~~~~~
:hidden:`LSTM`
~~~~~~~~~~~~~~
:hidden:`GRU`
~~~~~~~~~~~~~
:hidden:`RNNCell`
~~~~~~~~~~~~~~~~~
:hidden:`LSTMCell`
~~~~~~~~~~~~~~~~~~
:hidden:`GRUCell`
~~~~~~~~~~~~~~~~~


Linear layers
----------------------------------
:hidden:`Linear`
~~~~~~~~~~~~~~~~
:hidden:`Bilinear`
~~~~~~~~~~~~~~~~~~

Dropout layers
----------------------------------
:hidden:`Dropout`
~~~~~~~~~~~~~~~~~
:hidden:`Dropout2d`
~~~~~~~~~~~~~~~~~~~
:hidden:`Dropout3d`
~~~~~~~~~~~~~~~~~~~
:hidden:`AlphaDropout`
~~~~~~~~~~~~~~~~~~~~~~


Sparse layers
----------------------------------
:hidden:`Embedding`
~~~~~~~~~~~~~~~~~~~
:hidden:`EmbeddingBag`
~~~~~~~~~~~~~~~~~~~~~~


Distance functions
----------------------------------
:hidden:`CosineSimilarity`
~~~~~~~~~~~~~~~~~~~~~~~~~~
:hidden:`PairwiseDistance`
~~~~~~~~~~~~~~~~~~~~~~~~~~



Loss functions
----------------------------------
:hidden:`L1Loss`
~~~~~~~~~~~~~~~~
:hidden:`MSELoss`
~~~~~~~~~~~~~~~~~
:hidden:`CrossEntropyLoss`
~~~~~~~~~~~~~~~~~~~~~~~~~~
:hidden:`NLLLoss`
~~~~~~~~~~~~~~~~~
:hidden:`PoissonNLLLoss`
~~~~~~~~~~~~~~~~~~~~~~~~
:hidden:`KLDivLoss`
~~~~~~~~~~~~~~~~~~~
:hidden:`BCELoss`
~~~~~~~~~~~~~~~~~~~
:hidden:`BCEWithLogitsLoss`
~~~~~~~~~~~~~~~~~~~~~~~~~~~
:hidden:`MarginRankingLoss`
~~~~~~~~~~~~~~~~~~~~~~~~~~~
:hidden:`HingeEmbeddingLoss`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:hidden:`MultiLabelMarginLoss`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:hidden:`SmoothL1Loss`
~~~~~~~~~~~~~~~~~~~~~~
:hidden:`SoftMarginLoss`
~~~~~~~~~~~~~~~~~~~~~~~~
:hidden:`MultiLabelSoftMarginLoss`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:hidden:`CosineEmbeddingLoss`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:hidden:`MultiMarginLoss`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:hidden:`TripletMarginLoss`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



Vision layers
----------------

:hidden:`PixelShuffle`
~~~~~~~~~~~~~~~~~~~~~~
:hidden:`Upsample`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:hidden:`UpsamplingNearest2d`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:hidden:`UpsamplingBilinear2d`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



DataParallel layers (multi-GPU, distributed)
--------------------------------------------
:hidden:`DataParallel`
~~~~~~~~~~~~~~~~~~~~~~
:hidden:`DistributedDataParallel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Utilities
---------
:hidden:`clip_grad_norm`
~~~~~~~~~~~~~~~~~~~~~~~~
:hidden:`weight_norm`
~~~~~~~~~~~~~~~~~~~~~
:hidden:`remove_weight_norm`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:hidden:`PackedSequence`
~~~~~~~~~~~~~~~~~~~~~~~~
:hidden:`pack_padded_sequence`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:hidden:`pad_packed_sequence`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:hidden:`pad_sequence`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:hidden:`pack_sequence`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```

### Summary for torch.nn.functional

```
torch.nn.functional
===================

Convolution functions
----------------------------------
:hidden:`conv1d`
~~~~~~~~~~~~~~~~
:hidden:`conv2d`
~~~~~~~~~~~~~~~~
:hidden:`conv3d`
~~~~~~~~~~~~~~~~
:hidden:`conv_transpose1d`
~~~~~~~~~~~~~~~~~~~~~~~~~~
:hidden:`conv_transpose2d`
~~~~~~~~~~~~~~~~~~~~~~~~~~
:hidden:`conv_transpose3d`
~~~~~~~~~~~~~~~~~~~~~~~~~~


Pooling functions
----------------------------------
:hidden:`avg_pool1d`
:hidden:`avg_pool2d`
:hidden:`avg_pool3d`
:hidden:`max_pool1d`
:hidden:`max_pool2d`
:hidden:`max_pool3d`
:hidden:`max_unpool1d`
:hidden:`max_unpool2d`
:hidden:`max_unpool3d`
:hidden:`lp_pool2d`
:hidden:`adaptive_max_pool1d`
:hidden:`adaptive_max_pool2d`
:hidden:`adaptive_max_pool3d`
:hidden:`adaptive_avg_pool1d`
:hidden:`adaptive_avg_pool2d`
:hidden:`adaptive_avg_pool3d`


Non-linear activation functions
-------------------------------
:hidden:`threshold`
~~~~~~~~~~~~~~~~~~~
:hidden:`relu`
~~~~~~~~~~~~~~
:hidden:`hardtanh`
~~~~~~~~~~~~~~~~~~
:hidden:`relu6`
~~~~~~~~~~~~~~~
:hidden:`elu`
~~~~~~~~~~~~~
:hidden:`selu`
~~~~~~~~~~~~~~
:hidden:`leaky_relu`
~~~~~~~~~~~~~~~~~~~~
:hidden:`prelu`
~~~~~~~~~~~~~~~
:hidden:`rrelu`
~~~~~~~~~~~~~~~
:hidden:`glu`
~~~~~~~~~~~~~~~
:hidden:`logsigmoid`
~~~~~~~~~~~~~~~~~~~~
:hidden:`hardshrink`
~~~~~~~~~~~~~~~~~~~~
:hidden:`tanhshrink`
~~~~~~~~~~~~~~~~~~~~
:hidden:`softsign`
~~~~~~~~~~~~~~~~~~
:hidden:`softplus`
~~~~~~~~~~~~~~~~~~
:hidden:`softmin`
~~~~~~~~~~~~~~~~~
:hidden:`softmax`
~~~~~~~~~~~~~~~~~
:hidden:`softshrink`
~~~~~~~~~~~~~~~~~~~~
:hidden:`log_softmax`
~~~~~~~~~~~~~~~~~~~~~
:hidden:`tanh`
~~~~~~~~~~~~~~
:hidden:`sigmoid`
~~~~~~~~~~~~~~~~~


Normalization functions
-----------------------
:hidden:`batch_norm`
~~~~~~~~~~~~~~~~~~~~
:hidden:`local_response_norm`
~~~~~~~~~~~~~~~~~~~~
:hidden:`normalize`
~~~~~~~~~~~~~~~~~~~~


Linear functions
----------------
:hidden:`linear`
~~~~~~~~~~~~~~~~



Dropout functions
-----------------
:hidden:`dropout`
~~~~~~~~~~~~~~~~~
:hidden:`alpha_dropout`
~~~~~~~~~~~~~~~~~~~~~~~
:hidden:`dropout2d`
~~~~~~~~~~~~~~~~~~~
:hidden:`dropout3d`
~~~~~~~~~~~~~~~~~~~


Distance functions
----------------------------------
:hidden:`pairwise_distance`
~~~~~~~~~~~~~~~~~~~~~~~~~~~
:hidden:`cosine_similarity`
~~~~~~~~~~~~~~~~~~~~~~~~~~~


Loss functions
--------------
:hidden:`binary_cross_entropy`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:hidden:`poisson_nll_loss`
~~~~~~~~~~~~~~~~~~~~~~~~~~
:hidden:`cosine_embedding_loss`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:hidden:`cross_entropy`
~~~~~~~~~~~~~~~~~~~~~~~
:hidden:`hinge_embedding_loss`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:hidden:`kl_div`
~~~~~~~~~~~~~~~~
:hidden:`l1_loss`
~~~~~~~~~~~~~~~~~
:hidden:`mse_loss`
~~~~~~~~~~~~~~~~~~
:hidden:`margin_ranking_loss`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:hidden:`multilabel_margin_loss`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:hidden:`multilabel_soft_margin_loss`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:hidden:`multi_margin_loss`
~~~~~~~~~~~~~~~~~~~~~~~~~~~
:hidden:`nll_loss`
~~~~~~~~~~~~~~~~~~
:hidden:`binary_cross_entropy_with_logits`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:hidden:`smooth_l1_loss`
~~~~~~~~~~~~~~~~~~~~~~~~
:hidden:`soft_margin_loss`
~~~~~~~~~~~~~~~~~~~~~~~~~~
:hidden:`triplet_margin_loss`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Vision functions
----------------
:hidden:`pixel_shuffle`
~~~~~~~~~~~~~~~~~~~~~~~
:hidden:`pad`
~~~~~~~~~~~~~~~~~~~~~~~
:hidden:`upsample`
~~~~~~~~~~~~~~~~~~~~~~~
:hidden:`upsample_nearest`
~~~~~~~~~~~~~~~~~~~~~~~~~~
:hidden:`upsample_bilinear`
~~~~~~~~~~~~~~~~~~~~~~~~~~~
:hidden:`grid_sample`
~~~~~~~~~~~~~~~~~~~~~~~~~~~
:hidden:`affine_grid`
~~~~~~~~~~~~~~~~~~~~~~~~~~~
```