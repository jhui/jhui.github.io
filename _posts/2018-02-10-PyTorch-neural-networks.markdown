---
layout: post
comments: true
mathjax: true
priority: 1232
title: “PyTorch - Neural networks”
excerpt: “PyTorch - Neural networks”
date: 2018-02-09 14:00:00
---


### Neural Networks

In PyTorch, we use torch.nn to build layers. In _\_\_iniit\_\__, we configure different trainable layers including convolution and affine layers with nn.Conv2d and nn.Linear respectively. We create the method _forward_ to compute the network output. It contains functionals to link layers already configured in _\_\_iniit\_\__ to form a computation graph. Functionals include ReLU and max poolings. The difference between torch.nn and torch.nn.functional is very subtle which many other deep network frameworks do not make such a distinction. In PyTorch, torch.nn.functional tends to have no trainable parameters.

 To create a deep network:

```python
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))

        # 2 is ame as (2, 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:       # Get the products
            num_features *= s
        return num_features


net = Net()
print(net)
# Net(
#  (conv1): Conv2d (1, 6, kernel_size=(5, 5), stride=(1, 1))
#  (conv2): Conv2d (6, 16, kernel_size=(5, 5), stride=(1, 1))
#  (fc1): Linear(in_features=400, out_features=120)
#  (fc2): Linear(in_features=120, out_features=84)
#  (fc3): Linear(in_features=84, out_features=10)
#)
```

The learnable parameters of a model are returned by _net.parameters_. For example, _params_\[0\] returns the trainable parameters for conv1 which has the size of 6x1x5x5.

```python
params = list(net.parameters())
print(len(params))       # 10: 10 sets of trainable parameters

print(params[0].size())  # torch.Size([6, 1, 5, 5])

```

We compute the network output by:

```python
input = Variable(torch.randn(1, 1, 32, 32))
out = net(input)   # out's size: 1x10.
# Variable containing:
# 0.1268  0.0207  0.0857  0.1454 -0.0370  0.0030  0.0150 -0.0542  0.0512 -0.0550
# [torch.FloatTensor of size 1x10]
```

input here has a dimension of (batch size, # of channel, width, height). torch.nn processes batch data only. To support a single datapoint, use _input.unsqueeze(0)_ to convert a single datapoint to a batch with only one sample.

### Backward pass

To compute the backward pass for gradient, we first zero the gradient stored in the network. In some network design, we need to call _backward_ multiple times. In PyTorch, every time we backpropagate the gradient from a variable, the gradient is accumulative instead of being reset and replaced. For example in a generative adversary network GAN, we need an accumulated gradients from multiple backward passes: one for the generative part and one for the adversary part of the network. We need to reset the gradients only once but not between _backward_ calls. Hence, to accommodate such flexibility, we explicitly reset the gradient instead of _backward_ resets it every time.

```
net.zero_grad()
out.backward()
```

### Loss function

Create a mean square error loss function and backpropagate gradient based on the loss.

```python
output = net(input)
target = Variable(torch.arange(1, 11))   # Create a dummy true label Size 10.
criterion = nn.MSELoss()

# Compute the loss by MSE of the output and the true label
loss = criterion(output, target)         # Size 1

net.zero_grad()      # zeroes the gradient buffers of all parameters

loss.backward()

# Print the gradient for the bias parameters of the first convolution layer
print(net.conv1.bias.grad)

# Variable containing:
# -0.0007
# -0.0400
# 0.0184
# 0.1273
# -0.0080
# 0.0387
# [torch.FloatTensor of size 6]
```

If we follow loss in the backward direction using _grad\_fn_ attribute, we can see a graph of computations similar to:

```
nput -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
      -> view -> linear -> relu -> linear -> relu -> linear
      -> MSELoss
      -> loss
```

Here are some simple print out of the function chain.
```
loss = criterion(output, target)         # Size 1

print(loss.grad_fn)                      # <MseLossBackward object at 0x10d729908>
print(loss.grad_fn.next_functions[0][0]) # <AddmmBackward object at 0x10d729400>
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # <ExpandBackward object at 0x10fd39e48>
```

#### Update trainable parameters

To train the parameters, we create an optimizer and call _step_ to upgrade the parameters.

```python
import torch.optim as optim

# Create a SGD optimizer for gradient descent
optimizer = optim.SGD(net.parameters(), lr=0.01)

# Inside the training loop
# ...
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()        # Perform the training parameters update
```

We need to zero the gradient buffer once for every training iteration to reset the gradient computed by last data batch.
 
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

#### Summary
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