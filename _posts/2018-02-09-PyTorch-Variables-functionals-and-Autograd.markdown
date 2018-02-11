---
layout: post
comments: true
mathjax: true
priority: 1231
title: “PyTorch - Variables, functionals and Autograd.”
excerpt: “PyTorch - Variables, functionals and Autograd.”
date: 2018-02-09 14:00:00
---

### Variables

A **Variable** wraps a Tensor. It supports nearly all the API's defined by a Tensor. Variable also provides a _backward_ method to perform backpropagation. For example, to backpropagate a loss function to train model parameter $$x$$, we use a variable $$loss$$ to store the value computed by a loss function. Then, we call _loss.backward_ which computes the gradients $$\frac{\partial loss}{\partial x}$$ for all trainable parameters. PyTorch will store the gradient results back in the corresponding variable $$x$$.

Create a 2x2 Variable to store input data: 

```python
import torch
from torch.autograd import Variable

# Variables wrap a Tensor
x = Variable(torch.ones(2, 2), requires_grad=True)
# Variable containing:
# 1  1
# 1  1
# [torch.FloatTensor of size 2x2]
```

PyTorch executes the operations immediately. In TensorFlow, the execution is delayed until we execute it in a session later.

### Functions

We can add an operation to create another variable:

```python
y = x + 2            # Create y from an operation
# Variable containing:
# 3  3
# 3  3
# [torch.FloatTensor of size 2x2]

z = torch.add(x, y)  # Same as z = x + y
```

We can add more operations:

```
z = y * y * 2

out = z.mean()
# Variable containing:
# 2
# [torch.FloatTensor of size 1]
```

### Compute gradient

Autograd is a PyTorch package for the differentiation for all operations on Tensors. It performs the backpropagation starting from a variable. In deep learning, this variable often holds the value of the cost function. _backward_ executes the backward pass and computes all the backpropagation gradients automatically. We access indvidual gradient through the attributes _grad_ of a variable.  _x.grad_ below returns a 2x2 gradient tensor for $$\frac{\partial out}{\partial x}$$.

```python
out.backward()

print(x.grad)
# Variable containing:
# 3 3
# 3 3
# [torch.FloatTensor of size 2x2]
```

To check the resule, we compute the gradient manually:

$$
\begin{split}
\frac{\partial out}{\partial x_i} & = \frac{1}{4} \sum_j \frac{\partial z_j}{\partial x_i} \\
& = \frac{1}{4} \sum_j \frac{\partial 2 y_j^2}{\partial x_i} \\
& = \frac{1}{4} \sum_j 4 y_j \frac{\partial y_j }{\partial x_i} \\
& = \sum_j  (x_j + 2) \frac{\partial (x_j + 2) }{\partial x_i} \\
& = x_i + 2 \quad \quad  & \frac{\partial x_j }{\partial x_i} = 0 \text{ if } i \neq j \\
& = 3 \quad \quad  & \text{ for } x_i=1\\
\end{split}
$$

#### Dynamic computation graph

In PyTorch, the variables and functions build a dynamic graph of computation. For every variable operation, it creates at least a single Function node that connects to functions that created a Variable. The attribute _grad\_fn_ of a variable references the function that creates the variable. $$x$$ has no function but any variable created by an operation will have a function. 

```python
x = Variable(torch.ones(2, 2), requires_grad=True)
y = x + 1

print(x.grad_fn)     # None

print(y.grad_fn)     # The Function that create the Variable y
# <AddBackward0 object at 0x102995438>
```

When _backward_ is called, it follows backwards with the links created in the graph to backpropagate the gradient. 

#### Dynamic vs Static computation graph (PyTorch vs TensorFlow)

The TensorFlow computation graph is static. Operation executions are delayed until the graph is completed. TensorFlow defines a graph first with placeholders. Once all operations are added, we execute the graph in a session by feeding data into the placeholders. The computation graph is static because it cannot be changed afterwards. We can repeat this process with different batch of data but the graph remains the same.

By design, PyTorch uses a dynamic computation graph. Whenever we create a variable or operations, it is executed immediately. We can add and execute operations anytime before _backward_ is called. _backwards_ follows the graph backward to compute the gradients. Then the graph will be disposed. (the retain_graph flag can override this behavior but rarely suggested.) For the training data in the next iteration, a new graph is always used. We can use the same code to create the same structure, or create a graph with different operations. In NLP, we deal with variable length sentences. Instead of padding the sentence to a fixed length, we create graphs with different number of LSTM cells based on the sentence's length. 

We call this a define-by-run framework. which the backpropagation is based on what has be runing in the graph. Since we start a new graph for every iteration, the backpropagation can be different for each iteration.

#### Access data

We can access the raw data of a variable with _data_.

```python
x = torch.randn(3)
x = Variable(x, requires_grad=True)

y = x * 2
while y.data.norm() < 100:
    y = y * 2

print(y)
# Variable containing:
#  48.8215
# 162.7583
# -69.1980
# [torch.FloatTensor of size 3]
```

### Backward (non-scalar output)

_out_ below is a scalar and we do not need to specify any parameters for _backward_. By default, we backpropagate a gradient of 1.0 back.

```python
out = z.mean()
out.backward()    # Same as out.backward(torch.FloatTensor([1.0]))
```

_y_ below is a Tensor of size 3. _backward_ requires a Tensor to specify each backpropagation gradient if the variable is not a scalar. To match each element of _y_, _gradients_ needs to match the size of _y_. In some situtation, the gradient values are computed from the model predictions and the true labels.

```python
gradients = torch.FloatTensor([0.1, 1.0, 0.0001])
y.backward(gradients)

print(x.grad)
# Variable containing:
#  6.4000               - backpropagate gradient of 0.1
# 64.0000               - backpropagate gradient of 1.0
#  0.0064
# [torch.FloatTensor of size 3]
```

