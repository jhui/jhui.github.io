---
layout: post
comments: true
mathjax: true
priority: 1231
title: “PyTorch - Autograd”
excerpt: “PyTorch - Autograd”
date: 2018-02-09 14:00:00
---

Autograd is a PyTorch package for the differentiation for all operations on Tensors. This helps us to perform the backpropagation.

### Variables

A **Variable** wraps a Tensor. It has all the API's for a Tensor. _loss.backward_ performs backpropagation over the variable _loss_ that holding the loss function value. The variable $$x$$ also holds the gradient for $$\frac{\partial loss}{\partial x}$$.

Create a 2x2 Variable to store the data immediately:

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

### Functions

We can add an operation to create another variable:

```python
y = x + 2            # Create y from an operation
# Variable containing:
# 3  3
# 3  3
# [torch.FloatTensor of size 2x2]
```

grad_fn references the function that creates it. $$x$$ has no function but any variable created by an operation will have a function. 

```python
print(y.grad_fn)     # The Function that create the Variable y
# <AddBackward0 object at 0x102995438>

print(x.grad_fn)     # None
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

_backward_ processes the backward pass and compute the gradient automatically. The gradient w.r.t. $$x$$ is:

$$
\begin{split}
\frac{\partial out}{\partial x_i} & = \frac{1}{4} \sum_j \frac{\partial z_j}{\partial x_i} \\
& = \frac{1}{4} \sum_j \frac{\partial 2 y_j^2}{\partial x_i} \\
& = \frac{1}{4} \sum_j 4 y_j \frac{\partial y_j }{\partial x_i} \\
& = \sum_j  (x_j + 2) \frac{\partial (x_j + 2) }{\partial x_i} \\
& = (x_i + 2)  \\
& = 3 \quad \text{ for } x_i=1\\
\end{split}
$$

After calling _backward_, we use _grad_ to retrieve the gradients. x.grad returns a 2x2 gradient tensor for $$\frac{\partial out}{\partial x_i}$$.

```python
out.backward()

print(x.grad)
# Variable containing:
# 3 3
# 3 3
# [torch.FloatTensor of size 2x2]
```

#### Dynamic vs static computation graph

The variables and functions build a dynamic graph of computation. The TensorFlow computation graph is static. Operations are delayed until the graph is completed. TensorFlow defines a graph first with placeholders. Once all operations are added, we execute the graph in a session by feeding data into the placeholders. The computation graph is static because it cannot be changed afterwards. We can repeat this process with different batch of data but the graph remains the same.

By design, PyTorch uses a dynamic computation graph. Whenever we create a variable or operations, it is executed immediately. We can add and execute operations anytime before _backward_ is called.
When _backward_ is called, PyTorch performs the backward feed to compute the gradients. At this stage, the graph will be disposed. (the retain_graph flag can override this behavior but rarely suggested.) For the next batch of data, we can use the same code to create a new graph or create a graph differently. Hence, the graph can be changed for different data samples. In NLP, we deal with variable length sentences. Instead of padding the sentence to a fixed length, we create graphs with different number of LSTM cells based on the sentence's length. 

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

_out_ is a scalar and we do not need to specify any parameters for _backward_. By default, we backpropagate a gradient of 1.0 back.

```python
out = z.mean()
out.backward()    # Same as out.backward(torch.FloatTensor([1.0]))

```

But _y_ below is a Tensor of size 3, we need to specify the gradients that we want to backpropagate for each element of _y_. Hence, _gradients_ below has the same size as _y_.

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

