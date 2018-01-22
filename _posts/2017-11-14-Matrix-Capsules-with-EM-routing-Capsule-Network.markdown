---
layout: post
comments: true
mathjax: true
priority: -1000
title: “Understanding Matrix capsules with EM Routing (Based on Hinton's Capsule Networks)”
excerpt: “A simple tutorial in understanding Matrix capsules with EM Routing in Capsule Networks”
date: 2017-11-14 11:00:00
---

This article covers the second capsule network paper [_Matrix capsules with EM Routing_](https://openreview.net/pdf?id=HJWLfGWRb) based on Geoffrey Hinton's Capsule Networks. We will cover the basic of matrix capsules and apply EM (Expectation Maximization) routing to classify images with different viewpoints.  At the end, we will demonstrate the Matrix Capsule and EM routing implementation using TensorFlow.

### CNN challenges

In our [previous capsule article](https://jhui.github.io/2017/11/03/Dynamic-Routing-Between-Capsules/), we cover the challenges of CNN in exploring spatial relationship and discuss how capsule networks may address those short-comings. Let's recap some important challenges of CNN in classifying the same class of images but in different viewpoints. For example, classify a face correctly even with different orientations.

<div class="imgcap">
<img src="/assets/capsule/sface3.jpg" style="border:none;width:50%;">
</div>

Conceptually, the CNN trains neurons to handle different feature orientations (0°, 20°, -20° ) with a top level face detection neuron.

<div class="imgcap">
<img src="/assets/capsule/cnn1.jpg" style="border:none;width;">
</div>

To solve the problem, we add more convolution layers and features maps. Nevertheless this approach tends to memorize the dataset rather than generalize a solution. It requires a large volume of training data to cover different variants and to avoid overfitting. MNist dataset contains 55,000 training data. i.e. 5,500 samples per digits. However, it is unlikely that children need so many samples to learn numbers. Our existing deep learning models including CNN are inefficient in utilizing datapoints. 

### Adversaires
CNN is also vulnerable to adversaires by simply move, rotate or resize individual features.

<div class="imgcap">
<img src="/assets/capsule/sface2.jpg" style="border:none;width:20%;">
</div>

The following image can be mis-categorized as a gibbon in a CNN model by selectively making small changes into the original panda picture. The middle noisy picture is the changes made to the panda image on the left. After the change, the original CNN network mis-classifies the image as a gibbon.

<div class="imgcap">
<img src="/assets/capsule/a2.png" style="border:none;width:60%;">
</div>

(image source [OpenAi](https://blog.openai.com/adversarial-example-research/))

### Capsule

**A capsule captures the likeliness of a feature and its variant**. So the purpose of the capsule is not _only_ to detect a feature but also to train the model to learn the variants. So the same capsule can detect multiple variants.

<div class="imgcap">
<img src="/assets/capsule/c21.jpg" style="border:none;width;">
</div>

For example, the capsule above can detect a face rotated clockwise or counter clockwise:

<div class="imgcap">
<img src="/assets/capsule/c22.jpg" style="border:none;width;">
</div>

**Equivariance** is the detection of objects that can transform to each other.  Intuitively, the capsule network detects the face is rotated right 20° (equivariance) rather than realizes the face matched a variant that is rotated 20°. By forcing the model to learn the feature variant in a capsule, we _may_ extrapolate possible variants more effectively with less training data. In CNN, the final label is viewpoint invariant (i.e. the top neuron detects a face but losses the information in the angle of rotation). For equivariance, changes in the viewpoint lead to the corresponding changes in capsules (the angle of rotation). By maintaining the spatial orientation, it will be easier to avoid adversaires.

### Matrix capsule

The **matrix capsule** captures the activation (likeliness) and the 4x4 pose matrix.

<div class="imgcap">
<img src="/assets/capsule/capp.png" style="border:none;width:40%;">
</div>
(Source from the Matrix capsules with EM routing paper)

For example, the second row images below represent the same object above them with differen viewpoints. In matrix capsule, we train the model to capture the pose information (orientation, azimuths etc...). Of course, just like other deep learning methods, this is our intention even though it is never guaranteed.

<div class="imgcap">
<img src="/assets/capsule/data.png" style="border:none;width:60%">
</div>
(Source from the Matrix capsules with EM routing paper)

The objective of the EM (Expectation Maximization) routing is to group capsules to form a part-whole relationship using a clustering technique (EM). In machine learning, we use EM to cluster datapoints into different Gaussian distributions. For example, we cluster the datapoints below into two clusters modeled by two gaussian distributions.

<div class="imgcap">
<img src="/assets/ml/GM2.png" style="border:none;width:60%;">
</div>

In the face detection example, each of the mouth, eyes and nose detection capsules in the lower layer makes predictions (**votes**) on the pose matrices of its possible parent capsule(s). These votes are computed by multiplying its own pose matrix with a transformation matrix that we learn from the training data. The role of the EM routing is to cluster lower level capsules that produce similar votes to a parent capsule. For example, if the nose, mouth and eyes capsules all vote a similar pose matrix value for a capsule in the layer above, we cluster them together to build a higher level structure: the face capsule.

<div class="imgcap">
<img src="/assets/capsule/c3.jpg" style="border:none;width:80%">
</div>

Here is the visualization of the votes from the lower layer capsule.

<div class="imgcap">
<img src="/assets/capsule/cluster.jpg" style="border:none;width;width:70%">
</div>


> A higher level feature (a face) is detected by looking for agreement between votes from the capsules one layer below. We use EM routing to cluster capsules that have close proximity of the corresponding votes. 

### Gaussian mixture model & Expectation Maximization (EM)

We will take a short break to understand EM. A Gaussian mixture model clusters datapoints into a mixture of Gaussian distributions described by a mean $$\mu$$ and a standard deviation $$\sigma$$. Below, we cluster datapoints into the yellow and red cluster which each is described by a $$\mu$$ and a $$\sigma$$.

<div class="imgcap">
<img src="/assets/capsule/em.gif" style="border:none;width:40%">
</div>

(Image source wikipedia)

For a Gaussian mixture model with two clusters, we start with a random initialization of clusters $$ G_1 = (\mu_1, \sigma^2_1) $$ and $$ G_2 = (\mu_2, \sigma^2_2) $$. Expectation Maximization (EM) algorithm tries to fit the training datapoints into $$G_1$$ and $$G_2$$ and then re-compute $$\mu$$ and $$ \sigma$$ for $$G_1$$ and $$G_2$$ based on Gaussian distribution. The iteration continues until the solution converged such that the probability of seeing all datapoints is maximized with the final $$G_1$$ and $$G_2$$ distribution.

The probability of $$x$$ given (belong to) the cluster $$G_1$$ is:

$$
\begin{split}
P(x \vert G_1 ) & = \frac{1}{\sigma_1\sqrt{2\pi}}e^{-(x - \mu_1)^{2}/2\sigma_1^{2} } \\
\end{split}
$$

At each iteration, we start with 2 Gaussian distributions which we later re-calculate its $$\mu$$ and $$\sigma$$ based on the datapoints.
<div class="imgcap">
<img src="/assets/ml/GM1.png" style="border:none;width:60%">
</div>

Eventually, we will converge to two Gaussian distributions that maximize the likelihood of the observed datapoints.
<div class="imgcap">
<img src="/assets/ml/GM2.png" style="border:none;width:60%">
</div>

### Using EM for Routing-By-Agreement

Now, we will put it together to explain the EM routing for matrix capsule. A higher level feature (a face) is detected by looking for agreement between votes from the capsules one layer below. 

A **vote** $$v_{ij}$$ for capsule $$j$$ from capsule $$i$$ is computed by multipling the pose matrix $$M_i$$ of capsule $$i$$ with a **viewpoint invariant transformation** $$W_{ij}$$. 

$$
\begin{split}
&v_{ij} = M_iW_{ij} \quad 
\end{split}
$$

The probability that a capsule $$i$$ is assigned to capsule $$j$$ as a part-whole relationship is based on the proximity of the vote $$v_{ij}$$ to the votes $$(v_{oj} \ldots) $$ from other capsules. $$W_{ij}$$ is learned discriminatively through a cost function and the backpropagation. It learns not only what a face is composed of, and it also makes sure the pose information of the parent capsule matched with its sub-components after some transformation.

Here is the visualization of routing-by-agreement in matrix capsules. We group capsules with similar votes ($$ T_iT_{ij} \approx T_hT_{hj}$$) after transform the pose $$ T_i$$ and $$T_j$$ with a viewpoint invariant transformation. ($$T_{ij}$$ aka $$W_{ij}$$ and $$T_{hj}$$)
 
<div class="imgcap">
<img src="/assets/capsule/gh.png" style="border:none;width:80%">
</div>

(Source Geoffrey Hinton)

Even the viewpoint may change, the pose matrices (or votes) corresponding to the same high level structure (a face) will change in a co-ordinate way such that we can still cluster the same group of lower level capsules together. i.e. we do not need different transformation matrices for different viewpoints. The transformation matrix is viewpoint invariant.

<div class="imgcap">
<img src="/assets/capsule/cluster2.jpg" style="border:none;width:80%">
</div>

#### Capsule assignment

EM Routing clusters capsules to form a higher level structure. We also use EM routing to compute the **assignment probabilities** $$r_{ij}$$ to measure it quantitively. For example, the hand capsule is not part of the face capsule, the assignment probability between the face and the hand is zero.

<div class="imgcap">
<img src="/assets/capsule/c2.jpg" style="border:none;width:80%">
</div>

### Calculate capsule activation and pose matrix

Let $$ v^h_{ij} $$ be $$h$$-th dimensional component for the **vote** $$v_{ij}$$ from capsule $$i$$ to capsule $$j$$. $$ v_{ij} $$ is the product of the pose matrix ($$M_i$$) for capsule $$i$$ and the transformation matrix $$W_{ij}$$. 

$$
\begin{split}
&v_{ij} = M_iW_{ij} \quad 
\end{split}
$$

The capsule $$j$$ is modeled by a Gaussian $$G$$ ($$\mu^h$$ and $$\sigma^h$$ represents the mean and standard deviation for the h-th component). The probability distribution for $$ v^h_{ij} $$ based on this Gaussian distribution is (i.e. the probability that $$v^h_{ij}$$ belongs to the cluster $$G$$ using the Gaussian equation):

$$
\begin{split}
p^h_{i \vert j} & = \frac{1}{\sqrt{2 \pi ({\sigma^h_j})^2}} \exp{(- \frac{(v^h_{ij}-\mu^h_j)^2}{2 ({\sigma^h_j})^2})} \\
\ln(p^h_{i \vert j}) &= - \frac{(v^h_{ij}-\mu^h_j)^2}{2 ({\sigma^h_j})^2} - \ln(\sigma^h_j) - \frac{\ln(2 \pi)}{2} \\
\end{split}
$$

$$\ln(p_{i \vert j})$$ is the negative log likelihood of the vote $$v_{ij}$$ matching the pose matrix of the capsule $$j$$.

$$cost_{ij}$$ is the cost to activate the capsule $$j$$ by the lower layer capsule $$i$$. If the calculated cost is low, we increase the chance of activating $$j$$. If $$cost$$ is high, it implies the corresponding votes do not match the parent Gaussian distribution and it gives a lower chance to activate the parent capsule. 

$$cost$$ is the negative of the negative log likelihood. The h-th component of the cost for representing capsule $$i$$ by capsule $$j$$ is:

$$
cost^h_{ij} = - \ln(P^h_{i \vert j})
$$

Since capsules are not equally related to capsule $$j$$, we pro-rated the cost with the **assignment probabilities** $$r_{ij}$$. The cost for all lower layer capsules is:

$$
\begin{split}
cost^h_j &= \sum_i  r_{ij} cost^h_{ij} \\
&= \sum_i - r_{ij} \ln(p^h_{i \vert j}) \\
&= \frac{\sum_i r_{ij} (\sigma^h)^2}{2 (\sigma^h)^2} + (\ln(\sigma^h) + \frac{\ln(2 \pi)}{2}) \sum_i r_{ij} \\
&= (\ln(\sigma^h) + k) \sum_i r_{ij}  \quad \text{which k is a constant}
\end{split}
$$

To determine whether the capsule $$j$$ will be activated, we use the following equation:

$$
a_j = sigmoid(\lambda(b_j - \sum_h cost^h_j))
$$

In the original paper, "$$-b_j$$" is explained as the cost of describing the mean and variance of capsule j. From the perspective of routing by agreement, I sometimes interpret "b" as a threshold in which how close the votes on $$j$$ needed to be on each other to activate $$j$$. $$b_j$$ is learned discriminatively using a cost function and the backpropagation. In EM routing, we use multiple iterations (default 3 iterations) to cluster the datapoints. λ is an inverse temperature parameter which increases after each EM routing iteration. For example, λ is first initialized to 1 at the beginning of the iterations and increment by 1 after each routing iteration. The exact scheme is not discussed in the paper and we should experiment different schemes during the training. 

#### EM Routing

Here is the EM-routing in computing the capsule activation as well as the mean and the variance (Gaussian model) of the parent capsule.

<div class="imgcap">
<img src="/assets/capsule/al1.png" style="border:none;width:40%">
</div>

(Source from the Matrix capsules with EM routing paper)

We start with the activation $$a$$ for capsules in level L and their corresponding votes $$V$$ for level L+1. We initially set the assignment probability $$r_{ij}$$ to be uniformly distributed before the routing iterations. We call M-step to compute an updated Gaussian model ($$\mu$$, $$\sigma$$) with the current $$r_{ij}$$ and the activation for the capsules in layer L+1. Then we call E-step to recompute the assignment probabilities $$r_{ij}$$ based on the newly computed Gaussian values and the activations in Layer L+1. We re-iterate the process $$t$$ (default 3) times so we can cluster the capsules better (better $$r_{ij}$$) and use the last calculated $$a_j$$ as the activation of the capsule $$j$$ and the last calculated $$\mu^h_j$$ as the h-th component of the pose matrix. (we reshape the 16 components into a 4x4 pose matrix.) 

In M-step, we calculate $$\mu$$ and $$\sigma$$ based on the activation $$a_i$$ at Level L and the current $$r_{ij}$$ (which is updated by E-step). M-step also re-calcule the cost and the activation $$a_j$$ for the capsules $$j$$ in Level L+1. $$ \beta_{\nu} $$ and $$ \beta_{\alpha}$$ is trained discriminatively. λ is an inverse temperature parameter increased by 1 after each routing iteration in our code implementation. 

<div class="imgcap">
<img src="/assets/capsule/al2.png" style="border:none;width:90%">
</div>

In E-step, we re-calculate the assignment probability $$r_{ij}$$ based on the new $$\mu$$, $$\sigma$$ and $$a_j$$. The assignment is increased if the vote is closer to the $$\mu$$ of the updated Gaussian model.

<div class="imgcap">
<img src="/assets/capsule/al3.png" style="border:none;width:90%">
</div>

> We use the $$a_j$$ from the last m-step call in the iterations as the activation of the output capsule $$j$$ and $$ \mu^h_j $$ as the h-component (for h = 1 ... 4x4=16) of the corresponding pose matrix.

<div class="imgcap">
<img src="/assets/capsule/al1.png" style="border:none;width:40%">
</div>

#### Loss function (using Spread loss)

Instead of using

$$
z_j =  \sum_{i} W_i * x_i + b_{i}
$$

to calculate the activation of a neuron, matrix capsules use EM-routing to compute the activation of a capsule and its pose matrix. Nevertheless, the matrix capsules still heavily depend on the backpropagation in training the transformation matrix $$W_{ij}$$ and the parameters $$ \beta_{\nu} $$ and $$ \beta_{\alpha}$$.

The loss from the class $$i$$ (other than the true label $$t$$) is defined as

$$
L_i = (\max(0, m - (a_t - a_i)))^2 
$$

which $$a_t$$ is the activation of the target class (true label) and $$a_i$$ is the activation for class $$i$$. 

The total cost is:

$$
L = \sum_{i \neq t} (\max(0, m - (a_t - a_i)))^2 
$$


If the margin between the true label and the wrong class is smaller than $$m$$, we penalize it by the 
square of $$m - (a_t - a_i)$$. $$m$$ is initially start as 0.2 and linearly increased by 0.1 after each epoch training. $$m$$ will stop growing after reaching the maximum 0.9. Starting at a lower margin helps the training to avoid too many dead capsules during the early phase.

### Capsule Network

#### smallNORB

The smallNORB dataset has 5 toy classes: airplanes, cars, trucks, humans and animals. Every individual sample is pictured at 18 different azimuths (0-340), 9 elevations and 6 lighting conditions. This dataset is particular picked by the paper so it can study the classification of images with different viewpoints.

<div class="imgcap">
<img src="/assets/capsule/data.png" style="border:none;width:60%">
</div>

(Picture from the Matrix capsules with EM routing paper)

#### Architect

However, many code implementations start with the MNist dataset because the image size is smaller. So for our demonstration, we also pick the MNist dataset.

<div class="imgcap">
<img src="/assets/capsule/cape.png" style="border:none;">
</div>
(Picture from the Matrix capsules with EM routing paper)

In our implementation, A=32 channels, B=32 capsules, K=3 (3x3 kernels), C=32 capsules, D=32 capusles, E=10 classes.

Here is a brief description of each layer and the shape of their outputs:

| Layer Name | Apply | Output shape |
| --- | --- | --- | --- |
| MNist image |  |  28, 28, 1 |
| ReLU Conv1 | Regular Convolution (CNN) layer using 5x5 kernels with 32 output channels, stride 2 and padding | 14, 14, 32 |
| PrimaryCaps | Modified convolution layer with 1x1 kernels output 32 capsules. (each with a 4x4 Pose matrix and a scalar activation value) The convolution uses strides 1 and padding. <br>Requiring 32x32x(4x4+1) parameters. | <nobr>pose (14, 14, 32, 4, 4), </nobr> <br>activations (14, 14, 32) |
| ConvCaps1 | Capsule convolution with 3x3 kernels, strides 2 and no padding. <br>Requiring 3x3x32x32x4x4 parameters. | poses (6, 6, 32, 4, 4), activations (6, 6, 32) |
| ConvCaps2 | Capsule convolution with 3x3 kernels, strides 1 and no padding | poses (4, 4, 32, 4, 4), activations (4, 4, 32) |
| Class Capsules | Capsule with 1x1 kernel. Requiring 32x10x4x4 parameters.   | poses (10, 4, 4), activations (10) |


ReLU Conv1 is a regular convolution (CNN) layer using a 5x5 filter with a stride of 2 outputting 32 ($$A=32$$) channels (feature maps) using the ReLU activation.

In PrimaryCaps, we apply a 1x1 filter to transform each of the 32 channels into 32 ($$B=32$$) primary capsules. Each capsule contains a 4x4 pose matrix and a scalar for the activation. Therefore it takes $$ A \times B \times (4 \times 4 + 1) $$ 1x1 filters. We use the regular convolution layer to implement the PrimaryCaps. In CNN, each node outputs one scalar value. In the matrix capsule, we group $$4 \times 4 + 1$$ nodes to generate 1 capsule. 

PrimaryCaps is followed by a **convolution capsule layer** ConvCaps1 using a 3x3 filters ($$K=3$$) with a stride of 2. ConvCaps1 takes capsules as input and output capsules. The major difference between ConvCaps1 and a regular convolution layer is that a convolution capsule layer uses EM routing to compute the activation output and the pose matrices of the upper level capsules. 

The capsule output of ConvCaps1 is then feed into ConvCaps2. ConvCaps2 is another convolution capsule layer but with a stride of 1. 
 
The output capsules of ConvCaps2 are connected to the Class Capsules using a 1x1 filter and it has one capsule per class. (In MNist, we have 10 classes $$E=10$$) 

We use EM routing to compute the pose matrices and the output activations for ConvCaps1, ConvCaps2 and Class Capsules. In CNN, we slide the same filter over the spatial dimension in calculating the same feature map. i.e. we want to detect the same feature regardless of the location. In EM routing, we share the same transformation matrix $$W_{ij}$$ regardless of its spatial location in calculating the votes from capsule $$i$$ to capsule $$j$$. For example, from ConvCaps1 to ConvCaps2, we have 32 input capsules and 32 output capsules. The filter size is 3x3 and the capsule's pose matrix is 4x4. Hence, we require 3x3x32x32x4x4 parameters.

Here is the code in building our layers:
```python
def capsules_net(inputs, num_classes, iterations, batch_size, name='capsule_em'):
    """Define the Capsule Network model
    """

    with tf.variable_scope(name) as scope:
        # ReLU Conv1
        # Images shape (24, 28, 28, 1) -> conv 5x5 filters, 32 output channels, strides 2 with padding, ReLU
        # nets -> (?, 14, 14, 32)
        nets = conv2d(
            inputs,
            kernel=5, out_channels=32, stride=2, padding='SAME',
            activation_fn=tf.nn.relu, name='relu_conv1'
        )

        # PrimaryCaps
        # (?, 14, 14, 32) -> capsule 1x1 filter, 32 output capsule, strides 1 without padding
        # nets -> (poses (?, 14, 14, 32, 4, 4), activations (?, 14, 14, 32))
        nets = primary_caps(
            nets,
            kernel_size=1, out_capsules=32, stride=1, padding='VALID',
            pose_shape=[4, 4], name='primary_caps'
        )

        # ConvCaps1
        # (poses, activations) -> conv capsule, 3x3 kernels, strides 2, no padding
        # nets -> (poses (24, 6, 6, 32, 4, 4), activations (24, 6, 6, 32))
        nets = conv_capsule(
            nets, shape=[3, 3, 32, 32], strides=[1, 2, 2, 1], iterations=iterations,
            batch_size=batch_size, name='conv_caps1'
        )

        # ConvCaps2
        # (poses, activations) -> conv capsule, 3x3 kernels, strides 1, no padding
        # nets -> (poses (24, 4, 4, 32, 4, 4), activations (24, 4, 4, 32))
        nets = conv_capsule(
            nets, shape=[3, 3, 32, 32], strides=[1, 1, 1, 1], iterations=iterations,
            batch_size=batch_size, name='conv_caps2'
        )

        # Class capsules
        # (poses, activations) -> 1x1 convolution, 10 output capsules
        # nets -> (poses (24, 10, 4, 4), activations (24, 10))
        nets = class_capsules(nets, num_classes, iterations=iterations,
                              batch_size=batch_size, name='class_capsules')

        # poses (24, 10, 4, 4), activations (24, 10)
        poses, activations = nets

    return poses, activations
```

#### ReLU Conv1

ReLU Conv1 is a simple CNN layer. We use the TensorFlow slim API _slim.conv2d_ to create a CNN layer using a 3x3 kernel with stride 2 and ReLU. (We use the slim API so the code is more condense and easier to read.)
```python
def conv2d(inputs, kernel, out_channels, stride, padding, name, is_train=True, activation_fn=None):
  with slim.arg_scope([slim.conv2d], trainable=is_train):
    with tf.variable_scope(name) as scope:
      output = slim.conv2d(inputs,
                           num_outputs=out_channels,
                           kernel_size=[kernel, kernel], stride=stride, padding=padding,
                           scope=scope, activation_fn=activation_fn)
      tf.logging.info(f"{name} output shape: {output.get_shape()}")

  return output
```

#### PrimaryCaps

PrimaryCaps is not much difference from a CNN layer: instead of generating 1 scalar value, we generate out_capsules (32) capsules with 4 x 4  scalar values for the pose matrices and 1 scalar for the activation:
```python
def primary_caps(inputs, kernel_size, out_capsules, stride, padding, pose_shape, name):
    """This constructs a primary capsule layer using regular convolution layer.

    :param inputs: shape (N, H, W, C) (?, 14, 14, 32)
    :param kernel_size: Apply a filter of [kernel, kernel] [5x5]
    :param out_capsules: # of output capsule (32)
    :param stride: 1, 2, or ... (1)
    :param padding: padding: SAME or VALID.
    :param pose_shape: (4, 4)
    :param name: scope name

    :return: (poses, activations), (poses (?, 14, 14, 32, 4, 4), activations (?, 14, 14, 32))
    """

    with tf.variable_scope(name) as scope:
        # Generate the poses matrics for the 32 output capsules
        poses = conv2d(
            inputs,
            kernel_size, out_capsules * pose_shape[0] * pose_shape[1], stride, padding=padding,
            name='pose_stacked'
        )

        input_shape = inputs.get_shape()

        # Reshape 16 scalar values into a 4x4 matrix
        poses = tf.reshape(
            poses, shape=[-1, input_shape[-3], input_shape[-2], out_capsules, pose_shape[0], pose_shape[1]],
            name='poses'
        )

        # Generate the activation for the 32 output capsules
        activations = conv2d(
            inputs,
            kernel_size,
            out_capsules,
            stride,
            padding=padding,
            activation_fn=tf.sigmoid,
            name='activation'
        )

        tf.summary.histogram(
            'activations', activations
        )

    # poses (?, 14, 14, 32, 4, 4), activations (?, 14, 14, 32)
    return poses, activations
```
	
#### ConvCaps1, ConvCaps2

ConvCaps1 and ConvCaps are both convolution capsule with stride 2 and 1 respectively. In the comment of the following code, we trace the tensor shape for the ConvCaps1 layer.

The code involves 3 major steps:

* Use kernel_tile to tile (convolute) the pose matrices and the activation to be used later in voting and EM routing.
* Call mat_transform to generate the votes from the children "tiled" pose matrices and the transformation matrices.
* Call matrix_capsules_em_routing to compute the output capsules (pose and activation) of the parent capsule.

```python
def conv_capsule(inputs, shape, strides, iterations, batch_size, name):
  """This constructs a convolution capsule layer from a primary or convolution capsule layer.
      i: input capsules (32)
      o: output capsules (32)
      batch size: 24
      spatial dimension: 14x14
      kernel: 3x3
  :param inputs: a primary or convolution capsule layer with poses and activations
         pose: (24, 14, 14, 32, 4, 4)
         activation: (24, 14, 14, 32)
  :param shape: the shape of convolution operation kernel, [kh, kw, i, o] = (3, 3, 32, 32)
  :param strides: often [1, 2, 2, 1] (stride 2), or [1, 1, 1, 1] (stride 1).
  :param iterations: number of iterations in EM routing. 3
  :param name: name.

  :return: (poses, activations).

  """
  inputs_poses, inputs_activations = inputs

  with tf.variable_scope(name) as scope:

      stride = strides[1] # 2
      i_size = shape[-2] # 32
      o_size = shape[-1] # 32
      pose_size = inputs_poses.get_shape()[-1]  # 4

      # Tile the input capusles' pose matrices to the spatial dimension of the output capsules
      # Such that we can later multiple with the transformation matrices to generate the votes.
      inputs_poses = kernel_tile(inputs_poses, 3, stride)  # (?, 14, 14, 32, 4, 4) -> (?, 6, 6, 3x3=9, 32x16=512)

      # Tile the activations needed for the EM routing
      inputs_activations = kernel_tile(inputs_activations, 3, stride)  # (?, 14, 14, 32) -> (?, 6, 6, 9, 32)
      spatial_size = int(inputs_activations.get_shape()[1]) # 6

      # Reshape it for later operations
      inputs_poses = tf.reshape(inputs_poses, shape=[-1, 3 * 3 * i_size, 16])  # (?, 9x32=288, 16)
      inputs_activations = tf.reshape(inputs_activations, shape=[-1, spatial_size, spatial_size, 3 * 3 * i_size]) # (?, 6, 6, 9x32=288)

      with tf.variable_scope('votes') as scope:
          
          # Generate the votes by multiply it with the transformation matrices
          votes = mat_transform(inputs_poses, o_size, size=batch_size*spatial_size*spatial_size)  # (864, 288, 32, 16)
          
          # Reshape the vote for EM routing
          votes_shape = votes.get_shape()
          votes = tf.reshape(votes, shape=[batch_size, spatial_size, spatial_size, votes_shape[-3], votes_shape[-2], votes_shape[-1]]) # (24, 6, 6, 288, 32, 16)
          tf.logging.info(f"{name} votes shape: {votes.get_shape()}")

      with tf.variable_scope('routing') as scope:
          
          # beta_v and beta_a one for each output capsule: (1, 1, 1, 32)
          beta_v = tf.get_variable(
              name='beta_v', shape=[1, 1, 1, o_size], dtype=tf.float32,
              initializer=initializers.xavier_initializer()
          )
          beta_a = tf.get_variable(
              name='beta_a', shape=[1, 1, 1, o_size], dtype=tf.float32,
              initializer=initializers.xavier_initializer()
          )

          # Use EM routing to compute the pose and activation
          # votes (24, 6, 6, 3x3x32=288, 32, 16), inputs_activations (?, 6, 6, 288)
          # poses (24, 6, 6, 32, 16), activation (24, 6, 6, 32)
          poses, activations = matrix_capsules_em_routing(
              votes, inputs_activations, beta_v, beta_a, iterations, name='em_routing'
          )

          # Reshape it back to 4x4 pose matrix
          poses_shape = poses.get_shape()
          # (24, 6, 6, 32, 4, 4)
          poses = tf.reshape(
              poses, [
                  poses_shape[0], poses_shape[1], poses_shape[2], poses_shape[3], pose_size, pose_size
              ]
          )

      tf.logging.info(f"{name} pose shape: {poses.get_shape()}")
      tf.logging.info(f"{name} activations shape: {activations.get_shape()}")

      return poses, activations
```

kernel_tile use tiling and convolution to prepare the input pose matrices and the activations to the correct spatial dimension for voting and EM-routing. (The code is pretty hard to understand so readers may simply treat it as a black box if it is too hard.)
```python
def kernel_tile(input, kernel, stride):
    """This constructs a primary capsule layer using regular convolution layer.

    :param inputs: shape (?, 14, 14, 32, 4, 4)
    :param kernel: 3
    :param stride: 2

    :return output: (50, 5, 5, 3x3=9, 136)
    """

    # (?, 14, 14, 32x(16)=512)
    input_shape = input.get_shape()
    size = input_shape[4]*input_shape[5] if len(input_shape)>5 else 1
    input = tf.reshape(input, shape=[-1, input_shape[1], input_shape[2], input_shape[3]*size])


    input_shape = input.get_shape()
    tile_filter = np.zeros(shape=[kernel, kernel, input_shape[3],
                                  kernel * kernel], dtype=np.float32)
    for i in range(kernel):
        for j in range(kernel):
            tile_filter[i, j, :, i * kernel + j] = 1.0 # (3, 3, 512, 9)

    # (3, 3, 512, 9)
    tile_filter_op = tf.constant(tile_filter, dtype=tf.float32)

    # (?, 6, 6, 4608)
    output = tf.nn.depthwise_conv2d(input, tile_filter_op, strides=[
                                    1, stride, stride, 1], padding='VALID')

    output_shape = output.get_shape()
    output = tf.reshape(output, shape=[-1, output_shape[1], output_shape[2], input_shape[3], kernel * kernel])
    output = tf.transpose(output, perm=[0, 1, 2, 4, 3])

    # (?, 6, 6, 9, 512)
    return output
```

mat_transform extracts the transformation matrices parameters as a TensorFlow trainable variable $$w$$. It then multiplies with the "tiled" input pose matrices to generate the votes for the parent capsules.
```python
def mat_transform(input, output_cap_size, size):
    """Compute the vote.

    :param inputs: shape (size, 288, 16)
    :param output_cap_size: 32

    :return votes: (24, 5, 5, 3x3=9, 136)
    """

    caps_num_i = int(input.get_shape()[1]) # 288
    output = tf.reshape(input, shape=[size, caps_num_i, 1, 4, 4]) # (size, 288, 1, 4, 4)

    w = slim.variable('w', shape=[1, caps_num_i, output_cap_size, 4, 4], dtype=tf.float32,
                      initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0)) # (1, 288, 32, 4, 4)
    w = tf.tile(w, [size, 1, 1, 1, 1])  # (24, 288, 32, 4, 4)

    output = tf.tile(output, [1, 1, output_cap_size, 1, 1]) # (size, 288, 32, 4, 4)

    votes = tf.matmul(output, w) # (24, 288, 32, 4, 4)
    votes = tf.reshape(votes, [size, caps_num_i, output_cap_size, 16]) # (size, 288, 32, 16)

    return votes
```


#### EM routing coding

<div class="imgcap">
<img src="/assets/capsule/al1.png" style="border:none;width:40%">
</div>

Here is the code implementation for the EM routing which calling m_step and e_step alternatively. By default, we ran the iterations 3 times. The main purpose of the EM routing is to compute the pose matrices and the activations. In the last iteration, the m_step already complete the last calculation of those parameters. Therefore we can skip the e_step in the last iteration which mainly responsible for re-calculating the routing assignment.

```python
def matrix_capsules_em_routing(votes, i_activations, beta_v, beta_a, iterations, name):
  """The EM routing between input capsules (i) and output capsules (j).

  :param votes: (N, OH, OW, kh x kw x i, o, 4 x 4) = (24, 6, 6, 3x3*32=288, 32, 16)
  :param i_activation: activation from Level L (24, 6, 6, 288)
  :param beta_v: (1, 1, 1, 32)
  :param beta_a: (1, 1, 1, 32)
  :param iterations: number of iterations in EM routing, often 3.
  :param name: name.

  :return: (pose, activation) of output capsules.
  """

  votes_shape = votes.get_shape().as_list()

  with tf.variable_scope(name) as scope:

    # Match rr (routing assignment) shape, i_activations shape with votes shape for broadcasting in EM routing

    # rr: [3x3x32=288, 32, 1]
    # rr: routing matrix from each input capsule (i) to each output capsule (o)
    rr = tf.constant(
      1.0/votes_shape[-2], shape=votes_shape[-3:-1] + [1], dtype=tf.float32
    )

    # i_activations: expand_dims to (24, 6, 6, 288, 1, 1)
    i_activations = i_activations[..., tf.newaxis, tf.newaxis]

    # beta_v and beta_a: expand_dims to (1, 1, 1, 1, 32, 1]
    beta_v = beta_v[..., tf.newaxis, :, tf.newaxis]
    beta_a = beta_a[..., tf.newaxis, :, tf.newaxis]

    # inverse_temperature schedule (min, max)
    it_min = 1.0
    it_max = min(iterations, 3.0)
    for it in range(iterations):
      inverse_temperature = it_min + (it_max - it_min) * it / max(1.0, iterations - 1.0)
      o_mean, o_stdv, o_activations = m_step(
        rr, votes, i_activations, beta_v, beta_a, inverse_temperature=inverse_temperature
      )

      # We skip the e_step call in the last iteration because we only 
      # need to return the a_j and the mean from the m_stp in the last iteration
      # to compute the output capsule activation and pose matrices  
      if it < iterations - 1:
        rr = e_step(
          o_mean, o_stdv, o_activations, votes
        )

    # pose: (N, OH, OW, o 4 x 4) via squeeze o_mean (24, 6, 6, 32, 16)
    poses = tf.squeeze(o_mean, axis=-3)

    # activation: (N, OH, OW, o) via squeeze o_activationis [24, 6, 6, 32]
    activations = tf.squeeze(o_activations, axis=[-3, -1])

  return poses, activations
```
  
In the equation for the output capsule's activation $$a_j$$

 $$
 a_j = sigmoid(\lambda(b_j - \sum_h cost^h_j))
 $$
  
λ is an inverse temperature parameter. In our implementation, we start from 1 and increment it by 1 after each routing iteration. The original paper does not specify how λ is increased and you can experiment different schemes instead. Here is our source code:
```python
# inverse_temperature schedule (min, max)
it_min = 1.0
it_max = min(iterations, 3.0)
for it in range(iterations):
  inverse_temperature = it_min + (it_max - it_min) * it / max(1.0, iterations - 1.0)

  o_mean, o_stdv, o_activations = m_step(
    rr, votes, i_activations, beta_v, beta_a, inverse_temperature=inverse_temperature
  )
```

After the last iteration loop, $$a_j$$ is output as the final activation of the output capsule $$j$$. The mean $$ \mu^h_j $$ is used for the final value of the h-th component of the corresponding pose matrix. We later reshape those 16 components into a 4x4 pose matrix.
```python
# pose: (N, OH, OW, o 4 x 4) via squeeze o_mean (24, 6, 6, 32, 16)
poses = tf.squeeze(o_mean, axis=-3)

# activation: (N, OH, OW, o) via squeeze o_activationis [24, 6, 6, 32]
activations = tf.squeeze(o_activations, axis=[-3, -1])
```

#### m-steps

The algorithm for the m-steps.
<div class="imgcap">
<img src="/assets/capsule/al2.png" style="border:none;width:90%">
</div>

The following is the code listing of the m-step method. The comment traces the shape of the Tensors under the following conditions: a batch size of 24, 32 input capsules and output capsules, 3x3 kernels, 4x4=16 pose matrix and an output spatial dimension of 6x6.

m_step computes the mean and the variance of the parent capsules. Means and variances have the shape of (24, 6, 6, 1, 32, 16) and (24, 6, 6, 1, 32, 1) respectively in ConvCaps1.

```python
def m_step(rr, votes, i_activations, beta_v, beta_a, inverse_temperature):
  """The M-Step in EM Routing from input capsules i to output capsule j.
  i: input capsules (32)
  o: output capsules (32)
  h: 4x4 = 16
  output spatial dimension: 6x6
  :param rr: routing assignments. shape = (kh x kw x i, o, 1) =(3x3x32, 32, 1) = (288, 32, 1)
  :param votes. shape = (N, OH, OW, kh x kw x i, o, 4x4) = (24, 6, 6, 288, 32, 16)
  :param i_activations: input capsule activation (at Level L). (N, OH, OW, kh x kw x i, 1, 1) = (24, 6, 6, 288, 1, 1)
     with dimensions expanded to match votes for broadcasting.
  :param beta_v: Trainable parameters in computing cost (1, 1, 1, 1, 32, 1)
  :param beta_a: Trainable parameters in computing next level activation (1, 1, 1, 1, 32, 1)
  :param inverse_temperature: lambda, increase over each iteration by the caller.

  :return: (o_mean, o_stdv, o_activation)
  """

  rr_prime = rr * i_activations

  # rr_prime_sum: sum over all input capsule i
  rr_prime_sum = tf.reduce_sum(rr_prime, axis=-3, keep_dims=True, name='rr_prime_sum')

  # o_mean: (24, 6, 6, 1, 32, 16)
  o_mean = tf.reduce_sum(
    rr_prime * votes, axis=-3, keep_dims=True
  ) / rr_prime_sum

  # o_stdv: (24, 6, 6, 1, 32, 16)
  o_stdv = tf.sqrt(
    tf.reduce_sum(
      rr_prime * tf.square(votes - o_mean), axis=-3, keep_dims=True
    ) / rr_prime_sum
  )

  # o_cost_h: (24, 6, 6, 1, 32, 16)
  o_cost_h = (beta_v + tf.log(o_stdv + epsilon)) * rr_prime_sum

  # o_cost: (24, 6, 6, 1, 32, 1)
  # o_activations_cost = (24, 6, 6, 1, 32, 1)
  # yg: This is done for numeric stability.
  # It is the relative variance between each channel determined which one should activate.
  o_cost = tf.reduce_sum(o_cost_h, axis=-1, keep_dims=True)
  o_cost_mean = tf.reduce_mean(o_cost, axis=-2, keep_dims=True)
  o_cost_stdv = tf.sqrt(
    tf.reduce_sum(
      tf.square(o_cost - o_cost_mean), axis=-2, keep_dims=True
    ) / o_cost.get_shape().as_list()[-2]
  )
  o_activations_cost = beta_a + (o_cost_mean - o_cost) / (o_cost_stdv + epsilon)

  # (24, 6, 6, 1, 32, 1)
  o_activations = tf.sigmoid(
    inverse_temperature * o_activations_cost
  )

  return o_mean, o_stdv, o_activations
```

#### E-steps

The algorithm for the e-steps.
<div class="imgcap">
<img src="/assets/capsule/al3.png" style="border:none;width:90%">
</div>

e_step is mainly responsible for re-calculating the routing assignment (shape: 24, 6, 6, 288, 32, 1) after m_step updates the output activation $$a_j$$ and the Gaussian models with new $$\mu$$ and $$\sigma$$.

```python
def e_step(o_mean, o_stdv, o_activations, votes):
  """The E-Step in EM Routing.

  :param o_mean: (24, 6, 6, 1, 32, 16)
  :param o_stdv: (24, 6, 6, 1, 32, 16)
  :param o_activations: (24, 6, 6, 1, 32, 1)
  :param votes: (24, 6, 6, 288, 32, 16)

  :return: rr
  """

  o_p_unit0 = - tf.reduce_sum(
    tf.square(votes - o_mean) / (2 * tf.square(o_stdv)), axis=-1, keep_dims=True
  )

  o_p_unit2 = - tf.reduce_sum(
    tf.log(o_stdv + epsilon), axis=-1, keep_dims=True
  )

  # o_p is the probability density of the h-th component of the vote from i to j
  # (24, 6, 6, 1, 32, 16)
  o_p = o_p_unit0 + o_p_unit2

  # rr: (24, 6, 6, 288, 32, 1)cd

  zz = tf.log(o_activations + epsilon) + o_p
  rr = tf.nn.softmax(
    zz, dim=len(zz.get_shape().as_list())-2
  )

  return rr
```

#### Class capsules

Recall from a couple sections ago, the output of ConvCaps2 is feed into the Class capsules layer. The output poses for ConvCaps2 has a shape of (24, 4, 4, 32, 4, 4). 

* batch size of 24
* 4x4 spatial output
* 32 output channels
* 4x4 pose matrix

Instead of using a 3x3 filter in ConvCaps2. Class capsules use a 1x1 filter. Also, nstead of outputting a 2D spatial output (6x6 in ConvCaps1 and 4x4 in ConvCaps2), it outputs 10 capsules each representing one of the 10 classes in the MNist. The code structure for the class capsules is very similar to the conv_capsule. It makes call to compute the votes and then use EM routing to compute the capsule outputs.

```python
def class_capsules(inputs, num_classes, iterations, batch_size, name):
    """
    :param inputs: ((24, 4, 4, 32, 4, 4), (24, 4, 4, 32))
    :param num_classes: 10
    :param iterations: 3
    :param batch_size: 24
    :param name:
    :return poses, activations: poses (24, 10, 4, 4), activation (24, 10).
    """

    inputs_poses, inputs_activations = inputs # (24, 4, 4, 32, 4, 4), (24, 4, 4, 32)

    inputs_shape = inputs_poses.get_shape()
    spatial_size = int(inputs_shape[1])  # 4
    pose_size = int(inputs_shape[-1])    # 4
    i_size = int(inputs_shape[3])        # 32

    # inputs_poses (24*4*4=384, 32, 16)
    inputs_poses = tf.reshape(inputs_poses, shape=[batch_size*spatial_size*spatial_size, inputs_shape[-3], inputs_shape[-2]*inputs_shape[-2] ])

    with tf.variable_scope(name) as scope:
        with tf.variable_scope('votes') as scope:
            # inputs_poses (384, 32, 16)
            # votes: (384, 32, 10, 16)
            votes = mat_transform(inputs_poses, num_classes, size=batch_size*spatial_size*spatial_size)
            tf.logging.info(f"{name} votes shape: {votes.get_shape()}")

            # votes (24, 4, 4, 32, 10, 16)
            votes = tf.reshape(votes, shape=[batch_size, spatial_size, spatial_size, i_size, num_classes, pose_size*pose_size])

            # (24, 4, 4, 32, 10, 16)
            votes = coord_addition(votes, spatial_size, spatial_size)

            tf.logging.info(f"{name} votes shape with coord addition: {votes.get_shape()}")

        with tf.variable_scope('routing') as scope:
            # beta_v and beta_a one for each output capsule: (1, 10)
            beta_v = tf.get_variable(
                name='beta_v', shape=[1, num_classes], dtype=tf.float32,
                initializer=initializers.xavier_initializer()
            )
            beta_a = tf.get_variable(
                name='beta_a', shape=[1, num_classes], dtype=tf.float32,
                initializer=initializers.xavier_initializer()
            )

            # votes (24, 4, 4, 32, 10, 16) -> (24, 512, 10, 16)
            votes_shape = votes.get_shape()
            votes = tf.reshape(votes, shape=[batch_size, votes_shape[1] * votes_shape[2] * votes_shape[3], votes_shape[4], votes_shape[5]] )

            # inputs_activations (24, 4, 4, 32) -> (24, 512)
            inputs_activations = tf.reshape(inputs_activations, shape=[batch_size,
                                                                       votes_shape[1] * votes_shape[2] * votes_shape[3]])

            # votes (24, 512, 10, 16), inputs_activations (24, 512)
            # poses (24, 10, 16), activation (24, 10)
            poses, activations = matrix_capsules_em_routing(
                votes, inputs_activations, beta_v, beta_a, iterations, name='em_routing'
            )

        # poses (24, 10, 16) -> (24, 10, 4, 4)
        poses = tf.reshape(poses, shape=[batch_size, num_classes, pose_size, pose_size] )

        # poses (24, 10, 4, 4), activation (24, 10)
        return poses, activations
```

To integrate the spatial location of the predicted class in the class capsules, we add the scaled x, y coordinate of the center of the receptive field of each capsule in ConvCaps2 to the first two elements of the vote. This is called **Coordinate Addition**. According to the paper, this encourages the model to train the transformation matrix to produce values for those two elements to represent the position of the feature relative to the center of the capsule’s receptive field. The motivation of the coordinate addition is to have the first 2 elements of the votes ($$v^1_{jk}, v^2_{jk}$$) to predict the spatial coordinates (x, y) of the predicted class. 

By retaining the spatial information in the capsule, we move beyond simply checking the presence of a feature. We encourage the system to verify the spatial relationship of features to avoid adversaries. i.e. if the spatial orders of features are wrong, their corresponding votes should not match.

The pseudo code below illustrates the key idea. The pose matrices for the ConvCaps2 output has the shape of (24, **4, 4**, 32, 4, 4).  i.e. a spatial dimension of 4x4. We use the location of the capsule to locate the corresponding element in c1 below (a 4x4 matrix) which contains the scaled x coordinate of the center of the receptive field of the spatial capsule. We add this element value to $$v^1_{jk}$$. For the second element $$v^2_{jk}$$ of the vote, we repeat the same process using c2.

```
# Spatial output of ConvCaps2 is 4x4
v1 = [[[8.],  [12.], [16.], [20.]],
      [[8.],  [12.], [16.], [20.]],
      [[8.],  [12.], [16.], [20.]],
      [[8.],  [12.], [16.], [20.]],
     ]

v2 = [[[8.],  [8.],  [8.],  [8]],
      [[12.], [12.], [12.], [12.]],
      [[16.], [16.], [16.], [16.]],
      [[20.], [20.], [20.], [20.]]
         ]

c1 = np.array(v1, dtype=np.float32) / 28.0
c2 = np.array(v2, dtype=np.float32) / 28.0
```


Here is the final code which looks far more complicated in order to handle any image size.
```python
def coord_addition(votes, H, W):
    """Coordinate addition.

    :param votes: (24, 4, 4, 32, 10, 16)
    :param H, W: spaital height and width 4

    :return votes: (24, 4, 4, 32, 10, 16)
    """
    coordinate_offset_hh = tf.reshape(
      (tf.range(H, dtype=tf.float32) + 0.50) / H, [1, H, 1, 1, 1]
    )
    coordinate_offset_h0 = tf.constant(
      0.0, shape=[1, H, 1, 1, 1], dtype=tf.float32
    )
    coordinate_offset_h = tf.stack(
      [coordinate_offset_hh, coordinate_offset_h0] + [coordinate_offset_h0 for _ in range(14)], axis=-1
    )  # (1, 4, 1, 1, 1, 16)

    coordinate_offset_ww = tf.reshape(
      (tf.range(W, dtype=tf.float32) + 0.50) / W, [1, 1, W, 1, 1]
    )
    coordinate_offset_w0 = tf.constant(
      0.0, shape=[1, 1, W, 1, 1], dtype=tf.float32
    )
    coordinate_offset_w = tf.stack(
      [coordinate_offset_w0, coordinate_offset_ww] + [coordinate_offset_w0 for _ in range(14)], axis=-1
    ) # (1, 1, 4, 1, 1, 16)

    # (24, 4, 4, 32, 10, 16)
    votes = votes + coordinate_offset_h + coordinate_offset_w

    return votes
```

		  
### Spread loss

To compute the spread loss:

$$
L = \sum_{i \neq t} (\max(0, m - (a_t - a_i)))^2 
$$

The value of $$m$$ starts from 0.1. After each epoch, we increase the value by 0.1 until it reached the maximum of 0.9. This prevents too many dead capsules in the early phase of the training.
```python
def spread_loss(labels, activations, iterations_per_epoch, global_step, name):
    """Spread loss

    :param labels: (24, 10] in one-hot vector
    :param activations: [24, 10], activation for each class
    :param margin: increment from 0.2 to 0.9 during training

    :return: spread loss
    """

    # Margin schedule
    # Margin increase from 0.2 to 0.9 by an increment of 0.1 for every epoch
    margin = tf.train.piecewise_constant(
        tf.cast(global_step, dtype=tf.int32),
        boundaries=[
            (iterations_per_epoch * x) for x in range(1, 8)
        ],
        values=[
            x / 10.0 for x in range(2, 10)
        ]
    )

    activations_shape = activations.get_shape().as_list()

    with tf.variable_scope(name) as scope:
        # mask_t, mask_f Tensor (?, 10)
        mask_t = tf.equal(labels, 1)      # Mask for the true label
        mask_i = tf.equal(labels, 0)      # Mask for the non-true label

        # Activation for the true label
        # activations_t (?, 1)
        activations_t = tf.reshape(
            tf.boolean_mask(activations, mask_t), shape=(tf.shape(activations)[0], 1)
        )
        
        # Activation for the other classes
        # activations_i (?, 9)
        activations_i = tf.reshape(
            tf.boolean_mask(activations, mask_i), [tf.shape(activations)[0], activations_shape[1] - 1]
        )

        l = tf.reduce_sum(
            tf.square(
                tf.maximum(
                    0.0,
                    margin - (activations_t - activations_i)
                )
            )
        )
        tf.losses.add_loss(l)

        return l
```
				  
### Result

The following is the histogram of distances of votes to the mean of each of the 5 final capsules after each routing iteration. Each distance point is weighted by its assignment probability. With a human image as input, we expect, after 3 iterations, the difference is the smallest (closer to 0) for the human column. (any distances greater than 0.05 will not be shown here.)

<div class="imgcap">
<img src="/assets/capsule/iter.png" style="border:none;width:80%">
</div>

The error rate for the Capsule network is generally lower than a CNN model with similar number of layers as shown below. 

<div class="imgcap">
<img src="/assets/capsule/resu.png" style="border:none;width:50%">
</div>

(Source from the Matrix capsules with EM routing paper)

The core idea of FGSM (fast gradient sign method) adversary is to add some noise on every step of optimization to drift the classification away from the target class. It slightly modifies the image to maximize the error based on the gradient information. Matrix routing is shown to be less vulnerable to FGSM adversaries comparing to CNN.

### Visualization

The pose matrices in Class Capsules are interpreted as the latent representation of the image. By adjusting the first 2 dimension of the pose and reconstructing it through a decoder (similar to the one in the previous capsule article), we can visualize what the Capsule Network learns for the MNist data.

<div class="imgcap">
<img src="/assets/capsule/m2.png" style="border:none;width:80%">
</div>

(Source from the Matrix capsules with EM routing paper)

Some digits are slightly rotated or moved which demonstrate the Class Capsules are learning the pose information of the MNist dataset.

### Credits

Part of the code implementation in computing the vote and the EM-routing is modified from [Guang Yang](https://github.com/gyang274/capsulesEM) or [Suofei Zhang](https://github.com/www0wwwjs1/Matrix-Capsules-EM-Tensorflow) implementations. Our source code is located at the [github](https://github.com/jhui/machine_learning/tree/master/capsule_em). To run,

* Configure the mnist_config.py and cap_config.py according to your environment. 
* Run the download_and_convert_mnist.py before running train.py.

Please note that the source code is for illustration purpose with no support at this moment. Readers may refer to other implementations if you have problems to trouble shooting any issues that may come up. Zhang's implementation seems easier to understand but Yang implementation is closer to the original paper.



