---
layout: post
comments: true
mathjax: true
priority: -1000
title: “Understanding Matrix capsules with EM Routing (Based on Hinton's Capsule Networks)”
excerpt: “A simple tutorial in understanding Matrix capsules with EM Routing in Capsule Networks”
date: 2017-11-14 11:00:00
---

This article covers the second capsule network paper [_Matrix capsules with EM Routing_](https://openreview.net/pdf?id=HJWLfGWRb) based on Geoffrey Hinton's Capsule Networks. We will cover the basic of matrix capsules and apply EM routing to classify images with different viewpoints. The code demonstration is originated from [Guang Yang](https://github.com/gyang274/capsulesEM).

### CNN challenges

In our [previous capsule article](https://jhui.github.io/2017/11/03/Dynamic-Routing-Between-Capsules/), we cover the challenges of CNN in exploring spatial relationship and discuss how capsule networks may address those short-comings. Let's recap one of the challenge of CNN in handling different viewpoints like faces with different orientation. 

<div class="imgcap">
<img src="/assets/capsule/sface3.jpg" style="border:none;width:50%;">
</div>

Conceptually, the CNN trains neurons to handle different feature orientations with a top level face detection neuron.

<div class="imgcap">
<img src="/assets/capsule/cnn1.jpg" style="border:none;width;">
</div>

As indicated above, we add more convolution layers and features maps. Nevertheless this approach tends to memorize the dataset rather than generalize a solution. It requires a large volume of training data to cover different variants and to avoid overfitting. MNist dataset contains 55,000 training data. i.e. 5,500 samples per digits. However, it is unlikely that children need to read this large amount of samples to learn digits. Our existing deep learning models including CNN seem inefficient in utilizing datapoints. 

### Adversaires
CNN is also vulnerable to adversaires by simply move, rotate or resize individual features.

<div class="imgcap">
<img src="/assets/capsule/sface2.jpg" style="border:none;width:20%;">
</div>

The following image can be mis-categorized as a gibbon in a CNN model by selectively making small changes into the pixel value of the original panda picture. 

<div class="imgcap">
<img src="/assets/capsule/a2.png" style="border:none;width:60%;">
</div>

(image source [OpenAi](https://blog.openai.com/adversarial-example-research/))

### Capsule

A capsule captures the likeliness of a feature and its variant. So the purpose of the capsule is not only to detect a feature but also to train the model to learn the variants. So the same capsule can be used to detect multiple variants.

<div class="imgcap">
<img src="/assets/capsule/c21.jpg" style="border:none;width;">
</div>

For example, the same capsule can detect a face rotated 20° clockwise:

<div class="imgcap">
<img src="/assets/capsule/c22.jpg" style="border:none;width;">
</div>

**Equivariance** is the detection of objects that can transform to each other.  Intuitively, the capsule network detects the face is rotated right 20° (equivariance) rather than realizes the face matched a variant that is rotated 20°. By forcing the model to learn the feature variant in a capsule, we _may_ extrapolate possible variants more effectively with less training data. In CNN, the final label is viewpoint invariant (the top neuron detects a face but losses the information in the angle of rotation). For equivariance, changes in viewpoint lead to the corresponding changes in capsules which information in the angle of rotation is maintained.

### Matrix capsule

The matrix capsule captures the activation (likeliness) and the 4x4 pose matrix.

<div class="imgcap">
<img src="/assets/capsule/capp.png" style="border:none;width:40%;">
</div>
(Source from the Matrix capsules with EM routing paper)

For example, the second row images below represent the same object above them with differen viewpoints. In matrix capsule, we train the model hoping that the model will capture the pose information eventually from the training data. Of course, just like other deep learning methods, this is our intention even it is never guaranteed.

<div class="imgcap">
<img src="/assets/capsule/data.png" style="border:none;width:60%">
</div>
(Source from the Matrix capsules with EM routing paper)

The objective of the EM (Expectation Maximization) routing is to group capsules to form a part-whole relationship with a clustering technique (EM). In machine learning, we use EM to cluster datapoints into different Gaussian distributions. For example, we cluster the datapoints below into two clusters modeled by two gaussian distributions.

<div class="imgcap">
<img src="/assets/ml/GM2.png" style="border:none;width:60%;">
</div>

In the face detection example, each of the mouth, eyes and nose capsules in the lower layer makes predictions (**votes**) on the pose matrices of its possible parent capsule(s). These votes are computed as the multiplication of its pose matrix with a transformation matrix. The role of the EM routing is to cluster lower level capsules that produce similar votes. For example, if the nose, mouth and eyes capsules all vote a similar pose matrix value for a capsule in the layer above, we cluster them together to build a higher level structure: the face capsule.

<div class="imgcap">
<img src="/assets/capsule/c3.jpg" style="border:none;width:80%">
</div>

Here is the visualization of the votes from the lower layer capsule.

<div class="imgcap">
<img src="/assets/capsule/cluster.jpg" style="border:none;width;width:70%">
</div>


> A higher level feature (a face) is detected by looking for agreement between votes from the capsules one layer below. We use EM routing to cluster capsules that have close proximity of the corresponding votes. 

### Gaussian mixture model & Expectation Maximization (EM)

We will take a short break to understand EM. A Gaussian mixture model clusters datapoints into a mixture of Gaussian distributions described by a mean $$\mu$$ and a standard deviation $$\sigma$$.

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

Now, we go into details in clustering capsules. A higher level feature (a face) is detected by looking for agreement between votes from the capsules one layer below. A **vote** $$v_{ij}$$ for capsule $$j$$ from capsule $$i$$ is computed by multipling the pose matrix of capsule $$i$$ with a **viewpoint invariant transformation** $$W_{ij}$$. The probability that a capsule is assigned to capsule $$j$$ as a part-whole relationship is based on the proximity of the vote coming from that capsule to the votes coming from other capsules that are assigned to capsule $$j$$. $$W_{ij}$$ is learned discriminatively through cost function and backpropagation. It learns not only what a face composed of, and it also makes sure the pose information are matched with its sub-components after some transformation.

Here is the visualization of routing-by-agreement in matrix capsules. We try to group capsules with similar votes ($$ T_iT_{ij} \approx T_hT_{hj}$$) after transform the pose $$ T_i$$ and $$T_j$$ with a viewpoint invariant transformation. ($$T_{ij}$$ aka $$W_{ij}$$ and $$T_{hj}$$)
 
<div class="imgcap">
<img src="/assets/capsule/gh.png" style="border:none;width:80%">
</div>

(Source Geoffrey Hinton)

Even the viewpoint may change, the pose matrices (or votes) corresponding to the same high level structure (a face) will change in a co-ordinate way such that a cluster with the same capsules can be detected. Hence, the EM routing groups related capsules regardless of the viewpoint.

<div class="imgcap">
<img src="/assets/capsule/cluster2.jpg" style="border:none;width:80%">
</div>

#### Capsule assignment

EM Routing clusters related capsules to form a higher level structure. We also use EM routing to compute the **assignment probabilities** $$r_{ij}$$ to measure it quantitively. For example, the hand capsule is not part of the face capsule, the assignment probability between the face and the hand is zero.

<div class="imgcap">
<img src="/assets/capsule/c2.jpg" style="border:none;width:80%">
</div>

The value of $$r_{ij}$$ and the activation of a capsule is calculated iteratively using the EM routing detailed below. 

### Calculate capsule activation and pose matrix

Let $$ v^h_{ij} $$ be $$h$$-th dimensional component for the **vote** $$v_{ij}$$ from capsule $$i$$ to capsule $$j$$. $$ v_{ij} $$ is the product of the pose matrix ($$M_i$$) for capsule $$i$$ and the transformation matrix $$W_{ij}$$. 

$$
\begin{split}
&v_{ij} = M_iW_{ij} \quad 
\end{split}
$$

The capsule $$j$$ is modeled by a Gaussian $$G$$ ($$\mu^h$$ and $$\sigma^h$$ represents the mean and standard deviation for the h-th component). The probability distribution for $$ v^h_{ij} $$ based on this Gaussian distribution is (the probability that $$v^h_{ij}$$ belongs to the cluster $$G$$):

$$
\begin{split}
p^h_{i \vert j} & = \frac{1}{\sqrt{2 \pi ({\sigma^h_j})^2}} \exp{(- \frac{(v^h_{ij}-\mu^h_j)^2}{2 ({\sigma^h_j})^2})} \\
\ln(p^h_{i \vert j}) &= - \frac{(v^h_{ij}-\mu^h_j)^2}{2 ({\sigma^h_j})^2} - \ln(\sigma^h_j) - \frac{\ln(2 \pi)}{2} \\
\end{split}
$$

$$\ln(p_{i \vert j})$$ is the negative log likelihood of the vote $$v_{ij}$$ matching the pose matrix of the capsule $$j$$.

$$cost$$ calculates the cost to have the lower layer capsules being part of capsule $$j$$. If $$cost$$ is high, it implies the corresponding votes do not match the parent Gaussian distribution and it gives a lower chance to activate the parent capsule. $$cost$$ is the negative of the negative log likelihood. The h-th component of the cost for representing capsule $$i$$ by capsule $$j$$ is:

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

To determine whether the capsule $$j$$ is activated, we use the following equation:

$$
a_j = sigmoid(\lambda(b_j - \sum_h cost^h_j))
$$

In the original paper, "$$-b$$" is explained as the cost of describing the mean and variance of capsule j. From the perspective of routing by agreement, I sometimes interpret "b" as a threshold in which how far the votes on $$j$$ need to be agreed to activate $$j$$. $$b_j$$ is learned discriminatively using backpropagation. λ is an inverse temperature parameter which is updated later after each iteration. (Explain in later section)

Here is the EM-routing in computing the capsule activation as well as the mean and the variance of the capsule one layer above.

<div class="imgcap">
<img src="/assets/capsule/al1.png" style="border:none;width:40%">
</div>

(Source from the Matrix capsules with EM routing paper)

We start with the activation $$a$$ for capsules in level L and their corresponding votes $$V$$ for level L+1. We initially set the assignment probability $$r_{ij}$$ to be uniformly distributed before the iterations. We call M-step to compute an updated Gaussian model ($$\mu$$, $$\sigma$$) with the current $$r_{ij}$$ and the activation for the capsules in layer L+1. Then we call E-step to recompute the assignment probabilities $$r_{ij}$$ based on the newly computed Gaussian values and the activations in Layer L+1. We re-iterate the process $$t$$ (default 3) times to better cluster capsules together.

In M-step, we calculate $$\mu$$ and $$\sigma$$ based on the activation $$a_i$$ at Level L and the current $$r_{ij}$$ (which is updated by E-step). M-step also updates the activation for the capsules $$a_j$$ for Level L+1. $$ \beta_{\nu} $$ and $$ \beta_{\alpha}$$ is trained discriminatively. λ is an inverse temperature parameter. It increases after each iteration. The exact scheme is not discussed in the paper and we should experiment different schemes during the training. 

<div class="imgcap">
<img src="/assets/capsule/al2.png" style="border:none;width:90%">
</div>

In E-step, we re-calculate the assignment probability based on the new $$\mu$$ and $$\sigma$$. The assignment is increased if the vote is closer to the $$\mu$$ of the updated Gaussian model.

<div class="imgcap">
<img src="/assets/capsule/al3.png" style="border:none;width:90%">
</div>

> We use the $$a_j$$ from the last m-step call in the iterations as the activation of the output capsule $$j$$ and $$ \mu^h_j $$ as the h-component (for h = 1 ... 4x4=16) of the corresponding pose matrix.

<div class="imgcap">
<img src="/assets/capsule/al1.png" style="border:none;width:40%">
</div>

(Note: detail code implementation will be shown in the next section.)

### Capsule Network

#### smallNORB

The smallNORB dataset has 5 toy classes: airplanes, cars, trucks, humans and animals. Every individual sample is pictured at 18 different azimuths (0-340), 9 elevations and 6 lighting conditions. This dataset is particular picked such that we can study how a model can handle different viewpoints.

<div class="imgcap">
<img src="/assets/capsule/data.png" style="border:none;width:60%">
</div>

(Picture from the Matrix capsules with EM routing paper)

#### Architect

Now we use matrix capsule to classify our smallNORB data.

<div class="imgcap">
<img src="/assets/capsule/cape.png" style="border:none;">
</div>
(Picture from the Matrix capsules with EM routing paper)

ReLU Conv1 is a regular convolution layer with a 5x5 filter and a stride of 2 outputting 32 channels ($$A=32$$ feature maps) using ReLU activation.

In PrimryCaps, we apply a 1x1 filter to transform the 32 channels from  ReLU Conv1 into 32 ($$B=32$$) primary capsules. Each capsule contains a 4x4 pose matrix and a scalar for the activation. Therefore it takes $$ A \times B \times (4 \times 4 + 1) $$ 1x1 filters. PrimaryCaps is very similar to the regular convolution layer with the exception that it generates a capsule (4x4 pose matrix + 1 scalar activation) instead of a scalar value.

It then follows by a **convolution capsule layer** ConvCaps1 using a 3x3 filters ($$K=3$$) and a stride of 2. ConvCaps1 takes capsules as input and output capsules. The major difference between ConvCaps1 and a regular convolution is that it uses EM routing to compute the activation and the pose matrix for the next upper level capsules. 

The capsule output of ConvCaps1 contains the pose matrix and the activation. It is then feed into ConvCaps2. ConvCaps2 is another convolution capsule layer but with stride equal to 1. 
 
The output capsules of ConvCaps2 are connected to the Class Capsules using a fully connected layer instead of a convolution layer.   ConvCaps2  outputs one capsule per class. (In smallNORB, we have 5 classes $$E=5$$) 

Here is the code in building each layers:

```python
def capsules_net(inputs, num_classes, iterations, name='CapsuleEM-V0'):
  """Replicate the network in `Matrix Capsules with EM Routing.`
  """

  with tf.variable_scope(name) as scope:

    # [24, 28, 28, 1] -> conv 5x5 filters, strides 2, 32 channels -> [24, 14, 14, 32]
    nets = _conv2d_wrapper(
      inputs, shape=[5, 5, 1, 32], strides=[1, 2, 2, 1], padding='SAME', add_bias=True, activation_fn=tf.nn.relu, name='conv1'
    )

    # [24, 14, 14, 32] -> conv2d, 1x1, strides 1, channels 32x(4x4+1)
    # -> (poses (24, 14, 14, 32, 4, 4), activations (24, 14, 14, 32))
    nets = capsules_init(nets, shape=[1, 1, 32, 32], strides=[1, 1, 1, 1], padding='VALID', pose_shape=[4, 4], name='capsule_init')

    # (poses, activations) -> capsule-conv 3x3x32x32x4x4, strides 2
    # -> (poses (24, 6, 6, 32, 4, 4), activations 24, 6, 6, 32))
    nets = capsules_conv(nets, shape=[3, 3, 32, 32], strides=[1, 2, 2, 1], iterations=iterations, name='capsule_conv1')

    # (poses, activations) -> capsule-conv 3x3x32x32x4x4, strides 1 -> (poses, activations)
    # -> (poses (24, 4, 4, 32, 4, 4), activations 24, 4, 4, 32))
    nets = capsules_conv(
      nets, shape=[3, 3, 32, 32], strides=[1, 1, 1, 1], iterations=iterations, name='capsule_conv2'
    )

    # (poses, activations) -> capsule-fc 1x1x32x10x4x4 shared view
    # -> (poses (24, 10, 4, 4), activations 24, 10))
    nets = capsules_fc(nets, num_classes, iterations=iterations, name='capsule_fc')

    poses, activations = nets

  return poses, activations
```

And the outputs for each layer:

| Layer Name | Apply | Output shape |
| --- | --- | --- | --- |
| Raw image |  |  28, 28, 1 |
| ReLU Conv1 | Convolution layer with 5x5 kernels output 32 channels, stride 2, with padding | 14, 14, 32|
| PrimaryCaps | Convolution capsule layer with 1x1 kernels output 32x(4x4) for pose and 32x1 for activation with strides 1 and padding | pose (14, 14, 32, 4, 4) activations (14, 14, 32) |
| ConvCaps1 | Capsule convolution 3x3x32x32x4x4, strides 2 | poses (6, 6, 32, 4, 4), activations (6, 6, 32) |
| ConvCaps2 | Capsule convolution 3x3x32x32x4x4, strides 1 | poses (4, 4, 32, 4, 4), activations (4, 4, 32) |
| Class Capsules | Capsule FC 1x1x32x5x4x4  | poses (5, 4, 4), activations (5) |

#### ConvCaps1

To simplify the logic, we omit the code in reshaping some tensors below. The major operation for ConvCaps1

* Generate the votes for $$a_i$$.
* Use EM routing to calculate the pose matrix and the activation for the upper layer capsules.

```python
def capsules_conv(inputs, shape, strides, iterations, name):
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
  :param strides: often [1, 1, 1, 1], or [1, 2, 2, 1].
  :param iterations: number of iterations in EM routing, often 3.
  :param name: name.

  :return: (poses, activations) same as capsule_init().

  """
  # Extract the poses and activations
  inputs_poses, inputs_activations = inputs
  inputs_poses_shape = inputs_poses.get_shape().as_list()

  with tf.variable_scope(name) as scope:

    # kernel: (kh, kw, i, o, 4, 4) = (3, 3, 32, 32, 4, 4)
    kernel = _get_weights_wrapper(
      name='pose_view_transform_weights', shape=shape + [inputs_poses_shape[-1], inputs_poses_shape[-1]]
    )

    #  Convert inputs_poses to input_poses_patches so it can multiply with kernel to compute the vote
    # (24, 14, 14, 32, 4, 4) -> (24, 6,6, 3, 3, 32, 32, 4, 4)
    ...
    inputs_poses_patches = ... inputs_poses ...
    ...
	
    # Compute the vote
    # (24, 6, 6, 3, 3, 32, 32, 4, 4)
    votes = _matmul_broadcast(
      inputs_poses_patches, kernel, name='inputs_poses_patches_view_transformation'
    )

    # Reshape the votes -> (24, 6, 6, 3x3x32=288, 32, 16)
    ...
	
    # Reshape inputs_activations to i_activations
    # (24, 6, 6, 3x3x32=288)
    i_activations = ... inputs_activations ...

    # beta_v and beta_a one for each output capsule: (1, 1, 1, 32)
    beta_v = _get_weights_wrapper(
      name='beta_v', shape=[1, 1, 1, votes_shape[6]]
    )
    beta_a = _get_weights_wrapper(
      name='beta_a', shape=[1, 1, 1, votes_shape[6]]
    )

    # Use EM routing to compute the pose and activation
    # poses (24, 6, 6, 32, 16)
    # activation (24, 6, 6, 32)
    poses, activations = matrix_capsules_em_routing(
      votes, i_activations, beta_v, beta_a, iterations, name='em_routing'
    )

    # Reshpae poses
    # (24, 6, 6, 32, 4, 4)
    poses = tf.reshape(
      poses, [
        -1, votes_shape[1], votes_shape[2], votes_shape[6], votes_shape[7], votes_shape[8]
      ]
    )

  return poses, activations
```

#### EM routing coding

<div class="imgcap">
<img src="/assets/capsule/al1.png" style="border:none;width:40%">
</div>

Here is the code implementation for the EM routing. In the last iteration loop, $$a_j$$ is output as the activation of the output capsule $$j$$ and $$ \mu^h_j $$ as the h-component of the corresponding pose matrix.
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

    # note: match rr shape, i_activations shape with votes shape for broadcasting in EM routing
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
      # Skip the last iteration because we initialize rr to uniform distribution before each iteration
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
  
#### m-steps

The algorithm for the m-steps.
<div class="imgcap">
<img src="/assets/capsule/al2.png" style="border:none;width:90%">
</div>

The following providing the trace of the m-steps when creating the ConvCaps1 layer. (with a batch size N of 24, 32 input capsules and 32 output capsules, 3x3 kernels, 4x4=16 pose matrix and output spatial dimension of 6x6.) We compute the mean and variance with shape (24, 6, 6, 1, 32, 16) and the output activation of shape (24, 6, 6, 1, 32, 1). 
```python
def m_step(rr, votes, i_activations, beta_v, beta_a, inverse_temperature):
  """The M-Step in EM Routing from input capsules i to output capsule j.
  i: input capsules (32)
  o: output capsules (32)
  h: 4x4 = 16
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

#### e-steps

The algorithm for the e-steps.
<div class="imgcap">
<img src="/assets/capsule/al3.png" style="border:none;width:90%">
</div>

The code generating the new assignment probability (24, 6, 6, 288, 32, 1).
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

  # rr: (24, 6, 6, 288, 32, 1)
  zz = tf.log(o_activations + epsilon) + o_p
  rr = tf.nn.softmax(
    zz, dim=len(zz.get_shape().as_list())-2
  )

  return rr
```
	  
#### Class Capsules
	  
In CNN, a filter is shared in generate each filter map. So it detects a specific feature regardless of the location in the image. In Class Capsules, the transformation matrix is shared in extracting the same class. Class capsules apply one view transform weight matrix (4 x 4) to each input channel and the view transform matrix is shared across spatial locations. So the kernel labelled in D is 1x1 and the number of variables of weights is D x E x 4 x 4.
	  
To maintain the spatial location of capsule, we also adds the scaled x, y coordinate of the center of the receptive field of each capsule to the first two elements of the vote. This is called **Coordinate Addition**. Hence, the vote is changed from 16 (4x4) components to 18 components. We use the same EM-routing algorithm which take the 18 components instead of the 16 components
to compute the assignment probability and activation. This helps the transformations to produce those two elements that represent the position of the feature relative to the center of the capsule’s receptive field. 
	  
#### Loss function

The loss function is defined as

$$
L_i = (\max(0, m - (a_t - a_i)))^2 
$$

which $$a_t$$ is the activation of the target class and $$a_i$$ is the other classes. If the activation of a wrong class is closer than the margin $$m$$, we penalize it by the squared distance to the margin. $$m$$ is initially start as 0.2 and linearly increasing to 0.9 during training (say increase it after each epoch) to avoid dead capsules.

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

The core idea of FGSM (fast gradient sign method) adversary is to add some noise on every step of optimization to drift the classification away from the target class. We optimize the image to maximize the error based on the gradient information. Matrix routing is shown to be less vulnerable to FGSM adversaries comparing to CNN.

### Visualization

The pose matrices in Class Capsules are interpreted as the latent representation of the image. By adjusting the first 2 dimension of the pose and reconstructing it through a decoder (similar to the one in the previous capsule article), we can visualize what the Capsule Network learns for the MNist data.

<div class="imgcap">
<img src="/assets/capsule/m2.png" style="border:none;width:80%">
</div>

(Source from the Matrix capsules with EM routing paper)

Some digits are slightly rotated or moved which demonstrate the Class Capsules are learning the pose information of the MNist dataset.


