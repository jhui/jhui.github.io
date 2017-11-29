---
layout: post
comments: true
mathjax: true
priority: -1000
title: “Understanding Matrix capsules with EM Routing (Based on Hinton's Capsule Networks)”
excerpt: “A simple tutorial in understanding Matrix capsules with EM Routing in Capsule Networks”
date: 2017-11-14 11:00:00
---

This article covers the second capsule network paper [_Matrix capsules with EM Routing_](https://openreview.net/pdf?id=HJWLfGWRb) based on Geoffrey Hinton's Capsule Networks. We will cover the basic of matrix capsules and apply EM routing to classify images with different viewpoints.

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

A capsule captures the likeliness of a feature and its variant. So the purpose of the capsule is not only to detect a feature but also to train the model to learn the variants. 

<div class="imgcap">
<img src="/assets/capsule/c21.jpg" style="border:none;width;">
</div>

So the same capsule can detect the same object class with different orientations (for example, it detect a face with 0.9 likeliness and rotate 20° clockwise):

<div class="imgcap">
<img src="/assets/capsule/c22.jpg" style="border:none;width;">
</div>

**Equivariance** is the detection of objects that can transform to each other.  Intuitively, the capsule network detects the face is rotated right 20° (equivariance) rather than realizes the face matched a variant that is rotated 20°. By forcing the model to learn the feature variant in a capsule, we _may_ extrapolate possible variants more effectively with less training data. 

### Matrix capsule

The matrix capsule captures the activation (likeliness) and the 4x4 pose matrix.

<div class="imgcap">
<img src="/assets/capsule/capp.png" style="border:none;width:40%;">
</div>
(Source from the Matrix capsules with EM routing paper)


For example, the second row images below represent the same object above them with differen viewpoints. In matrix capsule, the pose matrix captures the viewpoint of the object. With deep learning training, we want to capture those information in the matrix capsule.

<div class="imgcap">
<img src="/assets/capsule/data.png" style="border:none;width:60%">
</div>
(Source from the Matrix capsules with EM routing paper)

The objective of the EM routing is to group capsules to form a part-whole relationship with a clustering technique (EM). In machine learning, we use EM to cluster datapoints into different Gaussian distributions. For example, we cluster the datapoints below into two clusters modeled by two gaussian distributions.

<div class="imgcap">
<img src="/assets/ml/GM2.png" style="border:none;width:60%;">
</div>

In the face detection example, each of the mouth, eyes and nose capsules in the lower layer makes predictions (**votes**) on the pose matrices of its possible parent capsule(s). These votes are computed as the multiplication of the pose matrix with a transformation matrix. The role of the EM routing is to cluster lower level capsules that produce similar votes. For example, if the nose, mouth and eyes capsules all vote a similar pose matrix value for a capsule in the layer above, we cluster them together to build a higher level structure: the face capsule.

<div class="imgcap">
<img src="/assets/capsule/c3.jpg" style="border:none;width:80%">
</div>

> A higher level feature (a face) is detected by looking for agreement between votes from the capsules one layer below. We use EM routing to cluster capsules that have close proximity of the corresponding votes. 

### Gaussian mixture model & Expectation Maximization (EM)

We will take a short break to understand EM. A Gaussian mixture model clusters datapoints into a mixture of Gaussian distributions.

<div class="imgcap">
<img src="/assets/capsule/em.gif" style="border:none;width:40%">
</div>

(Image source wikipedia)

For a Gaussian mixture model with two clusters, we start with a random initialization of clusters $$ G_1 = (\mu_1, \sigma^2_1) $$ and $$ G_2 = (\mu_2, \sigma^2_2) $$. Expectation Maximization (EM) algorithm tries to fit the training datapoints into $$G_1$$ and $$G_2$$ and then re-compute $$\mu$$ and $$ \sigma$$ for $$G_1$$ and $$G_2$$ based on Gaussian distribution. The iteration continues until the solution converged such that the probability of seeing all datapoints is maximized with $$G_1$$ and $$G_2$$ distribution.

The probability of $$x$$ given the cluster $$G_1$$ is:

$$
\begin{split}
P(x \vert G_1 ) & = \frac{1}{\sigma_1\sqrt{2\pi}}e^{-(x - \mu_1)^{2}/2\sigma_1^{2} } \\
\end{split}
$$

At each iteration, we start with 2 Gaussian distributions which we later re-calculate its $$\mu$$ and $$\sigma$$ based on the datapoints.
<div class="imgcap">
<img src="/assets/ml/GM1.png" style="border:none;width:60%">
</div>

Eventually, we will converge to two Gaussian distributions that maximize the likelihood of the datapoints.
<div class="imgcap">
<img src="/assets/ml/GM2.png" style="border:none;width:60%">
</div>

### Using EM for Routing-By-Agreement

A higher level feature (a face) is detected by looking for agreement between votes from the capsules one layer below. A **vote** $$v$$ is computed by multipling the pose matrix of capsule $$i$$ with a **viewpoint invariant transformation** $$W_{ic}$$ (from capsule $$i$$ to capsule $$c$$ above). The probability that a capsule is assigned to capsule $$c$$ as a part-whole relationship is based on the proximity of the vote coming from that capsule to the votes coming from other capsules that are assigned to capsule $$c$$. $$W_{ic}$$ is learned discriminatively through cost function and backpropagation. It learns not only what a face composed of, and it also makes sure the pose information are matched with its sub-components.

#### Capsule assignment

EM Routing determines how capsules are activated by capsules in the layer below. The **assignment probabilities** $$r_i$$ measures how much capsule $$i$$ is related with capsule $$c$$. For example, to eliminate the influence of the hand capsule on the face capsule, the assignment probability between the face and the hand is zero.

<div class="imgcap">
<img src="/assets/capsule/c2.jpg" style="border:none;width:80%">
</div>

The value of $$r_i$$ is calculated iteratively using the EM routing discussed below. 
### Calculate capsule activation

Let $$ v_{ih} $$ be the value on dimension $$h$$ of the **vote** from capsule $$i$$ to capsule $$c$$. (Note: we try to match the index scheme here with the one used in the technical paper even sometimes it looks confusing.) $$ v_{ih} $$ is the product of the pose matrix for capsule $$i$$ and the transformation matrix $$W_{ic}$$. The capsule $$c$$ is modeled by a Gaussian ($$\mu_h$$ and $$\sigma_h$$ for the dimension h). The probability distribution for $$ v_{ih} $$ based on Gaussian distribution is:

$$
\begin{split}
P_{ih} & = \frac{1}{\sqrt{2 \pi \sigma^2_h}} \exp{(- \frac{(v_{ih}-\mu_h)^2}{2 \sigma^2_h})} \\
\ln(P_{ih}) &= - \frac{(v_{ih}-\mu_h)^2}{2 \sigma^2_h} - \ln(\sigma_h) - \frac{\ln(2 \pi)}{2} \\
\end{split}
$$

Hence, $$\ln(P_{ih})$$ is the negative likelihood of the vote $$v_i$$ matching the pose matrix of the capsule $$c$$.

$$cost_h$$ calculates the cost of having the lower layer capsules being part of capsule $$c$$. If $$cost_h$$ is low, capsule $$i$$ is more likely to activate the face capsule. Since capsules are not equally related to capsule $$c$$, we pro-rated the cost with the **assignment probabilities** $$r_i$$. 


$$
\begin{split}
cost_h &= \sum_i - r_i \ln(P_{ih}) \\
&= \frac{\sum_i r_i \sigma^2_h}{2 \sigma^2_h} + (\ln(\sigma_h) + \frac{\ln(2 \pi)}{2}) \sum_i r_i \\
&= (\ln(\sigma_h) + k) \sum_i r_i  \quad \text{which k is a constant}
\end{split}
$$

The activation of the capsule $$c$$ is calculated by

$$
a_c = sigmoid(\lambda(b - \sum_h cost_h))
$$

which $$-b$$ represents the cost of describing the mean of capsule c and λ is an inverse temperature
parameter. $$b$$ is learned discriminatively using backpropagation. We use a fixed schedule for λ which is a hyper-parameter.

Here is the algorithm in computing the capsule activation as well as the mean of the capsule one layer above.

<div class="imgcap">
<img src="/assets/capsule/al1.png" style="border:none;width:50%">
</div>

(Source from the Matrix capsules with EM routing paper)

We start with the activation $$\alpha$$ for capsules in level L and their corresponding votes $$v$$ for level L+1. We initially set the assignment probability to be uniformly distributed. We call M-step to compute the the Gaussian model ($$\mu$$, $$\sigma$$) and the activation for the capsules in layer L+1. Then we call E-step to recompute the assignment probabilities $$r_i$$ based on how well the vote match with other capsules. We re-iterate the process $$t$$ (default 3) times to finalize the activation and $$\mu$$ for the capsules in level L+1.

In M-step, we calculate $$\mu$$ and $$\sigma$$ based on the activation, votes $$v$$ and the assignment probability for the lower layer capsules. Then we compute the new activation for the capsule. $$ \beta_{\nu} $$ and $$ \beta_{\alpha}$$ is trained discriminatively as stated before.

<div class="imgcap">
<img src="/assets/capsule/al2.png" style="border:none;width:90%">
</div>

In E-step, we re-calculate the probability based on the new $$\mu$$ and $$\sigma$$ and re-calculate the assignment probability. The assignment is increased if the vote is close to the $$\mu$$ of the new cluster.

<div class="imgcap">
<img src="/assets/capsule/al3.png" style="border:none;width:90%">
</div>

### Capsule Network

Now we can apply the matrix capsule in solving the smallNORB dataset.

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

We apply a 1x1 filter to transform the 32 channels into 32 ($$B=32$$) primary capsules which contain a 4x4 pose matrix and 1 scalar for the activation. Therefore it takes $$ A \times B \times (4 \times 4 + 1) $$ 1x1 filters.

It then follows by a convolution capsule layer ConvCaps1 with a 3x3 filters ($$K=3$$) and a stride of 2. ConvCaps1 is very similar to a regular convolution layer with the exception that it takes capsules as input and output capsules. ConvCaps2 is similar to ConvCaps1 except that ConvCaps2 has a stride of 1. ConvCaps2 connects to the Class Capsules which have one capsule per class. (5 classes $$E=5$$) 

In CNN, a filter is shared in generate each filter map. So it detects a specific feature regardless of the location in the image. In Class Capsules, the transformation matrix is shared in extracting the same capsule feature. (e.g. face) To maintain the spatial location of capsule, we also adds the scaled x, y coordinate of the center of the receptive field of each capsule to the first two elements of the vote. This is called **Coordinate Addition**. This helps the transformations to produce those two elements that represent the position of the feature relative to the center of the capsule’s receptive field. The routing is performed between adjacent capsule layers. For convolutional capsules, each capsule in layer L + 1 are connected to capsules within its receptive field in layer L only. 

#### Loss function

The loss function is defined as

$$
L_i = (\max(0, m - (a_t - a_i)))^2 
$$

which $$a_t$$ is the activation of the target class and $$a_i$$ is the other classes. If the activation of a wrong class is closer than the margin $$m$$, we penalize it by the squared distance to the margin. $$m$$ is initially start as 0.2 and linearly increasing to 0.9 to avoid dead capsules.

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


