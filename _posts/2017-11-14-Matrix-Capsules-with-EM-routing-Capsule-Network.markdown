---
layout: post
comments: true
mathjax: true
priority: 415
title: “Understanding Matrix capsules with EM Routing (Based on Hinton's Capsule Networks)”
excerpt: “A simple tutorial in understanding Matrix capsules with EM Routing in Capsule Networks”
date: 2017-11-14 11:00:00
---

This article covers the second capsule network paper [_Matrix capsules with EM Routing_](https://openreview.net/pdf?id=HJWLfGWRb) based on Geoffrey Hinton's Capsule Networks. We will cover the basic of matrix capsules and apply EM routing to group features to form a **part-whole relationship** (combining eyes, ears, mouth to form a face) in image classification problem.

### CNN challenges

In our [previous capsule article](https://jhui.github.io/2017/11/03/Dynamic-Routing-Between-Capsules/), we cover the challenges of CNN in exploring spatial relationship. Let's recap some of the challenges again.

The _simplest_ face recognition CNN model detects the presence of features like eyes, nose and mouth.

<div class="imgcap">
<img src="/assets/capsule/sface.jpg" style="border:none;width:20%;">
</div>
[(image source)](http://sharenoesis.com/article/draw-face/84)

The probability of detecting a face here depends on the probability of detecting individual features separately.

<div class="imgcap">
<img src="/assets/capsule/face3.jpg" style="border:none;width:50%;">
</div>

Nevertheless, this simple model is vulnerable to adversaries like misplacing the features. (As deep networks become popular in authentication like Apple FaceID, avoiding adversaries becomes very important.)

<div class="imgcap">
<img src="/assets/capsule/sface2.jpg" style="border:none;width:20%;">
</div>

Another shortcoming of CNN is handling different viewpoints. 

<div class="imgcap">
<img src="/assets/capsule/sface3.jpg" style="border:none;width:60%;">
</div>

To identify the three different viewpoints above as a face, a CNN model may build neurons in learning different feature orientations. 

<div class="imgcap">
<img src="/assets/capsule/cnn1.jpg" style="border:none;">
</div>

Nevertheless this tends to memorize the dataset rather than generalize a solution. It requires extensive training datapoints to have reasonable coverage of different variant combinations. MNist dataset contains 55,000 training data. i.e. 5,500 samples per digits. However, it is unlikely that children need to read this large amount of samples to learn digits. Our existing deep learning models including CNN seem inefficient in utilizing datapoints.



### Matrix capsule

In matrix capsule, we do not want to detect features in combinations with the viewpoint variants (spatial orientation). Instead, we want to detecting a face as simple as below regardless of the viewpoint variants.

<div class="imgcap">
<img src="/assets/capsule/c1.jpg" style="border:none;width:50%;">
</div>

A neuron in deep learning captures the likeliness of a feature. In a matrix capsule, it also captures a 4x4 pose matrix. 

<div class="imgcap">
<img src="/assets/capsule/capp.png" style="border:none;width:40%;">
</div>
(Source from the Matrix capsules with EM routing paper)

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

A higher level feature (a face) is detected by looking for agreement between votes from the capsules one layer below. A **vote** $$v$$ is computed by multipling the pose matrix of capsule $$i$$ with a **viewpoint invariant transformation** $$W_{ic}$$ (from capsule $$i$$ to capsule $$c$$ above). The probability that a capsule is assigned to capsule $$c$$ (as a part-whole relationship) is based on the proximity of the vote coming from that capsule to the votes coming from other capsules that are assigned to capsule $$c$$. 
   
$$W_{ic}$$ is learned discriminatively (through cost function and backpropagation). This linear transformation likely maps related features closer together while pull un-related features further away. Conceptually, capsules corresponding to votes with close proximity are grouped (clustered) together to be represented by capsule $$c$$.

<div class="imgcap">
<img src="/assets/capsule/adv2.jpg" style="border:none;width:70%">
</div>

> EM  routing is a Routing-By-Agreement because it groups capsules together that make similar votes.

In theory, it can explore the spatial ordering of features.  For example, it tranforms related features closer while maintains relative spatial information. But for the adversary images below, the transformation may actually pull the mouth away and therefore will not be grouped as part of the face.

<div class="imgcap">
<img src="/assets/capsule/adv.jpg" style="border:none;width:60%">
</div>

Even a viewpoint may change, the pose matrices for the eye, mouth and ears will change in a co-ordinate way such that the agreement between votes from different parts remains the same. EM locates cluster and have the benefit of grouping related features regardless of the viewpoint variant. (regardless of looking at a face from the front or slightly from the side) With EM routing, we should detect a face easier without over extensive training data with different viewpoints.

<div class="imgcap">
<img src="/assets/capsule/face21.jpg" style="border:none;width:40%">
</div>

> New capsules and routing algorithm will hopefully build higher level structures much easier and much effectively with less training data.

#### Capsule assignment

EM Routing determines how capsules are activated by capsules in the layer below. For example, the high activation in the eye, nose and mouth capsule should trigger the activation of the face capsule. The **assignment probabilities** $$r_i$$ measures how much capsule $$i$$ is related with capsule $$c$$. For example, to eliminate the influence of the hand capsule on the face capsule, the assignment probability between the face and the hand is zero.

<div class="imgcap">
<img src="/assets/capsule/c2.jpg" style="border:none;width:80%">
</div>

The value of $$r_i$$ is calculated iteratively using the EM routing discussed below. (Note: we try to match the index scheme here with the one used in the technical paper even sometimes it looks confusing.)

### Calculate capsule activation

Let $$ v_{ih} $$ be the value on dimension h of the vote from capsule $$i$$ to capsule $$c$$. $$ v_{ih} $$ is the product of the pose matrix for capsule $$i$$ and the transformation matrix $$W_{ic}$$. The capsule $$c$$ is modeled by a Gaussian ($$\mu_h$$ and $$\sigma_h$$ for the dimension h). The probability distribution for $$ v_{ih} $$ is (follow a Gaussian distribution):

$$
\begin{split}
P_{ih} & = \frac{1}{\sqrt{2 \pi \sigma^2_h}} \exp{(- \frac{(v_{ih}-\mu_h)^2}{2 \sigma^2_h})} \\
\ln(P_{ih}) &= - \frac{(v_{ih}-\mu_h)^2}{2 \sigma^2_h} - \ln(\sigma_h) - \frac{\ln(2 \pi)}{2} \\
\end{split}
$$

Hence, $$\ln(P_{ih})$$ is the negative likelihood of whether capsule $$i$$ should activate capsule $$c$$. 

$$
\begin{split}
cost_h &= \sum_i - r_i \ln(P_{ih}) \\
&= \frac{\sum_i r_i \sigma^2_h}{2 \sigma^2_h} + (\ln(\sigma_h) + \frac{\ln(2 \pi)}{2}) \sum_i r_i \\
&= (\ln(\sigma_h) + k) \sum_i r_i  \quad \text{which k is a constant}
\end{split}
$$

$$cost_h$$ calculates the cost of having the lower layer capsules being part of capsule $$c$$. Since capsules are not equally related to capsule $$c$$, we pro-rated the cost with the **assignment probabilities** $$r_i$$. If $$cost_h$$ is low, we are more likely to activate the face capsule.

The activation of the capsule $$c$$ is calculated by

$$
a_c = sigmoid(\lambda(b - \sum_h cost_h))
$$

which $$-b$$ represents the cost of describing the mean of capsule c and λ is an inverse temperature
parameter. $$b$$ will be learned discriminatively using backpropagation and we set a fixed schedule for λ which is a hyper-parameter.

Here is the algorithm in computing the capsule activation of the next level as well as the mean of the upper level capsule.

<div class="imgcap">
<img src="/assets/capsule/al1.png" style="border:none;width:40%">
</div>

We start with the activation $$\alpha$$ for capsules in level L and the vote $$v$$ computed for level l+1 from level l. We initially set the assignment probability to be uniformly distributed. We call M-step to compute the $$\mu$$, $$\sigma$$ and the activation for the capsules in layer L+1. Then we call E-step to recompute the assignment probabilities $$r_i$$ based on how well the prediction match with other capsules. We re-iterate the process $$t$$ (default 3) times to finalize the activation and $$\mu$$ for the capsules in level L+1.

In M-step, we calculate $$\mu$$ and $$\sigma$$ based on the activation, votes $$v$$ and the assignment probability for lower layer capsules. Then we compute the new activation for the capsule. $$ \beta_{\nu} $$ and $$ \beta_{\alpha}$$ is trained discriminatively. However, the paper is not clear on what are the input parameters in learning those values.

<div class="imgcap">
<img src="/assets/capsule/al2.png" style="border:none;width:80%">
</div>

In E-step, we re-calculate the probability based on the new $$\mu$$ and $$\sigma$$ and re-calculate the assignment probability. The assignment is increased if the vote is close to the $$\mu$$ of the new cluster.

<div class="imgcap">
<img src="/assets/capsule/al3.png" style="border:none;width:80%">
</div>

### Capsule Network

The architect of using Matrix capsule:

<div class="imgcap">
<img src="/assets/capsule/cape.png" style="border:none;width:80%">
</div>

ReLU Conv1 is a regular convolution layer with a 5x5 filter and a stride of 2 outputting 32 channels ($$A=32$$ feature maps) using ReLU activation.

We apply a 1x1 filter to transform the 32 channels into 32 ($$B=32$$) primary capsules which contain a 4x4 pose matrix and 1 scalar for the activation. Therefore it takes $$ A \times B \times (4 \times 4 + 1) $$ 1x1 filters.

It then follows by a convolution capsule layer ConvCaps1 with a 3x3 filters ($$K=3$$) and a stride of 2. ConvCaps1 is very similar to a regular convolution layer with the exception that it takes capsules as input and output capsules. ConvCaps2 is similar to ConvCaps1 except that ConvCaps2 has a stride of 1. ConvCaps2 connects to the Class Capsules which have one capsule per class. (5 classes $$E=5$$) 

In CNN, a filter is shared in generate each filter map. So it detects a specific feature regardless of the location in the image. In Class Capsules, the transformation matrix is shared in extracting the same capsule feature. (e.g. face) It also adds the scaled x, y coordinate of the center of the receptive field of each capsule to the first two elements of the vote. This is called **Coordinate Addition**. This helps the transformations to produce those two elements that represent the position of the feature relative to the center of the capsule’s receptive field.

The routing is performed between adjacent capsule layers. For convolutional capsules, each capsule in layer L + 1 are connected to capsules within its receptive field in layer L only. 

### Data

The smallNORB dataset has 5 toy classes: airplanes, cars, trucks, humans and animals. Every individual sample is pictured at 18 different azimuths (0-340), 9 elevations and 6 lighting conditions. This dataset is particular picked such that we can study how a model can handle different viewpoints.

<div class="imgcap">
<img src="/assets/capsule/data.png" style="border:none;width:80%">
</div>


### Visualization

The pose matrices in Class Capsules are interpreted as the latent representation of the image. By adjusting the first 2 dimension of the pose and reconstructing it through a decoder (similar to the one in the previous capsule article), we can visualize what the Capsule Network learns for the MNist data.

<div class="imgcap">
<img src="/assets/capsule/m2.png" style="border:none;width:80%">
</div>

(Source from the Matrix capsules with EM routing paper)

Some digits are slightly rotated or moved which demonstrate the Class Capsules are learning the pose information of the MNist dataset.




