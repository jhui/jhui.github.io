---
layout: post
comments: true
mathjax: true
title: “Convolution neural networks (CNN) tutorial”
excerpt: “Convolutional networks explore features by discover its spatial information. This tutorial will build CNN networks for visual recognition.”
date: 2017-03-01 12:00:00
---
**This is work in progress... The content needs major editing.**

### Overview
In a fully connected networks, all nodes in a layer is fully connected to all the nodes in the previous layer. This produces a complex model to explore all possible connections among nodes. But the complexity pays a high price of how easy to train the network and how deep the network can be. For spatial data like image, this complexity provides no additional benefits since most features are localized.

<div class="imgcap">
<img src="/assets/cnn/ppl.jpg" style="border:none;width:30%">
</div>

For face detection, the area of interested are all localized. Convolution neural networks apply small size filter to explore the images.The number of trainable parameters are significantly smaller and therefore allow CNN to use many filters to extract interested features. 

### Filters
Filters are frequently apply to images for different purposes. Human visual system applies edge detection filters to recognize object.

<div class="imgcap">
<img src="/assets/cnn/edge.png" style="border:none;">
</div>

For example, to blur an image, we can apply a 3x3 filter over every pixels in the image:
<div class="imgcap">
<img src="/assets/cnn/filter_b.png" style="border:none;">
</div>

To apply the filter to an image, we move the fiter 1 pixel at a time from left to right and top to bottom until we process every pixels.
<div class="imgcap">
<img src="/assets/cnn/stride.png" style="border:none;width:50%">
</div>

#### Stride and padding
However, we may encounter some problem on the edge. For example, on the top left corner, a filter may cover beyond the edge of an image. For a 3x3 filter, we may ignore the edge and geneate an output with width and height reduce by 2 pixels. Otherwise, we can pack extra 0 or replicate the edge of the origina image. All these settings are possible and configurable as "padding" in a CNN. 
<div class="imgcap">
<img src="/assets/cnn/padding.png" style="border:none;width:50%">
</div>

For a CNN, sometimes we do not move the filter only by 1 pixel. If we move the filter 2 pixels to the right, we call the "X stride" equal to 2.
<div class="imgcap">
<img src="/assets/cnn/stride2.png" style="border:none;width:50%">
</div>

Notice that both padding and stride may change the spatial dimension of the output. A stride of 2 in X direction will reduce X-dimension by 2. Without padding, the output shrink by N pixels which N is:

$$
N = \frac {\text{filter size} - 1} {2}
$$

### Convolution neural network (CNN)
A convolution neural network composes of convolution layers, polling layers and fully connected layers(FC). 

<div class="imgcap">
<img src="/assets/cnn/conv_layer.png" style="border:none;width:70%">
</div>

When we process the image, we apply many filters which each will geneate an output that we call **feature map**. If k features map are created, we call the feature maps have a depth of k.

<div class="imgcap">
<img src="/assets/cnn/conv_layer2.png" style="border:none;width:50%">
</div>


#### Convolution layers
We can apply multiple convolution filters to an image and then reduce each output with a maxium pool.

Apply maximum pool for sub-sampling:
<div class="imgcap">
<img src="/assets/cnn/pooling.png" style="border:none;">
</div>


#### Filters
Apply k filters:
<div class="imgcap">
<img src="/assets/cnn/filter_m.png" style="border:none;width:70%">
</div>






<div class="imgcap">
<img src="/assets/cnn/convolution_b1.png" style="border:none;">
</div>

<div class="imgcap">
<img src="/assets/cnn/convolution_b2.png" style="border:none;">
</div>

#### Convolutional pyramid


<div class="imgcap">
<img src="/assets/cnn/cnn3d.png" style="border:none;">
</div>

<div class="imgcap">
<img src="/assets/cnn/cnn3d2.png" style="border:none;">
</div>

#### Filter



#### Spatial dimension vs depth

<div class="imgcap">
<img src="/assets/cnn/cnn3d3.png" style="border:none;">
</div>

<div class="imgcap">
<img src="/assets/cnn/cnn3d4.png" style="border:none;">
</div>

### Fully connected network

<div class="imgcap">
<img src="/assets/cnn/cnn3d5.png" style="border:none;">
</div>


<div class="imgcap">
<img src="/assets/cnn/cnn3d6.png" style="border:none;">
</div>
