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
Filters are frequently apply to images for different purposes. Our visual system applies edge detection filters to recognize object.

<div class="imgcap">
<img src="/assets/cnn/edge.png" style="border:none;">
</div>

For example, to blur an image, we can apply a 3x3 filter as follows:
<div class="imgcap">
<img src="/assets/cnn/filter_b.png" style="border:none;">
</div>

### Convolution neural network (CNN)
A convolution neural network compose of convolution layers and fully connected layers.

#### Convolution layers

The convolution layer applies filters to the previous layer and sub-sampling the layer by maximum pool to reduce the spatial dimension. In the example below, we applies k filters to the original images and then reduce it spatial dimension by half using a 2x2 maximum pool.
<div class="imgcap">
<img src="/assets/cnn/conv_layer2.png" style="border:none;width:50%">
</div>

#### Filters
Apply k filters:
<div class="imgcap">
<img src="/assets/cnn/filter_m.png" style="border:none;width:70%">
</div>

Apply maximum pool for sub-sampling:
<div class="imgcap">
<img src="/assets/cnn/pooling.png" style="border:none;">
</div>





<div class="imgcap">
<img src="/assets/cnn/convolution_b1.png" style="border:none;">
</div>

<div class="imgcap">
<img src="/assets/cnn/convolution_b2.png" style="border:none;">
</div>

#### Convolutional pyramid
<div class="imgcap">
<img src="/assets/cnn/conv_layer.png" style="border:none;">
</div>


<div class="imgcap">
<img src="/assets/cnn/cnn3d.png" style="border:none;">
</div>

<div class="imgcap">
<img src="/assets/cnn/cnn3d2.png" style="border:none;">
</div>

#### Filter

<div class="imgcap">
<img src="/assets/cnn/padding.png" style="border:none;">
</div>


<div class="imgcap">
<img src="/assets/cnn/stride.png" style="border:none;">
</div>



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
