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
<img src="/assets/cnn/ppl.png" style="border:none;">
</div>





### Convolution neural netword (CNN)
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
<img src="/assets/cnn/filter.png" style="border:none;">
</div>

<div class="imgcap">
<img src="/assets/cnn/padding.png" style="border:none;">
</div>

<div class="imgcap">
<img src="/assets/cnn/filter_m.png" style="border:none;">
</div>

<div class="imgcap">
<img src="/assets/cnn/stride.png" style="border:none;">
</div>


#### Pooling
<div class="imgcap">
<img src="/assets/cnn/pooling.png" style="border:none;">
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
