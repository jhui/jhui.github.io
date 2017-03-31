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
<img src="/assets/cnn/filter_m.png" style="border:none;width:70%">
</div>

#### Pooling

To reduce the spatial dimension of a feature map, we apply maximum pool. A 2x2 maximum pool replace a 2x2 area by its maximum. After apply a 2x2 pool, we reduce the spatial dimension of a 4x4 input to a 2x2 output.
<div class="imgcap">
<img src="/assets/cnn/pooling.png" style="border:none;width:50%">
</div>

Here, we construct a CNN using convolution and pooling:
<div class="imgcap">
<img src="/assets/cnn/conv_layer2.png" style="border:none;width:50%">
</div>

Pooling is often used with a convolution layer. Therefore, we often consider it as part of the convolution layer rather than a separate layer. Other pooling like average pooling can be applied. However, for image classification, maximum pooling is more common.

### Multiple convolution layers

Like deep learning, the depth of the network increases the complexity of a model. A CNN network usually composes of many convolution layers. 
<div class="imgcap">
<img src="/assets/cnn/convolution_b1.png" style="border:none;width:70%">
</div>

The CNN above composes of 3 convolution layer. We start with a 32x32 pixel image with 3 channels (RGB). We first apply a 3x4 filters and a 2x2 max pooling. The output of this layer will be a 16x16x4 feature maps.  Here are the output dimension for each convolution layer:
<div class="imgcap">
<img src="/assets/cnn/cnn_chanl.png" style="border:none">
</div>

### Fully connected layers
After using convolution layers to extract the spatial features of an image, we apply fully connected layers for the final classification. First we flatten the output of the convolution layers. For example, if the final features maps have a dimension of 4x4x512, we will flaten it to an array of 4096 elements. We apply 2 more hidden layers here before we perform the final classification.

<div class="imgcap">
<img src="/assets/cnn/convolution_b2.png" style="border:none;">
</div>

### Convolutional pyramid

For each convolution layer, we reduce the spatial dimension while increase the depth of the feature maps. We call this convolutional pyramid 

<div class="imgcap">
<img src="/assets/cnn/cnn3d.png" style="border:none;">
</div>

Here, we reduce the spatial dimension of each convolution layer usually through pooling or with filter stride size greater that 1.
<div class="imgcap">
<img src="/assets/cnn/cnn3d4.png" style="border:none;width:50%">
</div>

While increase the depth of the feature map. The depth can be increased by the number of filters used.
<div class="imgcap">
<img src="/assets/cnn/cnn3d2.png" style="border:none;">
</div>

#### Google inceptions

In our previous discussion, the convolution filter in each layer is of the same size say 3x3. For GoogleNet, Google applies different size of filters to the input and concantente the feature maps together to increase the depth.

Here we have a 3x3 and a 1x1 filter. The first way generate 8 fetures map while the second one generate 2. We can concantentate them to form maps of depth 10. The inception idea is to increase the depth of the feature map by concantentate feature maps using different size of convolution filters and pooling. 
<div class="imgcap">
<img src="/assets/cnn/inception.png" style="border:none;depth:50%">
</div>

#### Non-linearity and optimization
Inceptions can be consider as one way to introduce non-linearity into the system. In many CNN, we apply similar layers we learned from deep learning after the convolution filters. This includes batch normalization and/or ReLU.

#### Fully connected network

After exploring the spatial relationship, we flatten the convolution layer output and connect it to a fully connected network:

<div class="imgcap">
<img src="/assets/cnn/cnn3d5.png" style="border:none;width:50%">
</div>

<div class="imgcap">
<img src="/assets/cnn/cnn3d6.png" style="border:none;width:50%">
</div>
