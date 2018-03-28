---
layout: post
comments: true
mathjax: true
priority: -750
title: “Fast R-CNN and Faster R-CNN”
excerpt: “Object detection using Fast R-CNN and Faster R-CNN.”
date: 2017-03-15 11:00:00
---

I am retiring this page now. 

• If you are interested in Fast R-CNN, Faster R-CNN & FPN, we have more updated information at [https://medium.com/@jonathan_hui/what-do-we-learn-from-region-based-object-detectors-faster-r-cnn-r-fcn-fpn-7e354377a7c9].
* If you are interested in single shot object detector like SSD and YOLO, please visit [https://medium.com/@jonathan_hui/what-do-we-learn-from-single-shot-object-detectors-ssd-yolo-fpn-focal-loss-3888677c5f4d].


### Object detection

Detecting objects and their locations are critical for many Artificial Intelligence (AI) applications. For example, in autonomous driving, it is important to realize what objects are in front of us, as well as identifying traffic lights and road signs. The license plate reader in a police car alerts the authority of stolen cars by locating the license plates and applying visual recognition. The following picture illustrates some of the objects detected with the corresponding boundary box and estimated certainty. **Object detection is about classifying objects and define a bounding box around them.** 

<div class="imgcap">
<img src="/assets/rcnn/img.jpg" style="border:none;">
</div>

### Regions with CNN features (R-CNN)

> The information in R-CNN and the corresponding diagrams are based on the paper [Rich feature hierarchies for accurate object detection and semantic segmentation, Ross Girshick et al.](https://arxiv.org/pdf/1311.2524.pdf)

#### R-CNN overview
<div class="imgcap">
<img src="/assets/rcnn/rcnn.png" style="border:none;width:80%">
</div>

With an image, we:
1. Extract region proposals: Use a region-extraction algorithm to propose about 2,000 objects' boundaries.
1. For each region proposal,
	1. Warp it to a size fitted for the CNN.
	1. Compute the CNN features.
	1. Classify what is the object in this region.

#### Use Selective search for region proposals

> The information in selective search is based on the paper [Segmentation as Selective Search for Object Recognition, van de Sande et al.](https://www.koen.me/research/pub/vandesande-iccv2011.pdf)

In selective search, we start with many tiny initial regions. We use a greedy algorithm to grow a region. First we locate two most similar regions and merge them together. Similarity $$S$$ between region $$a$$ and $$b$$ is defined as:

$$
S(a, b) =  S_{texture}(a, b) + S_{size} (a, b).
$$

where $$ S_{texture}(a, b) $$ measures the visual similarity, and $$S_{size} $$ prefers merging smaller regions together to avoid a single region from gobbling up all others one by one.

We continue merging regions until everything is combined together. In the first row, we show how we grow the regions, and the blue rectangles in the second rows show all possible region proposals we made during the merging. The green rectangle are the target objects that we want to detect.

<div class="imgcap">
<img src="/assets/rcnn/select.png" style="border:none;">
</div>
(Image source: van de Sande et al. ICCV'11)

#### Warping 

For every region proposal, we use a CNN to extract the features.  Since a CNN takes a fixed-size image, we wrap a proposed region into a 227 x 227 RGB images. 

<div class="imgcap">
<img src="/assets/rcnn/rcnn3.jpg" style="border:none;width:40%">
</div>

#### Extracting features with a CNN

This will then process by a CNN to extract a 4096-dimensional feature:

<div class="imgcap">
<img src="/assets/rcnn/rcnn2.jpg" style="border:none;width:60%">
</div>

#### Classification

We then apply a SVM classifier to identify the object:

<div class="imgcap">
<img src="/assets/rcnn/rcnn4.jpg" style="border:none;width:50%">
</div>

#### Putting it together

<div class="imgcap">
<img src="/assets/rcnn/rcnnf.png" style="border:none;width:80%">
</div>

#### Bounding box regressor

The original boundary box proposal may need further refinement. We apply a regressor to calculate the final red box from the initial blue proposal region.

<div class="imgcap">
<img src="/assets/rcnn/c2.png" style="border:none;width:40%">
</div>

<div class="imgcap">
<img src="/assets/rcnn/bound2.jpg" style="border:none;width:80%">
</div>

Here, the R-CNN classifies objects in a picture and produces the corresponding boundary box.
<div class="imgcap">
<img src="/assets/rcnn/bound.png" style="border:none;width:80%">
</div>

### SPPnet & Fast R-CNN

> The information and diagrams in this section come from "Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition", Kaiming He et al for SPPnet and "Fast CNN", Ross Girshick (ICCV'15) for Fast R-CNN

**R-CNN is slow in training & inference.** We have 2,000 proposals which each of them needed to be processed by a CNN to extract features, and therefore, R-CNN will repeat the ConvNet 2,000 times to extract features.
<div class="imgcap">
<img src="/assets/rcnn/bound3.png" style="border:none;width:60%">
</div>

The feature maps in a CNN convolution layer represent spatial features in an image. Why don't we generate the proposals from the feature maps instead? 

<div class="imgcap">
<img src="/assets/rcnn/st1.png" style="border:none;width:70%">
</div>

Hence, instead of converting 2,000 regions into the corresponding features maps, we convert the whole image once and generate regional proposals from them.
<div class="imgcap">
<img src="/assets/rcnn/frcnn.png" style="border:none">
</div>

#### SPPnet

Here is the visualization of the features maps in a CNN.
<div class="imgcap">
<img src="/assets/rcnn/map.png" style="border:none;width:50%">
</div>

SPPnet uses a regional proposal method to generate region of interests (**RoIs**). The blue rectangular below shows one possible region of interest:
<div class="imgcap">
<img src="/assets/rcnn/map2.png" style="border:none;width:60%">
</div>

> Both papers do not restrict themselves to any region proposal methods.

Here we warp region of interests (RoIs) into spatial pyramid pooling (SPP) layers.
<div class="imgcap">
<img src="/assets/rcnn/sp2.png" style="border:none;width:70%">
</div>

Each spatial pyramid layer is in a different scale, and we use maximum pooling to warp the original RoI to the target map on the right.
<div class="imgcap">
<img src="/assets/rcnn/pyr.png" style="border:none;width:50%">
</div>

We pass it to a fully-connected network, and use a SVM for classification and a linear regressor for the bounding box.
<div class="imgcap">
<img src="/assets/rcnn/spp3.png" style="border:none;width:70%">
</div>

#### Fast R-CNN

Same as SPPnet, we use the features maps at the CNN layer "conv5" for region proposals. However, instead of generating a pyramid of layers, Fast R-CNN warps ROIs into one single layer using the RoI pooling.
<div class="imgcap">
<img src="/assets/rcnn/st2.png" style="border:none;width:70%">
</div>

The RoI pooling layer uses max pooling to convert the features in a region of interest into a small feature map of H × W. Both H & W (e.g., 7 × 7) are tunable hyper-parameters. 

You can consider Fast R-CNN is a special case of SPPNet. Instead of multiple layers, Fast R-CNN only use one layer.
<div class="imgcap">
<img src="/assets/rcnn/pyr2.png" style="border:none;width:50%">
</div>

It is feed into a fully-connected network for classification using linear regression and softmax. The bounding box is further refined with a linear regression.
<div class="imgcap">
<img src="/assets/rcnn/st4.png" style="border:none;width:70%">
</div>

<div class="imgcap">
<img src="/assets/rcnn/pool.png" style="border:none;width:70%">
</div>

The key difference between SPPnet and Fast R-CNN is that SPPnet cannot update parameters below SPP layer during training:
<div class="imgcap">
<img src="/assets/rcnn/sp4.png" style="border:none;width:70%">
</div>

In Fast R-CNN, all parameters including the CNN can be trained together. All the parameters are trained together with a log loss function from the class classification and a L1 loss function from the boundary box prediction. 
<div class="imgcap">
<img src="/assets/rcnn/st6.png" style="border:none;width:70%">
</div>

### Faster R-CNN

> The information and some diagrams in this section are based on the paper [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks, Shaoqing Ren etc al](https://arxiv.org/pdf/1506.01497.pdf)

Both SPPnet and Fast R-CNN requires a region proposal method. 
<div class="imgcap">
<img src="/assets/rcnn/st7.png" style="border:none;width:70%">
</div>

The difference between Fast R-CNN and Faster R-CNN is that we do not use a special region proposal method to create region proposals. Instead, we train a region proposal network that takes the feature maps as input and outputs region proposals. These proposals are then feed into the RoI pooling layer in the Fast R-CNN.

<div class="imgcap">
<img src="/assets/rcnn/st8.png" style="border:none;">
</div>

The region proposal network is a convolution network. The region proposal network uses the feature map of the "conv5" layer as input. It slides a 3x3 spatial windows over the features maps with depth K. For each sliding window, we output a vector with 256 features. Those features are feed into 2 fully-connected networks to compute:
* 2 scores representing how likely it is an object or non-object/background.
* A boundary box.

<div class="imgcap">
<img src="/assets/rcnn/anchor1.png" style="border:none;width:70%">
</div>

We then feed the region proposals to the RoI layer of the Fast R-CNN.
<div class="imgcap">
<img src="/assets/rcnn/space.png" style="border:none;width:50%">
</div>

#### Multiple anchor boxes

In the example above, we generate 1 proposal per sliding window.  The region proposal network can scale or change the aspect ratio of the window to generate more proposals. In Faster R-CNN, 9 anchor boxes (on the right) are generated per anchor. 
<div class="imgcap">
<img src="/assets/rcnn/anchor3.png" style="border:none;width:75%">
</div>

### Implementations

The implementations of Faster R-CNN can be found at:
* [Ross Girshick Github](https://github.com/rbgirshick/py-faster-rcnn) and
* [Shaoqing Ren Github](https://github.com/ShaoqingRen/faster_rcnn)





