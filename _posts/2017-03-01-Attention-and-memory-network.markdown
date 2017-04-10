---
layout: post
comments: true
mathjax: true
title: “Soft & hard attention and memory network ”
excerpt: “How to use attention to improve deep network learning, and use memory network for Q&A?”
date: 2017-03-01 12:00:00
---
**This is work in progress... The content needs major editing.**

### Attention

In cognitive psychology, selective attention restricts our attention to particular objects in the environment. It helps us focus, so we can tune out irrelevant information and concentrate on what really matters. For example, when we cross a busy street, our attention is to avoid hitting people. To transcript the following picture, we concentrate on the people walking towards my direction. One possible transcript is "A man holding a couple plastic container is walking down a street towards me." Once, we have a visual fixation on the man, we continue exploring other relevant details including what is he doing, where is he.

<div class="imgcap">
<img src="/assets/att/attention.jpg" style="border:none;;">
</div>


We have the caption to start with a "start" token, we can model the next word in the caption roughly as: 

$$
\text{next word} = f(image, \text{last word})
$$

Applying the RNN techniques, we can rewrite the model more precisely as:

$$
h_{t} = f(x, h_{t-1})
$$

$$
y_{t} = g(h_{t})
$$


which $$ x $$ is the image and $$ h_{t} $$ is the hidden state to predict the "next word" $$ y_{t} $$ at time step $$ t $$. As we learn from the selective attention, this model is over generalized, and can be more effective in focusing on a smaller region.

$$
h_{t} = f(attention(x, h_{t-1}), h_{t-1} )
$$

which $$ attention $$ is a function to compute where should we focus based on the image and the last word.

### Image caption model with LSTM

Let's have a quick review of image caption using LSTM. We use a CNN to extract the features $$ x $$ of an image, and feed it to every LSTM cells. To make prediction, each LSTM cell takes in the hidden state from the previous time step $$ h_{t-1} $$ and the image features $$ x $$ to make a hidden state $$ h_{t} $$. We pass $$ h_{t} $$ to say an affine operation to make a prediction of the caption word $$ y_{t} $$. 

<div class="imgcap">
<img src="/assets/att/rnn.png" style="border:none;;">
</div>

### Attention

The key difference between a LSTM model and the one with attention is that we are not paying attention to the whole picture. Instead, we focus on selective areas under the current context. For example, at the beginning of the caption creation, we start with an empty context and with our first attention at the man walking towards us. Next, we update the context to be "a man", and pay attention to a local area to find what is he holding. By continue exploring the areas in focus and updating the context, we generate the image caption. Mathematically, we are trying to replace the LSTM model,

$$
h_{t} = f(x, h_{t-1})
$$

with an attention module $$attention$$:

$$
h_{t} = f(attention(x, h_{t-1}), h_{t-1} )
$$

The attention module have 2 major inputs:
* a context, and
* image features in each localized areas.

For the context, we use the hidden state $$ h_{t-1} $$ of the previous time step as our context. For image features, instead of feeding the whole image to the module, we have images divided into regions. In a LSTM system, we process an image with CNN and use one of the output from the fully connected layer as images features. Nevertheless, we loss the spatial locality information. For attention, we use the output of one of the convolution layer which spatial information is preserved.

<div class="imgcap">
<img src="/assets/att/cnn3d2.png" style="border:none;;">
</div>

Here, the feature maps in the second convolution layers are divided into 4 which closely resemble the top right & left and the bottom right & left of the original pictures. In the following diagram, we replace the image $$ x $$ as an input to a LSTM model with an attention module with the context $$ h_{t-1} $$ and 4 regions of images $$ (x_1, x_2, x_3, x_{4}) $$ from the CNN.

<div class="imgcap">
<img src="/assets/att/att2.png" style="border:none;;">
</div>

<div class="imgcap">
<img src="/assets/att/context.png" style="border:none;;">
</div>

Here, for completness, we put back the attention modules back into the LSTM models.
<div class="imgcap">
<img src="/assets/att/att3.png" style="border:none;;">
</div>

### Soft attention

A soft attention identifies areas of focus by finding the weight $$ \alpha $$ of attention that the attention module focus on in each region. For example, we multiple $$ \alpha $$ with the image to display the area of focus and the predicted caption word at that time step.

<div class="imgcap">
<img src="/assets/att/attention2.png" style="border:none;;">
</div>

$$
s = \tanh(W_{c} C + W_{x} X )
$$

$$
\alpha = softmax(s)
$$

<div class="imgcap">
<img src="/assets/att/soft.png" style="border:none;;">
</div>

### Hard attention

<div class="imgcap">
<img src="/assets/att/hard.png" style="border:none;;">
</div>






