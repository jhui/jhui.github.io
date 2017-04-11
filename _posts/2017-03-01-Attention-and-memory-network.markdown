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

In cognitive science, selective attention restricts our attention to particular objects in the environment. It helps us focus, so we can tune out irrelevant information and concentrate on what really matters. For example, when we cross a busy street, our attention is to avoid hitting people. To transcript the following picture, one possibility is "A man holding a couple plastic containers is walking down a street towards me." Once, we have a visual fixation, we continue exploring other relevant details including what is he doing, where is he.

<div class="imgcap">
<img src="/assets/att/attention.jpg" style="border:none;;">
</div>


We generate an image caption starting with a "start" token. We predict the next word in the caption with the following formula: 

$$
\text{next word} = f(image, \text{last word})
$$

Applying the RNN techniques, we rewrite the model more precisely as:

$$
h_{t} = f(x, h_{t-1})
$$

$$
\text{next word} = y_{t} = g(h_{t})
$$


which $$ x $$ is the image and $$ h_{t} $$ is the hidden state to predict the "next word" $$ y_{t} $$ at time step $$ t $$. As we learn from the selective attention, this model is over generalized, and can be more effective if we replace the image with more focused information.

$$
h_{t} = f(attention(x, h_{t-1}), h_{t-1} )
$$

which $$ attention $$ is a function to generate more focus image features.

### Image caption model with LSTM

Let's have a quick review of the image caption using LSTM. We use a CNN to extract the features $$ x $$ of an image, and feed it to every LSTM cells. Each LSTM cell takes in the hidden state $$ h_{t-1} $$ from the previous time step and the image features $$ x $$ to calculate a new hidden state. We pass $$ h_{t} $$ to say an affine operation to make a prediction on the next caption word $$ y_{t} $$. 

<div class="imgcap">
<img src="/assets/att/rnn.png" style="border:none;width:80%;">
</div>

### Attention

The key difference between a LSTM model and the one with attention is that attention pays attention to specific area. For example, at the beginning of the caption creation, we start with an empty context. Our first attention is on the man walking towards us, and we make the first word prediction "A". We update the context to be "A" with a continous focus on the man. Once we predict the second word as "man", we move our attention to what he is holding. By continue exploring the attention area and updating the context, we generate an image caption. Mathematically, we are trying to replace the image $$ x $$ in LSTM model,

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

For the context, we use the hidden state $$ h_{t-1} $$ of the previous time step as our context. To generate image features, instead of feeding the whole image to the module, we have images divided into regions. In a LSTM system, we process an image with CNN and use one of the fully connected layer output as input features. Nevertheless, this is not adequate for attention since some spatial information has lost. Instead, we use the output of one of the convolution layer which spatial information is still preserved.

<div class="imgcap">
<img src="/assets/att/cnn3d2.png" style="border:none;;">
</div>

Here, the feature maps in the second convolution layers are divided into 4 which closely resemble the top right & left and the bottom right & left of the original pictures. In the following diagram, we replace the image $$ x $$ as an input to a LSTM model with an attention module.

<div class="imgcap">
<img src="/assets/att/att2.png" style="border:none;width:80%;">
</div>

The attention module takes the context $$ h_{t-1} $$ and 4 regions of images $$ (x_1, x_2, x_3, x_{4}) $$ from the CNN to comput the new image features used by the LSTM.
<div class="imgcap">
<img src="/assets/att/context.png" style="border:none;;">
</div>

The following is the complete flow of the LSTM model using attentions.
<div class="imgcap">
<img src="/assets/att/att3.png" style="border:none;;">
</div>

### Soft attention

We can implement attention with soft attention or hard attention. In soft attention, instead of using the image as an input to the LSTM, we input a weighted image account for attention. Soft attention computes weights to be multiplied with input features. If we plot those weighted features, we will visualize areas with high attention. For example, with the context $$h_{t-1} $$ representing "A man holding a couple plastic", soft attention highlights the plastic container area to predict the word "container".

<div class="imgcap">
<img src="/assets/att/attention2.png" style="border:none;;">
</div>

With our CNN outputs $$ x_1, x_2, x_3 \text{ and } x_4 $$, each feature map covers a sub-section of an image. With $$ x_i $$, we compute a score $$ s_{i} $$ to measure the attention level under the current context:

$$
s_{i} = \tanh(W_{c} C + W_{x} X_{i} ) = \tanh(W_{c} h_{t-1} + W_{x} x_{i} )
$$

We pass $$ s_{i} $$ to a softmax to normalize it. This becomes a weight $$ \alpha $$ to measure the attention relative to each other.

$$
\alpha_i = softmax(s_1, s_2, \dots, s_{n}, \dots)
$$

With softmax, $$ \alpha_{i} $$ adds up to 1 and therefore we can use it to compute a weighted average $$ x_{i} $$ to replace $$ x $$. Since the weight $$ \alpha_{i} $$ is higher in high attention area, $$ Z $$ is more "focus" that $$ x $$ as a LSTM input.

$$
Z = \sum_{i} \alpha_{i} x_{i}
$$

<div class="imgcap">
<img src="/assets/att/soft.png" style="border:none;;">
</div>

### Hard attention

In soft attention, we compute a weight $$ \alpha_{i} $$ for each $$ x_{i}$$, and use it to calculate a weighted average of $$ x $$ as the input to the LSTM module. $$ \alpha_{i} $$ adds up to 1 which can also be interpreted as the chance that $$ x_{i} $$ covers our current "attention area". So instead of a weighted average, hard attention uses $$ s_{i} $$ as a sample rate to pick $$ x_{i} $$ as the input to LSTM. Hard attention replaces a deterministic method with a stochastic sampling model. To calculate the gradient descent correctly in the backpropagation, we need to perform many samplings and average out our results using the Monte Carlo method. Monte Carlo performs many end-to-end episodes to compute an average for all sampling results. Soft attention assumes a weighted average is a good approximation to our "attention objects" while hard attention makes no such assumptions but requires a lot of samplings to make it accurate. 

> Soft attention is more popular because the backpropagation seems more effective.

<div class="imgcap">
<img src="/assets/att/hard.png" style="border:none;">
</div>






