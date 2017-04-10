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
\text{next word} = f(\text{last word}, image)
$$

Applying the RNN techniques, we can rewrite the model more precisely as:

$$
h_{t} = f(x, h_{t-1})
$$

which $$ x $$ is the image and $$ h_{t} $$ is the hidden state to predicit the "next word" at time step $$ t $$. As we learn from the selective attention, this model is over generalized, and can be more effective in focusing on a smaller region.

$$
h_{t} = f(h_{t-1}, attention(x, h_{t-1}) )
$$

which $$ attention $$ is a network to compute where should we focus based on the image and the last word.

### Image caption model with RNN

Let's have a quick review of image caption using RNN. We use a CNN to extract the feature $$ x $$ of an image, and feed it to every LSTM cells. To make prediction, each LSTM cell takes in the hidden state from the previous time step $$ h_{t-1} $$ and the image featuers $$ x $$ to make a hidden state $$ h_{t} $$. We pass $$ h_{t} $$ to say an affine operation to make a prediction of the caption word.


<div class="imgcap">
<img src="/assets/att/rnn.png" style="border:none;;">
</div>





