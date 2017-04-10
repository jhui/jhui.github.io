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

Selective attention restricts our attention to particular objects in the environment. It helps us focus, so we can tune out irrelevant information and concentrate on what really matters. For example, when we cross a busy street, our attention is on avoiding hitting people. To transcript the following pictures, we focus on the people who is walking towards my direction and the relevant details. One possible transcription is "A man holding a couple plastic container is walking down a street towards me." Once, we have a visual fixation on the man, we explore other related details including what is he doing, where is he.

<div class="imgcap">
<img src="/assets/att/attention.jpg" style="border:none;;">
</div>

We can model the image captioning problem roughly as: 

$$
\text{next word} = f(\text{last word}, image)
$$

With a RNN, we can rewrite the model more precisely as:

$$
h_{t} = f(h_{t-1}, x_{i})
$$

which $$ x_{i} $$ is the image and $$ h_{t} $$ is the hidden state to generate the "next word". As we learn from selective attention, this model is over generalized, and may be more effective to predict the next word in a caption by focusing a smaller region in the image.

$$
h_{t} = f(h_{t-1}, attention(h_{t-1}, x_{i}) )
$$

 




