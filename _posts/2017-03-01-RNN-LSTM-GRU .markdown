---
layout: post
comments: true
mathjax: true
title: “RNN, LSTM and GRU tutorial”
excerpt: “This tutorial covers the RNN, LSTM and GRU networks that are widely popular for deep learning in NLP.”
date: 2017-03-01 12:00:00
---
**This is work in progress... The content needs major editing.**

### Recurrent Neural Network (RNN)

If Convolution networks are deep networks for images, recurrent networks are the networks for the time sequence data. LSTM and GRU networks are popular for the natural language processing (NLP). But before discussing those networks, we cover the basis a Recurrent neural network first.

In a fully connected network, we model h as a function of X 
$$
f(X_i)
$$

For time sequence data, RNN has an extra parameter 
$$
h_{t-1}
$$
which is the output in the previous time step.

<div class="imgcap">
<img src="/assets/rnn/rnn_b.png" style="border:none;width:60%;">
</div>

So at time step t, we take the output at t-1 and input at t to compute on next prediction
$$
h(t)
$$

$$
h_t = f(x_t, h_{t-1})
$$

<div class="imgcap">
<img src="/assets/rnn/rnn_b3.png" style="border:none;width:40%;">
</div>

For example, we can unroll a RNN network from time step t-1 to t+1 below:
<div class="imgcap">
<img src="/assets/rnn/rnn_b2.png" style="border:none;width:40%;">
</div>

#### Create image caption using RNN
Consider we want to use deep learning to create captions for an image.
<div class="imgcap">
<img src="/assets/rnn/cap.png" style="border:none;">
</div>

<div class="imgcap">
<img src="/assets/rnn/cap2.png" style="border:none;;">
</div>

<div class="imgcap">
<img src="/assets/rnn/cap3.png" style="border:none;;">
</div>

<div class="imgcap">
<img src="/assets/rnn/cap4.png" style="border:none;;">
</div>

<div class="imgcap">
<img src="/assets/rnn/cap5.png" style="border:none;;">
</div>

<div class="imgcap">
<img src="/assets/rnn/cap6.png" style="border:none;;">
</div>

<div class="imgcap">
<img src="/assets/rnn/cap7.png" style="border:none;;">
</div>

<div class="imgcap">
<img src="/assets/rnn/cap8.png" style="border:none;;">
</div>

<div class="imgcap">
<img src="/assets/rnn/cap9.png" style="border:none;;">
</div>

<div class="imgcap">
<img src="/assets/rnn/cap10.png" style="border:none;;">
</div>

<div class="imgcap">
<img src="/assets/rnn/cap11.png" style="border:none;;">
</div>



```python
input_dim   = 512   # CNN feature dimension: 512  
hidden_dim  = 512   # Hidden state dimension: 512
wordvec_dim = 256  			   
```

```python
# W_proj: (input_dim, hidden_dim)
W_proj  = np.random.randn(input_dim, hidden_dim)
W_proj /= np.sqrt(input_dim)
b_proj  = np.zeros(hidden_dim)
```

```python
# Initialize CNN -> hidden state projection parameters
# h0: (N, hidden_dim)
h0 = features.dot(W_proj) + b_proj
```

```python
W_embed  = np.random.randn(vocab_size, wordvec_dim)
W_embed /= 100
```

```python
# captions:    (N, 17) each contains a word index (0 to (vocab_size 1004 - 1))
# captions_out (N, 16)
# mask:        (N, 16)

# W_embed (vocab_size, wordvec_dim)
# captions_in: (N, 16) each captions_in contain at most 16 words.
# x: (N, 16, wordvec_dim)
x, cache_embed = word_embedding_forward(captions_in, W_embed)
```

```python
# h: (N, 16, hidden_dim)
# Wx: (wordvec_dim, hidden_dim)
# Wh: (hidden_dim, hidden_dim)
h, cache_rnn = rnn_forward(x, h0, Wx, Wh, b)
```

```python
# W_vocal: (hidden_dim, vocab_size 1004)
# scores: (N, 16, vocab_size 1004)
scores, cache_scores = temporal_affine_forward(h, W_vocab, b_vocab)
loss, dscores = temporal_softmax_loss(scores, captions_out, mask)
```

### Long Short Term Memory network (LSTM)

### Gated Recurrent Units (GRU)
