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

If Convolution networks are deep networks exploring the spatial information, recurrent networks are the correspnding network for the time sequence data. This type of networks are popular with natural language processing (NLP).

<div class="imgcap">
<img src="/assets/rnn/mnist.png" style="border:none; width:40%;">
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
