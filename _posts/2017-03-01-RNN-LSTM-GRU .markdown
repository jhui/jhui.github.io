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

If Convolution networks are deep networks for images, recurrent networks are the networks for the time sequence data, like speeh or natural language. For example, the more advanced LSTM and GRU networks are popular for the natural language processing (NLP). But before discussing LSTM or GRU, we will look into a simplier network called Recurrent neural netwok (RNN).

In a fully connected network, we model h as 
$$
h = f(X_i)
$$
.

For time sequence data, besides the current input, we remember the output from the last time step to make a prediction.

$$
h_t = f(x_t, h_{t-1})
$$

<div class="imgcap">
<img src="/assets/rnn/rnn_b.png" style="border:none;width:45%;">
</div>

So at time step t, we take both the output at 
$$
t-1 
$$
and input at 
$$
t
$$ 
to make the next prediction
$$
h_t
$$
<div class="imgcap">
<img src="/assets/rnn/rnn_b3.png" style="border:none;width:35%;">
</div>

For example, the following diagram unroll a RNN for time step 
$$
t-1
$$ 
 to 
$$
t+1
$$
:
<div class="imgcap">
<img src="/assets/rnn/rnn_b2.png" style="border:none;width:60%;">
</div>

#### Create image caption using RNN
How to create captions for an image? For example, we may input a school bus image into the RNN and expect it to create a caption like:
<div class="imgcap">
<img src="/assets/rnn/cap.png" style="border:none;">
</div>
During the training, we
1. Use a CNN network to capture features of an image.
2. Multiple the features with a matrix to generate
$$
h_0
$$
3. Feed 
$$ 
h_0
$$
to the RNN.
4. Use a word vector to convert a word to a vector.
5. Feed the word vector and
$$
h_0
$$ to the RNN.
$$
h_1 = f(X_1, h_0)
$$
6. Use a matrix to map 
$$
h
$$
to the final predicted word.
7. Move to the next time step with 
$$
h_1
$$ 
and the last predicted word as input.

<div class="imgcap">
<img src="/assets/rnn/cap12.png" style="border:none;;">
</div>

#### Capture image features
We pass the image into a CNN and use one of the activation layer in the fully connected (FC) network to initialize the RNN. For example, in the picture below, we pick the output of the first FC layer which has a shape of (512,) as the features used to create captions.

<div class="imgcap">
<img src="/assets/rnn/cnn.png" style="border:none;;">
</div>

We multiple the CNN features with a matrix to compute
$$
h_0
$$
.

<div class="imgcap">
<img src="/assets/rnn/cap2.png" style="border:none;">
</div>

We will use 
$$
h_0
$$
for the RNN to compute
$$
h_1 = f(h_0, X_0)
$$

<div class="imgcap">
<img src="/assets/rnn/cap8.png" style="border:none;width:80%;">
</div>

The shape of CNN features (N, 512) and h (N, 512) which N is the batch size:
```python
input_dim   = 512   # CNN features dimension: 512  
hidden_dim  = 512   # Hidden state dimension: 512
```

Define the matrix to project the CNN features to 
$$
h_0
$$
.

```python
# W_proj: (input_dim, hidden_dim)
W_proj  = np.random.randn(input_dim, hidden_dim)
W_proj /= np.sqrt(input_dim)
b_proj  = np.zeros(hidden_dim)
```

Compute
$$
h_0
$$
.
```python
# Initialize CNN -> hidden state projection parameters
# h0: (N, hidden_dim)
h0 = features.dot(W_proj) + b_proj
```

#### Map words to RNN
In our training data, it contains both the images and captions. It also have a dictionary which map a vocabulary word to an integer. Words are stored as a word index in the training dataset. For example, the caption "A yellow school bus idles near a park" may stored as "1 5 3401 3461 78 5634 87 5 111 2" which 1 represents start of a caption, 5 represents 'a', 3401 represents 'yellow' etc... 

The RNN does not use the word index directly. Instead, through an word embedding lookup table, the word index is converted to a vector of wordvec_dim. The RNN will take this vector and 
$$
h_{t-1}
$$ to compute
$$
h_t
$$

<div class="imgcap">
<img src="/assets/rnn/encode.png" style="border:none;width:70%;">
</div>


We will create a lookup table to
<div class="imgcap">
<img src="/assets/rnn/cap9.png" style="border:none;;">
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


We mulitiply 
$$
h
$$ 
with another matrix to generate a score for each word in the vocabulary. This fit into a softmax function to calculate the loss in the training or to make prediction.


<div class="imgcap">
<img src="/assets/rnn/cap11.png" style="border:none;;">
</div>


#### Making prediction

We will use the CNN to generate features for the image.
<div class="imgcap">
<img src="/assets/rnn/cap4.png" style="border:none;width:80%;">
</div>

At time step 1, we feed the RNN with the word "start". The RNN computes the value 
$$
h_1
$$
which will multiply with 
$$
W_{vocab}
$$
to generate score for each word in the vocabulary (1004). We will make the first word prediction by select the one with the highest score (i.e. "A"). At time step 2, we will fit "A" into the time step 2. With 
$$
h_1
$$ 
computed at time step 1, we then made the second preduction "bus".
```python
scores, _ = affine_forward(next_h, W_vocab, b_vocab)
captions[:, t] = scores.argmax(axis=1)
prev_word = captions[:, t].reshape(N, 1)
```
	
<div class="imgcap">
<img src="/assets/rnn/cap7.png" style="border:none;;">
</div>

```python
input_dim   = 512   # CNN features dimension: 512  
hidden_dim  = 512   # Hidden state dimension: 512
wordvec_dim = 256  			   
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
