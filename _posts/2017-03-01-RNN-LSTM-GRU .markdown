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


For time sequence data, besides the current input, we remember the output from the last time step to make a prediction.

$$
h_t = f(x_t, h_{t-1})
$$

<div class="imgcap">
<img src="/assets/rnn/rnn_b.png" style="border:none;width:50%;">
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
Let's use a real example for RNN. How to create captions for an image? For example, we take a school bus image into the RNN and output a caption like:
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
4. Use a word embedding lookup table to convert a word to a vector. (a.k.a word2vec)
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
to the final predicted word say "A".
7. Move to the next time step with 
$$
h_1
$$ 
and the last predicted word "A" as input.

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
for the first time step 1.

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

Define the shape of CNN features (N, 512) and h (N, 512) which N is the batch size:
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
Our training data contains both the images and captions. It also have a dictionary which map a vocabulary word to an integer. Words in the dataset are stored as word indexes in the training dataset. For example, the caption "A yellow school bus idles near a park" may stored as "1 5 3401 3461 78 5634 87 5 111 2" which 1 represents the "start" of a caption, 5 represents 'a', 3401 represents 'yellow' etc... 

The RNN does not use the word index directly. Instead, through a word embedding lookup table
$$
W_{embed}
$$
, the word index is converted to a vector with length wordvec_dim. The RNN will take this vector
$$
X_t
$$ 
and 
$$
h_{t-1}
$$ to compute
$$
h_t
$$

<div class="imgcap">
<img src="/assets/rnn/cap9.png" style="border:none;;">
</div>

The following illustrates how a word is stored in a training dataset and convert to a word vector to be consumed by the RNN.
<div class="imgcap">
<img src="/assets/rnn/encode.png" style="border:none;width:70%;">
</div>


Here is the code to convert an input caption word to the word vector x.
```python
wordvec_dim = 256  			   
```

```python
W_embed  = np.random.randn(vocab_size, wordvec_dim)
W_embed /= 100
```

```python
# captions:    (N, 17) The caption represent "<start> A yellow school bus idles near a park <end> <null> ... <null>" represent in word index. 
# captions_in  (N, 16) The caption feed into the RNN (X) = captions without the last word.
# captions_out (N, 16) The true caption output: the caption without "<start>"

# W_embed (vocab_size, wordvec_dim)
# captions_in: (N, 16) each captions_in contain at most 16 words.
# x: (N, 16, wordvec_dim)
x, cache_embed = word_embedding_forward(captions_in, W_embed)
```

#### RNN
<div class="imgcap">
<img src="/assets/rnn/score.png" style="border:none;width:50%;">
</div>

We pass the word vector
$$
X_0
$$
into the RNN to make the first word prediction. The output of the RNN 
$$
h_1
$$
is then multipy with 
$$
W_{vocab}
$$
to generate scores for each word in the vocabulary for prediction. We compute the softmax loss with the scores with the true caption
$$
\text{captions_out} \big[ 0  \big]
$$.

<div class="imgcap">
<img src="/assets/rnn/score_1.png" style="border:none;">
</div>

The coding in computing the predicted caption and the softmax score.
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

#### Time step 0
Here is how we train with the image feature
$$
h_0
$$
and the first word 'start' and the prediction 'A'.

<div class="imgcap">
<img src="/assets/rnn/cap11.png" style="border:none;width:70%;">
</div>

#### Complete flow

After we completed the first time step.  we move onto the next time step with
$$
h_1
$$
and the next caption word 'A'. Note that, for training, we do not use our best prediction as the next input
$$
X_t
$$
. We always use the captions provided by the training set.  i.e., even the word 'A' has a very low score in our prediction, we still use the work 'A' as the next input word since the whole purpose is to optimize the RNN with the true caption at the lowest cost.

<div class="imgcap">
<img src="/assets/rnn/cap13.png" style="border:none;;">
</div>

Here is the detail complete flow in training 1 sample data.
<div class="imgcap">
<img src="/assets/rnn/cap3.png" style="border:none;;">
</div>

Here is the code for the forward feed, backpropagation and the loss.
```python
  def loss(self, features, captions):
	# For training, say the caption is "<start> A yellow bus idles near a park"
	# captions_in is the Xt input: "<start> A yellow bus idles near a"
	# captions_out is the true label: "A yellow bus idles near a park"
    captions_in = captions[:, :-1]
    captions_out = captions[:, 1:]
    
    mask = (captions_out != self._null)

	# Retrieve the trainable parameters
    W_proj, b_proj = self.params['W_proj'], self.params['b_proj']    
    W_embed = self.params['W_embed']
    Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']
    W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']
    
    loss, grads = 0.0, {}
    # vocab_size = 1004
    # T          = 16
    #
    # features    : (N, input_dim)
    # W_proj      : (input_dim, hidden_dim)
    # h0          : (N, hidden_dim)
    #
    # x           : (N, T, wordvec_dim)
    # captions_in : (N, T) of word index
    # W-embed     : (vacab_size, wordvec_dim)
    #
    # h           : (N, 16, hidden_dim)
    # Wx          : (wordvec_dim, hidden_dim)
    # Wh          : (hidden_dim, hidden_dim)
    #
    # scores      : (N, 16, vocab_size)
    # W_vocab     : (hidden_dim, vocab_size)

	# Compute h0 from the image features.
    h0 = features.dot(W_proj) + b_proj

	# Find the word vector of the input caption word.
    x, cache_embed = word_embedding_forward(captions_in, W_embed)

	# Forward feed for the RNN
    h, cache_rnn = rnn_forward(x, h0, Wx, Wh, b)

    # Compute the scores for each words in the vocabulary
    scores, cache_scores = temporal_affine_forward(h, W_vocab, b_vocab)
	
	# Compute the softmax loss
    loss, dscores = temporal_softmax_loss(scores, captions_out, mask)

    # Perform the backpropagation
    dh, grads['W_vocab'], grads['b_vocab'] = temporal_affine_backward(dscores, cache_scores)
    dx, dh0, grads['Wx'], grads['Wh'], grads['b'] = rnn_backward(dh, cache_rnn)
    grads['W_embed'] = word_embedding_backward(dx, cache_embed)
    grads['b_proj'] = np.sum(dh0, axis=0)
    grads['W_proj'] = features.T.dot(dh0)
    
    return loss, grads
```

#### Making prediction

We will use the CNN to generate features for the image and map it to  
$$
h_0
$$
with 
$$
W_{proj}
$$
.
<div class="imgcap">
<img src="/assets/rnn/cap4.png" style="border:none;width:80%;">
</div>

At time step 1, we feed the RNN with "start". The RNN computes the value 
$$
h_1
$$
which will multiply with 
$$
W_{vocab}
$$
to generate score for each word in the vocabulary (1004). We will make the first word prediction by select the one with the highest score (say, "A"). At time step 2, we will fit "A" as an input into the time step 2. With 
$$
h_1
$$ 
computed at time step 1, we then made the second preduction "bus".
	
<div class="imgcap">
<img src="/assets/rnn/cap7.png" style="border:none;;">
</div>

Here we compute the score and set the caption word at time step t to be the word with the highest score. We set prev_word to our prediction which will be used in the next time step.
```python
scores, _ = affine_forward(next_h, W_vocab, b_vocab)
captions[:, t] = scores.argmax(axis=1)
prev_word = captions[:, t].reshape(N, 1)
```

Here is the full code making the prediction with comments:
```python
def sample(self, features, max_length=30):
    N = features.shape[0]
    captions = self._null * np.ones((N, max_length), dtype=np.int32)

    # Retrive all trainable parameters
    W_proj, b_proj = self.params['W_proj'], self.params['b_proj']
    W_embed = self.params['W_embed']
    Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']
    W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']
    
	# N is the size of the data to test
    # prev_word : (N, 1)
    #
    # next_h    : (N, hidden_dim)
    # features  : (N, input_dim)
    # W_proj    : (input_dim, hidden_dim)
    #
    # embed     : (N, 1, wordvec_dim)
    # W-embed   : (vacab_size, wordvec_dim)
    #
    # next_c    : (N, hidden_dim*4) for LSTM
    #
    # scores    : (N, vocab_size)
    # W_vocab     : (hidden_dim, vocab_size)
    #
    # captions  : (N, max_length)

    # Set the first word as "<start>"
    prev_word = self._start * np.ones((N, 1), dtype=np.int32)

    # Compute h0
    next_h, affine_cache = affine_forward(features, W_proj, b_proj)

    H, _ = Wh.shape
	# for each time step
    for t in range(max_length):
	  # Compute the word vector.
      embed, embed_cache = word_embedding_forward(prev_word, W_embed)
	  # Compute h from the RNN
      next_h, cache = rnn_step_forward(np.squeeze(embed), next_h, Wx, Wh, b)
	  # Map h to scores for each vocabulary word
      scores, _ = affine_forward(next_h, W_vocab, b_vocab)
	  # Set the caption word at time t.
      captions[:, t] = scores.argmax(axis=1)
	  # Set it to be the next word input in next time step.
      prev_word = captions[:, t].reshape(N, 1)

    return captions
```

<div class="imgcap">
<img src="/assets/rnn/cap5.png" style="border:none;;">
</div>

### Long Short Term Memory network (LSTM)

<div class="imgcap">
<img src="/assets/rnn/cap6.png" style="border:none;;">
</div>

### Gated Recurrent Units (GRU)
