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

If convolution networks are deep networks for images, recurrent networks are networks for the time sequence data, like speeh or natural language. For example, the more advanced LSTM and GRU networks are popular for the natural language processing (NLP). But to illustrate the core ideas, we will look into a simplier network called Recurrent neural netwok (RNN).

In a fully connected network, we model h as 

$$
h = f(X_i)
$$


For time sequence data, besides the input, we maintain a hidden state representing the features in the previous time sequence. Hence, to make prediction at time step t, we takes both input $$ X_t $$ and the hidden state from the previous time step $$ h_{t-1}$$ to compute:

$$
h_t = f(x_t, h_{t-1})
$$

<div class="imgcap">
<img src="/assets/rnn/rnn_b.png" style="border:none;width:60%;">
</div>

The following unroll the time step 
$$
t
$$
which take the hidden state
$$
h_{t-1} 
$$
and input
$$
X_t
$$ 
to compute 
$$
h_t
$$
<div class="imgcap">
<img src="/assets/rnn/rnn_b3.png" style="border:none;width:35%;">
</div>

Here we unroll a RNN from time step 
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

In a FC network, 
$$
h
$$
servers as the output of the network. In RNN, 
$$
h
$$
servers 2 purposes: the hidden state for the previous sequence data as well as producing a prediction. Here we map the hidden state 
$$
h_t
$$
to a final prediction. For example, multiply 
$$
h_t
$$
with the matrix
$$
W
$$
to produce the desired predictions
$$
Y
$$.
 
<div class="imgcap">
<img src="/assets/rnn/cap14.png" style="border:none;width:30%;">
</div>

#### Create image caption using RNN
We will study a real example in explaing the RNN. For example, we input a school bus image into the RNN and output a caption like "A yellow school bus idles near a park." Our RNN will read an image and create an image caption.
<div class="imgcap">
<img src="/assets/rnn/cap.png" style="border:none;">
</div>
During the RNN training, we
1. Use a CNN network to capture features of an image.
2. Multiple the features with a trainable matrix to generate
$$
h_0
$$
3. Feed 
$$ 
h_0
$$
to the RNN.
4. Use a word embedding lookup table to convert a word to a word vector 
$$
X_1
$$
. (a.k.a word2vec)
5. Feed the word vector and
$$
h_0
$$ to the RNN.
$$
h_1 = f(X_1, h_0)
$$
6. Use a trainable matrix to map 
$$
h
$$
to scores which predict the probabilities of
$$
word_i
$$
to be the next caption word.
7. Move to the next time step with 
$$
h_1
$$ 
and the word "A" as input.

<div class="imgcap">
<img src="/assets/rnn/cap12.png" style="border:none;;">
</div>

#### Capture image features
We pass the image into a CNN and use one of the activation layer in the fully connected (FC) network to initialize the RNN. For example, in the picture below, we pick the input of the second FC layer which has a shape of (512,) as the features used to create captions.

<div class="imgcap">
<img src="/assets/rnn/cnn.png" style="border:none;;">
</div>

We multiple the CNN features with a trainable matrix to compute
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
h_1 = f(h_0, X_1)
$$

<div class="imgcap">
<img src="/assets/rnn/cap8.png" style="border:none;width:80%;">
</div>

Define the shape of CNN features (N, 512) and h (N, 512) which N is the batch size in training:
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

Compute $$ h_0 $$ by multipy the image features with $$ W_{proj} $$.
```python
# Initialize CNN -> hidden state projection parameters
# h0: (N, hidden_dim)
h0 = features.dot(W_proj) + b_proj
```

#### Map words to RNN
Our training data contains both the images and captions. It also have a dictionary which map a vocabulary word to an integer. Caption words in the dataset are stored as word indexes. For example, the caption "A yellow school bus idles near a park." may stored as "1 5 3401 3461 78 5634 87 5 111 2" which 1 represents the "start" of a caption, 5 represents 'a', 3401 represents 'yellow'  and 2 represents the "end" of a caption.

> In this tutorial, we called the captions provided in the training dataset: true caption.

However, the RNN does not use the word index directly. Instead, through a word embedding lookup table (word2vec)
$$
W_{embed}
$$
, the word index is converted to a vector of length wordvec_dim. The RNN will take this vector
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
<img src="/assets/rnn/cap9.png" style="border:none;width:45%;">
</div>

>  word2vec is a method to map a word to a vector say with 256 values. The mapping maintains the semantic relationship among words. The embedding lookup table is trained with deep learning.

When we create the training data, we convert words to the corresponding word index using a vocabulary dictionary. In runtime, we map the word index to a word vector.
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
# T = 16: Number of unroll time step.
# captions:    (N, T+1) The caption represent "<start> A yellow school bus idles near a park <end> <null> ... <null>" represent in word index. 
# captions_in  (N, T) The caption feed into the RNN (X) = captions without the last word.
# captions_out (N, T) The true caption output: the caption without "<start>"

# W_embed (vocab_size, wordvec_dim)
# captions_in: (N, T) each captions_in contain at most 16 words.
# x: (N, T, wordvec_dim)
x, cache_embed = word_embedding_forward(captions_in, W_embed)
```

Loop up the word vector of x from the lookup table W.
```python
def word_embedding_forward(x, W):
  out, cache = None, None
  N, T = x.shape
  V, D = W.shape
  out = W[x]
  cache = (V, x)
  return out, cache  
```
  

#### RNN
<div class="imgcap">
<img src="/assets/rnn/score.png" style="border:none;width:40%;">
</div>

We pass the word vector
$$
X_0
$$
into the RNN. The output of the RNN 
$$
h_1
$$
is then multipy with 
$$
W_{vocab}
$$
to generate scores for each word in the vocabulary. For example, if we have 10004 words in the vocabulary, it will generate 10004 scores predicting how likely each word will be the next word in the caption. With the true caption and the scores, we compute the softmax loss of the RNN. 
<div class="imgcap">
<img src="/assets/rnn/score_1.png" style="border:none;">
</div>

> We provide codes for readers want more concrete details. Nevertheless, fully understanding of the code is not needed or suggested.

We comput $$ h_t $$ 
by feeding the RNN with $$ X_t $$ and $$ h_{t-1} $$.
We then map $$ h_t $$ to scores which are used to compute the softmax cost.
```python
# h: (N, 16, hidden_dim)
# Wx: (wordvec_dim, hidden_dim)
# Wh: (hidden_dim, hidden_dim)
h, cache_rnn = rnn_forward(x, h0, Wx, Wh, b)

# W_vocal: (hidden_dim, vocab_size 1004)
# scores: (N, 16, vocab_size 1004)
scores, cache_scores = temporal_affine_forward(h, W_vocab, b_vocab)
loss, dscores = temporal_softmax_loss(scores, captions_out, mask)
```

#### rnn_forward

<div class="imgcap">
<img src="/assets/rnn/cap13.png" style="border:none;width:40%;">
</div>

rnn_forward simply unroll the RNN T time steps and update 
$$
h_t
$$
with each RNN computation. At each step, it takes the $$ h $$ from the previous step and use the true captions provided by the training set as input $$ X_t $$.  i.e., even the word 'A' has a very low score in our previous step prediction, we still use the work 'A' as the next input word since the whole purpose is to optimize the RNN with the true caption.

```python
def rnn_forward(x, h0, Wx, Wh, b):
  h, cache = None, None
  N, T, D = x.shape
  H = h0.shape[1]
  h = np.zeros((N, T, H))
  state = {}
  state[-1] = h0
  cache_step = [None] * T

  for t in range(T):
    xt = x[:, t, :]
    state[t], cache_step[t] = rnn_step_forward(xt, state[t-1], Wx, Wh, b)
    h[:, t, :] = state[t]

  cache = (cache_step, D)
  return h, cache
```


For each RNN step, we multiple $$ h_{t-1} $$ with $$ W_h $$ and $$ x_{t} $$ with $$ W_x $$ to generate 
$$ h_t $$

```python
def rnn_step_forward(x, prev_h, Wx, Wh, b):
  next_h, cache = None, None
  state = np.dot(x, Wx) + np.dot(prev_h, Wh) + b
  next_h = np.tanh(state)

  cache = x, prev_h, Wx, Wh, state
  return next_h, cache
```    

#### Scores

After finding $$ h_t $$, we compute the scores by multiply
$$
W_{vocab}
$$
with
$$
h_t
$$
```python
 def temporal_affine_forward(x, w, b):
   N, T, D = x.shape
   M = b.shape[0]
   out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
   cache = x, w, b, out
   return out, cache
```

#### Softmax cost

For each words in the vocabulary (1004 words), we predict the probability of each word to be the next caption word. Then we compute the softmax cost to train the RNN later.
```python
def temporal_softmax_loss(x, y, mask):
  N, T, V = x.shape
  
  x_flat = x.reshape(N * T, V)
  y_flat = y.reshape(N * T)
  mask_flat = mask.reshape(N * T)
  
  probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
  dx_flat = probs.copy()
  dx_flat[np.arange(N * T), y_flat] -= 1
  dx_flat /= N
  dx_flat *= mask_flat[:, None]
  
  dx = dx_flat.reshape(N, T, V)
  
  return loss, dx
```
    
#### Time step 0
Here is how we train with the image feature
$$
h_0
$$
and the first word 'start' to compute the softmax loss.

<div class="imgcap">
<img src="/assets/rnn/cap11.png" style="border:none;width:70%;">
</div>

#### Complete flow

After we completed the first time step.  we move onto the next time step with
$$
h_1
$$
and the next true caption word 'A'. Note that, for training, we do not use our best prediction as the input
$$
X_t
$$
. 

Here is the detail complete flow in training 1 sample data.
<div class="imgcap">
<img src="/assets/rnn/cap3.png" style="border:none;;">
</div>

Here is the code listing for the forward feed, backpropagation and the loss.
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
to generate scores for each word in the vocabulary (1004). We will make the first word prediction by select the one with the highest score (say, "A"). At time step 2, we will fit the highest score prediction "A" as an input into the time step 2. With 
$$
h_1
$$ 
computed at time step 1 and "A", we made the second preduction "bus".
	
<div class="imgcap">
<img src="/assets/rnn/cap7.png" style="border:none;;">
</div>

Here we compute the score and set the caption word at time step t to be the word with the highest score. We set prev_word to this prediction which will be used in the next time step.
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

Finally here is the final detail flows:
<div class="imgcap">
<img src="/assets/rnn/cap5.png" style="border:none;;">
</div>

### Long Short Term Memory network (LSTM)

$$ h_t $$ in RNN serves 2 purpose:
* Make an output prediction, and
* Be a hidden state extracting the information in the sequence data process so far.

This actually serve 2 different purposes and therefore LSTM breaks $$ h_t $$ according to the roles above. The hidden state of the LSTM cell will now be $$ C $$.

<div class="imgcap">
<img src="/assets/rnn/lstm.png" style="border:none;width:50%;">
</div>

#### LSTM gates
In LSTM, we want a mechansim to selectively allow what information to remember and what information to ignore. Therefore we construct different gates with value between 0 to 1, and multiple it with the original value. For example, a gate with 0 means no information to pass through and a gate with 1 means everything is passing through.

$$ 
\text{value} = \text{gate} \cdot \text{value}
$$

In LSTM, we have 3 different gates but all with the same form:

$$
gate = g(X_t, h_{t-1}) = \sigma (W_{x} X_t + W_{h} h_{t-1} + b) 
$$

which $$ \sigma $$ is the sigmoid function.

> All gates have different set of W and b. But people feel lost in the LSTM equations without realize its simplicity. So we just assume they all take different set of W and b for now.

You may also find a lot of W, b in later equations, but all are belong to the same pattern:

$$
z(X_t, h_{t-1}) = (W_{x} X_t + W_{h} h_{t-1} + b) 
$$

Once realize that, all the LSTM equations are pretty simple.

#### Updating C
<div class="imgcap">
<img src="/assets/rnn/lstm2.png" style="border:none;width:20%;">
</div>

To update C, we constructs 2 gates:
* forget gate: a gate to forget previous hidden state informatin $$ C_{t-1} $$.
* input gate: a gate to allow what current information $$ \tilde{C} $$ will add to $$ C $$.

$$
gate_{forget} = \sigma (W_{x} X_t + W_{h} h_{t-1} + b) 
$$

$$
gate_{input} = \sigma (W_{x} X_t + W_{h} h_{t-1} + b) 
$$

In RNN, the mechanism to update $$ h_t $$ is pretty simple:

$$
h_t = f(X_t, h_{t-1})
$$

But in LSTM, there are 2 steps.
* Compute what new information $$ \tilde{C} $$ may generate in time step t
* Forget some old information $$ C_{t-1}  $$ and add back some from $$ \tilde{C} $$.

$$
\tilde{C} = \tanh (W_{x} X_t + W_{h} h_{t-1} + b) 
$$

$$
C_t = gate_{forget} \cdot C_{t-1} + gate_{input} \cdot \tilde{C}
$$

#### Update h
<div class="imgcap">
<img src="/assets/rnn/lstm1.png" style="border:none;width:20%;">
</div>

To update $$ h_{t} $$, we compute a new output gate and compute the new $$ h_t $$

$$
gate_{out} = \sigma (W_{x} X_t + W_{h} h_{t-1} + b) 
$$
 
 $$
 h_t = gate_{out} \cdot \tanh (C_t)
 $$
 
 
#### Image captures with LSTM
Now we can have an optional to use a LSTM network instead of RNN. 
```python
if self.cell_type == 'rnn':
  h, cache_rnn = rnn_forward(x, h0, Wx, Wh, b)
else:
  h, cache_rnn = lstm_forward(x, h0, Wx, Wh, b)
``` 

```python
def lstm_forward(x, h0, Wx, Wh, b):
  h, cache = None, None
  N, T, D = x.shape
  H, _ = Wh.shape
  next_h = h0
  next_c = np.zeros((N, H))

  cache_step = [None] * T

  h = np.zeros((N, T, H))
  for t in range(T):
    xt = x[:, t, :]
    next_h, next_c, cache_step[t] = lstm_step_forward(xt, next_h, next_c, Wx, Wh, b)
    h[:, t, :] = next_h
  cache = (cache_step, D)
  
  return h, cache
```


```python
def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
  next_h, next_c, cache = None, None, None
  N, H = prev_h.shape
  a = x.dot(Wx) + prev_h.dot(Wh) + b
  ai = a[:, :H]
  af = a[:, H:2 * H]
  ao = a[:, 2 * H:3 * H]
  au = a[:, 3 * H:]
  ig = sigmoid(ai)
  fg = sigmoid(af)
  og = sigmoid(ao)
  update = np.tanh(au)
  next_c = fg * prev_c + ig * update
  next_h = og * np.tanh(next_c)

  cache = (next_c, og, ig, fg, og, update, ai, af, ao, au, Wx, x, Wh, prev_h, prev_c)
  
  return next_h, next_c, cache
```

<div class="imgcap">
<img src="/assets/rnn/cap6.png" style="border:none;;">
</div>

### Gated Recurrent Units (GRU)
