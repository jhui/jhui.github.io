---
layout: post
comments: true
mathjax: true
title: “Memory network (MemNN) & End to end memory network (MemN2N)”
excerpt: “Use a memory network to store knowledge for inferencing.”
date: 2017-03-15 11:00:00
---
**This is work in progress...**

### Memory network

Virtual assistances are pretty good at answering a one-line query but fail badly in carrying out a conversation. Here is a verbal exchange demonstrating the challenge ahead of us with virtual assistances:

* Me: Can you find me some restaurants?
* Assistance: I find a few places within 0.25 mile. The first one is Caffé Opera on Lincoln Street. The second one is ...
* Me: Can you make a reservation at the first restaurant? 
* Assistance: Ok. Let's make a reservation for the Sushi Tom restaurant on the First Street.

Why can't the virtual assistance follow my instruction to book the Caffé Opera? It is because the virtual assistance does not remember our conversation and simply response our second query without the context of the previous conversation. Therefore, the best it can do is to find a restaurant that relates to the word "First", and it finds a restaurant located on First Street. Memory networks address this issue by remembering information processed so far.

> The description on the memory networks (MemNN) is based on [Memory networks, Jason Weston etc.](https://arxiv.org/pdf/1410.3916.pdf)

Consider the follow "story" statements:

- Joe went to the kitchen. 
- Fred went to the kitchen. 
- Joe picked up the milk.
- Joe traveled to the office. 
- Joe left the milk. 
- Joe went to the bathroom.

#### Storing information into the memory

First, we store all the statements in the memory $$ m $$:

| Memory slot | Sentence|
|-----|-----------|
| 1 | Joe went to the kitchen. |
| 2 | Fred went to the kitchen. |
| 3 | Joe picked up the milk. |
| 4 | Joe traveled to the office. | 
| 5 | Joe left the milk. |
| 6 | Joe went to the bathroom. |

#### Answering a query
To answer the query $$q$$ "where is the milk now?", we compute our first inference based on: 

$$ o_1 = O_1(q, m) = \underset{i=1, \dots, N}{\arg\max} s_0(q, m_{i}) $$

where $$s_{0} $$ is a function that scores the match between an input $$x$$ and $$m_{i} $$, and $$ o_1 $$ is the index to the memory $$m$$ with the best match. Therefore, $$ m_{o_1} $$ is our best match in the first inference: "Joe left the milk."

Then, we make a second inference based on $$ \big[ q \text{ "where is the milk now"} , m_{o_1} \text{" Joe left the milk."} \big]$$

$$ o_2 = O_2(x, m) = \underset{i=1, \dots, N}{\arg\max} s_o(\big[ q, m_{o_{1}} \big], m_{i}) $$

where $$ m_{o_2}  $$ will be "Joe traveled to the office.". 

We combine the query, and the inference results as $$ o $$:

$$  o = \big[ q, m_{o_{1}} , m_{o_{2}}  \big] = \big[ \text{ "where is the milk now"} ,  \text{" Joe left the milk."},  \text{" Joe travelled to the office."}$$

To generate a final response $$ r $$ from $$o$$:

$$ r = \underset{w \in W}{\arg\max} s_r(\big[ q, m_{o_{1}}, m_{o_{2}}   \big], w) $$

where $$W$$ is all the words in the dictionary, and $$ s_{r} $$ is another function that scores the match between $$ \big[ q, m_{o_{1}}, m_{o_{2}} \big] $$ and a word $$w$$. In our example, the final response $$ r $$ is the word "office".

> We assume we can find the answer with 2 rounds of  inference.

#### Encoding the input
We use bags of words to represent our input text. First, we start with a vocabulary of a size 
$$ \left| W \right| $$ 
.

To encode the query "where is the milk now" with bags of words:

| Text | ... | is | Joe | left | milk | now | office | the | to | travelled | where | ...|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| where is the milk now | ... | 1 | 0 | 0  | 1 | 1 | 0 | 1 | 0 | 0  | 1 | ... |

For better performance, we use 3 different set of bags to encode $$ q $$, $$ m_{o_{1}} $$ and $$m_{o_{2}}$$ separately, i.e., we encode "Joe" in $$q$$ as "Joe_1" and the same word in $$ m_{o_{1}} $$ as "Joe_2":

$$ q $$: Where is the milk now?

$$m_{o_{1}}$$: Joe left the milk.

$$m_{o_{2}}$$: Joe travelled to the office.

To encode $$ \big[ q, m_{o_{1}}, m_{o_{2}} \big] $$ above, we use:

| ... | Joe_1 | milk_1 | ... | Joe_2 | milk_2 | ... | Joe_3 | milk_3 | ... ||
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| | 0 | 1 | |  1  | 1  |  | 1 | 0 | |

Hence, each sentence is converted to a bag of words of size
$$ 3 \left| W \right| $$ 
.

#### Compute the scoring function

We use word embeddings $$ U $$ to convert a sentence with a bag of words with size 
$$ 3 \left| W \right| $$ 
into a dense word embedding with a size $$n$$. We evaluate both score functions $$s_{o} $$ and $$ s_{r} $$  with:

$$
s_{o}(x, y) = Φ_{x}(x)^T{U_{o}}^TU_{o}Φ_{y}(y)
$$

$$
s_{r}(x, y) = Φ_{x}(x)^T{U_{r}}^TU_{r}Φ_{y}(y)
$$

which $$ U $$ is trained with a margin loss function.

### Margin loss function

We will train the parameters in $$ U_{o} $$ and $$ U_{r} $$ with the marginal loss function:

$$
\sum_{\bar{f}\neq m_{o_{1}}} \max(0, γ - s_{o}(x, m_{o_{1}}) + s_{o}(x, \bar{f})) +
$$

$$
\sum_{\bar{f}\neq m_{o_{2}}} \max(0, γ - s_{o}(\big[ x, m_{o_{1}} \big], m_{o_{2}} ) + s_{o}(\big[ x, m_{o_{1}} \big], \bar{f'})) +
$$

$$
\sum_{\bar{r}\neq r} \max(0, γ - s_{r}(\big[ x, m_{o_{1}}, m_{o_{2}} \big], r ) + s_{r}(\big[ x, m_{o_{1}}, m_{o_{2}} \big], \bar{r} )
$$

where $$ \bar{f}, \bar{f'} $$ and $$ \bar{r} $$ are other possible predictions other than the true label. 

### Huge memory networks

For a system with huge memory, computing every score for each memory entry is expensive. Alternatively, after computing the word embedding $$ U_{0} $$, we apply K-clustering to divide the word embedding space into K clusters. We map each input $$ x $$ into its corresponding cluster and perform inference within that cluster space only.

### End to end memory network (MemN2N)

> The description, as well as the diagrams, on the end to end memory networks (MemN2N) are based on [End-To-End Memory Networks, Sainbayar Sukhbaatar etc.](https://arxiv.org/abs/1503.08895)

We start with a query "where is the milk now?". It is encoded with bags of words using a vector of size $$ V $$. (which $$ V $$ is the size of the vocabulary.) In the simplest case, we use Embedding $$B$$ ($$d \times V$$) to convert the vector to a word embedding of size d.

$$
u = embedding_B(q)
$$

For each memory entry $$ x_{i} $$, we use another Embedding $$A$$ ($$d \times V$$) to convert it to $$ m_{i} $$ of size d. 

$$
m_{i} = embedding_A(x_{i})
$$

<div class="imgcap">
<img src="/assets/mem/max10.png" style="border:none;width:70%;">
</div>

The match between $$u$$ and each memory $$m_{i} $$ is computed by taking the inner product followed by a softmax:

$$
p_{i} = softmax(u^T m_{i}).
$$


<div class="imgcap">
<img src="/assets/mem/max11.png" style="border:none;width:70%;">
</div>

We use a third Embedding $$ C $$ to encode $$x_{i}$$ as $$ c_{i} $$

$$
c_{i} = embedding_C(x_{i})
$$

The output is computed as:

$$
o = \sum_{i} p_{i} c_{i} . 
$$

<div class="imgcap">
<img src="/assets/mem/max12.png" style="border:none;width:70%;">
</div>

We multiply the sum of $$ o $$ and $$ u $$ with matrix W ($$ V \times d $$). The result is passed to a softmax function to predict the final answer.

$$
\hat{a} = softmax(W(o + u)) 
$$

<div class="imgcap">
<img src="/assets/mem/max13.png" style="border:none;width:60%;">
</div>

Here, we combine all the steps into one diagram:
<div class="imgcap">
<img src="/assets/mem/max1.png" style="border:none;">
</div>

### Multiple layer

Similar to RNN, we can stack multiple layers to form a complex network. At each layer $$ i $$, it has its own embedding $$ A_{i} $$ and embedding $$ C_{i} $$. The input at layer $$k+1$$ is computed by:

$$
u^{k+1} = u^k + o^k.
$$

<div class="imgcap">
<img src="/assets/mem/max2.png" style="border:none;width:60%;">
</div>
















