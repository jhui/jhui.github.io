---
layout: post
comments: true
mathjax: true
priority: 410000
title: “Machine learning - Notes”
excerpt: “Machine learning - Notes”
date: 2017-01-15 12:00:00
---

### Entropy

A staff misses an all-hands meeting and ask what is he or she missing. A co-worker said nothing. It is not nothing is covered but probably just the same message over and over again. In this scenario, $$P(message_a)=1$$ and the message contains no information since it is all predictable. In information theory, entropy measures the amount of information which expresses as the number of bits to encode it. In Huffman encoding, since letters does not occur randomly in a text, we can use fewer bits to it.

> In information theory, information and random-ness are positively correlated. High entropy equals high randomness and more bits to encode it.

We define entropy as:

$$
H(y) = - \sum_X p(y) \log p(y)
$$

The entropy of a coin peaks when $$p(head)=p(tail)=0.5$$. 

For a fair coin, 

$$ H(X) = - p(head) \cdot \log_2(p(head)) - p(tail) \cdot log_2(p(tail)) =  - \log_2 \frac{1}{2} = 1 $$ 

and therefore we can use 1 bit of 0 to represent head and 1 bit of 1 to represent tail.

So for a fair dice, $$ H(X) = \log_2 6 \approx 2.59 $$. It means a fair dice gives more surprise than a fair coin.

### Cross entropy

If entropy measures the minimum of bits to encode information using the most optimized scheme. Cross entropy measures the minimum of bits to encode $$Y$$ using the wrong optimized scheme from $$\hat{Y}$$.

Cross entropy is defined as:

$$
H(y, \hat{y}) = - \sum_X p(y) \log p(\hat{y})
$$

> Cross entropy is encoding y with the distribution from $$\hat{y}$$.

### KL Divergence

KL Divergence measures the difference between 2 probability distribution as

$$
D_{KL}(p \vert \vert q) = \sum_{i=1}^N p(i) \log \frac{p(i)}{q(i)} 
$$

<div class="imgcap">
<img src="/assets/ml/kl.png" style="border:none;width:80%">
</div>
Diagram source Wikipedia.

Recall:

$$
\begin{split}
H(p) & = - \sum p \log p \\
H(p, q) & = - \sum p \log q \\
D_{KL}(p \vert \vert q) & = \sum p \log \frac{p}{q} \\
\end{split}
$$

Compute cross entropy:

$$
\begin{split}
H(p, q) & = - \sum p \log q \\
   & = - \sum p \log p + \sum p \log p - \sum p \log q \\
   & = H(p) + \sum p \log \frac{p}{q} \\
   & = H(p) + D_{KL}(p \vert \vert q) 		 
\end{split}
$$

So cross entropy is the sum of entropy and KL-divergence.

$$
\begin{split}
H(p, q) & =  H(p) + D_{KL}(p \vert \vert q) 		 
\end{split}
$$


### Maximum Likelihood Estimation

Maximum Likelihood Estimation (MLE) is the same as minimize KL Divergence:

$$
\begin{split}
\hat\theta & = \arg\max_{\theta} \prod^N_{i=1} p(x_i \vert \theta ) \\
& = \arg\max_{\theta} \sum^N_{i=1} \log p(x_i \vert \theta ) \\
& = \arg\max_{\theta} \frac{1}{N} \sum^N_{i=1} \log p(x_i \vert \theta ) - \frac{1}{N} \sum^N_{i=1} \log p(x_i \vert \theta_0 ) \\
& = \arg\max_{\theta} \sum^N_{i=1} \log \frac {p(x_i \vert \theta )}{p(x_i \vert \theta_0 )} \\
& = \arg\max_{\theta} \sum_{x_i \in X} P(x_i \vert \theta_0) \log \frac {p(x_i \vert \theta )}{p(x_i \vert 
\theta_0 )} \\
& \implies \arg\min_{\theta}  D_{KL}(P(x_i \vert \theta_0) \vert \vert P(x_i \vert \theta)) \\
\end{split}
$$

which $$\theta_0$$ is the ground truth.

$$
\begin{split}
\hat\theta & = \arg\max_{\theta} \sum_{x_i \in X} P(x_i \vert \theta_0) \log \frac {p(x_i \vert \theta )}{p(x_i \vert \theta_0 )} \\
& = \arg\min_{\theta} \sum_{x_i \in X} P(x_i \vert \theta_0) \log p(x_i \vert \theta_0 ) -  P(x_i \vert \theta_0) \log p(x_i \vert \theta ) \\ 
& \implies \arg\min_{\theta} H(real world) - H(model)\\
\end{split}
$$

### Bias and variance

There are two sources of error. The variance is error sensitivity to small changes in the training set. It is often the result of overfitting: powerful model but not enough data. On the other hand, bias happens when the model is not powerful enough to make accurate prediction (underfitting). Usually a high variance but low bias model makes in-consistence prediction when trained with different batches of input. But the average prediction is close to the true value. The orange dots below are predictions make by a deep network. The predictions made by a highly variance model generate predictions widely spread around the true value. A highly variance model make consistence predictions but it is off from the true value.

<div class="imgcap">
<img src="/assets/ml/var.png" style="border:none;width:60%">
</div>

We define bias and variance as:

$$
\begin{split}
bias(\hat\theta) & = \mathbb{E}_{p(D\vert \theta_0)} (\hat\theta) - \theta_0 = \overline\theta - \theta_0 \\
\mathbb{V} (\hat\theta) & = \mathbb{E}_{p(D\vert \theta_0)} (\hat\theta - \overline\theta)^2 \\
\end{split}
$$

Here we proof that the mean square error is actually compose of a bias and a variance.

$$
\begin{split}
MSE & = \mathbb{E} (\hat\theta - \theta_0)^2 \\
& = \mathbb{E} [2 (\overline\theta^2 - \theta_0 \overline\theta - \overline\theta \hat \theta -  \theta_0 \hat\theta) + (\hat\theta - \theta_0)^2] \\
& = \mathbb{E} [\overline\theta^2 - 2 \theta_0 \overline\theta + \overline\theta^2] + \mathbb{E} [\overline\theta^2 - 2 \overline\theta \hat \theta + \overline\theta^2)] \\
 & = (\overline\theta - \theta_0)^2 + \mathbb{E} (\hat\theta - \overline\theta)^2 \\
& = bias(\hat\theta)^2 + \mathbb{V} (\hat\theta) \\
\end{split}
$$

Given:

$$
\begin{split}
& \mathbb{E} (\overline\theta^2 - \theta_0 \overline\theta - \overline\theta \hat \theta -  \theta_0 \hat\theta) \\
& = \overline\theta^2 - \theta_0\overline\theta - \overline\theta^2 - \theta_0\overline\theta \\
& = 0 \\
\end{split}
$$
 
> Simple model has high bias. Overfitting a high complexity model causes high variance.


### Norms
L0-norm (0 if x is 0, otherwise it is 1)

$$
L_0 = \begin{cases}
                        0 \quad \text{ if } x_i = 0 \\
                        1 \quad \text{otherwise}
                    \end{cases}
$$

L1-norm (Manhattan distance)

$$
\begin{split}
L_1 & = \| x \|_1 =  \sum^d_{i=0} \vert x_i \vert  \\
\| x - y \|_1 & =  \sum^d_{i=0} \vert x_i - y_i \vert  \\
\end{split}
$$

L2-norm (Euclidian distance)

$$
\begin{split}
L_2  & = \| x \|_2 = \| x \| =  \sqrt{\sum^d_{i=0} x_i^2}  \\
L_2^2  & =  \sum^d_{i=0} x_i^2 = x^Tx  \\
\| x - y \| & =  \sqrt{\sum^d_{i=0} (x_i - y_i)^2}  \\
\end{split}
$$

Lp-norm

$$
\begin{split}
L_p  & = \| x \|_p =  \sqrt[p]{\sum^d_{i=0} x_i^p}  \\
\end{split}
$$

$$\text{L}_\infty$$-norm

$$
\begin{split}
L_\infty (x) &  =  max(\vert x_i \vert)  \\
\end{split}
$$

### L1 regularization

$$
\begin{split}
J(W) & = \frac{1}{2} \| xw - y \|^2 + \frac{\lambda}{2} \vert w \vert
& = MSE + \text{ regularization cost }
\end{split}
$$

<div class="imgcap">
<img src="/assets/ml/L1.png" style="border:none;width:40%">
</div>

L1 regularization has a tendency to push $$w_i$$ to exactly 0. Therefore, L1-regularization increases the sparsity of $$W$$.


### L2 regularization

$$
\begin{split}
J(W) & = \frac{1}{2} \| xw - y \|^2 + \frac{\lambda}{2} \| w \|^2
& = MSE + \text{ regularization cost }
\end{split}
$$

$$ W^*_a$$ is where regularization cost is 0 and $$ W^*_b $$ is where MSE is minimum. The optimal solution for $$J$$ is where the concentric circle meet with the eclipse. This is the same as minimizing mean square error with the L2-norm constraint.

<div class="imgcap">
<img src="/assets/ml/L2.png" style="border:none;width:40%">
</div>

MSE with L2-regularization is also called **ridge regression**.

### Gaussian distribution/Normal distribution

$$
P(x) = \frac{1}{\sigma\sqrt{2\pi}}e^{-(x - \mu)^{2}/2\sigma^{2} } 
$$

$$
x \sim \mathcal{N}{\left(
\mu 
,
\sigma^2
\right)}
$$

### Binomial distributions

$$
P(x;p,n) = \left( \begin{array}{c} n \\ x \end{array} \right) p^{x}(1 - p)^{n-x}
$$
 
<div class="imgcap">
<img src="/assets/ml/bdist.png" style="border:none;width:50%">
</div>
Source: wiki

The Gaussian distribution is the limiting case for the binomial distribution with:

$$
\begin{split}
\mu & = n p \\
\sigma^2 & = n p (1-p) 
\end{split}
$$

### Poisson Distribution

Assuming a rare event with an event rate $$\lambda$$, the probability of observing x events within an interval $$t$$ is:

$$
P(x) = \frac{e^{-\lambda t} (\lambda t)^x}{x!}
$$

Example: If there were 2 earthquakes per 10 years, what is the chance of having 3 earth quakes in 20 years.

$$
\begin{split}
\lambda t & = 2 \cdot (\frac{20}{10}) = 4 \\
P(x) & = \frac{e^{-\lambda t} (\lambda t)^x}{x!} \\
P(3) & = \frac{e^{-4} \cdot 4^3}{3!}
\end{split}
$$

Given:

$$
\begin{split}
prob. & = p  = \frac{v}{N}  \\
P(x \vert N, p) & = \frac{N!}{x! (N-x)!} p^x(1-p)^{N-x} \\
\end{split}
$$

Proof:

$$
\begin{split}
P(x \vert v) & = \lim_{N\to\infty} P(x|N, v) \\
&= \lim_{N\to\infty} \frac{N!}{x! (N-x)!} (\frac{v}{N})^x(1-\frac{v}{N})^{N-x} \\
&= \lim_{N\to\infty} \frac{N(N-1)\cdots(N-x+1)}{N^x} \frac{v^x}{x!}(1-\frac{v}{N})^N(1-\frac{v}{N})^{-x} \\
&= 1 \cdot \frac{v^x}{x!} \cdot e^{-v} \cdot 1 & \text{Given } v \ll N \\
&= \frac{e^{-v} v^x }{x!}   \\
&= \frac{e^{-\lambda t} (\lambda t)^x }{x!}   & \text{Given }  v = \lambda t \\\\ 
\end{split}
$$

### Beta distribution

The definition of a beta distribution is:

$$
\begin{align} 
P(\theta \vert a, b) = \frac{\theta^{a-1} (1-\theta)^{b-1}} {B(a, b)}
\end{align}
$$

For discret variable, the beta function $$B$$ is defined as:

$$
\begin{align} 
B(a, b) & = \frac{\Gamma(a) \Gamma(b)} {\Gamma(a + b)} \\
\Gamma(a) & = (a-1)!
\end{align}
$$

For continuos variable, the beta function is:

$$
\begin{align} 
B(a, b) = \int^1_0 \theta^{a-1} (1-\theta)^{b-1} d\theta
\end{align}
$$

Here are the beta distribution for different values of a and b. For $$a=b=1$$, the probability is uniformly distributed:  
<div class="imgcap">
<img src="/assets/ml/c1.png" style="border:none;width:20%">
</div>

For $$a=10, b=1$$:
<div class="imgcap">
<img src="/assets/ml/c2.png" style="border:none;width:20%">
</div>

For $$a=1, b=10$$:
<div class="imgcap">
<img src="/assets/ml/c3.png" style="border:none;width:20%">
</div>

For $$a=b=0.5$$:
<div class="imgcap">
<img src="/assets/ml/c4.png" style="border:none;width:20%">
</div>

For $$a=2, b=3$$:
<div class="imgcap">
<img src="/assets/ml/c5.png" style="border:none;width:20%">
</div>

### Probabilities

Basic:

$$
\begin{split}
P(A, B) &= P(B) P(A \vert B) \\
P(A \vert B) &= \frac{P(A, B)}{P(B)} \\
P(A, B \vert C) &= P(A \vert B, C) P(B \vert C) \\ 
\end{split}
$$

Given 2 events $$ x_i, x_j $$ are independent:

$$
\begin{align} 
P(x_i, x_j ) & = P(x_i) P(x_j ) \\
P(x_i, x_j \vert y) & = P(x_i \vert y) P(x_j \vert y) \\
\end{align} 
$$

#### Bayes' theorem

$$
\begin{split}
P(A \vert B) & = \frac{P(B \vert A) P(A)}{P(B)} \\
P(A \vert B) & = \frac{P(B \vert A) P(A)}{\sum^n_i P(B, A_i) } = \frac{P(B \vert A) P(A)}{\sum^n_i P(B, \vert A_i) P(A_i) }
\end{split}
$$

Apply Bayes' theorem to conditional probability

$$
P(A \vert B, C) = \frac{P(B \vert A, C) P(A \vert C)}{ P(B \vert C)}
$$

Proof:

$$
\begin{split}
P(A \vert B,C) & =  \frac{P(B,C \vert A) P(A)} {P(B,C)} \\
& = \frac{P(B \vert A, C) P(C \vert A) P(A)}{P(B,C)} \\ 
& = \frac{P(B \vert A, C) \frac{P(C \vert A) P(A)}{P(C)}}{\frac{P(B,C)}{P(C)}} \\ 
& = \frac{P(B \vert A, C) P(A \vert C)}{ P(B \vert C)} \\
\end{split}
$$

#### Naive Bayes' theorem

Naive Bayes' theorem assume $$ x_i$$ and $$x_j$$ are independent. i.e.

$$
\begin{align} 
P(x_i, x_j \vert y) & = P(x_i \vert y) P(x_j \vert y) \\
\end{align} 
$$

$$
\begin{split}
P(y \vert x_1, x_2, \cdots , x_n) & = \frac{P(x_1, x_2, \cdots , x_n \vert y) P(y)}{P(x_1, x_2, \cdots , x_n)} \\
& = \frac{P(x_1 \vert y) P(x_2 \vert y) \cdots  P(x_n \vert y) P(y)}{P(x_1, x_2, \cdots , x_n)} \\
& \propto P(x_1 \vert y) P(x_2 \vert y) \cdots  P(x_n \vert y) P(y) \\
\end{split}
$$

We often ignore the marginal property (the denominator) in Naive Bayes theorem because it is constant and therefore not important when we are optimizing or comparing the parameters for the model.

### Terms

#### Term frequency–Inverse document frequency (tf-idf)

tf-idf measures the frequency of a term in a document corrected by how common the term in the documents.

Term frequency, Inverse document frequency:

$$
\begin{align} 
tf(t,d) & = \text{Frequency of a term in a document} \\
idf(t) & = \log \frac{\text{number of documents}}{\text{number of documents containing the term }} \\
tf-idf(t, d) & = tf(t,d) \cdot idf(t) \\
\end{align} 
$$

where n is the number of documents and the number of documents containing the term.

#### Skip-gram model and Continuous Bag-of-Words model (CBOW) 

Skip-gram model tries to predict each context word from its target word. For example, in the sentence:

"The quick brown fox jumped over the lazy dog."

The target word "fox" have 2 context words in a bigram model (2-gram). The training data (input, label) will look like: (quick, the), (quick, brown), (brown, quick), (brown, fox).

The continuous bag-of-words is the opposite of Skip-gram model. It predicts the target word from the context word.

#### Unsupervised learning

Unsupervised learning tries to understand the relationship and the latent structure of the input data. In contrast to supervised learning, unsupervised training set contains input data but not the labels.

Example of unsupervised learning;

Clustering
* K-means
* Density based clustering - DBSCAN
Gaussian mixture models 
	*Expectation–maximization algorithm (EM)
Anomaly detection
* Gaussian model
* Clustering
Deep Networks
* Generative Adversarial Networks
*  Restricted Boltzmann machines
Latent factor model/Matrix factorization
       * Principal component analysis
	* Singular value decomposition
	* Non-negative matrix factorization
Manifold/Visualization
	* MDS, IsoMap, t-sne
Self-organized map
Association rule
	
