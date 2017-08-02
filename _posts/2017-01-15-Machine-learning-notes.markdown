---
layout: post
comments: true
mathjax: true
priority: 110000
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

#### Linear regression with gaussian distribution

Using linear regression with gaussian distribution with $$ \mu = x\theta$$:


$$
y_i \sim \mathcal{N}(x_i^T\theta, \sigma^2) = x_i^T\theta + \mathcal{N}(0, \sigma_2)
$$

Likelihood:

$$
\begin{split}
p(y \vert x, \theta, \sigma) & = \prod_{i=1}^n p(y_i \vert x_i, \theta, \sigma) \\
& = \prod_{i=1}^n (2 \pi \sigma^2)^{-1/2}e^{- \frac{1}{2 \sigma^2}(y_i - x^T_i \theta)^2} \\
& = (2 \pi \sigma^2)^{-n/2}e^{- \frac{1}{2 \sigma^2} \sum^n_{i=1}(y_i - x^T_i \theta)^2} \\
& = (2 \pi \sigma^2)^{-n/2}e^{- \frac{1}{2 \sigma^2} (y - x \theta)^T(y - x \theta)} \\
\end{split}
$$

MLE by optimizing the log likelihood $$l$$:

$$
\begin{split}
l(\theta) & = \log(p(y \vert x, \theta, \sigma)) \\
& = -\frac{n}{2} \log(2 \pi \sigma^2) - \frac{1}{2 \sigma^2}(y - x \theta)^T(y - x \theta) \\
\\
\frac{\partial l(\theta)}{\partial \theta} & = 0 - \frac{1}{2 \sigma^2} [0 - 2 x^Ty + 2x^Tx\theta] = 0\\
\hat\theta & = (x^Tx)^{-1}x^Ty
\end{split}
$$

The inverse of $$x$$ may not be well conditioned. We can add a $$\delta$$ to improve the solution:

$$
\begin{split}
\hat\theta & = (x^Tx)^{-1}x^Ty \\
\hat\theta & = (x^Tx + \delta^2 I )^{-1}x^Ty \\
\end{split}
$$

Let's assume the cost function is: (A MSE + regularization.)

$$
J(\theta) = (y -x\theta)^T(y-x\theta) + \delta^2\theta^T\theta 
$$

The following proves that $$\hat\theta$$ is the solution for the cost function above.

$$
\begin{split}
\frac{\partial{J(\theta)}}{\partial \theta} & = 2 x^Tx\theta - 2x^Ty + 2 \delta^2 I \theta = 0 \\
& = (x^Tx + \delta^2 I ) \theta = x^Ty
\end{split}
$$

We can rewrite the regularization as an optimization constraint which the $$ \vert \vert \theta \vert \vert $$ needs to smaller than $$ t(\delta) $$.

$$
\min_{\theta^T\theta \le t(\delta)} (y-x\theta)^T(y-x\theta)
$$

### Bayesian linear regression

We can also apply Bayesian inference with prior $$\mathcal{N}(\theta \vert \theta_0, V_0)$$ and likelihood $$ \mathcal{N}(y \vert x\theta, \sigma^2 I ) $$ to compute the posterior.

$$
\begin{split}
p( \theta \vert x, y, \sigma^2) & \sim \mathcal{N}(\theta \vert \theta_0, V_0) \mathcal{N}(y \vert x\theta, \sigma^2 I ) = \mathcal{N}(\theta \vert \theta_n, V_n) \\
\theta_n & = V_n V^{-1}_0 \theta_0 + \frac{1}{\sigma^2}V_nx^Ty \\
V^{-1}_n & = V^{-1}_0 + \frac{1}{\sigma^2}x^Tx
\end{split}
$$

Comparison between MLE linear regression and Bayesian linear regression
<div class="imgcap">
<img src="/assets/ml/ar.png" style="border:none;width:80%">
</div>
Source Nando de Freitas, UBC machine learning class.

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

$$
\begin{split}
L_\infty (x) &  =  max(\vert x_i \vert)  \\
\end{split}
$$

L0-norm

$$
L_0 = \begin{cases}
                        0 \quad \text{ if } x_i = 0 \\
                        1 \quad \text{otherwise}
                    \end{cases}
$$

### L2 regularization

$$
\begin{split}
J(W) & = \frac{1}{2} \| xw - y \|^2 + \frac{\lambda}{2} \| w \|^2
& = MSE + \text{ regularization cost }
\end{split}
$$

$$ W^*_a$$ is where regularization cost is 0 and $$ W^*_b $$ is where MSE is minimum. The optimal solution for $$J$$ is where the concentric circle meet with the eclipse.

<div class="imgcap">
<img src="/assets/ml/L2.png" style="border:none;width:35%">
</div>

This is the same as minimizing mean square error with the L2-norm constraint.

L2-regularize MSE is also called **ridge regression**.

### L1 regularization

$$
\begin{split}
J(W) & = \frac{1}{2} \| xw - y \|^2 + \frac{\lambda}{2} \vert w \vert
& = MSE + \text{ regularization cost }
\end{split}
$$

<div class="imgcap">
<img src="/assets/ml/L1.png" style="border:none;width:35%">
</div>

L1 regularization has a tendency to push $$w_i$$ to 0. ie L2-regularization increases the sparsity of $$w$$.

