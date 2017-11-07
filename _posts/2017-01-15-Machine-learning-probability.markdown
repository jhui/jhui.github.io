---
layout: post
comments: true
mathjax: true
priority: 422000
title: “Machine learning - Information theory, probability and game theory.”
excerpt: “Machine learning - Information theory, probability and game theory.”
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

**Maximum Likelihood Estimation (MLE) is the same as minimize KL Divergence.**

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

### Bernoulli distributions

$$
\begin{split}
P(X=1) &= \phi \\
P(X=0) &= 1 - \phi \\
\mathbf{E}_x[x] & = \phi \\
Var_x (x) & = \phi (1 - \phi) \\
\end{split}
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

### Exponential and Laplace distribution

$$
p(x; λ) = λ1_{x≥0}exp (−λx) .
$$

which $$1_{x≥0}$$ is 1 if $$ x≥0 $$. Otherwise 0.

$$
Laplace(x; µ, γ) = \frac{1}{2γ} exp(− \frac{\vert x − µ\vert}{γ} )
$$

Both have a sharp point at 0 which sometimes used for machine learning.

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

#### Probability Mass Functions with discrete variables

$$
\sum_i P (x = x_i) = 1
$$

####  Probability Density Functions with continuous variables

$$
\int p(x)dx = 1
$$

Note: lower case $$p$$ for probability density functions

#### Marginal probability

$$
\begin{split}
P(x) &= \sum_y P(x, y) \\
&= \sum_y P(x \vert y) P(y) \\
\end{split}
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

### Chain rule

$$
P (x^1, . . . , x^n) = P (x^1) \prod^n_{i=2} P (x^i | x^1, . . . , x^{i−1}) \\
P (a, b, c) = P (a \vert b, c)P (b \vert c)P (c) \\
$$

### Expectation

$$
\mathbf{E}_{x \sim P} [f(x)] = \sum_x P(x)f(x)
$$

$$
\mathbf{E}_{x \sim P} [f(x)] = \int p(x)f(x)dx
$$

### Variance and covariance

$$
Var(f(x)) = \mathbf{E} [(f(x) − \mathbf{E}[f(x)])^2]
$$

$$
Cov(f(x), g(y)) = \mathbf{E} [(f(x) − \mathbf{E} [f(x)]) (g(y) − \mathbf{E} [g(y)])]
$$

$$
Cov(x_i, x_i) = Var(x_i)
$$

### Nash Equilibrium

In game theory, the Nash Equilibrium is when no player will change its strategy after considering all possible strategy of opponents. i.e. if we reveal every strategy of all players and no one will change their strategy for a better gain, the Nash equilibrium is reached. A game can have 0, 1 or multiple Nash Equilibria. 

#### The Prisoner's Dilemma

In the prisoner's dilemma problem, police arrests 2 suspects but only have evidence to charger them for a lesser crime with 1 month jail time. But if one of them confess, the other party will receive a 12 months jail time and the one confess will be released. Yet, if both confess, both will receive a jail time of 6 months.

<div class="imgcap">
<img src="/assets/ml/nash.png" style="border:none;width:80%">
</div>

For Mary, if she thinks Peter will keep quiet, her best strategy will be confess to receive no jail time instead of 1 month.

<div class="imgcap">
<img src="/assets/ml/nash2.png" style="border:none;width:80%">
</div>

On the other hand, if she thinks Peter will confess, her best strategy will be confess also to get 6 months jail time.
<div class="imgcap">
<img src="/assets/ml/nash3.png" style="border:none;width:80%">
</div>

In either cases, she should confess. Similarly, Peter should confess also. Therefore (-6, -6) is the Nash Equilibrium even (-1, -1) is the least jail time combined. Why (-1, -1) is not a Nash Equilibrium? Because if Mary knows Peter will keep quiet, she can switch to confess and get a lesser sentence which Peter will response by confessing the crime also. (Providing that Peter and Mary cannot co-ordinate their strategy.)

### Jensen-Shannon Divergence

It measures how distinguishable two or more distributions are from each other.

$$
JSD{X || Y} = H(\frac{X + Y}{2}) - \frac{H(X) + H(Y)}{2}
$$
	




  