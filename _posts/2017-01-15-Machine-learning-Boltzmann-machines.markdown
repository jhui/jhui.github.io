---
layout: post
comments: true
mathjax: true
priority: 970000
title: “Machine learning - Restricted Boltzmann Machines.”
excerpt: “Machine learning - Restricted Boltzmann Machines.”
date: 2017-01-15 12:00:00
---

(Working in progress)

### Maxwell–Boltzmann distribution

Maxwell–Boltzmann distribution models the distribution of speeds of the gas molecules at a certain temperature:

<div class="imgcap">
<img src="/assets/know/bolt.png" style="border:none;width:40%">
</div>

(Source Wikipedia)

Here is the Maxwell–Boltzmann distribution equation: 

$$
\begin{split}
f(E) & = A e^{\frac{-E}{kT}} \\
\end{split}
$$

$$f(E)$$ gives the probability density (PDF) of molecules with energy $$E$$ which is a function of the temperature $$T$$. The probability density decreases exponentially with $$E$$. As the temperature increases, the probability of a molecule having higher energy also increases. Since the number of molecules remains constant, the peak is therefore lower. Maxwell–Boltzmann distribution equation often expresses as a a function of the molecular velocity:

$$
\begin{split}
f(v) & = \sqrt{ \left (\frac{m}{2\pi kT} \right)^{3}  } 4\pi v^2  e^{\frac{-mv^2}{2kT}}  \\
\end{split}
$$

Maxwell–Boltzmann distribution can be generalized as:

$$
\begin{split}
P(x) & = \frac{e^{-E(x)}}{Z}  \quad \text{where } Z = {\sum_{x^{'}} e^{-E(x^{'}) }}  \text{ renormalize the value to } [0, 1].\\
\end{split}
$$

In machine learning, the Boltzmann Machie and the Restricted Boltzmann Machie use the Maxwell–Boltzmann distribution to model a datapoint $$x$$. 

### Boltzmann Machine

A Boltzmann Machine projects an input data $$x$$ from a higher dimensional space to a lower dimensional space, forming a condensed representation of the data: latent factors. It contains visible units ($$A, B, C, D, \dots$$) for data input $$x$$ and hidden units (blue nodes) which are the latent factors of the input $$x$$. All nodes are connected together with a bi-directional weight $$W_{ij}$$. 

<div class="imgcap">
<img src="/assets/know/bolt2.png" style="border:none;width:30%">
</div>
(Source Wikipedia)

We use $$ W_{ij} $$ to model the connection between unit $$i$$ and $$j$$ with binary state $$s_i$$ and $$s_j$$ $$ \in \{0, 1\}$$. If $$s_i$$ and $$s_j$$ are the same, we want $$W_{ij}>0$$. Otherwise, we want $$W_{ij}<0$$. 

The energy between unit $$i$$ and $$j$$ is defined as:

$$
E(i, j) = - W_{ij} s_i s_j - b_i s_i
$$

The energy function of the system is the sum of all units:

$$
\begin{split}
E &= - \sum_{i < j} W_{ij}s_i s_j - \sum_i \theta_i s_i  \quad \text{ or} \quad E(x) & = -x^T U x - b^T x \\
\end{split}
$$

If $$s_i=s_j=1$$, we want $$W_{ij}$$ to be positive. If it is right, the energy drops by "- W_{ij} s_i s_j". Otherwise, the energy will increase. Hence, the energy function is equivalent to a cost function in training a model.

Using Boltzmann distribution, the PDF for $$x$$ is 

$$
\begin{split}
P(x) & = \sum_{x^{'}} P(x, x^{'}) \\
 & = \sum_{x^{'}} \frac{e^{-E(x, x^{'})}}{Z} \\
\end{split}
$$

Define free energy as:

$$
\begin{split}
\mathcal{F}(x) & = -\log \sum_{x^{'}} e^{-E(x, x^{'})} \\
\end{split}
$$

So

$$
\begin{split}
P(x) &= \frac{e^{-\mathcal{F}(x)}}{Z} \quad \text{where } Z = \sum_{x^{'}} e^{-\mathcal{F}(x^{'})} \\
\end{split}
$$

Take the negative log:

$$
\begin{split}
- \log p(x) & = \mathcal{F}(x) + \log Z \\
& = \mathcal{F}(x) + \log (\sum_{x^{'}} e^{-\mathcal{F}(x^{'})})  \\
- \frac{\partial \log p(x)}{\partial \theta} & = \frac{\partial \mathcal{F}(x)}{\partial \theta} + \frac{1}{\sum_{x^{'}} e^{-\mathcal{F}(x^{'})}} \frac{\partial  }{\partial \theta} \sum_{x^{'}} e^{-\mathcal{F}(x^{'})} \\
 & = \frac{\partial \mathcal{F}(x)}{\partial \theta}  - \sum_{x^{'}} \frac{e^{-\mathcal{F}(x^{'})}}{\sum_{x^{'}} e^{-\mathcal{F}(x^{'})}}  \frac{\partial \mathcal{F}(x^{'})}{\partial \theta}  \\
 & = \frac{\partial \mathcal{F}(x)}{\partial \theta}  - \sum_{x^{'}} \frac{e^{-\mathcal{F}(x^{'})}}{Z}  \frac{\partial \mathcal{F}(x^{'})}{\partial \theta}  \\
& = \frac{\partial \mathcal{F}(x)}{\partial \theta} - \sum_{x^{'}} p(x^{'}) \frac{\partial \mathcal{F}(x^{'})}{\partial \theta} \\
\end{split}
$$

### Restricted Boltzmann Machine (RBM)

<div class="imgcap">
<img src="/assets/know/bolt3.png" style="border:none;width:30%">
</div>

([Source](http://deeplearning.net/tutorial/rbm.html))


To determine the activation of the unit $$i$$, we compute the sum of all votes from all neighbors $$j$$ and then add a bias. 

$$z = \sum_j W_{ij} a_j + b_i$$

The result is passed to a sigmoid function to determine the probability $$p_{on} = \sigma(z)$$. We sample from $$p_{on}$$ to set $$s_i$$ to 1 or 0.  

