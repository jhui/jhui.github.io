---
layout: post
comments: true
mathjax: true
priority: 970000
title: “Machine learning - Restricted Boltzmann Machines.”
excerpt: “Machine learning - Restricted Boltzmann Machines.”
date: 2017-01-15 12:00:00
---

(Personal notes)

### Maxwell–Boltzmann distribution

Maxwell–Boltzmann distribution models the distribution of speeds of the gas molecules at a certain temperature:

<div class="imgcap">
<img src="/assets/know/bolt.png" style="border:none;width:40%">
</div>

(Source Wikipedia)

Here is the Maxwell–Boltzmann distribution equation: 

$$
\begin{split}
f(v) & = \sqrt{ \left (\frac{m}{2\pi kT} \right)^{3}  } 4\pi v^2  e^{\frac{-mv^2}{2kT}}  \\
\end{split}
$$

$$f(v)$$ gives the probability density function (PDF) of molecules with velocity $$v$$ which is a function of the temperature $$T$$. The probability density decreases exponentially with $$v$$. As the temperature increases, the probability of a molecule having higher velocity also increases. Since the number of molecules remains constant, the peak is therefore lower. 

### Energy based model

In Maxwell–Boltzmann statistics, the probability distribution is defined using an energy function $$E(x)$$:

$$
\begin{split}
P(x) & = \frac{e^{-E(x)}}{Z} \\ 
\end{split}
$$

where $$ Z = {\sum_{x^{'}} e^{-E(x^{'}) }}$$ is called the **partition function**, and it renormalizes the probability between 0 and 1.
 
By defining an energy function $$E(x)$$ for an energy based model like the Boltzmann Machie or the Restricted Boltzmann Machie, we can compute the probability distribution $$P(x)$$.

### Boltzmann Machine

A Boltzmann Machine projects an input data $$x$$ from a higher dimensional space to a lower dimensional space, forming a condensed representation of the data: latent factors. It contains visible units ($$A, B, C, D, \dots$$) for data input $$x$$ and hidden units (blue nodes) which are the latent factors of the input $$x$$. All nodes are connected together with a bi-directional weight $$W_{ij}$$. 

<div class="imgcap">
<img src="/assets/know/bolt2.png" style="border:none;width:30%">
</div>
(Source Wikipedia)

$$s_i \in \{0, 1 \}$$ is the binary state for unit $$i$$. We use $$ W_{ij} $$ to model the connection between unit $$i$$ and $$j$$. If $$s_i$$ and $$s_j$$ are the same, we want $$W_{ij}>0$$. Otherwise, we want $$W_{ij}<0$$. Intuitively, $$W_{ij}$$ indicates whether two units are positively or negatively related. If it is negatively related, one activation may turn off the other. 

The energy between unit $$i$$ and $$j$$ is defined as:

$$
E(i, j) = - W_{ij} s_i s_j - b_i s_i
$$

So the energy is increased if $$s_i = s_j = 1$$ and $$W_{ij}$$ is wrong. i.e. $$W_{ij} <0 $$.

The energy function of the system is the sum of all units:

$$
\begin{split}
E &= - \sum_{i < j} W_{ij}s_i s_j - \sum_i \theta_i s_i  \quad \text{ or} \quad E(x) & = -x^T U x - b^T x \\
\end{split}
$$

> The energy function is equivalent to the cost function in deep learning.

Using Boltzmann statistics, the PDF for $$x$$ is 

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

And,

$$
\begin{split}
e^{-\mathcal{F}(x) }  = \sum_{x^{'}} e^{-E(x, x^{'})} \\
\end{split}
$$

Therefore,

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
\end{split}
$$

Its gradient is:

$$
\begin{split}
- \frac{\partial \log p(x)}{\partial \theta} & = \frac{\partial \mathcal{F}(x)}{\partial \theta} + \frac{1}{\sum_{x^{'}} e^{-\mathcal{F}(x^{'})}} \frac{\partial  }{\partial \theta} \sum_{x^{'}} e^{-\mathcal{F}(x^{'})} \\
 & = \frac{\partial \mathcal{F}(x)}{\partial \theta}  - \sum_{x^{'}} \frac{e^{-\mathcal{F}(x^{'})}}{\sum_{x^{'}} e^{-\mathcal{F}(x^{'})}}  \frac{\partial \mathcal{F}(x^{'})}{\partial \theta}  \\
 & = \frac{\partial \mathcal{F}(x)}{\partial \theta}  - \sum_{x^{'}} \frac{e^{-\mathcal{F}(x^{'})}}{Z}  \frac{\partial \mathcal{F}(x^{'})}{\partial \theta}  \\
& = \frac{\partial \mathcal{F}(x)}{\partial \theta} - \sum_{x^{'}} p(x^{'}) \frac{\partial \mathcal{F}(x^{'})}{\partial \theta} \\
& = \frac{\partial \mathcal{F}(x)}{\partial \theta} - \mathbb{E}_{x^{'} \sim p} [ \frac{\partial \mathcal{F}(x^{'})}{\partial \theta}  ] \\
\end{split}
$$

### Restricted Boltzmann Machine (RBM)

In Boltzmann Machines, visible units or hidden units are fully connected with each other. In Restricted Boltzmann Machine (RBM), units in the same layer are not connected. The units in one layer is only fully connected with units in the next layer.

<div class="imgcap">
<img src="/assets/know/bolt3.png" style="border:none;width:30%">
</div>

([Source](http://deeplearning.net/tutorial/rbm.html))

The energy function for an RBM:

$$
\begin{split}
E(v, h) = -a^T v - b^T h - v^T W h \\ 
\end{split}
$$

Probability:

$$
\begin{split}
P(v, h) & = \frac{1}{Z} e^{-E(v, h)} \\
P(v) &= \frac{1}{Z} \sum_h e^{-E(v, h)}
\end{split}
$$

Conditional probability:

$$
\begin{split}
P(h \vert v) & = \prod_i P(h_i \vert v) \\
P(v \vert h) & = \prod_j P(v_j \vert h) \\
\end{split}
$$

$$
\begin{split}
P(h_j = 1 \vert v) &= \sigma \big( b_j + \sum_i W_{ij} v_i  \big) \\
P(v_i = 1 \vert h) &= \sigma \big( a_i + \sum_j W_{ij} h_j  \big) \\
\end{split}
$$

$$
\begin{split}
P(v^k_i = 1 \h) = \frac{ e^{ a^k_i + \sum_j W^k_{ij} h_j } }{Z}
\end{split}
$$

