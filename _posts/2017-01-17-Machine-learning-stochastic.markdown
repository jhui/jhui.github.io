---
layout: post
comments: true
mathjax: true
priority: 100000
title: “Machine learning - Stochastic process - Naive Bayes and Gaussian Process”
excerpt: “Machine learning (Naive Bayes and Gaussian Process)”
date: 2017-03-08 12:00:00
---

### Bayes' theorem

Bayes' theorem calculates the conditional probability (Probability of A given B):

$$
P(A \vert B) = \frac{P(B \vert A) P(A)}{P(B)}
$$

Sometimes the result of the Bayes' theorem can surprise you . Let's say there is an un-common decease that only 0.1% of the population has it. We develop a test of 99% accuracy for positive and negative result. What is the chance that you have this decease if your test is positive. The answer is only 9% which is much smaller than you may think. The intuition is that even the test looks accurate, it generates more false positive than true positive because the decease is not common. With Bayes' theorem, we can demonstrate why it is only 9%.

(Note: 'd' stands for having the decease and '+' stands for testing positive.)

$$
\begin{equation}
\begin{split}
P( d \vert + ) &= \frac{P(+ \vert d) P(d)}{P(+)} \\
\\
&= \frac{P(+ \vert d) P(d)}{P(+, d) + P(+, \neg d)} \\
\\
&= \frac{P(+ \vert d) P(d)}{P(+ \vert d)P(d) + P(+ \vert \neg d)P(\neg d)} \\
\\
& = \frac{0.99 * 0.001}{0.99*0.001 + 0.01 *0.999} \\
\\
& = 0.0902 \\
\end{split}
\end{equation}
$$


Proof of the Bayes theorem:

$$
\begin{equation}
\begin{split}
P(A,B) & = P(A \vert B) P(B) \\
\\
P(A \vert B) & = \frac{P(A,B)}{P(B)}, \quad P(B \vert A) & = \frac{P(A,B)}{P(A)} \\
\\
\frac{P(A \vert B)}{P(B \vert A)} &= \frac{P(A)}{P(B)} \\
\\
P(A \vert B) &= \frac{P(B \vert A) P(A)}{P(B)}
\end{split}
\end{equation}
$$

### Naive Bayes Classifier

Naive Bayes Classifier classifies objects $$Y$$ given observation $$ x_1, x_2, \dots, x_n $$ based on Bayes' theorem. For example, if we draw an object from a basket, if it is red and round $$(x)$$, what is the chance that it is an apple $$(Y)$$?

$$
\begin{equation}
\begin{split}
P(Y \vert x_1, x_2, \dots, x_n) & = \frac{P(x_1, x_2, \dots, x_n \vert Y) P(Y)}{P(x_1, x_2, \dots, x_n)} 
\end{split}
\end{equation}
$$
 
 Assume $${x_i}$$ and $${x_j}$$ are independent of each others. (An red object does not increase the chance that it is round.) i.e.: $$  P(x_1, x_2, \dots \vert Y) = P(x_1 \vert Y) P(x_2 \vert Y) \dots $$
 
 $$
\begin{equation}
\begin{split}
 P(Y \vert x_1, x_2, \dots, x_n) & = \frac{P(x_1 \vert Y) P(x_2 \vert Y) ... P( x_n \vert Y) P(Y)}{P(x_1, x_2, \dots, x_n)} \\
 \\
 & = \frac{P(Y) \prod^n_{i=1} P( x_i \vert Y) }{P(x_1, x_2, \dots, x_n)}  \\
 \\
 & \propto P(Y) \prod^n_{i=1} P( x_i \vert Y)
\end{split}
\end{equation}
$$

#### Example
Let's say we want to determine whether an object we pick in a basket is an apple or a grape given the object is red, round & sweet

$$
\begin{equation}
\begin{split}
P(apple \vert red, round, sweet) & = \frac{P(red, round, sweet \vert apple) P(apple)}{P(red, round, sweet)} \\
\\
& = \frac{P(red | apple) P(round | apple) P(sweet | apple ) P(apple)}{P(red, round, sweet)} \\
\end{split}
\end{equation}
$$

$$ P(red, round, sweet) $$ is the same regardless of which object to pick, therefore we ignore the denominator in comparing object's probability.
\begin{equation}
\begin{split}
P(apple \vert red, round, sweet) & \propto P(red | apple) P(round | apple) P(sweet | apple ) P(apple)
\end{split}
\end{equation}
 
Say, half of the object in the basket are apples, and 

$$\text{Given: } P(red \vert apple) = 0.7, P(round \vert apple) = 0.98, P(sweet \vert apple) = 0.9 $$

The chance that the object is an apple:

$$
P(apple \vert red, round, sweet) \propto 0.7 \cdot 0.98 \cdot 0.9 \cdot 0.5 = 0.6174
$$

If a quarter of the object in the basket are grapes, the corresponding chance for a grape is:

$$\text{Given: } P(red \vert grape) = 0.6, P(round \vert grape) = 0.95, P(sweet \vert grape) = 0.3 $$

$$
P(grape \vert red, round, sweet) \propto 0.6 \cdot 0.95 \cdot 0.3 \cdot 0.25 = 0.04
$$

So if an object is red, round and sweet, it is likely an apple.

### Bayes inference

Let's go through an example in determine the infection rate ($$ \theta $$) for Flu. On the first day of the month, we got 10 Flu testing results back. We call this **evidence** $$(x)$$. For example, our new evidence shows that 4 samples are positive. $$ (x=4 \text{ out of 10}) $$.  We can conclude that ($$ \theta = 4/10 = 0.4 $$). Nevertheless we want to study further by calculating the probability of $$ x $$ given a specific $$\theta$$.

The chance of $$ x $$ given $$\theta$$ is the **likelihood** of our evidence. In simple term, likelihood is how likely the evidence (4 samples are positive) will happen with different infection rate. Based on simple probability theory:

$$
\begin{align} 
P(x \vert \theta) &= {n \choose x} \theta^x(1-\theta)^{n-x}\\ &= \dfrac{10!}{4!(10-4)!}\theta^4 (1-\theta)^{10-4}\\  
\end{align}
$$

We plot the probability for values $$\theta$$ range from 0 to 1. **Maximum likelihood estimation (MLE)** is where $$\theta$$ has the highest probability. Here MLE is 0.4. i.e. the most likely value for $$\theta$$ with the evidence $$x$$ is 0.4.
<div class="imgcap">
<img src="/assets/ml/inf.png" style="border:none;width:60%">
</div>

We called $$P(x \vert \theta)$$ the **likelihood**. The blue line above is the likelihood. 

> In a stochastic process, the computed value is not a scalar but a probability distribution. We may predict $$\theta$$ has a mean value of 0.3 with a variance 0.1. If we need a scalar value, we sample it from the probability distribution.

We had also collected 100 months of data on the infection rate for flu. This data forms a **belief** of the Flu infection rate which we usually call **prior** $$ P(\theta) $$. The orange line below is $$ P(\theta) $$. It is the prior belief of the probability distribution for $$ \theta $$. The distribution centers around 0.14 meaning we generally believe the average infection rate of Flu is 0.14. The blue line is the probability distribution of $$ \theta $$ just based on the new evidence. (likelihood) Obviously the new evidence is different from the prior (belief).

<div class="imgcap">
<img src="/assets/ml/inf2.png" style="border:none;width:60%">
</div>

We either suspect that the much higher infection rate of the new evidence is caused by low sampling size of the new evidence or we encounter a new strength of Flu that we need to re-adjust the **prior**.

#### Recap the terminology

>  People discuss Bayes with a lot of terminologies. We pause a while to summarize the terms again.

$$
P(H \vert D) = \frac{P(D \vert H) P(H)}{P(D)}
$$

**Evidence** is some data we observed. For example, 4 samples out of 10 are infected. We can treat **belief** as a hypothesis. For example, we can start with a belief of 0.14 infection rate with a variance 0.02 and later use Bayes inference to refine it with data that we collect.

| **posterior probability**| $$P(H \vert D) $$ | The refined belief with additional given new evidence. <br> The new belief after we collect 1000 samples. |
| **likelihood** | <nobr>$$ P(D \vert H) $$ </nobr>| The probability of the evidence given the belief. <br> The chance of have 4 positive samples out of 10 for different values of the infection rate.|
| **prior** | <nobr>$$ P(H) $$ </nobr>| The probability of the belief prior to new evidence. <br> Our hypothesis which will later combine with new data to refine it to a posterior.|
| **marginal probability** | <nobr>$$ P(D) $$ </nobr>| The probability of seeing that data. <br> The probability of seeing 4 positive samples under all possible infection rate values. |

#### Bayesian inference (continue)

> Bayesian inference try to draw better prediction based on evidence and prior belief.

With Bayes inference,  we calculate the **posterior** probability $$P(\theta \vert x)$$ using Bayes theory. We re-calibrate our belief given the new evidence:

$$
P(\theta \vert x) = \frac{P(x \vert \theta) P(\theta)}{P(x)}
$$

Here is the plot of the posterior. The yellow line is the prior $$P(\theta)$$, the blue line is the likelihood. The green line is the posterior calculated with the Bayes theory.
<div class="imgcap">
<img src="/assets/ml/inf3.png" style="border:none;width:100%">
</div>

It is clear that the posterior moves closer to the likelihood with the new evidence. The posterior 

$$
P(\theta \vert x) = \frac{P(x \vert \theta) P(\theta)}{P(x)}
$$

depends on the prior and the likelihood. Even the prior peaks at 0.14 (the red line). The new evidence has a likelihood that contradict it. The likelihood at the red line is lower and therefore their posterior (their multiplication), shown as the green dot, is lower. At the vertical green line, even the prior is lower but it is compensated by the higher likelihood and therefore the peak of the posterior moves towards the likelihood.

<div class="imgcap">
<img src="/assets/ml/inf3c.png" style="border:none;width:100%">
</div>

In short, with new evidence, we shift our belief towards the direction of the newly observed data. We re-calibrate the belief with new data.

In Bayes inference, we can start with certain belief for the probability distribution of $$\theta$$. Alternatively, we can start with a random guess say it is uniformed distributed like the orange line below. e.g. the chance that $$\theta=0.1$$ is the same as $$ \theta=0.9 $$. The first calculated posterior with this prior will equal to the likelihood.

<div class="imgcap">
<img src="/assets/ml/infr.png" style="border:none;width:60%">
</div>

With new round of evidence, we compute the new posterior given the evidence. This posterior becomes our belief (prior) in the next iteration when another round of evidence is collected. After many iterations, our prior will converge to the true probability distribution of $$\theta$$.

How far the posterior will move towards to the new evidence? It depends on the size of the evidence. Obviously, a large sampling size will move the posterior closer. The following plot indicates that the larger the size of the evidence, the further it moves towards the evidence. The variance decreases also and we are more certain on its value.

<div class="imgcap">
<img src="/assets/ml/inf4.png" style="border:none;width:100%">
</div>

### Programming

The source code can be find [here.](https://github.com/jhui/machine_learning/blob/master/machine_learning/bayesian_inference.py) which use PyMC3 as the Bayes inference engine.

Here is some coding of creating a prior and calculate the posterior afterwards. Then we sample 5000 data from the posterior distribution. We use pymc3 for the Bayes inference model. Since the code is self-explanatory, we encourage you understand the concept directly from the code.
```python
# Markov chain Monte Carlo model
# Create an evidence. 7 infection out of 10 people
people_count = 10
chance_of_infection = 0.4
infection_count = int(people_count * chance_of_infection)  # Number of infections

# Create a model using pymc3
with pm.Model() as model:
   # A placeholder
   people_count = np.array([people_count])
   infections = np.array([infection_count])

   # We use a beta distribution to model the prior.
   # The beta distribution takes in 2 parameters. 
   # For example, if both is 1, the distribution is uniformly distributed.
   # We have 10, 60 which is the distribution we used in the prior in plot.
   theta_prior = pm.Beta('prior', 10, 60)
   
   # We create a model with our evidence and the prior (assuming bi-normial distribution)
   observations = pm.Binomial('obs', n = people_count
                                   , p = theta_prior
                                   , observed = infections)

   # We use the strategy of maximize a posterior
   start = pm.find_MAP()

   # Sampling 5000 data from the calculated posterior probability.
   trace = pm.sample(5000, start=start)
   
   # Coding in plotting the graph
   ...
```

### Gaussian distribution
We are used to define a function $$ y = f(x) $$ with input $$x$$ to compute $$y$$ deterministic. Alternatively, we can use a stochastic model to define such relationship.  For example, given a GPA of 3.8 for a student, the student will earn an average of $70K salary with a variance of $10K.

$$\text{Probability density function (PDF)} = p(X=3.8) = f(3.8) $$. 
<div class="imgcap">
<img src="/assets/ml/gpa.png" style="border:none;width:40%">
</div>

> Note:  $$f$$ is a probability distribution - not a scalar value ($$\mu = $70K \text{ and } \sigma=$10k$$).

In the following diagram, we assume $$p(X=x)$$ follows a gaussian distribution: 
<div class="imgcap">
<img src="/assets/gm/g0.png" style="border:none;width:60%">
</div>

$$
\text{Probability density function (PDF) of a Gaussian dist.} = p(X=x) = f(x) = \frac{e^{-(x - \mu)^{2}/(2\sigma^{2}) }} {\sigma\sqrt{2\pi}}
$$

We can sample data using the PDF of a Gaussian distribution:

$$
x \sim \mathcal{N}{\left(
\mu 
,
\sigma
\right)}
$$

68% of data will be within 1 $$  \sigma $$ from the $$ \mu $$ and 95% within 2 $$  \sigma $$.

> In many real world examples, data follows a gaussian distribution. 

Now, we try to model relationship like $$ y = f(x) $$. For example, we want to model the relationship between the body height and the body weight for San Francisco residents. We collect the information from 1000 adult residents and plot the data below with each red dot represents 1 person:

<div class="imgcap">
<img src="/assets/gm/auto.png" style="border:none;width:80%">
</div>

The corresponding of $$ PDF = probability(height=h, weight=w)$$ in 3D:

<div class="imgcap">
<img src="/assets/gm/auto2.png" style="border:none;width:60%">
</div>

Let us generalize the problem to a multivariate gaussian distribution.

$$
f(x) = f(x_1, x_2, \dots, x_p)
$$

For a multivariate vector:

$$
x = \begin{pmatrix}
x_1 \\
\vdots \\
x_p
\end{pmatrix}
$$

The PDF of a multivariate Gaussian distribution is:
<div class="imgcap">
<img src="/assets/gm/g1.png" style="border:none;width:60%">
</div>

with covariance matrix $$ \sum $$:

$$
\sum = \begin{pmatrix}
    E[(x_{1} - \mu_{1})(x_{1} - \mu_{1})] & E[(x_{1} - \mu_{1})(x_{2} - \mu_{2})] & \dots  & E[(x_{1} - \mu_{1})(x_{p} - \mu_{p})] \\
    E[(x_{2} - \mu_{2})(x_{1} - \mu_{1})] & E[(x_{2} - \mu_{2})(x_{2} - \mu_{2})] & \dots  & E[(x_{2} - \mu_{2})(x_{p} - \mu_{p})] \\
    \vdots & \vdots & \ddots & \vdots \\
    E[(x_{p} - \mu_{p})(x_{1} - \mu_{1})] & E[(x_{p} - \mu_{p})(x_{2} - \mu_{2})] & \dots  & E[(x_{n} - \mu_{p})(x_{p} - \mu_{p})]
\end{pmatrix}
$$

which $$E$$ is the expected value for all data points. We will explain this with an example later.

The notation for sampling becomes:

$$
x 
\sim \mathcal{N}{\left(
\mu
,
\sum
\right)}
$$

which

$$
x =
\begin{pmatrix}
x_1 \\
\vdots \\
x_p
\end{pmatrix}
$$

$$
\mu =
\begin{pmatrix}
\mu_1 \\
\vdots \\
\mu_p
\end{pmatrix}
$$

Let's go back to our weight and height example to illustrate it.

$$
x = \begin{pmatrix}
weight \\
height 
\end{pmatrix}
$$

From our data, we compute the mean weight = 190 lb and mean height = 70 inches:

$$
\mu = \begin{pmatrix}
190 \\
70
\end{pmatrix}
$$

What is the covariance matrix for? Each element in the covariance matrix measures how one variable is related to another.  For example, $$ E_{21} $$ measures how $$\text{height } (x_2)$$ is related with $$ \text{weight} (x_2)$$. If weight increases with height, we will expect $$E_{21}$$ to be positive.

$$
E_{21} = E[(x_{2} - \mu_{2})(x_{1} - \mu_{1})] 
$$

Let's get into detail on computing $$ E_{21} $$ above. To simplify, we consider we got only 2 data points (200 lb, 80 inches) and (180 lb, 60 inches). 

$$
 E_{21} = E[(x_{2} - \mu_{2})(x_{1} - \mu_{1})] = E[(x_{height} - 70)(x_{weight} - 190)]
$$

$$
E_{21} = E[(x_{height} - 70)(x_{weight} - 190)] = \frac{1}{2} \left( ( 80 - 70) \times (200 - 190)  + ( 60 - 70) \times (180 - 190)  \right)
$$

After computing all 1000 data, here is the value of $$ \sum $$:

$$
\sum = \begin{pmatrix}
    100 & 25 \\
    25 & 50 \\
\end{pmatrix}
$$

$$
x \sim \mathcal{N}{\left(
\begin{pmatrix}
190 \\
70
\end{pmatrix}
,
\begin{pmatrix}
    100 & 25 \\
    25 & 50 \\
\end{pmatrix}
\right)}
$$

$$ E_{21} $$ measures the co-relationship between variables $$x_2$$ and $$x_1$$. Positive values means both are positively related. With no surprise, $$ E_{21} $$ is positive because weight increases with height. If two variables are independent of each other, it should be 0 like:

$$
\sum = \begin{pmatrix}
    100 & 0 \\
    0 & 50 \\
\end{pmatrix}
$$

#### Calculate the probability of $$x < value$$

The covariance variable $$\sum$$ can have the form:

$$
\sum = \begin{pmatrix}
    \sigma^2_1 & \rho \sigma_1 \sigma_2 \\
    \rho \sigma_1 \sigma_2 & \sigma^2_2 \\
\end{pmatrix}
$$

The probability of $$X_2 \le z$$ given $$ X_2 = x $$ :
 
$$
P(X_1 \le z\,|\, X_2 = x) = \Phi\left(\frac{z - \rho x}{\sqrt{1-\rho^2}}\right)
$$

$$
\Phi(x) = \int^\infty_{x=0} \frac{e^{-(x - \mu)^{2}/(2\sigma^{2}) }} {\sigma\sqrt{2\pi}}
$$

#### Coding

Here we sample data from a 2-variable gaussian distribution. From the covariance matrix, we can tell x is positively related with y as $$\sum_{21} \text{ and } \sum_{12}$$ is positive.
```python
mean = [0, 2]
cov = [[1, 2], [3, 1]]

x, y = np.random.multivariate_normal(mean, cov, 5000).T
plt.plot(x, y, 'x')
plt.axis('equal')
plt.show()
```

<div class="imgcap">
<img src="/assets/ml/d1.png" style="border:none;width:60%">
</div>

Here, we plot the probability distribution for (y, x).
```python
from scipy.stats import multivariate_normal

x, y = np.mgrid[-1:1:.01, -1:1:.01]  # x (200, 200) y (200, 200)
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x; pos[:, :, 1] = y   # pos (200, 200, 2)

mean = [-0.4, -0.3]
cov = [[2.1, 0.2], [0.4, 0.5]]
rv = multivariate_normal(mean, cov)
p = rv.pdf(pos)                      # (200, 200)
plt.contourf(x, y, p)
plt.show()
```

<div class="imgcap">
<img src="/assets/ml/sc2.png" style="border:none;width:60%">
</div>

### Multivariate Gaussian Theorem

Given a Gaussian Distribution for

$$
x = \begin{pmatrix}
x_1\\
x_2 
\end{pmatrix}
\sim \mathcal{N}{\left( \mu , \sigma \right)}
$$

The posterior conditional for $$p(x_1 \vert x_2) $$ is given below. This formular is particular important for the Gaussian process in the later section. For example, if we have samples of 1000 graduates with their GPAs and their salaries, we can use this theorem to predict salary given a GPA $$ P(salary \vert GPA)$$ by creating a Gaussian distribution model with our 1000 training datapoints. 

[Proof of this theorem can be found here.](https://stats.stackexchange.com/questions/30588/deriving-the-conditional-distributions-of-a-multivariate-normal-distribution) We will not go into details on the proof. But with the assumption that $$x$$ is Gaussian distributed, The co-relation of $$x_1$$ and $$x_2$$ is defined by $$\mu$$ and $$\sum$$. So given the value of $$x_2$$, we can restrict the possible values of $$x_1$$ and form its distribution.
 
<div class="imgcap">
<img src="/assets/ml/th1.png" style="border:none;width:80%">
</div>
Source: Nando de Freitas UBC Machine learning class.

In the equation above, we can predict $$ P(salary \vert GPA_i)$$ by creating a gaussian distribution for $$ GPA_i$$ with $$ \mu_{salary \vert GPA_{1}}  $$ and $$ \sum_{salary \vert GPA_{1}}$$. For example, with a 3.4 GPA, we can predict a graduate can earn a mean salary of $65K with a variance of $5K. The details on computing the mean and the variance will discuss next.

### Gaussian Process (GP)

The intuition of the Gaussian Process is simple. If 2 points have similar input, their output should be similar. If we get closer and closer to a training datapoint, we are more certain about the prediction.

If a student with a GPA 3.5 earns $70K a year, another student with a GPA 3.45 should earn something very similar. In GP, we use our training dataset to build a Gaussian distribution to make prediction. For each prediction, we output a mean and a $$\sigma$$. For example, we can predict a student with GPA 3.3 earns $$\mu = $65K$$ and $$ \sigma= $5K$$ and a student with GPA 2.5 earns $$\mu = $50K$$ and $$ \sigma= $15K$$.
$$\sigma$$ is proportional to the un-certainty. Here, we are more certain about the salary for a GPA of 3.3 because there is a training datapoint close by. (GPA 3.5)

In GP, instead of computing $$\sum$$, we compute $$K$$ trying to measure the similarity between datapoint $$x^i$$ and $$x^j$$.

| K | $$x^1$$ | $$x^2$$ | ... | $$x^n$$ |
| $$x^1$$ | 1 | $$k(x^1, x^2)$$ | | $$k(x^1, x^n)$$|
| $$x^2$$ | $$k(2^1, x^1)$$ | 1 | | $$k(x^2, x^n)$$ |
| ... | 
| $$x^n$$ | $$k(x^n, x^1)$$ | $$k(x^n, x^2)$$ | | 1 |

which kernel $$k$$ is a function measuring the similarity of the 2 datapoints (1 means the same). There are many possible kernel functions, we will use an exponential square distance for our kernel.

$$
K_{i,j} = k(x^i, x^j) = e^{-\frac{1}{2}(x^i - x^j)^2}
$$

Note: In previous section $$x_1 $$ means the weight of a datapoint. Here $$x^1 $$ means datapoint 1.  $$X^x_1$$ means the weight of datapoint 3.

With all the training data, we can create a Gaussian model:

$$
\begin{pmatrix}
f 
\end{pmatrix}

\sim \mathcal{N}{\left( \mu , K \right)}
$$

Let's demonstrate it again with our 2 datapoints (150lb, 66 inches) and (200lb, 72 inches). Here we are building a Gaussian model in predicting weight from height.

$$
\begin{pmatrix}
150 \\
200
\end{pmatrix}

\sim \mathcal{N}{\left(
175
,
\begin{pmatrix}
K_{11} & K_{12}\\
K_{21} & K_{22}
\end{pmatrix}
\right)}
$$

with 175 as the mean of the weight and $$K_{i,j} = k(x^i, x^j) $$ measures the similarity of the height of the datapoints $$x^i, x^j $$. The notation above just mean we can sample a vector $$f$$ on weight

$$
f = \begin{pmatrix}
150 \\
200
\end{pmatrix}
$$

from $$\mathcal{N}$$ which is model by the datapoint (150, 66) and (200, 72).

Now let's say we want to predict $$ f^1, f^2 $$ for input $$x^{1}_*, x^{2}_*$$. The model change to:

$$
\begin{pmatrix}
150 \\
200 \\
f^1 \\
f^2 \\
\end{pmatrix}

\sim \mathcal{N}{\left(
175
,
\begin{pmatrix}
K_{11} & K_{12} & K_{13} & K_{14} \\
K_{21} & K_{22} & K_{23} & K_{24} \\
K_{31} & K_{32} & K_{33} & K_{34} \\
K_{41} & K_{42} & K_{43} & K_{44} \\
\end{pmatrix}
\right)}
$$

Let's understand what does it mean again. For example, we have a vector contain 4 persons height 

$$
x = (66, 72, 67, 68) 
$$ 

We can use $$\mathcal{N}$$ to sample the possible weight for these people:

$$ (150, 200, f^1, f^2) $$

We know the first 2 values from the training data and we try to compute the distribution for $$ f^1 \text{and } f^2 $$. (what is their $$ \mu$$ and $$\sigma$$.) Now, instead of predicting just 2 values, we can have input over a range of values

$$x = (66, 72, 66.01, 66.02, 66.03, ..., 89.99, ... ) $$ 

and use $$\mathcal{N}$$ to sample vector:

$$ (150, 200, f^1, f^2, f^3, ...) $$

For example, our first output sample from $$\mathcal{N}$$ is

$$(150, 200, 150.1, 150.2, 149.6, ...) $$ 

We can plot the output against input. The line below looks like a regular deterministic function. i.e. $$weight = g(height)$$.

<div class="imgcap">
<img src="/assets/ml/w.png" style="border:none;width:40%">
</div>

We can sample from $$\mathcal{N}$$ 2 more times and each one generate 1 solid line below. 

<div class="imgcap">
<img src="/assets/ml/w2.png" style="border:none;width:40%">
</div>

We see that the 2 training data forces $$\mathcal{N}$$ to have all lines interest at the blue dots. The Gaussian model $$\mathcal{N}$$ generates lines of samples to map a height to a weight. As mentioned before, each line behave like a non-stochastic function. Therefore, in mathematical term, GP is always charactered as building a Gaussian model to discribe the distribution of functions. (lines) 

If we keep sampling, we will start visually recognize the mean and the range of value for each $$x_i$$. For example, the red dot and the blue line below indicate the average and the range of weight for people 67 inches tall.

<div class="imgcap">
<img src="/assets/ml/w3.png" style="border:none;width:70%">
</div>

We are not going to solve the problem by sampling. Instead we will solve it analytically.

Back to

$$
\begin{pmatrix}
150 \\
200 \\
f^1 \\
f^2 \\
\end{pmatrix}

\sim \mathcal{N}{\left(
175
,
\begin{pmatrix}
K_{11} & K_{12} & K_{13} & K_{14} \\
K_{21} & K_{22} & K_{23} & K_{24} \\
K_{31} & K_{32} & K_{33} & K_{34} \\
K_{41} & K_{42} & K_{43} & K_{44} \\
\end{pmatrix}
\right)}
$$

We can generalize the expression as follow which $$f$$ is the label (weight) of the training set and $$f_{*}$$ is the weights that we want to predict for $$x_*$$:

$$
\begin{pmatrix}
f \\
f_{*}
\end{pmatrix}
$$

Now we need to solve $$ p(f_* \vert f) $$ with the Gaussian model.

$$
\begin{pmatrix}
f \\
f_{*}
\end{pmatrix}
\sim \mathcal{N}{\left(
\begin{pmatrix}
\mu \\
\mu_{*}
\end{pmatrix}
,
\begin{pmatrix}
K & K_{s}\\
K_{s}^T & K_{ss}\\
\end{pmatrix}
\right)}
$$


Recall from the previous section on Multivariate Gaussian Theorem, if we have a model 

$$
\begin{pmatrix}
x_1 \\
x_2 
\end{pmatrix}

\sim \mathcal{N}{\left(
\begin{pmatrix}
\mu_1 \\
\mu_2
\end{pmatrix}
,
\begin{pmatrix}
\sum_{11} & \sum_{12} \\
\sum_{21}  & \sum_{22} \\
\end{pmatrix}
\right)}
$$


We can solve $$p(x_1 \vert x_2) $$ by:
<div class="imgcap">
<img src="/assets/ml/ss1.png" style="border:none;width:50%">
</div>

Now, we apply it to solve 

$$
 p(f_{*} \vert f)
$$

For the training dataset, let input $$x$$ has the gaussian distribution:

$$
\begin{split}
f & \sim \mathcal{N}{\left(\mu, \sigma^2\right)} \\
& \sim \mu + \sigma(\mathcal{N}{\left(0, 1\right)}) \\
\end{split}
$$

And let the Gaussian distribution for $$f_{*}$$ be:

$$
\begin{split}
f_{*} & \sim \mathcal{N}{\left(\overline{\mu}_{*}, \overline{\Sigma}_{*}\right)   } \\
& \sim \overline{\mu}_{*} + L\mathcal{N}{(0, I)}
\end{split}
$$

which L is defined as

$$
LL^T = \overline{\Sigma}_{*}
$$

and from the Multivariate Gaussian Theorem:

$$
\begin{split}
p(f_*\vert  x_*,  x,  f) & = \mathcal N( \mu_{*} + K_{ s}^T  K^{-1}( f -  \mu ),  K_{ss} -  K_{s}^T  K^{-1}  K_{s}) \\
\overline{\mu}_{*} & = \mu_{*} + K_{ s}^T  K^{-1}( f -  \mu ) \\
\overline{\Sigma}_{*} & = K_{ss} -  K_{s}^T  K^{-1}  K_{s}
\end{split}
$$

In the coding example below, we are solving the equation above by modeling training data created from $$ y = \sin(x)$$. In this example, $$ \mu=\mu_{*}=0 $$ as the mean value of a $$sin$$ function is 0. Our equation will therefore simplify to:

$$
\begin{split}
\overline{\mu}_{*} & = K_{ s}^T  K^{-1} f  \\
\overline{\Sigma}_{*} & = K_{ss} -  K_{s}^T  K^{-1}  K_{s}\\
K & = L L^T \quad \text{(use Cholesky to decompose K)}
\end{split}
$$

With matrix A and vector b, we can use linear algebra to solve x.

$$ 
Ax = b
$$

Will we use the notation

$$
x = A \backslash b 
$$

to demonstrate we are applying linear alegra to solve x from A and b.

$$
\begin{split}
\text{Let u be } u &= L^{-1} f \\
Lu &= f \\
  u &= L \backslash f \\
\text{Let v be } v &= L^{-T} u = L^{-T} (L \backslash f) \\
  L^{T}v &= u \\
   v &= L^{T} \backslash u    \\
   &= L^{T} \backslash (L \backslash f)      
\end{split}
$$

Apply $$K = LL^{T} $$ and the equation above

$$
\begin{split}
\overline{\mu}_{*} & = K_{ s}^T  K^{-1} f  \\
 & = K_{ s}^T (LL^{T})^{-1} f \\
 & = K_{ s}^T L^{-T}L^{-1} f \\
 & = K_{ s}^T L^{-T} (L \backslash f) \\
 & = K_{ s}^T L^T \backslash ( L \backslash f ) \\
\end{split}
$$

First we are going to prepare our training data and label it with a $$\sin$$ function. The training data contains 5 datapoints. $$(x_i=-4, -3, -2, -1 \text{ and} 1)$$. 
```python
Xtrain = np.array([-4, -3, -2, -1, 1]).reshape(5,1)
ytrain = np.sin(Xtrain)      # Our output labels.
```

Testing data: We create 50 new datapoint linear distributed between -5 and 5 to be predicted by the Gaussian process.
```python
# 50 Test data
n = 50
Xtest = np.linspace(-5, 5, n).reshape(-1,1)
```

Here we define a kernel measure the similarity of 2 datapoint using an exponential square kernel.
```python
# A kernel function (aka Gaussian) measuring the similarity between a and b. 1 means the same.
def kernel(a, b, param):
    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    return np.exp(-.5 * (1/param) * sqdist)
```

To compute the Kernel ($$K, K_s, K_{ss}) $$
```python
K = kernel(Xtrain, Xtrain, param)                        # Shape (5, 5)
K_s = kernel(Xtrain, Xtest, param)                       # Shape (5, 50)
K_ss = kernel(Xtest, Xtest, param)                       # Kss Shape (50, 50)
```

We will use the Cholesky decomposition to compute $$ K = LL^T$$.
```python
L = np.linalg.cholesky(K + 0.00005*np.eye(len(Xtrain)))  # Shape (5, 5)
```

Compute the mean output for our prediction $$\overline{\mu}_*$$. As we assumem $$ \mu_{*} = \mu = 0$$, the equation becomes:

$$
\begin{split}
\overline{\mu}_{*} & = K_{ s}^T L^T \backslash ( L \backslash f )
\end{split}
$$


```python
L = np.linalg.cholesky(K + 0.00005*np.eye(len(Xtrain)))  # Add some nose to make the solution stable 
                                                         # Shape (5, 5)

# Compute the mean at our test points.
Lk = np.linalg.solve(L, K_s)                             # Shape (5, 50)
mu = np.dot(Lk.T, np.linalg.solve(L, ytrain)).reshape((n,)) # Shape (50, )
```

Compute $$\sigma$$
```python
# Compute the standard deviation.
s2 = np.diag(K_ss) - np.sum(Lk**2, axis=0)               # Shape (50, )
stdv = np.sqrt(s2)                                       # Shape (50, )
```

Sample $$f_*$$ so we can plot it.

$$
f_*  \sim \overline{\mu}_{*} + L\mathcal{N}{(0, I)}
$$

$$
\begin{split}
\overline{\Sigma}_{*} & = K_{ss} -  K_{s}^T  K^{-1}  K_{s} \\
 \overline{\Sigma}_{*}  & = LL^T \\
\end{split}
$$

```python
L = np.linalg.cholesky(K_ss + 1e-6*np.eye(n) - np.dot(Lk.T, Lk))    # Shape (50, 50)
f_post = mu.reshape(-1,1) + np.dot(L, np.random.normal(size=(n,5))) # Shape (50, 3)
```

We sample 3 possible output shown in orange, blue and green line. The gray area is where within 2 $$\sigma$$ of $$\mu$$. Blue dot is our training dataset. Here, at blue dot, $$sigma$$ is closer to 0. For points between the training datapoints, $$\sigma$$ increases reflect its un-certainty because it is not close to the training dataset points. When we move beyond $$x=1$$, we do not have any more training data and result in large $$\sigma$$.

<div class="imgcap">
<img src="/assets/ml/gp4.png" style="border:none;width:100%">
</div>

Here is another plot of posterior after seeing 5 evidence. The blue dot is where we have training datapoint and the gray area demonstrate the un-certainty (variance) of the prediction.
<div class="imgcap">
<img src="/assets/ml/s4.png" style="border:none;width:100%">
</div>

Source wikipedia.
