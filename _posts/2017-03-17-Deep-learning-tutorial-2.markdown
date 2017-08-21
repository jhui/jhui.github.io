---
layout: post
comments: true
mathjax: true
priority: 200
title: “Deep learning without going down the rabbit holes. (Part 2)”
excerpt: “Part 2 of the deep learning.”
date: 2017-03-17 14:00:00
---

[Part 1 of the deep learning can be found here.](https://jhui.github.io/2017/03/18/Deep-learning-tutorial/)
### Overfit

In part one, we prepare a model with 4 layers of computational nodes. The solutions for $$ W $$ are not unique. Should we prefer one solution over the other?

```
Layer 1:
      [[ 1.10727659,  0.22189273,  0.13302861,  0.2646622 ,  0.2835898 ],
       [ 0.23522207,  0.01791731, -0.01386124,  0.28925567,  0.187561  ]]
	   
Layer 2:
 	[[ 0.9450821 ,  0.14869831,  0.07685842,  0.23896402,  0.15320876],
       [ 0.33076781,  0.02230716,  0.01925127,  0.30486342,  0.10669098],
       [ 0.18483084, -0.03456052, -0.01830806,  0.28216702,  0.07498301],
       [ 0.11560201,  0.05810744,  0.021574  ,  0.10670155,  0.11009798],
       [ 0.17446553,  0.12954657,  0.03042245,  0.03142454,  0.04630127]]), 
	   
Layer 3:	   
	[[ 0.79405847,  0.10679984,  0.00465651,  0.20686431,  0.11202472],
       [ 0.31141474,  0.01717969,  0.00995529,  0.30057041,  0.10141655],
       [ 0.13030365, -0.09887915,  0.0265004 ,  0.29536237,  0.07935725],
       [ 0.07790114,  0.04409276,  0.01333717,  0.10145275,  0.10112565],
       [ 0.12152267,  0.11339623,  0.00993313,  0.02115832,  0.03268988]]), 
	   
Layer 4:	  
	[[ 0.67123192],
       [ 0.48754364],
       [-0.2018187 ],
       [-0.03501616],
       [ 0.07363663]])]
```

This leads us to a very important topic. When we increase the complexity of our model, we risk the chance of modeling the noise also. If we do not have enough sample data to cancel out the noise, we make bad predictions. But, even without the noise, we can still have a bad model. Let’s walk through an example. We start with training samples with input values and output range from 0 to 20. How will you model an equation to link the data points below?

<div class="imgcap">
<img src="/assets/dl/d1.png" style="border:none;width:70%">
</div>

One possibility is
$$
y = x
$$ which is simple and just misses 2 on the left and 2 on the right.
<div class="imgcap">
<img src="/assets/dl/d2.png" style="border:none;width:70%">
</div>

But, when we show it to our android, Pieter, who has a much higher computational capability than us, he models it as:

$$
y = 1.9  \cdot 10^{-7}  x^9 - 1.6 \cdot 10^{-5} x^8 + 5.6 \cdot 10^{-4} x^7 - 0.01 x^6  + 0.11 x^5 - 0.63 x^4 + 1.9  x^3 - 2.19  x^2 + 0.9 x - 0.0082
$$

<div class="imgcap">
<img src="/assets/dl/d3.png" style="border:none;width:70%">
</div>

Which model is correct? The answer is “don’t know”. Someone thinks the first one is simpler. Simple explanations deserve more credit. But, if you show it to a stock broker, they may say the second curve looks closer to the market closing price of a stock.
Instead, we should ask whether our model is too “custom tailored” to the training dataset and fails to make generalized predictions. Even the second model fits the training data completely, but it can make poor predictions if the true model is proven to be a simple straight line.

> When we have a complex model but not enough data, we can easily **overfit** the model which memorizes the training data but fails to make generalized predictions.

#### Validation
**Machine learning is about making predictions.** A model that has 100% accuracy in training can still be a bad model. For that, we split our data into three parts: say 80% for training, 10% for validation and 10% for testing. During training, we use the training dataset to build models with different hyperparameters. We run those models with the validation dataset and pick the one with the highest accuracy. This strategy works if the validation dataset is similar to what we want to predict. But, as a last safeguard, we use the 10% testing data for a final insanity check. This testing data is for a final verification, but not for model selection. If your testing result is dramatically different from the validation result, the data should be randomized more, or more data should be collected.

> Validation data is used to select hyperparameters and models with the highest accuracy. The testing data should never use to select a model. It should be used very infrequently just for insanity check.

#### Visualization 

Let's us visualize some of the overfit problem. We can train a model to create a boundary to separate the blue dots from the white dots below. A complex model (the left picture) can produce ad hoc boundaries compared to a low complexity model. In the yellow-circled area, if the two left white dot samples is missing in our training dataset, a complex model may create an odd shaped boundary just to include this white dot. A low complexity model produces a smoother surface, which sometimes makes more generalized predictions. A complex model is vulnerable to outliers. For example, the white dot in the green circle may be incorrectly labeled as white. In a simple model, the boundary is much smoother and may simply ignore this outlier.

<div class="imgcap">
<img src="/assets/dl/of.png" style="border:none;width:90%">
</div>

Recall from Pieter’s equation, our sample data can be modeled nicely with the following equations:

$$
y = 1.9  \cdot 10^{-7}  x^9 - 1.6 \cdot 10^{-5} x^8 + 5.6 \cdot 10^{-4} x^7 - 0.01 x^6  + 0.11 x^5 - 0.63 x^4 + 1.9  x^3 - 2.19  x^2 + 0.9 x - 0.0082
$$

In fact, there are infinite solutions having different polynomial orders $$ x^k$$.

Compared with the linear model $$ y = x $$, the
$$ \| coefficient \| $$ 
in Pieter’s equation is higher but it seems they are cancelling out each other. In addition, the higher the order, the harder to train the model because of the much bigger search space for the parameters. In additional, some areas in the search space have very steep gradients that making training tough.

Let us create a polynomial model with order 5 to fit our sample data.

$$
y = c_5 x^5 + c_4 x^4 + c_3 x^3 + c_2 x^2 + c_1 x + c_{0}
$$

We need more iterations to train this model and it is less accurate than a model with order three.

<div class="imgcap">
<img src="/assets/dl/p1.png" style="border:none;width:60%">
</div>

So why don’t we focus on making a model with the right complexity. In real life problems, a complex model is the only way to push accuracy to an acceptable level. A better solution is to introduce methods to reduce overfitting rather than make the model simpler. One simple solution is to add more training data such that it is much harder to overfit. Here, double our training data produces a model closer to a straight line. Unfortunately, labeling large training dataset in real problems can be expensive.

<div class="imgcap">
<img src="/assets/dl/p2.png" style="border:none;width:60%">
</div>

> Complex models sometimes create ad hoc ways to fit training data that does not generalize. Increase the volume of training data  reduces overfit effectively but can be expensive to do.

### Regularization

**Regularization punishes over-complexity.** As we have observed before, there are many solutions to a DL problem. To have a close fit, however, the coefficient usually has a larger magnitude. 

$$
\|c\| = \sqrt{(c_5^2 + c_3^2 + c_3^2 + c_2^2 + c_1^2 + c_{0}^2)}
$$

Overfit model tends to have larger magnitude. To have our training not to overfit the training data, we add a penalty in the cost function to penalize large magnitude. In this example, we use a L2 norm (**L2 regularization**)
$$ ||W|| $$
as the penalty. 

$$
J = MSE + \lambda \cdot \|W\|
$$

This is called **regularization**. Here, we introduce another hyperparameter called regularization factor $$ \lambda $$ to penalize overfitting.

> Regularization favors less complex model if the cost reduction is less significant.

After many repetitions of trial, we pick $$ \lambda $$ to be 1. With the regularization, our model makes better predictions.

<div class="imgcap">
<img src="/assets/dl/p3.png" style="border:none;width:60%">
</div>

Like other hyperparameters, the selection process is trial and error. In fact, we use a relatively high $$ \lambda $$
because there are only a few trainable parameters in the model. In real life problems, $$ \lambda $$ is much lower because the number of trainable parameters are usually in millions.

There is another interesting observation. The loss may jump up sharply and drop to the previous level after a few thousand iterations. 
```
Iteration 87000 [2.5431744485195127]
Iteration 88000 [2.525734745522529]
Iteration 89000 [223.88938197268865]
Iteration 90000 [195.08231216279583]
Iteration 91000 [3.0582387198108449]
Iteration 92000 [2.4587727305339286]
```

If we look into the equation, we realize the gradient

$$
y = x_{i}
$$

$$
\frac{\partial y}{\partial c_i}   = i  x^{i-1}
$$

can be very steep which cause the learning rate problem discussed before. For example, from iteration 10,000 to 11,000, the coefficient for $$ x^5 $$
changes slightly from -0.000038 to -0.000021 but the cost jumps from 82 to 34,312. (It takes 2000 iterations to undo the jump.)

```
Iteration 10000 [82.128486144319155, 
 array([[  1.66841311e+00],
       [  5.15883024e-01],
       [  1.05449372e-01],
       [ -1.15910560e-01],
       [  1.32065116e-02],
       [ -3.83863265e-04]])]
Iteration11000 [34312.355686493174, 
  array([[  1.82722611e+00],
       [  5.83582807e-01],
       [  1.28332499e-01],
       [ -1.11263342e-01],
       [  1.24705708e-02],
       [ -2.05131433e-04]])]
```

When we build our model, we try out a polynomial model with order of nine. Even after a long training, the model still makes poor predictions. We decide to start with a model with order of three and increase it gradually. This is another example to demonstrate why we should start with a simple model first. At seven, we find the model is too hard to train. The following is what a seven-order model predicts:
<div class="imgcap">
<img src="/assets/dl/p4.png" style="border:none;width:60%">
</div>

### Diminishing and exploding gradient

**Cannot train a model if gradients explode or diminish.** From our previous example, we demonstrate the importance of tracing the gradient at different layers to troubleshoot problems. In our online dating model, we log
$$ \| gradient \| $$ 
for each layers.

```python
iteration 0: loss=45.6
layer 0: gradient = 226.1446016395799
layer 1: gradient = 566.6340440894377
layer 2: gradient = 371.4818585197662
layer 3: gradient = 371.7283667292019
iteration 10000: loss=12.28
layer 0: gradient = 39.087735791986816
layer 1: gradient = 70.66776450168192
layer 2: gradient = 40.95339598248693
layer 3: gradient = 49.27868977928858
...
iteration 90000: loss=11.78
layer 0: gradient = 8.695315741501654
layer 1: gradient = 13.149909360278247
layer 2: gradient = 9.97983678446837
layer 3: gradient = 7.053793667949491
```

A couple of things we need to monitor. Is the magnitude too high or too small? If the magnitude remains high, some parameters may be oscillating.

```
iteration 0: ... dW1=1.183e+04 dW2=5.929e+06 ...
iteration 200: ... dW1=4.458e+147 dW2=2.203e+150 ...
iteration 400: ... dW1=1.656e+291 dW2=8.184e+293 ...
iteration 600: ... dW1=nan dW2=nan ...
```

If the gradient is small, the network learns slowly. In the following log, the gradient diminishes from the right layer (layer 6) to the left layer (layer 0). Layer 0 almost learns nothing.
```
iteration 0: loss=553.5
layer 0: gradient = 2.337481559834108e-05
layer 1: gradient = 0.00010808796151264163
layer 2: gradient = 0.0012733936924033608
layer 3: gradient = 0.01758514040640722
layer 4: gradient = 0.20165907211476816
layer 5: gradient = 3.3937365923146308
layer 6: gradient = 49.335409914253
iteration 1000: loss=170.4
layer 0: gradient = 0.0005143399278199742
layer 1: gradient = 0.0031069449720360883
layer 2: gradient = 0.03744160389724748
layer 3: gradient = 0.7458109132993136
layer 4: gradient = 5.552521662655173
layer 5: gradient = 16.857110777922465
layer 6: gradient = 37.77102597043024
iteration 2000: loss=75.93
layer 0: gradient = 4.881626633589997e-05
layer 1: gradient = 0.0015526594728625706
layer 2: gradient = 0.01648262093048127
layer 3: gradient = 0.35776408953278077
layer 4: gradient = 1.6930852548061421
layer 5: gradient = 4.064949014764085
layer 6: gradient = 12.7578637206897
```

Many deep networks suffer from this gradient diminishing problem. Let’s come back to backpropagation to understand the problems.

<div class="imgcap">
<img src="/assets/dl/chain.png" style="border:none;width:70%">
</div>

The gradient descent is computed as:

$$
\frac{\partial J}{\partial l_{1}} = \frac{\partial J}{\partial l_{2}} \frac{\partial l_{2}}{\partial l_{1}}  = \frac{\partial J}{\partial l_{3}} \frac{\partial l_{3}}{\partial l_{2}}  \frac{\partial l_{2}}{\partial l_{1}} 
$$ 

$$
\frac{\partial J}{\partial l_{1}} = \frac{\partial J}{\partial l_{10}} \frac{\partial l_{10}}{\partial l_{9}} \cdots  \frac{\partial l_{2}}{\partial l_{1}} 
$$ 

As indicated, the gradient descent depends on the loss $$ \frac{\partial J}{\partial l} $$ as well as the gradients $$ \frac{\partial l_{k+1}}{\partial l_{k}}, \frac{\partial l_{k}}{\partial l_{k-1}} \dots $$. Let's look at a sigmoid activation function. If $$ x $$ is higher than 5 or smaller than -5, the gradient is close to 0. Hence, in these regions the node learns close to nothing regardless of the loss.

<div class="imgcap">
<img src="/assets/dl/sigmoid2.png" style="border:none;width:80%">
</div>

The derivative of a sigmoid function behaves like a gate to the loss signal. If the input is > 5 or <-5, the derivative is so small, it blocks most of the loss signal to propagate backwards. So, nodes on its left sides learn little.

In addition, the chain rule in the gradient descent has a multiplication effect. If we multiple numbers smaller than one, it diminishes quickly. On the contrary, if we multiple numbers greater than one, it explodes.

$$ 
0.1 \cdot 0.1 \cdot 0.1 \cdot 0.1 \cdot 0.1 = 0.00001 
$$

$$ 
5 \cdot 5 \cdot 5 \cdot 5 \cdot 5 = 3125
$$

The gradient of deep network may diminish quickly or explode. To solve this problem, we have designed new computation nodes that can bypass an individual layer. Microsoft Resnet (2015) has 152 layers. Instead of backpropagate the cost through a long chain of nodes, Resnet have both pathways to backpropagate it through the convolution layer or by-passed through the arc.
(Source Kaiming He, Xiangyu Zhang … etc)

<div class="imgcap">
<img src="/assets/dl/resnet.png" style="border:none;width:80%">
</div>

<div class="imgcap">
<img src="/assets/dl/resnet2.png" style="border:none;width:40%">
</div>

Let's get into details in demonstrating a node called LSTM that used in the natural language processing (NLP). In LSTM, the state of a cell is updated by

$$
C_t = gate_{forget} \cdot C_{t-1} + gate_{input} \cdot \tilde{C}
$$

If we want to bypass this node such that the input $$C_{t-1}$$ becomes the output $$C_t$$, we just need to set $$ gate_ {forget}=1 $$ and $$ gate_{input} = 0 $$. So one way to address the diminishing gradient problem is to design a computation node with pathway that can bypass a node.

> Diminish gradient can be mitigated by designing computation nodes that providing a bypassing pathway for the node.

#### Gradient clipping

To avoid gradient explosion, we can apply gradient clipping to restrict values of the gradient.

In the example below, we use TensorFlow ( a popular deep learning platform from Google) to clip the gradient if it is too big.
```python
params = tf.trainable_variables()
opt = tf.train.GradientDescentOptimizer(learning_rate)
for b in xrange(time_steps):
    gradients = tf.gradients(losses[b], params)
    clipped_gradients, norm = tf.clip_by_global_norm(gradients, 5.0)
```

The gradients are rescaled according to the ratio $$\frac{5}{max(\| gradient \|, \| global \text{ } gradients \|)}$$
```
# t is each individual gradient
# global_norm is the norm for all gradients
# clip_norm is set to 5.0
global_norm = sqrt(sum([l2norm(t)**2 for t in t_list]))
t_list[i] * clip_norm / max(global_norm, clip_norm)
```

### Classification

A very important part of deep learning is classification. We have mentioned object recognition before. These are classification problems asking the question: what is this? For example, for Android Pieter to safely walk in a street, he needs to learn what is a traffic light, and is there a pedestrian walking towards him. Classification applies to non-visual problems also. We classify whether an email is a spam or should we approve/disapprove a loan etc…

<div class="imgcap">
<img src="/assets/dl/street2.jpg" style="border:none;width:40%">
</div>

Like solving regression problem using DL, we use a deep network to compute a value. In classification, we call this **a score**. We apply a classifier to convert the score to a probability value between 0 and 1. To train the network, the training dataset provides the answers to the classification which we call **true label**. (like classifying an image as a school bus, a truck or an airplane)

#### Logistic function (sigmoid function)

A score computed by a network takes on any value. A classifier squashes it to a probability value. For example, to classify whether an email is a spam or the medical test is positive, we apply a **logistic function** (also called a **sigmoid function**) to the score. If the output probability is lower than 0.5, we predict "no", otherwise we predict "yes".

$$
p = \sigma(score) = \frac{1}{1 + e^{-score}}
$$

<div class="imgcap">
<img src="/assets/dl/sigmoid.png" style="border:none;width:40%">
</div>


#### Softmax classifier

For many classification problems, we categorize an input to one of the many classes. For example, we can classify an image to one of the 100 possible object classes. We use a softmax classifier to compute K probabilities, one per class for an input image (the combined probabilities remains 1).

<div class="imgcap">
<img src="/assets/dl/deep_learner2.jpg" style="border:none;width:70%;">
</div>

The network computes K scores per image. The probability that an image belongs to the class $$ i $$ will be.

$$
p_i =  \frac{e^{score_i}}{\sum_{c \in y} e^{score_c}} 
$$

For example, the school bus above may have a score of (3.2, 0.8, 0) for the class school bus, truck and airplane respectively. The probability for the corresponding class is

$$
p_{\text{bus}} =  \frac{e^{3.2}}{ e^{3.2} + e^{0.8} + e^0} = 0.88
$$

$$
p_{\text{truck}} =  \frac{e^{0.2}}{ e^{3.2} + e^{0.8} + e^0} = 0.08
$$

$$
p_{\text{airplane}} =  \frac{e^0}{ e^{3.2} + e^{0.8} + e^0} = 0.04
$$

```python
def softmax(z):
    z -= np.max(z)
    return np.exp(z) / np.sum(np.exp(z))

a = np.array([3.2, 0.8, 0])   # [ 0.88379809  0.08017635  0.03602556]
print(softmax(a))
```

To avoid the numerical stability problem caused by adding large exponential values, we subtract the inputs by its maximum. Adding or subtract a number from the input does not change the probability value in softmax.

$$
softmax(z) = \frac{e^{z_i -C}}{\sum e^{z_c  - C}} =  \frac{e^{-C} e^{z_i}}{e^{-C} \sum e^{z_c}} = \frac{e^{z_i}}{\sum e^{z_c}}
$$

```python
z -= np.max(z)
```

**logits** is defined as a mean to measure odd.

$$
logits = \log(\frac{p}{1-p})
$$

If we combine the softmax equation with the logits equation, it is easy to see that the score is the logit. 

$$
p = softmax(score) = \frac{e^{z_i}}{\sum e^{z_c}} \\
logits = score
$$

That is why in many literatures and APIs, logit and score are interchangeable when a softmax classifier is used. However, more than one functions that map scores to probabilities will meet the definition of logits. Sigmoid function is one of them.

> Softmax is the most common classifier among others.

#### SVM classifier

A SVM classifier uses the maximum margin loss (Hinge loss) and the L2-regularization for the cost function:

$$
J = \sum_{j\neq y_i} \max(0, score_j - score_{y_i} + 1) + \frac{\lambda}{2} \| W \|^2
$$

which $$ y_i$$ is the true label for datapoint $$i$$.

The class having the highest score will be the class predicted. The maximum margin loss function will be discussed later. Its main objective is to create a boundary to separate classes with the largest possible margin.

<div class="imgcap">
<img src="/assets/ml/svm2.png" style="border:none;width:40%">
</div>

#### Softmax vs SVM 

Softmax outputs probabilities which have better interpretations than scores in SVM. The value of a score or the difference between 2 score has little interpretable values. In SVM, if a datapoints is not inside a margin, it has 0 data cost. SVM cares more about in fixing wrong predictions or predictions that are not too certain. Softmax add data cost even the predictions are close. In practice, the performance between both are small.


### Entropy 

Entropy measures the amount of information. In data compression, it represents the minimum number of bits in representing data. By definition, entropy is defined as:

$$
H(y) = \sum_{i \in S} y_i \log \frac{1}{y_i} = -\sum_{i \in S} y_i \log y_{i}
$$

Suppose we have a string composed of "a", "b" or "c" with a 25% chance of occurrence for "a", 25% for "b", and 
50% for "c". The entropy is:

$$ 
\begin{align}
H & = 0.25 \log(\frac{1}{0.25}) + 0.25 \log(\frac{1}{0.25})  + 0.5 \log(\frac{1}{0.5}) \\
H  &= 0.25 \cdot 2 + 0.25 \cdot 2  +  0.5 \cdot 1 \\
& = 1.5 \\
\end{align}
$$

We will use bit 0 to represent 'c' and bits 10 for 'a' and bits 11 for 'b'. In average, we need 1.5 bits per character to represent a string.

#### Cross entropy

$$
H(y, \hat{y}) = \sum_i y_i \log \frac{1}{\hat{y}_i} = -\sum_i y_i \log \hat{y}_i
$$

If entropy measures the minimum of bits to encode information using the most optimized scheme. Cross entropy measures the minimum of bits to encode $$y$$ using the wrong optimized scheme from $$\hat{y}$$. The cross entropy is always higher than entropy unless both distributions are the same: you need more bits to encode the information if you use a less optimized scheme. 

#### KL Divergence:

$$
\mbox{KL}(y~||~\hat{y}) = \sum_i y_i \log \frac{y_i}{\hat{y}_i}
$$

KL divergence is simply cross entropy minus entropy (the extra bits needed to encode the data):

$$
\begin{align}
\mbox{KL}(y~||~\hat{y}) &= \sum_i y_i \log \frac{y_i}{\hat{y}_i} \\
&= \sum_i y_i \log \frac{1}{\hat{y}_i} - \sum_i y_i \log \frac{1}{y_i}  \\
&= H(y, \hat{y}) - H(y) 
\end{align}
$$

In machine learning, KL Divergence measures the difference between 2 probability distributions. 

<div class="imgcap">
<img src="/assets/ml/kl.png" style="border:none;width:80%">
</div>

Source wikipedia.

It becomes a very good cost function to penalize the difference between the true label and the prediction made by the model. In our previous example, we classify a picture as either a bus, a truck or an airplane (bus, truck, airplane). The true label probability distribution for a bus is (1, 0, 0) and our model prediction can be (0.88, 0.08, 0.04). The KL Divergence becomes a very good cost function to train our model.

> KL Divergence is even better for stochastic processes when the true label $$y_i$$ is stochastic rather than deterministic. (true label values smaller than 1)

Recall:

$$
\begin{align}
\mbox{KL}(y~||~\hat{y}) &= H(y, \hat{y}) - H(y) \\
\end{align}
$$

The entropy of the true label is constant, therefore $$\frac{\partial{H(y)}}{\partial{w}} = 0 $$.

$$
\begin{align}
\mbox{KL}(y~||~\hat{y}) &= H(y, \hat{y}) - H(y) \\
\frac{\partial{\mbox{KL}(y~||~\hat{y})}} {\partial{w}} &= \frac{\partial{H(y, \hat{y})}}{\partial{w}} - \frac{\partial{H(y)}}{\partial{w}} \\ 
\frac{\partial{\mbox{KL}(y~||~\hat{y})}} {\partial{w}} &= \frac{\partial{H(y, \hat{y})}}{\partial{w}} \\
\end{align}
$$

i.e. Optimizing KL-divergence is the same as optimizing cross entropy.

> We can optimize the simpler cross entropy instead of the KL-divergence.

The cross entropy cost of our example is:

$$
\begin{align}
H(y, \hat{y}) &= -\sum_i y_i \log \hat{y}_i \\
&= - 1 \log{0.88} - 0 \cdot \log{0.08} - 0 \cdot \log{0.04} =   - \log{0.88} \\
\end{align}
$$

Since $$y_i$$ is always equal to 1 for the true label and 0 otherwise, the cross entropy can be simplified as:

$$
\begin{align}
H(y, \hat{y}) &= -\sum_i y_i \log \hat{y}_i \\
&= -\sum_i \log \hat{y}_i \\
\end{align}
$$

> Cross entropy is a very popular cost function for deep learning.


### Maximum likelihood estimation (MLE)

In the last section, we state without proof that KL divergence and cross entropy is a good cost function. Let's derive it. What is our objective in training a model? Our objective is to train the model parameters so that the likelihood of our model is maximized (MLE). We want to find $$W$$ such that our training dataset is most possible comparing to other $$W$$. The probability of our model matching the true labels of all training datapoints is:

$$
p(y | x, W) =  \prod_{i =1}^n p(y_{i} |  x_{i}, W)
$$

for training datapoints $$x_1, x_2, \cdots x_n$$ and true label $$y_1, y_2, \cdots y_n$$

Recall:

$$
p(y_{i} |  x_{i}, W) = \hat{y_i}
$$

For our first datapoint, $$ y_{1} = (1, 0, 0) $$ and $$ \hat{y_{1}} $$ = $$ (0.88, 0.08, 0.04)$$.

$$
p(y_{1} |  x_{1}, W) = 0.88
$$

#### Negative log-likelihood (NLL)

Logarithm is a monotonic function. Hence optimize $$J$$ is the same as optimize $$\log{J}$$. However, $$ \log{(prob.)}$$ change the sign and the direction of the function. Maximum likelihood translates to minimize the negative log-likelihood (NLL).

> Maximize the "maximum likelihood estimation" (MLE) is the same as minimizing the negative log-likelihood. 

i.e. Maximize:

$$
MLE \equiv p(y | x, W) =  \prod_{i =1}^n p(y_{i} |  x_{i}, W)
$$

is the same as minimize

$$
\begin{align}
nll & \equiv  - \log{p(y | x, W)} \\
& = - \log{ \prod_{i =1}^n p(y_{i} |  x_{i}, W) } \\
& = - \sum_{i =1}^n \log{ p(y_{i} |  x_{i}, W) } \\
& = - \sum_{i =1}^n \log{ \hat{y_{i}}}  \\
\end{align}
$$

With no coincident, this is the cross entropy in our previous example. 

> NLL is a very common method in finding the cost function.

#### Logistic loss

As an exercise to demonstrate NLL, we derive the logistic loss from NLL.

In logistic regression, we compute the probability by

$$
{p(y_{i} |  x_{i}, W)} = \sigma(z_{i}) = \frac{1}{1 + e^{-z_i}}
$$

Apply NLL,

$$
\text{nnl} = - \log {p(y_{i} |  x_{i}, W)} = - \log{ \frac{1}{1 + e^{-z_j}} } = - \log{1} + \log (1 + e^{-z_j}) 
$$

This becomes the logistic loss:

$$
\text{nnl} = \sum\limits_{i}^n    \log (1 + e^{- z}) 
$$

$$
\text{nnl} = \sum\limits_{i}^n \log (1 + e^{- y_i W^T x_{i}}) 
$$

#### NLL for softmax

This is the Softmax cost function defined as the NLL and the corresponding gradient:

$$
\begin{align}
J(w) &= -  \sum_{i=1}^{N}  \log p(\hat{y}^i = y^i \vert x^i, w ) \\
\nabla_{score_{k}} J &= \begin{cases}
                        p - 1 \quad & \hat{y} = y \\
                        p & \text{otherwise}
                    \end{cases}
\end{align}
$$

```python
def softmax_loss(x, y):
    probs = np.exp(x - np.max(x, axis=1, keepdims=True)) # Subtract from the max for better numeric stability.
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = x.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
```

### Cost function
**Deep learning is about knowing your costs. Good cost function builds good models.** But, how do you define the cost? San Francisco is about 400 miles from Los Angeles. It costs about $80 for the gas. When you order food from a restaurant, they do not deliver to homes more than a few miles away. From their perspective, the cost grows exponentially with distance. So, there are many definitions of cost. In DL, our objective is not knowing the value of the cost, but finding a set of $$W$$ to make cost the lowest. Therefore, we have much flexibility for the cost function as long as we can find this set of $$W$$. For example, some cost functions ignore outliers better than others. There are other important considerations including how easy can we optimize the cost function. 

#### Cross entropy cost function
Apply NLL to find the cost function for a classification problem:

$$
p(y | x, W) =  \prod_n p(y_{i} |  x{i}, W)
$$

Recall from the previous section:

$$
\text{nll} =  - \sum_n \log {\hat{y_{i}}}
$$

> A cross entropy cost function and a softmax classifier is one of the most popular combinations in solving classification problems.

#### SVM loss (aka Hinge loss or Max margin loss)

$$
J = \sum_{j\neq y_i} \max(0, score_j - score_{y_i} + 1)
$$

If the margin between the score of the true label and the score of a class is less than 1, we add it to the cost.  For example, with a score of (8, 14, 9.5, 10) and the true label is the last entry:

$$
\begin{align}
J & = max(0, 8 - 10 + 1) + max(0, 14 - 10 + 1) + max(0, 9.5 - 10 + 1) \\
& = 0 + 5 + 0.5 = 5.5
\end{align}
$$

For SVM, the cost function creates a boundary with a maximum margin to separate classes. 

<div class="imgcap">
<img src="/assets/ml/svm2.png" style="border:none;width:40%">
</div>

For training points that are predicted correctly and outside the margin, the cost is 0. Within the margin, all training datapoints add a penalty to the cost function. i.e. If you prediction is certain and it is correct, there is no penalty. However, we penalize predictions that are in-correct or below some certainty.

<div class="imgcap">
<img src="/assets/ml/svm22.png" style="border:none;width:40%">
</div>

#### Mean square error (MSE)

$$
MSE = \frac{1}{N} \sum_i (h_i - y_{i})^2
$$

We have used MSE for regression problems before. We can use MSE in classification. But, in practice, we use cross entropy loss instead. Classification uses a classifier to squash scores to a probability between 0 and 1. The mapping is not linear. For a sigmod classifier, a large range of values is squeezed to 0 or 1 (for $$x < -5$$ or $$ > 5$$). As shown before, the partial derivative is close to 0 for those region. Based on the chain rule in the back propagation:

$$
\frac{\partial J}{\partial score} = \frac{\partial J}{\partial out} \frac{\partial out}{\partial score}
$$

The cost signal is hard to propagate backward in those regions regardless of cost. However, there is a way to solve this issue. Even the partial derivative of the sigmoid function in those regions is small but we can make 
$$ 
\frac{\partial J}{\partial out} 
$$ 
very large if the prediction is bad. The sigmoid function squashes values exponentially. We need a cost function that punishes bad predictions in the same scale to counter that. Squaring the error in MSE is not good enough. Cross entropy punishes bad predictions exponentially. That is why the cross entropy cost function trains better than MSE in the classification problems.

### Regularization

As mentioned before, we add regularization cost into the cost function to avoid overfitting. How does it impact our models? The idea of regularization is to train a model that makes generalized predictions. It forces the model not to memorize the small bits of an individual sample that is not part of the generalized features. The most effective solution is to add more training data. But it can be expensive to label data. Part of the overfit problem is that the model is too powerful. We can reduce the complexity of the model by reducing the number of features. We can remove features that can be directly or indirectly derived by others. We can eliminate some layers or switch to a design that explores better on the locality of the information. For example, using CNN for images to explore the spatial locality and LSTM for temporal locality. 

#### L0, L1, L2 regularization

Large W tends to overfit. L2 regularization adds the L2 norm to a cost function as the regularization cost.

$$
J = error + \lambda \cdot \|W\|
$$

$$
\|W\|_2 =  \sqrt{\sum\limits_{i} \sum\limits_{j} W_{ij}^2}
$$

Beside L2 norm, we can use L0 or L1 norm.

L0 regularization

$$
|| W ||_0 = \sum \begin{cases} 1, & \mbox{if } w \neq 0 \\ 0, & \mbox{otherwise} \end{cases}
$$

L1 regularization

$$
|| W ||_1 = \sum |w|
$$

#### Comparison among L0, L1 and L2 regularization

The derivative of L2 norm is easy to compute and it is smooth which works well with gradient descent. Hence L2 regularization is very popular. $$W^*_b$$ is the optimal $$W$$ without the regularization. By adding the L2 regularization (the circle), we shift the optimal point $$W^*$$ closer to 0. 

<div class="imgcap">
<img src="/assets/ml/L2.png" style="border:none;width:30%">
</div>

The shape of the L1 regularization is a rotated square instead of a circle since regularization cost is measured as $$ \vert W \vert$$ but not $$ \| W \| $$. L1 regularization tries to push some $$W^*_i$$ to zero rather than closer to 0. So it increase the sparsity of $$w$$.

<div class="imgcap">
<img src="/assets/ml/L1.png" style="border:none;width:35%">
</div>

L0 regularization adds a constant cost to the regular cost (blue curve) except at $$w=0$$. The green curve is the new cost which have a sudden drop at 0 to to the green dot. Therefore, it can push more $$w$$ to 0 comparing to L1 regularization.

<div class="imgcap">
<img src="/assets/ml/L00.png" style="border:none;width:50%">
</div>

>  L1 and L0 regularization promotes sparsity for $$w$$ which can be desirable to avoid overfitting when we have too many input features. 


#### Dropout

A non-intuitive regularization method called dropout discourages weights with large values. To avoid overfit, we do not want weights to be too dominating. By randomly dropping connections from one layer to the other layer, we force the network not to depend too much on a single node and try to draw prediction from many different pathways. This has an effect similar to forcing the weights smaller.

In the following diagram, for each iteration during training, we randomly drop off some connections.
<div class="imgcap">
<img src="/assets/dl/drop.png" style="border:none;width:40%">
</div>

Here is the code to implement the dropout for the forward feed and back propagation. In the forward feed, it takes a parameter on the percentage of nodes to be dropout. Notice that, dropout applies to training only.
```python
def dropout_forward(x, dropout_param):
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])

  mask = None
  out = None

  if mode == 'train':
    drop = 1 - p
    mask = (np.random.rand(*x.shape) < drop) / drop
    out = x * mask
  elif mode == 'test':
    out = x

  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)

  return out, cache

def dropout_backward(dout, cache):
  dropout_param, mask = cache
  mode = dropout_param['mode']
  
  if mode == 'train':
    dx = dout * mask
  elif mode == 'test':
    dx = dout
  return dx
```

> Dropout forces the decision not to dependent on few features. It behave like other regularization in constraining the magnitude of $$W$$.

### Deep learning network (Fully-connected layers) CIFAR-10

Let's put together everything to solve the CIFRA-10. CIFAR-10 is a computer vision dataset for object classification. It has 60,000 32x32 color images belonging to one of 10 object classes, with 6000 images per class.

(Source Alex Krizhevsky)
<div class="imgcap">
<img src="/assets/dl/cifra.png" style="border:none;width:60%">
</div>

We implement a fully connected network similar to the following to classify the CIFRA images. In our implementation, we allow the user to control how many hidden layers to create and the number of nodes per layer.
<div class="imgcap">
<img src="/assets/dl/fc_net.png" style="border:none;width:40%">
</div>

Let’s have some boiler plate code that we did before. This is the forward feed and the back propagation code for
$$
y = Wx + b
$$
and the ReLU.
```python
def affine_forward(x, w, b):
    out = x.reshape(x.shape[0], -1).dot(w) + b
    cache = (x, w, b)
    return out, cache

def affine_backward(dout, cache):
    x, w, b = cache
    dx = dout.dot(w.T).reshape(x.shape)
    dw = x.reshape(x.shape[0], -1).T.dot(dout)
    db = np.sum(dout, axis=0)
    return dx, dw, db

def relu_forward(x):
    out = np.maximum(0, x)
    cache = x
    return out, cache

def relu_backward(dout, cache):
    dx, x = None, cache
    dx = dout
    dx[x < 0] = 0
    return dx
```

We combine them to form a forward feed and a backpropagation "affine relu" layer. (Each affine relu layer is used as a hidden layer.)
```python
def affine_relu_forward(x, w, b):
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache

def affine_relu_backward(dout, cache):
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db
```

This is the softmax function.
```python
def softmax_loss(x, y):
    probs = np.exp(x - np.max(x, axis=1, keepdims=True)) # Subtract from the max for better numeric stability.
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = x.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
```

We are creating a FullyConnectedNet network with 3 hidden layers with (100, 50, 25) nodes respectively. We also initialize $$w$$ with a random normal distribution.
```python
class FullyConnectedNet(object):

    def __init__(self, hidden_dims, input_dim=3 * 32 * 32, num_classes=10, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32):
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        layers = [input_dim] + hidden_dims + [num_classes]
        # Initialize the W & b for each layers 
        for i in range(self.num_layers):
            self.params['W%d' % (i + 1)] = np.random.randn(layers[i], layers[i + 1]) * weight_scale
            self.params['b%d' % (i + 1)] = np.zeros(layers[i + 1])

        # Cast all parameters to the correct datatype
        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)
			
			
model = FullyConnectedNet([100, 50, 25], weight_scale=5e-2, dtype=np.float64)			
```

Here is the key part in computing the loss in the model class. We do a feed forward for the hidden and output layers. Use softmax to compute the loss. Add L2 regularization cost and compute all the gradients using the backpropagation.
```python
def loss(self, X, y=None):
    X = X.astype(self.dtype)
    # We reuse the same method for prediction. So we have the train mode and the test mode (make prediction.)
    mode = 'test' if y is None else 'train'

    layer = [None] * (1 + self.num_layers)
    cache_layer = [None] * (1 + self.num_layers)

    layer[0] = X

    # Feed forward for each layer define in FullyConnectedNet([100, 50, 25], ...)
    for i in range(1, self.num_layers):
        # Retrieve the W & b
        W = self.params['W%d' % i]
        b = self.params['b%d' % i]
        # Feed forward for one affine relu layer
        layer[i], cache_layer[i] = affine_relu_forward(layer[i - 1], W, b)

    last_W_name = 'W%d' % self.num_layers
    last_b_name = 'b%d' % self.num_layers
    # From the last hidden layer to the output layer, we do an affine op but not ReLU
    scores, cache_scores = affine_forward(layer[self.num_layers - 1],
                                          self.params[last_W_name],
                                          self.params[last_b_name])

    # If just making prediction, we return the scores										  
    if mode == 'test':
        return scores

    loss, grads = 0.0, {}
    # Compute the loss
    loss, dscores = softmax_loss(scores, y)

    # For each layer, add the regularization loss
    for i in range(self.num_layers):
        loss += 0.5 * self.reg * np.sum(self.params['W%d' % (i + 1)] ** 2)

    # Back progagation the output to the last hidden layer
    dx = [None] * (1 + self.num_layers)
    dx[self.num_layers], grads[last_W_name], grads[last_b_name] = affine_backward(dscores, cache_scores)
    grads[last_W_name] += self.reg * self.params[last_W_name]

    # Back progagation from the last hidden layer to the first hidden layer
    for i in reversed(range(1, self.num_layers)):
        dx[i], grads['W%d' % i], grads['b%d' % i] = affine_relu_backward(dx[i + 1], cache_layer[i])
        grads['W%d' % i] += self.reg * self.params['W%d' % i]

    return loss, grads
```

Here is the code of training the model. The _Solver_ takes in the model we created and performs the gradient descent with the training dataset to train the model. All deep learning software platform provides similar services. People use those built-in services to optimize $$w$$.  Since we had covered the gradient descent, we will not list the code for the _Solver_ here. 

```python
model = FullyConnectedNet([100, 50, 25],
              weight_scale=weight_scale, dtype=np.float64)
solver = Solver(model, data,
                print_every=100, num_epochs=10, batch_size=200,
                update_rule='adam',
                optim_config={
                  'learning_rate': learning_rate,
                }
         )
solver.train()
```

And this is the code of making predictions and calculating the accuracy.
```python
X_test, y_test, X_val, y_val = data['X_test'], data['y_test'], data['X_val'], data['y_val']

y_test_pred = np.argmax(model.loss(X_test), axis=1)
y_val_pred = np.argmax(model.loss(X_val), axis=1)
print('Validation set accuracy: ', (y_val_pred == y_val).mean())
print('Test set accuracy: ', (y_test_pred == y_test).mean())
```


**Deep learning is about creating a model by learning from data**. We have solved a visual recognition problem that is otherwise difficult to solve. Instead of coding all the rules, which is impossible for the CIFRA problem, we create a FC network to learn the model from data. The accuracy of our model can be further improved by adding convolution layer (CNN) on top of the FC network. CNN will be covered in a separate article.

### MNist
One of the first deep learning datasets that most people learn is the MNist. It is a dataset for handwritten numbers from 0 to 9.

<div class="imgcap">
<img src="/assets/dl/mnist.gif" style="border:none;width:40%">
</div>

We will implement code with the TensorFlow to solve the problem with 98%+ accuracy.

Unlike the code with numpy, TensorFlow constructs a graph describing the network first. Here, we declare a placeholder for our input features (the pixel values of the image) and the labels. The real data will be provided later in the execution phase. 
```python
x = tf.placeholder(tf.float32, [None, 784])
labels = tf.placeholder(tf.float32, [None, 10])  # True label.

```

We declare $$ W $$ and $$ b $$ as variables. Those are trainable variables that we want to learn from the training dataset. In the declaration, we also declare the method to initialize the variables. Nevertheless, the initialization is performed in the execution phase later.
```python
W1 = tf.Variable(tf.truncated_normal([784, 256], stddev=np.sqrt(2.0 / 784)))
b1 = tf.Variable(tf.zeros([256]))
W2 = tf.Variable(tf.truncated_normal([256, 100], stddev=np.sqrt(2.0 / 256)))
b2 = tf.Variable(tf.zeros([100]))
W3 = tf.Variable(tf.truncated_normal([100, 10], stddev=np.sqrt(2.0 / 100)))
b3 = tf.Variable(tf.zeros([10]))
```

We define 2 hidden layers. Each has a matrix multiplication operation followed by ReLU. Then we define an output layer with the matrix multiplication only. Once again, this is just declaration and no operation is performed yet.
```python
### Building a model
# Create a fully connected network with 2 hidden layers
# 2 hidden layers using relu (z = max(0, x)) as an activation function.
h1 = tf.nn.relu(tf.matmul(x, W1) + b1)
h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)
y = tf.matmul(h2, W3) + b3
```

Now, we define our loss function including a cross entropy cost with L2 regularization penalties for our $$ W $$. We use an Adam optimizer (named as train_step) to perform the gradient descent to optimize $$W$$.  We also have a placeholder for $$ \lambda $$ so a user can supply the value later to control the regularization.
```
# Cost function & optimizer
# Use a cross entropy cost fuction with a L2 regularization.
lmbda = tf.placeholder(tf.float32)
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=y) +
        lmbda * (tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3)))
train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy)
```

Once the declaration code is completed for the computation graph, we created a session to execute the graph and train the network for 10,000 iterations. For each iteration, we retrieve the next batch of training data and run the operation node "train_step" (i.e. the Adam optimizer we defined above). TensorFlow runs the operation as well as all operations that it depends on. 
```python
# Create an operation to initialize the variable
init = tf.global_variables_initializer()
# Now we create a session to execute the operations.
with tf.Session() as sess:
    sess.run(init)
    # Train
    for _ in range(10000):
      batch_xs, batch_ys = mnist.train.next_batch(100)
      sess.run(train_step, feed_dict={x: batch_xs, labels: batch_ys, lmbda:5e-5})
```

Once the training is complete, we create 2 more operations. The "correct_prediction" compare the predictions with the true labels. “Accuracy” computes the accuracy of our prediction.
```python
with tf.Session() as sess:
    ...
	
    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
```

We run the accuracy operation with our testing dataset and print out the results.
```python
with tf.Session() as sess:
    ...
	
    print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                          labels: mnist.test.labels}))
```

For completeness, here is the code listing. This file depends on "tensorflow.examples.tutorials.mnist" which is used to read the MNist data.
```python
import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np

FLAGS = None


def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  ### Building a model
  # Create a fully connected network with 2 hidden layers
  # Initialize the weight with a normal distribution.
  x = tf.placeholder(tf.float32, [None, 784])
  labels = tf.placeholder(tf.float32, [None, 10])  # True label.
  
  W1 = tf.Variable(tf.truncated_normal([784, 256], stddev=np.sqrt(2.0 / 784)))
  b1 = tf.Variable(tf.zeros([256]))
  W2 = tf.Variable(tf.truncated_normal([256, 100], stddev=np.sqrt(2.0 / 256)))
  b2 = tf.Variable(tf.zeros([100]))
  W3 = tf.Variable(tf.truncated_normal([100, 10], stddev=np.sqrt(2.0 / 100)))
  b3 = tf.Variable(tf.zeros([10]))


  # 2 hidden layers using relu (z = max(0, x)) as an activation function.
  h1 = tf.nn.relu(tf.matmul(x, W1) + b1)
  h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)
  y = tf.matmul(h2, W3) + b3

  # Cost function & optimizer
  # Use a cross entropy cost fuction with a L2 regularization.
  lmbda = tf.placeholder(tf.float32)
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=y) +
         lmbda * (tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3)))
  train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy)

  init = tf.global_variables_initializer()
  with tf.Session() as sess:
      sess.run(init)
      # Train
      for _ in range(10000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, labels: batch_ys, lmbda:5e-5})

      # Test trained model
      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(labels, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                          labels: mnist.test.labels}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

# 0.9816

```

This code demonstrates the power to solve a complex visual problem with few lines of DL code. With 10,000 iterations, we achieved an accuracy above 98%.


### Weight initialization

Weight initialization is one critical area in implementing a deep network. If you start the parameters incorrectly, you may not even beat the random odd of guessing. Sometimes people accidentally initialize $$W$$ with 0s. This is close to having every neurons dead and we will not be able to backpropagate the loss. We want some non-symmetry on $$W$$. So nodes are learning different features. To introduce such non-symmetry, we initialize $$W$$ with a Gaussian distribution with $$\mu = 0$$. Initially, we do not want the output of the matrix multiplication falls into the saturation regions of the activation function. In these regions, cost is in-sensitive to changes in the input. Their partial derivatives are close to 0 and kill the gradient in the backpropagation. $$W$$ learns too slow in saturation regions.

 Let's generate 20,000 values of $$ W $$ with $$\mu = 0$$ and $$\sigma^2 = 1 $$. Here is the distribution of $$W$$.

$$ \sigma^2 = 0.998 $$

<div class="imgcap">
<img src="/assets/dl/var1.png" style="border:none;width:40%">
</div>

Let's create a node with 1000 input. We initialize half of the $$x$$ as 1 and the other half as 0. We generate 20,000 values of $$y$$ with $$w$$ initialized above.

$$
y = Wx + b
$$

The variance of the output jumps almost 500 times and the plot is almost flat.

$$ \sigma^2 = 497.6 $$

<div class="imgcap">
<img src="/assets/dl/plotgg.png" style="border:none;width:40%">
</div>

If we feed this output of the matrix multiplication into the activation function. Huge portion of the values are in the saturation region for any activation functions we discussed as far. Instead, we should ask if we want the input of the activation to be Gaussian distributed with $$\mu = 0$$ and $$\sigma^2 = 1 $$, how $$W$$ should be distributed?

$$
\begin{align}
Var(z) & = Var(\sum^n_{i=1} w_i x_i) \\
&= \sum^n_{i=1} Var(w_i x_i)  \quad & \text{property of variance}\\
&= \sum^n_{i=1} [E(w_i)]^2Var(x_i) + [E(x_i)]^2Var(w_i) + Var(w_i)Var(x_i) \quad & \text{property of variance} \\
&= \sum^n_{i=1} Var(w_i)Var(x_i) \quad & \text{both } x_i, w_i \text{ have 0 means} \\
&= (n \cdot Var(w))Var(x) \quad & \text{property of variance}\\
&= Var(\frac{w}{\sqrt{n}})Var(x) \quad & \text{property of variance}\\
\end{align}
$$

Hence, if we want the input to the activation function to have $$\mu = 0$$ and $$\sigma^2 = 1 $$, we need to scale $$w$$ by $$ \frac{1}{\sqrt{n}} $$. With ReLU activation, we want to scale it with

$$
\sqrt {\frac {2}{\text{number of input}}} 
$$

```python
w = np.random.randn(n) * sqrt(2.0/n)
```

### Training parameters

In previous sections, we discuss many problems in training a network and how bad learning rate produces bad predictions. We now come back to the gradient descent and discuss different methods for updating the trainable parameters. This is not an easy topic. Fortunately, people can simply use most of the built-in optimization methods provided by the DL software platforms.

#### Rate of decay
To maintain a constant learning rate is not a good idea. In later phase, we want much smaller steps to avoid oscillation. After some initial training, we can start decaying the learning rate for every N iterations. For example, after 10,000 iterations, the learning rate will be decay by the formula below for every 20,000 iterations:

$$
\text{learning rate} = \text{learning rate} \cdot \text{decay factor}
$$

where decay factor is another hyperparameter say 0.95.

#### Momentum update
We mentioned that gradient descent is like dropping a ball in a bowl. But our gradient descent adjusts the parameters by the gradient of the current location of $$ W $$ only. In the physical world, the movement of the ball depends on the location and also on the momentum of the ball. We could adjust $$ W $$ by the gradient and its history rather than throwing all the history away. If we recall the stochastic gradient descent, it follows a zip zap pattern rather than a smooth curve. With this historical information, we can make stochastic gradient or mini-batch gradient to behave more smoothly.

Here we introduce a variable $$v$$, which behaves like the momentum in the physical word. In each iteration, we update $$ v $$ by keeping a portion of v minus the change caused by the gradient at that location. $$ mu $$ controls how much history information to keep, and this will be another hyperparameter. Researchers may describe $$ mu $$ 
as fraction. If you recall the oscillation problem before, this actually becomes a damper to stop the oscillations. Momentum based gradient descent often has a smoother path and settles to a minimal closer and faster.

```python
v = mu * v - learning_rate * dw
w += v
```

#### Nesterov Momentum

In Momentum update, we use the current location of $$w$$ to compute $$dw$$. In Nesterov Momentum, the current location is replaced with a look-ahead location: the location where the "ball" should go without taking the current location into account. Nesterov Momentum converges better than Momentum update.

In the program below, instead of using location $$w$$ to compute $$dw$$, we use location $$w_ahead$$ to compute $$dw$$.
```python
w_ahead = w + mu * v
v = mu * v - learning_rate * dw_ahead
w += v
```

In practice, people use the following formula:
```python
v_prev = v
v = mu * v - learning_rate * dw 
w += -mu * v_prev + (1 + mu) * v 
```

#### Adagrad
 If the input features are not scaled correctly, it is impossible to find the right learning rate that works for all the features. This indicates the learning rate needs to be self-adapted for each tunable parameter. One way to do it is to remember how much change has made to a specific $$ W_i $$. We should reduce the rate of change for parameters that already have many updates. This mitigates the oscillation problem because it acts like a damper again. In Adagrad, we reduce the learning rate with a ratio inversely proportional to the L2 norm of all previous gradients $$dw_i$$.
  
```python
cache += dw**2
w += - learning_rate * dw / (np.sqrt(cache) + 1e-7) # add a tiny value to avoid division by 0.
```

> Parameter based methods like Adagrad reduce the learning rate for parameters according to their accumulated changes.

#### RMSprop

Like Adagrad, RMSprop adjusts the learning rate according to the history of the gradients. But it uses a slightly different formular with the introduction of another hyperparameter decay_rate.
```python
cache = decay_rate * cache + (1 - decay_rate) * dw**2
w += - learning_rate * dx / (np.sqrt(cache) + 1e-7)
```

> RMSprop behaves like Adagrad but have mechanism to forget older changes.

#### Adam

Adam combines the concepts of momentum with RMSprop:
```python
m = beta1*m + (1-beta1)*dw
v = beta2*v + (1-beta2)*(dw**2)
w += - learning_rate * m / (np.sqrt(v) + 1e-7)
```

> Adam combines both momentum and parameter based gradient descent. Adam is the most often used method now.

Here is an example of using Adam Optimizer in TensorFlow
```python
loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits)) \
           + lmbda * tf.nn.l2_loss(weights1) + lmbda * tf.nn.l2_loss(weights2) \
           + lmbda * tf.nn.l2_loss(weights3) + lmbda * tf.nn.l2_loss(weights4)

optimizer = tf.train.AdamOptimizer(0.0005).minimize(loss)

...

with tf.Session(graph=graph) as session:
    ...
    for step in range(num_steps):
        ...
        _, l, predictions = session.run(
            [optimizer], feed_dict=feed_dict)
```

#### Visualization of Gradient descent methods

Here are some animations produced by Alec Radford in demonstrating how the gradient descent behaves for different algorithms. Regular gradient descent (red) learns the slowest. Momentum based algorithm has a tendency to overshoot the target initially. Adagrad learns faster initially than RMSprop. But RMSprop provides a mechanism to erase old history which make learning more adaptable later.

<div class="imgcap">
<img src="/assets/dl/a1.gif" style="border:none;width:70%">
</div>

<div class="imgcap">
<img src="/assets/dl/a2.gif" style="border:none;width:70%">
</div>

### Feature Scaling (normalization)

As discussed before, we want the feature input to be scaled correctly (normalized). If the features do not have similar scale, it will be hard for the gradient descent to work: the training parameters oscillate like the red arrow.

<div class="imgcap">
<img src="/assets/dl/gauss_s.jpg" style="border:none;width:40%">
</div>

For example, with 2 input features, we want the shape to be as close to a circle as possible.
<div class="imgcap">
<img src="/assets/dl/gauss_shape.jpg" style="border:none;">
</div>

To achieve that we normalize the features in the dataset to have zero mean and unit variance. 

$$
z = \frac{x - \mu}{\sigma}
$$

For images, we normalize every pixel independently. We compute a mean and a variance at each pixel location for the entire training dataset. Therefore, for an image with NxN pixels, we use NxN means and variances to normalize the image.

$$
z_{ij} = \frac{x_{ij} - \mu_{ij}}{\sigma{ij}}
$$

In practice, we do not read all the training data at once to compute the mean or variance. We compute a running mean during the training. Here is the formula for the running mean:

$$
\mu_{n} = \mu_{n-1}  + k \cdot (x_{i}-\mu_{n-1})
$$

where $$k$$ is a small constant.

> Use the running mean and variance from training dataset to normalize validation and testing data.

#### Whitening

In machine learning, we can train a model faster if features are not correlated (whitened). In a dating application, someone may prefer a tall person but not too thin. If we rescale the weight and height independent of each other, the rescaled features are still co-related. From the rescaled weight, we can tell easily whether a person is heavier than the average population. But usually, we are more interested to know whether the person is thinner in the same height group which is very hard to tell from the rescaled weight alone.

We can express the co-relations between features $$x_i$$ and $$ x_{j} $$ in a covariance matrix:

$$
\sum = \begin{bmatrix}
    E[(x_{1} - \mu_{1})(x_{1} - \mu_{1})] & E[(x_{1} - \mu_{1})(x_{2} - \mu_{2})] & \dots  & E[(x_{1} - \mu_{1})(x_{n} - \mu_{n})] \\
    E[(x_{2} - \mu_{2})(x_{1} - \mu_{1})] & E[(x_{2} - \mu_{2})(x_{2} - \mu_{2})] & \dots  & E[(x_{2} - \mu_{2})(x_{n} - \mu_{n})] \\
    \vdots & \vdots & \ddots & \vdots \\
    E[(x_{n} - \mu_{n})(x_{1} - \mu_{1})] & E[(x_{n} - \mu_{n})(x_{2} - \mu_{2})] & \dots  & E[(x_{n} - \mu_{n})(x_{n} - \mu_{n})]
\end{bmatrix}
$$

Which $$ E $$ is the expected value.

Consider 2 data samples (10, 20) and (32, 52). The mean of $$ x_1 $$ is $$ \mu_1 = \frac {10+32}{2} = 21 $$ and $$ \mu_2 = 36 $$

The expected value of the first element in the second row will be:

$$
E[(x_{2} - \mu_{2})(x_{1} - \mu_{1})] = \frac {(20 - 36)(10 - 21) + (52 - 36)(32 - 21)} {2}
$$

From the covariance matrix $$ \sum $$, we can find a matrix $$W$$ to convert $$x$$ to $$x^{'}$$ whose features are un-correlated with each other. For example, the contour plotting on the left below indicates weight and height are correlated (weight increase with height). On the right, the features are whitened with their correlation-ship recalibrate (taken out). For example, if the new $$weight^{'}$$ feature is smaller than 0, we know the person is thinner in the same height group.

<div class="imgcap">
<img src="/assets/dl/gaussf.jpg" style="border:none;width:50%">
</div>

> Whitening un-correlates features that make training more effective.

This sounds complicated, but can be done easily by Numpy linear algebra library.
```python
X -= np.mean(X, axis = 0)    
cov = np.dot(X.T, X) / X.shape[0]

U,S,V = np.linalg.svd(cov)
Xdecorelate = np.dot(X, U)
Xwhite = Xdecorelate / np.sqrt(S + 1e-5)
```

> Image data usually require 0 centered but does not require whitening.

### Batch normalization

We have emphasized so many times the benefits of having features with mean = 0 and $$ \sigma^2=1 $$ 

But, why do we stop at the input layer only. Batch normalization renormalizes a layer output. For example, we renormalized the output of the affine layer (matrix multiplication) before feeding it into the ReLU.
```python
def affine_batchnorm_relu_forward(x, w, b, gamma, beta, bn_param):
  h, h_cache = affine_forward(x, w, b)
  norm, norm_cache = batchnorm_forward(h, gamma, beta, bn_param)
  relu, relu_cache = relu_forward(norm)
  cache = (h_cache, norm_cache, relu_cache)
  return relu, cache
```

We apply the normalization formula below:

$$
z = \frac{x - \mu}{\sigma}
$$

which the mean and variance is computed from the current mini-batch data. 

We feed $$z$$ to a linear equation with the trainable scalar values $$ \gamma $$ and $$ \beta$$ (1 pair for each normalized layer). 

$$
out = \gamma z + \beta
$$

The normalization can be undone if $$ gamma = \sigma $$ and $$ \beta = \mu $$. We initialize $$\gamma = 1$$ and $$\beta =0 $$, so the input is normalized and therefore learns faster, and the parameters will be learned during the training.

```python
def batchnorm_forward(x, gamma, beta, bn_param):
    sample_mean = np.mean(x, axis=0)
    sample_var = np.var(x, axis=0)

    sqrt_var = np.sqrt(sample_var + eps)
    xmu = (x - sample_mean)
    xhat = xmu / sqrt_var

    out = gamma * xhat + beta
```

Batch normalization solves a problem called internal covariate shift. As weights are updated, the distribution of outputs at each layer changes. Batch normalization normalized data at each layer again. So we can use a higher learning rate that speed up learning.
Batch normalization also help regularize $$W$$.

In the training, we use the mean and variance of the current training sample. But for testing, we do not use the mean/variance of the testing data. Instead, we record a running mean & variance during the training and apply it in validation or testing.
```python
running_mean = momentum * running_mean + (1 - momentum) * sample_mean
running_var = momentum * running_var + (1 - momentum) * sample_var
```

We use this running mean and variance to normalize our testing data.
```python
xhat = (x - running_mean) / np.sqrt(running_var + eps)
out = gamma * xhat + beta
```

> Batch normalization is common practice to use before or after the activation functions including CNN layers.

This is the code to implement batch normalization in TensorFlow:
```python
w_bn = tf.Variable(w_initial)
z_bn = tf.matmul(x, w_bn)

bn_mean, bn_var = tf.nn.moments(z_bn, [0])
scale = tf.Variable(tf.ones([100]))
beta = tf.Variable(tf.zeros([100]))

bn_layer = tf.nn.batch_normalization(z_bn, bn_mean, bn_var, beta, scale, 1e-3)
l_bn = tf.nn.relu(bn_layer)
```


### Hyperparameter tuning

Because we know very little in our model for real life problems, the hyperparameter tuning is usually a try and error process. Some parameters are dependent on each other and cannot tune separately. Sometimes those relationship is subtle. For example, the regularization rate changes the shape of the cost function. Therefore it impacts how we tune the learning rate. We can create a mesh of values to tune both parameters together. For example, with learning rates of 1e-1, 1e-2, … and 1e-8 and regularization rate of 1e-3, 1e-4, ... and 1e-6, we have a potential of 8x4 combinations to test (1e-1, 1e-3), (1e-1, 1e-3), …, (1e-8, 1e-5) and (1e-8, 1e-6) ). We may not want to use an exactly rectangular shape of a mesh. For example, we may want to slight deviations at each mesh point with the hope that some irregularity may help us to explore more information.

<div class="imgcap">
<img src="/assets/dl/mesh.png" style="border:none;width:60%">
</div>

> Start tuning parameters from coarse grain with fewer iterations before fine tuning.

### Troubleshooting

Many places can go wrong when training a deep network. Here are some simple tips:
* Unit test the forward pass and back propagation code.
	* At the beginning, test with non-random data.
* Compare the backpropagation result with the naive gradient check.
* Always start with a simple network that works. 
	* Increasing accuracy should not be the first priority.
	* Handling multiple challenges in a complex network is not the way to go. Issues grow exponentially in DL.
* Create simple scenarios to verify the network:
	* Compute the loss for random guesses. It should match the loss at early training when the model is no better than a random guess.
	* Train with a small dataset with few iterations.
	* Verify if loss drops and/or accuracies increase during training.
	* Drop regularization - training accuracies should go up.	
	* Overfit with a small dataset to see if the loss is small or 0. (Turn off the regularization cost temporarily.)
* Do not waste time on a large dataset with long iterations during early development.
* Verify how trainable parameters are initialized.	
* Always keep track of the shape of the data and document it in the code.
* Display and verify some training samples and the predictions.

#### Monitor loss
We want to plot the cost vs iterations. Monitor the loss to see its trend:
* If loss goes up early, the learning rate is way too high.
* If loss drops fast and flattens very quickly, the learning rate is high.
* If loss drops too slow, the learning rate is too slow.

<div class="imgcap">
<img src="/assets/dl/mont1.png" style="border:none;width:70%">
</div>

#### Train vs validation accuracy
Plot out accuracy between validation and training to identify overfit issues.
* If validation error is much lower than the training error, the model is overfit.

<div class="imgcap">
<img src="/assets/dl/mont2.png" style="border:none;width:70%">
</div>

#### Monitor Gradient descent

Monitor the updates to $$W$$ ratio:

$$
\frac{\| \alpha \cdot dw \|}{\| W \|}
$$

* If the ratio is $$ > 1e-3 $$, consider lower the learning rate.
* If the ratio is $$ < 1e-3 $$, consider increase the learning rate.

Plot weight, activation and gradient histograms for all layers.
* Verify if you are having gradient diminishing or explode problem.
* Identify layers that have very low gradients. 
* Verify whether you have too many saturated nodes.

<div class="imgcap">
<img src="/assets/dl/hhist.png" style="border:none;width:40%">
</div>

#### Visualize filters and activation

For visualization problem, we can visualize the $$W$$ matrix in 2D for FC network, or the filters for CNN in the first couple layers. This helps us to identify what type of features that the model is extracting. The first layer should extract simple structures like edge and color.

<div class="imgcap">
<img src="/assets/cnn/cnnfilter.png" style="border:none;width:70%">
</div>

We can locate pictures that have the highest activation for each feature map. Once again, it verifies what features are used in building the model. 

In later layers, we should see more complex structures evolved:
<div class="imgcap">
<img src="/assets/dl/pp22.png" style="border:none;width:50%">
</div>

<div class="imgcap">
<img src="/assets/cnn/cnnlayer_4.png" style="border:none;width:50%">
</div>

(Source from Visualizing and Understanding Convolutional Networks, Matthew D Zeiler et al.)

### Data augmentation

We have focused on the mechanics of the DL. One significant improvement for the network training is to have more data. This avoids overfitting and has better coverage of your feature spaces. However, getting labeled dataset can be expensive. One alternative is data augmentation. For example, for visual recognition, we can flip the image, slightly rotate or skew the images with software libraries. This helps us to avoid overfitting and produces generalized predictions invariant of the spatial location of the objects. Semi-learning may even expand the classification training dataset further by allowing some data without labels to be used as training data if the model can classify them with high certainty.

> Very simple effort to augment your data can have a significant impact on the training.

### Model ensembles

So far, we try to find the best models. In machine learning, we can take a vote from different decision trees to make the final prediction. This is based on the assumption that mistakes are localized. There is a smaller chance for 2 different models to make the same mistake. In DL, each training starts with random guesses and therefore the models optimized are not unique.  We can pick the best models after repeating the training multiple times. We take votes from those models for the final predictions. This requires us to run the program multiple times, and can be prohibitively expensive. Alternatively, we run the training once and checkpoints multiple times. We pick the best models from the checkpoints. We can also use the validation phase to pick our best models to ensemble. Instead of running multiple models, we can also run a running average for our training parameters. We can have one vote per model, taking an average or use weights based on the confidence level for each prediction.

> A lot of production system uses model ensembles to push the accuracy up for few percentage points.

### Convolution Net (CNN) & Long short term memory (LSTM)

FC network is rarely used alone. Exploring all possible connections among nodes in the previous layer provides a complex model that is wasteful with small returns. A lot of information is localized. For an image, we want to extract features from neighboring pixels. CNN applies filters to explore localized features, and then apply FC to make predictions. LSTM applies time feedback loop to extract time sequence information. CNN & LSTM add complexity to the model to explore localized information. Topics here covers most of the fundermental of CNN and LSTM. 

> Two separated articles on CNN and LSTM will explain each methods in details.

### Credits
For the CIFRA 10 example, we start with assignment 2 in the Stanford class "CS231n Convolutional Neural Networks for Visual Recognition". We start with some skeleton codes provided by the assignment and put it into our code to complete the assignment.






















