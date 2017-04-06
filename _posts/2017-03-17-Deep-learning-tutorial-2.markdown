---
layout: post
comments: true
mathjax: true
title: “Deep learning without going down the rabbit holes. (Part 2)”
excerpt: “Part 2 of the deep learning.”
date: 2017-03-17 14:00:00
---
**This is work in progress...**

[Part 1 of the deep learning can be found here](https://jhui.github.io/2017/03/18/Deep-learning-tutorial/)
### Overfit

In part 1, we prepare a simple model with 4 layers of computation nodes. The solutions for $$ W $$ are not unique. Should we prefer one solution over the other, or should we select smaller values for $$ W $$? Are part of the network cancel out each other?

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

This lead us to a very important topic.  When we increase the complexity of our model, we risk the chance of modeling the noise into the model. If we do not have enough sample data to cancel out the noise, we make bad predictions. But even without the noise, we can still have a bad model. Let's walk through an example. We start with training samples with input values and output range from 0 to 20. How will you create an equation to link the data points below.
<div class="imgcap">
<img src="/assets/dl/d1.png" style="border:none;width:70%">
</div>

One possiblity is
$$
y = x
$$ which is simple and just miss 2 on the left and 2 on the right.
<div class="imgcap">
<img src="/assets/dl/d2.png" style="border:none;width:70%">
</div>

But when we show it to our Android Pieter, which has much higher computation capability than us, he models it as:

$$
y = 1.9  \cdot 10^{-7}  x^9 - 1.6 \cdot 10^{-5} x^8 + 5.6 \cdot 10^{-4} x^7 - 0.01 x^6  + 0.11 x^5 - 0.63 x^4 + 1.9  x^3 - 2.19  x^2 + 0.9 x - 0.0082
$$

<div class="imgcap">
<img src="/assets/dl/d3.png" style="border:none;width:70%">
</div>

Which model is correct? The answer is "don't know". Someone thinks the first one is simplier, and simple explanation deserves more credits. But if you show it to a stock broker, they may say the second curve looks closer to the market closing price of a stock. 

Instead, we should ask whether our model is too "custom tailor" for the training data, and fails to make generalized predictions. The second curve fits the sample data 100% but will make poor predictions if the true model is a straight line.

#### Validation
**Machine learning is about making predictions.** A model that has 100% accuracy in training can be a bad model. For that, we split our testing data into 3 parts: say 80% for training, 10% for validation and 10% for testing. During training, we use the training dataset to build models with different hyperparameters. We run those models with the validation dataset and pick the one with the highest accuracy. This strategy works if the validation dataset is similar to what we want to predict. But as a last safeguard, we use the 10% testing data for a final insanity check. This testing data is for one last verification but not for model selection. If your testing result is dramatically different from the validation result, we need to randomize the data more, or to collect more data.

#### Visualization 
We can train a model to create a boundary to separate the blue dots from the white dots below. A complex model can produce sophisticated boundaries comparing to a low complexity model. In the yellow circled area, if we miss the 2 left white dot samples in our training, a complex model may create an odd shape boundary just to include this white dot. A low complexity model may produce a smoother surface which by chance may include those 2 white dots. A complex model may mishandle outliers. For example, a complex model should ignore the white dot in the green circle like a simple model does.

<div class="imgcap">
<img src="/assets/dl/of.png" style="border:none;width:60%">
</div>

Recall from Pieter's equation, our sample data can be model nicely with the following equations:

$$
y = 1.9  \cdot 10^{-7}  x^9 - 1.6 \cdot 10^{-5} x^8 + 5.6 \cdot 10^{-4} x^7 - 0.01 x^6  + 0.11 x^5 - 0.63 x^4 + 1.9  x^3 - 2.19  x^2 + 0.9 x - 0.0082
$$

In fact, there are infinite answers using as many polynomial orders $$ x^k \dots $$

Comparing with the linear model $$ y = x $$, we realize that the
$$ || coefficient || $$ 
in Pieter equation is higher. In additional, the higher the order, it will be harder to train the model because we have a bigger search space for the parameters. In additional, some areas of the search space have a very steep gradient.

Let us create a polynomal model with order 5 to fit our sample data.

$$
y = c_5 x^5 + c_4 x^4 + c_3 x^3 + c_2 x^2 + c_1 x + c_{0}
$$

We need more iterations to train this model and result is less accurate than a model with order 3.
<div class="imgcap">
<img src="/assets/dl/p1.png" style="border:none;width:60%">
</div>

So why don't we focus on making a model with the right complexity. In real life problems, a complex model is the only way to push accuracy to an acceptable level, and overfitting is un-avoidable. You may even find it harder to train a model if the model is over simplified. A better solution is to introduce methods to reduce overfitting rather than make the model simpiler. One simple solution is to add more sample data such that it is much harder to overfit. Here, double the sample data produces a model closer to a straight line. Unfortunately, labeling large training dataset in real problems can be expensive.
<div class="imgcap">
<img src="/assets/dl/p2.png" style="border:none;width:60%">
</div>

### Regularization

As we observe before, there are many solutions to a DL problem but in order to have a close fit, the coefficient of our training parameters will have larger magnitude. 

$$
||c|| = \sqrt{(c_5^2 + c_3^2 + c_3^2 + c_2^2 + c_1^2 + c_{0}^2)}
$$

For example, if we set $$ c_{2}, c_{3}, c_{4} \text{ and } c_{5} $$  to 0, we get a simple straight line model. To have our training not to overfit the training data, we add a penalty in the cost function to penalize large magnitude.

$$
J = \text{mean square error} + \lambda \cdot ||W||
$$

This is called **regularization**. Here we introduce another hyperparameter called regularization factor $$ \lambda $$ to penalize overfitting.

In this example, we use a L2 norm (**L2 regularization**)
$$ ||W|| $$
 as the penality. 

After many try and error, we pick $$ \lambda $$ to be 1. With the regularization, our model makes better predictions.

<div class="imgcap">
<img src="/assets/dl/p3.png" style="border:none;width:60%">
</div>

Like other hyperparameters for training, the process is try and error. In fact we use a relative high $$ \lambda $$
in this problem because there are only a few trainable parameters in the model. In real life problems, $$ \lambda $$ is lower because we are dealing with larger amount of trainable parameters.

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

can be very steep which cause the learning rate problem discussed before. The cost here escalates very high which takes many iterations to undo. For example, from iteration 10,000 to 11,000, the coefficient for $$ x^5 $$
changes from -0.000038 to -0.000021 but the cost jumps from 82 to 34,312.

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

When we build our model, we also try out a polynomial model with order of 9. Even after a long training, the model still makes poors predictions. We decide to start with a model with order of 3, and increase it gradually. This is another example to demonstrate why we should start with a simple model first. At 7, we find the model is too hard to train. The following is what a 7 order model predicts:
<div class="imgcap">
<img src="/assets/dl/p4.png" style="border:none;width:60%">
</div>

### Diminishing and exploding gradient

From our previous example, we demonstrate how important to trace the gradient at different layer to troubleshoot problem. In our online dating model, we log 
$$ || gradient || $$ 
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

There are a couple things that we need to monitor. Are the magnitude too high or too small? If the magnitude is too high at later stage of training, the gradient descent is having problem to find the minima. Some parameters may be oscillating. For example, when we have the scaling problem with the features (year of education and monthly income), the gradient is so huge that the model learns nothing.
```
iteration 0: ... dW1=1.183e+04 dW2=5.929e+06 ...
iteration 200: ... dW1=4.458e+147 dW2=2.203e+150 ...
iteration 400: ... dW1=1.656e+291 dW2=8.184e+293 ...
iteration 600: ... dW1=nan dW2=nan ...
```

 If gradient is small, the network learns slowly. In the following log, the gradient diminishing from the right layer (layer 6) to the left layer (layer 0). Layer 0 is learning much slower than layer 6.
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

A network with many deep layers may suffer from this gradient diminishing problem. Let's come back to backpropagation to understand the problems.

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

As indicated, the gradient descent depends on the loss $$ \frac{\partial J}{\partial l} $$ as well as the gradients $$ \frac{\partial l_{k+1}}{\partial l_{k}}, \frac{\partial l_{k}}{\partial l_{k-1}} \dots $$. Let's look at a sigmoid activation function. If $$ x $$ is higher than 5 or smaller than -5, the gradient is close to 0. Hence, in those region, the node learns close to nothing regardless of the loss.

<div class="imgcap">
<img src="/assets/dl/sigmoid2.png" style="border:none;width:80%">
</div>

We can visualize the derivative of a sigmoid function behaves like a gate to the loss signal. If the input is > 5 or <-5, the derviative is so small, it blocks most of the loss signal to propagage backward. So nodes on its left sides learn little. 

In addition, the chain rule in the gradient descent has a multiplication effect. If we multiple numbers smaller than one, it diminishes quickly. On the contrary, if we multiple numbers greater than one, it explodes. 

$$ 
0.1 \cdot 0.1 \cdot 0.1 \cdot 0.1 \cdot 0.1 = 0.00001 
$$

$$ 
5 \cdot 5 \cdot 5 \cdot 5 \cdot 5 = 3125
$$

So if the network design and the initial parameters have some symmetry that output similar values,  the gradient may diminish quickly or explode. However, we cannot say with certainty on when and how it may happen because we lack a full understanding between the maths of gradient descent and a complex model. Nevertheless, the empirical data for deep network indicates it is a problem.

Microsoft Resnet (2015) has 152 layers. A lot of natural language process (NLP) problems are vulnerable to diminishing and exploding gradients. How can they address the issue? This is the network design for Resnet. Instead of one long chain of nodes, a mechanism is built to bypass a layer to make learning faster. (Source Kaiming He, Xiangyu Zhang ... etc)
<div class="imgcap">
<img src="/assets/dl/resnet.png" style="border:none;width:40%">
</div>

<div class="imgcap">
<img src="/assets/dl/resnet2.png" style="border:none;width:40%">
</div>

As always in DL, an idea often looks far complicated in a diagram or a equation. In LSTM, the state of a cell is updated by

$$
C_t = gate_{forget} \cdot C_{t-1} + gate_{input} \cdot \tilde{C}
$$

Bypassing a layer can visualize as feeding the input to the output directly $$ out = in $$. For $$ C_t $$ to be the same as $$ C_{t-1} $$, we need $$ gate_ {forget}=1 $$ and $$ gate_{input} = 0 $$. So one way to addressing the diminishing gradient problem is to design a different function used in the node.

#### Gradient clipping
To avoid gradient explosion, we can apply gradient clipping to restrict values of the gradient.

Here, we use TensorFlow for coding. TensorFlow is an open source machine learning software from Google. In real life problem, Numpy is important in data preparation but people use a software package like TensorFlow to implement deep networks.

Here we set the maxium clip norm to be 5.0.
```python
params = tf.trainable_variables()
opt = tf.train.GradientDescentOptimizer(learning_rate)
for b in xrange(time_steps):
    gradients = tf.gradients(losses[b], params)
    clipped_gradients, norm = tf.clip_by_global_norm(gradients, 5.0)
```

If the gradient reaches above 5.0, the gradients are rescaled according to the ratio $$ \frac{5}{\text{norm of the gradient}} $$. (L2 norm is the length of the vector.)
```
global_norm = sqrt(sum([l2norm(t)**2 for t in t_list]))
t_list[i] * clip_norm / max(global_norm, clip_norm)
````

### Classification

A very important part of deep learning is classification. We mention face detection and object recognition before. These are classification problems asking the question: what is this? For example, for Android Pieter to safely walk in a street, he needs to learn what is a traffic light, is the pedestrian faceing him or not. Classification applies to non-visual problems also. We classify whether an email is a spam, or approve/disapprove a loan etc...

<div class="imgcap">
<img src="/assets/dl/street2.jpg" style="border:none;width:40%">
</div>

Like solving regression problem using DL, we use a deep network to compute a value. In classification, we call this **a score**. We apply a classifier to convert the score to a probability value. To train the network, the training dataset provides the answers to the classification (like classify an image as a school bus/truck/airplane) which we call **true label**.

#### Logistic function (sigmoid function)

A score compute by a network takes on any value. A classifier squashes it to a probabilty value between 0 and 1. For a "yes" or "no" type of prediction (the email is/is not a spamm, the drug test is positive/negative), we apply a **logistic function** (also called a sigmoid function) to the score value. If the output probability is lower than 0.5, we predict "no", otherwise we predict "yes".

$$
p = \sigma(score) = \frac{1}{1 + e^{-score}}
$$

<div class="imgcap">
<img src="/assets/dl/sigmoid.png" style="border:none;width:40%">
</div>


#### Softmax classifier

For many classification problem, we categorize an input to one of the many classes. For example, we can classify an image to one of the 100 possbile object classes. We use softmax classifier to compute K probabilites, one per class for an input image (the combined probabilities remains 1).

<div class="imgcap">
<img src="/assets/dl/deep_learner2.jpg" style="border:none;width:70%;">
</div>

The network computes K scores per image. The probability that an image belongs to the class $$ i $$ will be.

$$
p_i =  \frac{e^{score_i}}{\sum e^{score_c}} 
$$

For example, the school bus above may have a score of (3.2, 0.8, 0) for the class school bus, truck and airplane respectively. The probability for the correponding class is

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

To avoid the numerical stability problem caused by adding large expotential values, we subract the inputs by its max. Adding or subtract a number from the input produces the same probabilities in softmax. 

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
p = softmax(score) = \frac{e^{z_i}}{\sum e^{z_c}}
$$

That is why many literatures and APIs use the term logit for score when softmax is used. However, there are more than one function that map scores to probabiities and meet the definition of logits. Sigmoid function is one of them.

> Softmax is the most common classifier among others.

#### SVM classifier

Linear SVM classifer applies a linear classifier to map input to K scores, one per class. 

$$
\hat{y} = W x + b
$$

The class having the highest score will be the class prediction. To train the network, SVM loss is used. We will discuss the SVM in the cost function section later. Its main objective is to create a boundary to separate classes with the largest possible margin.

<div class="imgcap">
<img src="/assets/dl/SVM.png" style="border:none;width:40%">
</div>

### Entropy 

> We have introduced a few lingos so you can understand some common DL terms. This section is on the information theory which you can browse through it quickly.

With a probabilistic model, we want a cost function that works with probability predictions. We need to take a break to the information theory on entropy. It may worth the time because entropy is heavily used in machine learning, 

Say we have a string "abcc", "a" and "b" occurs 25% (0.25) of the time and "c" with 50% (0.5). Entropy defines the minimum amount of bits to represent the string. For the most frequent character, we use fewer bits to represent it.

Entropy:

$$
H(y) = \sum_i y_i \log \frac{1}{y_i} = -\sum_i y_i \log y_{i}
$$

The string "abcc" needs 1.5 bit per character. Here is our encoding scheme: 0 represents 'c', binary number 01 for 'a' and 10 for 'b'.  The average number of bit to represent the string is 

$$ H = 0.25 \log(\frac{1}{0.25}) + 0.25 \log(\frac{1}{0.25})  + 0.5 \log(\frac{1}{0.5}) $$

$$ H  = 0.25 \cdot 2 \cdot 2 + 1 \cdot 0.5 = 1.5 $$

```python
b = -( 0.25 * math.log2(0.25) + 0.25 * math.log2(0.25) + 0.5 * math.log2(0.5) )   # 1.5 bit
```

Entropy is also a measure of randomness (disorder). A fair dice, comparing with a biased dice, has more randomness with even distribution of outcomes. A biased dice is more predictable and therefore less entropy. In entropy, randomness means more information since it requires more bits to represent the information. For example, it takes more time to descibe the details inside a messy room.

The entropy of a biased coin and a fair coin.
```python
b = -(1.0 * math.log2(1.0))                           # 0 : A 2-head coin
b = -(0.5 * math.log2(0.5) + 0.5 * math.log2(0.5)  )  # 1 bit: 0 for head 1 for tail.
```

#### Cross entropy

$$
H(y, \hat{y}) = \sum_i y_i \log \frac{1}{\hat{y}_i} = -\sum_i y_i \log \hat{y}_i
$$

Cross entropy is the amount of bits to encode y but use the $$ \hat{y} $$ distribution to compute the encode scheme.The cross entropy is always higher than entropy until both distributions are the same. You need more bits to encode the information if you use a less optimized scheme. In classification, we make predictions using probabilities $$ \hat{y} $$ (0.88 chance for a school bus, 0.8 for a truck and 0.4 for an airplane.) 

#### KL Divergence:

$$
\mbox{KL}(y~||~\hat{y}) = \sum_i y_i \log \frac{1}{\hat{y}_i} - \sum_i y_i \log \frac{1}{y_i} = \sum_i y_i \log \frac{y_i}{\hat{y}_i}
$$

KL divergence is simply cross entropy - entropy: the extra bits need to encode the information. In machine learning, KL Divergenence estimates the difference between 2 distributions. Since $$ y $$ is the true labels which does not change, we can treat it as a constant. Therefore finding a model to minimze KL divergence (the difference between probability distribution (1, 0, 0) & (0.88, 0.08, 0.04)) is the same as minimze the cross entropy. We, therefore, compute cross entropy most of the time.

> Use cross entropy to optimize a model.

### Maximum likelihood estimation (MLE)

> This section is for background information. There are some probability lingos. It is good to know but not required.

What is our objective in training a model? Our objective is to tune our trainable parameters so that the likelihood of our model is maximized (MLE). In plain terms, we want to train the parameters $$ W $$ such that the prediction for the training data is as close to the labels.

The likelihood is define as the probabilty of making a prediction to be the same as the true labels give the input $$X$$ and $$W$$. If we can find the $$ W $$ to maximize the likelihood for all training data, we find our model. In probability, we write it as

$$
p(y | x, W) =  \prod_i p(y_{i} |  x_{i}, W)
$$

For each sample $$i$$,

$$
p(y_{i} |  x_{i}, W) = \hat{y_i}
$$

which $$ y_{1} = (1, 0, 0) $$ in our example (100% for school bus and 0% chance otherwise), and $$ \hat{y_{1}} $$ = $$ (0.88, 0.08, 0.04)$$.

#### Neagtive log-likelihood (NLL)

We want to take the log of the MLE because we can treat the product terms as additions which is easier. Log is a monotonic increase function, and therefore, finding $$ x $$ to maximize $$ f(x) $$ is the same as find $$x$$ to maximize $$ \log(f(x))$$.

Since probability is between 0 and 1 and its log is negative, we take a negative sign to make it positive. 

$$
nll = - \log {p(y_{i} |  x_{i}, W)} = - \log{ \hat{y_{i}}}
$$

We call this the negative log-likelihood. To maximize the "maximum likelihood estimation" (MLE) is the same as minimizing the negative log-likihood. 

**To train a network, we find $$W $$ to minimizing the negative log-likelihood**.

#### Logistic loss

As an exercise to demonstrate NLL, we dervive the logistic loss (or even the mean square error) from NLL.

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
\text{nnl} = \sum\limits_{i}    \log (1 + e^{- z}) 
$$

$$
\text{nnl} = \sum\limits_{i} \log (1 + e^{- y_i W^T x_{i}}) 
$$

### Cost function
**Deep learning is knowing your cost.** But how to define the cost? San Francisco is about 400 miles from Los Angels. It costs about $60 for the gas. When you order food from a restaurant, they do not deliver to home more than a few miles away. From their prespective, the cost grows expotentially with distance. So there are many definitions of cost. In DL, our objective is not knowing the value of the cost, but find a set of $$W$$ to make cost the lowest. Therefore, we have many flexibility for the cost function as long as we can find this set of $$W$$. There are objectives to punish outliners heavily. But there are other important consideration include how easy and how fast to optimize the cost function. Does the cost function solve the gradient diminishing problem?

#### Cross entropy cost function
Apply NLL to find the cost function for a classification problem:

$$
p(y | x, W) =  \prod_n p(y_{i} |  x{i}, W)
$$

$$
\text{nll} = - \log {(p(y | W, x)} = - \sum_n \log {p(y_{i} | W, x_{i})}
$$

$$
\text{nll} =  - \sum_n \log {\hat{y_{i}}}
$$

Since $$ y_{i} = (0, \cdots, 1, \dots, 0, 0) $$ We can put back $$ y_{i} $$, and the equation becomes the cross entropy.

$$
\text{nll} = - \sum_n \sum_i y_{i} \log {\hat{y_{i}}} 
$$

As a summary, the cross entropy cost function can be written as

$$
H(y, \hat{y}) = \sum_i y_i \log \frac{1}{\hat{y}_i} = -\sum_i y_i \log \hat{y}_i
$$

or as negative log likelihood

$$
\text{nll} =  - \sum_n \log {\hat{y_{i}}}
$$

> Cross entroy cost function with softmax classifier is one of the most popular combination to solve classification problems.

#### SVM loss (also called Hinge loss or Max margin loss)

$$
J = \sum_{j\neq y_i} \max(0, score_j - score_{y_i} + 1)
$$

If the margin between the score of a class and that of the true label is greater than -1, we add it to the cost.  For example, a score of (8, 14, 9.5, 10) with the last entry being the true label.

$$
J = max(0, 8 - 10 + 1) + max(0, 14 - 10 + 1) + max(0, 9.5 - 10 + 1)
$$

$$
J = 0 + 5 + 0.5 = 5.5
$$

For SVM, the cost function creates a boundary with the maximum margin to separate classes.

<div class="imgcap">
<img src="/assets/dl/SVM.png" style="border:none;width:40%">
</div>

#### Mean square error (MSE)

$$
\text{mean square error} = \frac{1}{N} \sum_i (h_i - y_{i})^2
$$

We use MSE for regression problems before. We can use MSE in classification. But in practice, we use cross entropy loss. Classification uses a classifier to squash values to a probability between 0 and 1. The mapping is not linear. For a sigmod classfier, a large range of value (less than -5 or greater than 5) is squeezed to 0 or 1. As shown before, those areas have close to 0 partial derviative. Based on the chain rule in the back propagation 

$$
\frac{\partial J}{\partial score} = \frac{\partial J}{\partial out} \frac{\partial out}{\partial score}
$$

The loss signal is hard to propage backward in those region regardless of loss because $$ \frac{\partial out}{\partial score} \approx 0$$ . However, there is a way to solve this issue. The partial derviative of the sigmod function can be small but we can make 
$$ 
\frac{\partial J}{\partial out} 
$$ 
very large if the prediction is bad. The sigmod function squashes values expontentially. We need a cost function that punishes bad predictions in the same scale to counter that. Squaring the error does not make it. Cross entropy punishes bad prediction expotentially. That is why cross entropy cost function trains better than MSE in the classification problems.

### Deep learing network (Fully-connected layers) CIFAR-10

Let's put together everything to solve the CIFRA-10. CIFAR-10 is a computer vision dataset for object classification. It has 60,000 32x32 color images belong to one of 10 object classes, with 6000 images per class.

(Source Alex Krizhevsky)
<div class="imgcap">
<img src="/assets/dl/cifra.png" style="border:none;width:40%">
</div>

We implement a fully connected network similar to the following to classify the CIFRA images to the corresponding classes. In our implmentation, we allow user to control how many hidden layers to create and the number of nodes per layer.
<div class="imgcap">
<img src="/assets/dl/fc_net.png" style="border:none;width:40%">
</div>

Let's have some boiler plate code that did before. This is the forward feed and the backpropagation code for 
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

We combine them to form a forward feed and a backpropagation "affine relu" layer.
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

Our softmax function
```python
def softmax_loss(x, y):
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = x.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
```

We are creating a FullyConnectedNet network with 3 hidden layers with (100, 50, 25) nodes respectively.
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

Here is the key part in computing the loss. We do a feed forward, use softmax to compute the lost, and compute all the gradients for the backpropagation. The code should be familiar, and the comment should be self-explanable.
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
	# From the last hidden layer to the output layer, we do not perform ReLU
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

Here is the code of training the model:
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

And the code of making predictions:
```python
X_test, y_test, X_val, y_val = data['X_test'], data['y_test'], data['X_val'], data['y_val']

y_test_pred = np.argmax(model.loss(X_test), axis=1)
y_val_pred = np.argmax(model.loss(X_val), axis=1)
print('Validation set accuracy: ', (y_val_pred == y_val).mean())
print('Test set accuracy: ', (y_test_pred == y_test).mean())
```

We will not show the code of preparing the data and the code performing the gradient descent. For the gradient descent, people use out of the box libraries to solve it. We will demonstrate it using TensorFlow in the next section.

**Deep learning is about create a model by learning from data**. We have solved a visual recognition problem that otherwise hard to solve. Instead of coding all the rules which is impossible for the CIFRA problem, we create a FC network to learn the model from data. The accuracy of our model is reasonable using the FC layer, but easily beat by adding convolution layers. Hence, we will not spend more effort for now.

### MNist
One of the first deep learning dataset that most people learn is the MNist. It is a dataset for handwritten numbers from 0 to 9.

<div class="imgcap">
<img src="/assets/dl/mnist.gif" style="border:none;width:40%">
</div>

We will show the TensorFlow code to solve the problem with 98%+ accuracy.

Unlike the code with numpy, TensorFlow constructs a graph describing the network first. Here we declare a placeholder for our input features (the pixel values of the image) and the ture labels which will be provided later in the training. 
```python
x = tf.placeholder(tf.float32, [None, 784])
labels = tf.placeholder(tf.float32, [None, 10])  # True label.

```

We declare $$ W $$ and $$ b $$ as variables and provide methods to initiate it later.  
```python
W1 = tf.Variable(tf.truncated_normal([784, 256], stddev=np.sqrt(2.0 / 784)))
b1 = tf.Variable(tf.zeros([256]))
W2 = tf.Variable(tf.truncated_normal([256, 100], stddev=np.sqrt(2.0 / 256)))
b2 = tf.Variable(tf.zeros([100]))
W3 = tf.Variable(tf.truncated_normal([100, 10], stddev=np.sqrt(2.0 / 100)))
b3 = tf.Variable(tf.zeros([10]))
```

We define 2 hidden layers. Each has a matrix multiplication operation followed by ReLU. Then another operation multiplies it with a matrix. Note, so far we are just making declaration, no variables are initialized and no matrix multiplication is done.
```python
### Building a model
# Create a fully connected network with 2 hidden layers
# 2 hidden layers using relu (z = max(0, x)) as an activation function.
h1 = tf.nn.relu(tf.matmul(x, W1) + b1)
h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)
y = tf.matmul(h2, W3) + b3
```

Now we define our loss function including a cross entropy and the regularization penalty for our $$ W $$. We use Adam as an optimizer for the gradient descent.  We also have a placeholder for $$ \lambda $$ so user can supply it later to control the regularization.
```
# Cost function & optimizer
# Use a cross entropy cost fuction with a L2 regularization.
lmbda = tf.placeholder(tf.float32)
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=y) +
        lmbda * (tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3)))
train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy)
```

We create a session to execute the graph and train the network for 10,000 iteratons. For each iteration, we retrieve the next batch of sample data, and run the operation "train_step" (the Adam optimizer). TensorFlow runs the operation as well as all operations that it depends on. 
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

Once the training is complete, we create 2 more operations, the correct_prediction compare the predictions with the true lables. "accuracy" computes how many predictions are correct.
```python
with tf.Session() as sess:
    ...
	
    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
```

We run the accuracy operation with our testing dataset and print out the result.
```python
with tf.Session() as sess:
    ...
	
    print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                          labels: mnist.test.labels}))
```

For completness, here is the code listing. This file depends on "tensorflow.examples.tutorials.mnist" which is used to read the MNist data.
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

This code demonstrates how powerful to solve a complex visual problem with few lines of DL code. With 10,000 iterations, we achieve an accuracy above 98%.

### Regularization

We apply regularization to overcome overfit. The idea is to train a model that makes generalized predictions. It forces the model not to memorize the small bits of an individual sample that is not part of the genearlized features. Part of the overfit problem is that the model is too powerful. We can reduce the complexity of the model by reducing the number of features: remove features that can be direct or indirect derived by others. We can also eliminate some layers, or switch to a design that explores better on the locality of the information. For example, we use CNN for images to explore the spatial locality and LSTM for NPL. Overfit can be overcomed by adding more training data also.

#### L0, L1, L2 regularization

Large W tends to overfit. L2 regularization adds the L2 norm to a cost function as the regularization cost.

$$
J = \text{data cost} + \lambda \cdot ||W||
$$

$$
||W||_2 =  \sqrt{\sum\limits_{i} \sum\limits_{j} (W_{ij}^2)}
$$

There are different functions to compute the regularization cost of tunable parameters:

L0 regularization

$$
|| W ||_0 = \sum \begin{cases} 1, & \mbox{if } w \neq 0 \\ 0, & \mbox{otherwise} \end{cases}
$$

L1 regularization

$$
|| W ||_1 = \sum |w|
$$

L0, L1 and L2 regularization penalize on $$ W $$ but on different extends. L2 puts the highest attention (or penalty) on large parameters and L0 pays attention on non-zero prameters.  L0 penalizes non-zero parameters which forces more parameters to be 0, i.e. increase the sparsity of the parameters. L2 focuses on large parameters, and therefore have more non-zero parameters. L1 regularization also rewards sparsity. Sparsity of parameters effectively reduces the features in making prediction. Sometimes it is used as feature selections. L2 is more popular but some problem domains may prefer higher sparsity.


#### Dropout
A non-intuitive regularization method is called dropout. L2 regularization discourages weights with large values. To avoid overfit, we may not want some weights to be too dominating. By randomly dropping connections from one layer to the other layer, we force the network not to depend too much on a single node and try to learn from many different ways. This has an effect similar to forcing the weights smaller.

In the following diagram, for each iteration, we randomly drop off the connection during training.
<div class="imgcap">
<img src="/assets/dl/drop.png" style="border:none;width:40%">
</div>

Here is the code to implement the dropout for the forward feed and backpropagation. In the forward feed, it takes a parameter on the percentage of nodes to be dropout. Notice that, dropout applies to training only.
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


### Weight initialization

Weight initialization is one important area in implementing a network. If you start the parameters incorrectly, you may not even beat the random odd of guessing. The initial input to the activation function (or non-linear functions) should not fall into its low partial derivative areas. Otherwise, the network learns slowly. Sometimes those parameters are accidentally initialized with 0s. This is close to turning every neuron dead, and not able to backpropagate the loss correctly. In fact, you do not want the output values to the next layer to look the same. Some non-symmetry is preferable, otherwise, the loss will evenly distribute back to previous layers. To introduce such non-symmetry, we initialize those parameters with gaussian distribution with mean = 0.  But should we use $$ \sigma = 1$$ 

The value to the activation is produced by $$ z = Wx + b $$. We will demonstrate that if $$ W $$ has $$ \sigma = 1$$, $$z$$ will be much flatter with high variance value.

We generate 20,000 values of $$ W $$ with mean = 0 and $$\sigma = 1 $$. Then we plot the distribution of W and re-calculate the variance again.

$$ \sigma = 0.998 $$

<div class="imgcap">
<img src="/assets/dl/var1.png" style="border:none;width:40%">
</div>

We generate 20,000 values of $$ y $$ using our familiar formula with 1000 of input $$x$$ which half of them are 1 and another half are 0:

$$
y = Wx + b
$$

We plot the distribution of $$ y $$.

$$ \sigma = 497.6 $$

<div class="imgcap">
<img src="/assets/dl/var2.png" style="border:none;width:40%">
</div>

The plot has different scales for the x and y-dimension. If we decrease the scale in y-direction a little bit, you will realize the plot is actually close to flat.

<div class="imgcap">
<img src="/assets/dl/var3.png" style="border:none;width:40%">
</div>

The plot is flat. i.e. it is more uniform. Therefore to generate an input to the activation function with a gaussian distribution of mean = 0 and $$\sigma = 1$$, we need to take into the account of the number of inputs to the node. Hence we use the following formula for the variance:

$$
\frac {2}{\sqrt{\text{number of input}}}
$$

Note: Some research paper indicates using 2 as numerator has better performance than 1.

### Training parameters

In previous sections, we discuss many problems in training a network, and how bad learning rate produces bad predictions. We now come back to the gradient descent and discuss different methods for updating the trainable parameters. This is not an easy topic because the shape of the cost function can be very different in different problem domains. Fortunately, most DL software libraries provide many different optimization methods. We will cover a couple core concepts here.

#### Rate decay
To maintain a constant learning rate is not be a good idea. It is like using a saw to finish the last part of making a table. One common way to reduce the rate is after some initial phase, we start to decay the learning rate for every N iterations. For example, after 10,000 iterations, the learning rate will be decay by the formula below for every 20,000 iterations:

$$
\text{learning rate} = \text{learning rate} \cdot \text{decay factor}
$$

which decay factor is another hyperparameter say 0.95.

#### Momentum update
We mention gradient descent is like dropping a ball in a bowl. But our gradient descent adjusts the parameters by the gradient of the current location of $$ W $$ only. In the physical world, the movement of the ball depends on the location but also the velocity of the ball. We could adjust $$ W $$ by the gradient and its history rather than throw all the information away. If we recall the stochastic gradient descent, it follows a zip zap pattern rather than a smooth curve. With this history information, we can make stochastic gradient or mini-batch gradient to behave more smoothly.

Here we introduce a variable $$v$$ which behaves like the velocity (momentum) in the physical word. In each iteration, we update $$ v $$ by keeping a portion of v minus the change caused by the gradient at that location. $$ mu $$ controls how much history information to keep, and this will be another hyperparameter. Researchers may describe $$ mu_ $$ 
as a fraction. If you recall the parameter oscillation problem before, this actually becomes a damper to stop the oscillations. Momentum based gradient descent often have a smoother path and settle to a minima closer and faster.

```python
v = mu * v - learning_rate * dw
w += v
```

#### Adagrad
 If the input features are not scaled correctly, it is impossible to find the right learning rate that works for all the features. This indicates the learning rate needs to self-adapt for each tunable parameter. One way to do it is to remember how much change has made to a specific $$ W_i $$. We will reduce the parameter change if that parameter has been changed frequently. This will absolutely help the oscillation problem because it acts like a damper again. In Adagrad, it was done slightly difference by allowing the rate change to drop inversely by the L2 norm of all the previous gradients $$ dw_i $$.
 
```python
cache += dw**2
w += - learning_rate * dw / (np.sqrt(cache) + 1e-7) # add a tiny value to avoid division by 0.
```

#### RMSprop

RMSprop uses a similar concepts with the following formula and the hyperparameter decay_rate to control how much previous histories yo keep.
```python
cache = decay_rate * cache + (1 - decay_rate) * dw**2
w += - learning_rate * dx / (np.sqrt(cache) + 1e-7)
```

#### Adam

Adam combines the concepts of momentum with RMSprop:
```python
m = beta1*m + (1-beta1)*dw
v = beta2*v + (1-beta2)*(dw**2)
w += - learning_rate * m / (np.sqrt(v) + 1e-7)
```

> Adam is the most often used method now.

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

#### Visualization

Here is some animations produced by Alec Radford in demonstrating how the gradient descent behaves for different algorithms. Regular gradient desent (red) learns the slowest and different algorithms have different descending patterns and speed. One more interesting point is how some algorithms overshoot the minima or oscailate around it.

<div class="imgcap">
<img src="/assets/dl/a1.gif" style="border:none;width:70%">
</div>

<div class="imgcap">
<img src="/assets/dl/a2.gif" style="border:none;width:70%">
</div>

### Feature Scaling (normalization)

As we find out, we want the feature input to the network to be scaled correctly (normalized). If the features do not have the proper scale, it will be much harder for the gradient descent to work. The training parameters may oscaillate.

<div class="imgcap">
<img src="/assets/dl/gauss_s.jpg" style="border:none;width:40%">
</div>

For example, with 2 input features, we want the shape to be as close to a circle as possible.
<div class="imgcap">
<img src="/assets/dl/gauss_shape.jpg" style="border:none;">
</div>

We normalize the features in the dataset to have zero mean and unit variance. 

$$
z = \frac{x - \mu}{\sigma}
$$

For image, we normalize every pixels independently. We compute a mean and a variance at each pixel location for the whole training dataset. Therefore, for an image with NxN pixels, we use NxN means and variances to normalize the image.

$$
z_{ij} = \frac{x_{ij} - \mu_{ij}}{\sigma{ij}}
$$

In practice, we do not read all the trainning data at once to compute the mean or variance. We compute a running mean during the training. Here is the formula for the running mean:

$$
\mu_{n} = \mu_{n-1}  + k \cdot (x_{i}-\mu_{n-1})
$$

which $$k$$ is a small constant.

#### Whitening

In machine learning, we prefer features to be un-related. For example, in a dating application, a person may prefer a tall person but not too thin. However, weight and heigth are co-related. A taller person is heavier than a shorter person in average. Re-scaling these features independently can only tell whether a person is lighter than average in the population, but not whether the person is thin. Weigth increases with height:
<div class="imgcap">
<img src="/assets/dl/gauss.jpg" style="border:none;">
</div>

A network learns faster if features are un-related. We express the co-relations between a feature $$x_i$$ and $$ x_{j} $$ in terms of a covariance matrix below:

$$
\sum = \begin{bmatrix}
    E[(x_{1} - \mu_{1})(x_{1} - \mu_{1})] & E[(x_{1} - \mu_{1})(x_{2} - \mu_{2})] & \dots  & E[(x_{1} - \mu_{1})(x_{n} - \mu_{n})] \\
    E[(x_{2} - \mu_{2})(x_{1} - \mu_{1})] & E[(x_{2} - \mu_{2})(x_{2} - \mu_{2})] & \dots  & E[(x_{2} - \mu_{2})(x_{n} - \mu_{n})] \\
    \vdots & \vdots & \ddots & \vdots \\
    E[(x_{n} - \mu_{n})(x_{1} - \mu_{1})] & E[(x_{n} - \mu_{n})(x_{2} - \mu_{2})] & \dots  & E[(x_{n} - \mu_{n})(x_{n} - \mu_{n})]
\end{bmatrix}
$$

Which $$ E $$ is the expected value.

Consider 2 data samples : (10, 20) and (32, 52). 
The mean of $$ x_1 $$ will be $$ \mu_1 = \frac {10+32}{2} = 21 $$ and $$ \mu_2 = 36 $$

The expected value of the first element in the second row will be:

$$
E[(x_{2} - \mu_{2})(x_{1} - \mu_{1})] = \frac {(20 - 36)(10 - 21) + (52 - 36)(32 - 21)} {2}
$$

From the covariance matrix $$ \sum $$, we find a matrix $$W$$ by $$ \sum $$ to convert the input $$ X $$ to $$ Y = W X $$. The purpose of whitening is to change the feature distribution from the left to the right one.

<div class="imgcap">
<img src="/assets/dl/gaussf.jpg" style="border:none;width:50%">
</div>

This sounds complicated but can be done by numpy linear algebra library
```python
X -= np.mean(X, axis = 0)    
cov = np.dot(X.T, X) / X.shape[0]

U,S,V = np.linalg.svd(cov)
Xdecorelate = np.dot(X, U)
Xwhite = Xdecorelate / np.sqrt(S + 1e-5)
```

> Image data usually require 0 centered but does not require whitening.

### Batch normalization

We have emphasized so many times the benefits of having features with mean = 0 and $$ \sigma=1 $$ 

But why we only stop at the input layer only. Batch normalization re-normalizes a layer output. For example,  we re-normalizes the output of the linear layer before feeding it into the ReLU.
```python
def affine_batchnorm_relu_forward(x, w, b, gamma, beta, bn_param):
  h, h_cache = affine_forward(x, w, b)
  norm, norm_cache = batchnorm_forward(h, gamma, beta, bn_param)
  relu, relu_cache = relu_forward(norm)
  cache = (h_cache, norm_cache, relu_cache)
  return relu, cache
```

We apply the normalization using the formula before:

$$
z = \frac{x - \mu}{\sigma}
$$

which during the training, we use the mean and variance computed from the current mini-batch samples. We then feed the output to a linear equation with the trainable scalar values $$ \gamma $$ and $$ \beta$$ (1 pair for each normalized layer). 

$$
out = \gamma z + \beta
$$

If $$ gamma = \sigma $$ and $$ \beta = \mu $$, we can see the normalization can be undo.

```python
def batchnorm_forward(x, gamma, beta, bn_param):
    sample_mean = np.mean(x, axis=0)
    sample_var = np.var(x, axis=0)

    sqrt_var = np.sqrt(sample_var + eps)
    xmu = (x - sample_mean)
    xhat = xmu / sqrt_var

    out = gamma * xhat + beta
```

In the training, we use the mean and variance of the current training sample. But for testing, we do not use the mean/variance of the testing data. Instead, we record a running mean & variance during training and apply it.
```python
running_mean = momentum * running_mean + (1 - momentum) * sample_mean
running_var = momentum * running_var + (1 - momentum) * sample_var
```

We normalize the input during testing with the running mean/variance in the training.
```python
xhat = (x - running_mean) / np.sqrt(running_var + eps)
out = gamma * xhat + beta
```

### Hyperparameter tuning

Because the model is such a black box to us in real life problems, the hyperparameter tuning is usually a try and error. Some parameters are dependent on each other and cannot tune separately. Sometimes the relationship is subtle. For example, the regularization rate changes the shape of the cost function, and therefore impacts how we tune the learning rate. We can create a mesh of values to be used for tuning. For example, with learning rates of (1e-1, 1e-2, ... 1e-8) and regularization of (1e-3, 1e-4, .. 1e-6), we have a potential of 8x4 combinations to test ( (1e-1, 1e-3), (1e-1, 1e-3), ..., (1e-8, 1e-5), (1e-8, 1e-6) ). We may not want to use an exactly rectangular shape of a mesh. For example, we may want to slightly deviate at each mesh point with the hope that some irregularity may help us to explore more information.

<div class="imgcap">
<img src="/assets/dl/mesh.png" style="border:none;width:40%">
</div>

> Start tune parameters from coarse grain with fewer iterations before fine tuning.

### Troubleshooting

Many places can go wrong when training a deep network. Here are some simple tips:
* Unit test the forward pass and back propagation code.
	* At the beginning, test with non-random data.
* Compare the backpropagation result with the naive gradient check.
* Always start with a simple network that works. 
	* Push accuracy up should not be the first priority.
	* Handle multiple battle fronts in a complex network is not the way to go. Issues grow exponentially in DL.
* Create simple scenarios to verify the network:
	* Train with a small dataset with few iterations.
	* Compare the loss/accuracy value with the corresponding value of a random guess. 
	* Verify if loss drops and/or accuracies increase during training.
	* Drop regularization - training accuracies should go up.	
	* Overfit with a small dataset to see if the loss is small.
* Monitor or plot out the loss closely to see its trend.
* Keep track of the norm of W and gradient (and ratios) preferable in key layers. Looks for gradient vanishing/exploding problems.
* Do not waste time on a large dataset with long iterations during early development.
* Verify how trainable parameters are initialized.	
* Always keep track of the shape of the data and document it in the code.
* Display and verify some training samples and the predictions.
* Plot out accuracy between validation and training to identify overfit issues.
* Plot activation/gradient histograms for all layers. If initialization is not done correctly, there should be a lot of dead/saturated nodes.
* For visualization problem, try to display the filter in an early layer and the activations.

### CNN & LSTM
FC network is rarely used alone. Exploring all possible connections among nodes in the previous layer provides an complex model that is wasteful with small returns. A lot of information is localized. For an image, we want to extract features from neighboring pixels. CNN applies filters to explore localized features, and then apply FC to make predictions. LSTM applies time feedback loop to extract time sequence information. CNN & LSTM make changes to the design of a computation node and how it is connected. The core part of DL remains the same and learning CNN after FC is easier since the foundation is the same. Nevertheless, you will go nowhere in learning DL without CNN and/or LSTM. Hence, we have provided a separate discussion on both CNN and LSTM.

### Data argumentation

We have focused on the mechanics of the DL. One significant improvement for network training is to have more data. This avoids overfitting and has better coverage of your feature spaces. However, getting labeled samples can be expensive. One alternative is data argumentation. For example, for visual recognition, we can flip the image, slightly rotate or skew the images with software libraries. This helps us to avoid overfitting and produces generalized predictions invariant of the spatial location of the objects. Some research may even expand further by allowing some data without labels to be used as training data if they produce a very high score with the model.

> Very simple effort to argument your data can have a significant impact on the training.

### Model ensembles

So far, we try to find the best models. In machine learning, we can take a vote from different decision trees to make the final prediction. This base on the assumption that mistakes can be localized. There is a smaller chance for 2 different models to make the same mistake. In DL, each training starts with random guesses and therefore the models usually are not unique.  We can pick the best models after training the networks multiple times. We can vote from different models to make the final predictions. This requires us to run the program multiple times, and can be prohibitively expensive. The alternative, we can run the training once and pick the best models during the latter phase of the training. We can have one vote per model, taking an average or use weights based on the confidence level of each prediction.

### Credits
For the CIFRA 10 example, we start with assignment 2 in the Stanford class "CS231n Convolutional Neural Networks for Visual Recognition". We start with some skeleton codes provided by the assignment and put into our code to complete the assignment.






















