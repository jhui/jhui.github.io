---
layout: post
comments: true
mathjax: true
priority: 550
title: “Generative models - GAN, Variational Autoencoders”
excerpt: “Generative models”
date: 2017-03-06 14:00:00
---

### Discriminative models

Generative models generate images from scratch. Before diving into the generative models, we look into some prior methods of generating images. In a discriminative model, we draw conclusion on something we observe. For example, we train a CNN discriminative model to classify a picture. 

$$ y = f(image) $$

<div class="imgcap">
<img src="/assets/cnn/cnn.png" style="border:none;width:60%">
</div>

#### Class visualization

Often, we want to visualize what features a network tries to learn in the classification process. First, we train a classification CNN model. Then we genearte a random image and feed forward to the network. Instead of backpropagate the gradient to train $$W$$, we backpropagate the gradient to make $$image$$ to look like the target class. i.e., use backpropagation to change the $$image$$ to increase the score of the target class. In order to do so, we change $$ \frac{\partial J}{\partial score_i} $$ manually to:

$$
\frac{\partial J}{\partial score_i}=
    \left\{
    \begin{array}{lr}
      1,& i=target \\
      0,& i \neq target 
    \end{array}
    \right\}
$$

, and reiterate the feed forward and backward many times. Here is the skeleton code to generate an image from a pre-trained CNN _model_ for the target class _target_y_.

```phtyon
def class_visualization(target_y, model, learning_rate, l2_reg, num_iterations):

    # Generate a random image
    X = np.random.randn(1, 3, 64, 64)
    for t in xrange(num_iterations):
        dX = None
        scores, cache = model.forward(X, mode='test')

        # Artifically set the dscores for our target to 1, otherwise 0.
        dscores = np.zeros_like(scores)
        dscores[0, target_y] = 1.0

        # Backpropagate
        dX, grads = model.backward(dscores, cache)
        dX -= 2 * l2_reg * X
		
        # Change the image with the gradient descent.
        X += learning_rate * dX
    return X
```

To make it works better, we add clipping, jittering and blurring:
```phtyon
def class_visualization(target_y, model, learning_rate, l2_reg, num_iterations, blur_every, max_jitter):
    X = np.random.randn(1, 3, 64, 64)
    for t in xrange(num_iterations):
        # Add the jitter
        ox, oy = np.random.randint(-max_jitter, max_jitter + 1, 2)
        X = np.roll(np.roll(X, ox, -1), oy, -2)

        dX = None
        scores, cache = model.forward(X, mode='test')

        dscores = np.zeros_like(scores)
        dscores[0, target_y] = 1.0
        dX, grads = model.backward(dscores, cache)
        dX -= 2 * l2_reg * X

        X += learning_rate * dX

        # Undo the jitter
        X = np.roll(np.roll(X, -ox, -1), -oy, -2)

        # As a regularizer, clip the image
        X = np.clip(X, -data['mean_image'], 255.0 - data['mean_image'])

        # As a regularizer, periodically blur the image
        if t % blur_every == 0:
            X = blur_image(X)
    return X
```


Here is our attempt to generate a spider image starting from random noises.
<div class="imgcap">
<img src="/assets/gm/spider.png" style="border:none;width:30%">
</div>

#### Artistic Style Transfer

Artistic style transfer applies the style of one image to another. We start with a picture as our style:
<div class="imgcap">
<img src="/assets/gm/starry1.jpg" style="border:none;width:40%">
</div>

and transfer the style to another image
<div class="imgcap">
<img src="/assets/gm/starry2.jpg" style="border:none;width:40%">
</div>

to create
<div class="imgcap">
<img src="/assets/gm/starry.png" style="border:none;width:40%">
</div>
[Image source](https://github.com/jcjohnson/neural-style)

Here, we pass the first and the second image to the CNN respectively. We extract the corresponding features of those image at some layers deep down in the network. We subtract the difference and use it as the gradient for backpropagation. 
```python
out_feats, cache = model.forward(X, end=layer)
dout = 2 * (out_feats - target_feats)
dX, grads = model.backward(dout, cache)
dX += 2 * l2_reg * np.sum(X**2, axis=0)

X -= learning_rate * dX
``` 

#### Google DeepDream

Google DeepDream uses a CNN to find and enhance features within an image. It forward feed an image to a CNN network to extract features at a particular layer. It later backpropagate the gradient by explicitly changing the gradient to its activation:

$$ 
\frac{\partial J}{\partial score_i} = activation
$$ 

This exaggerate the features at the chosen layer of the network.

```python
out, cache = model.forward(X, end=layer)
dX, grads = model.backward(out, cache)
X += learning_rate * dX
```

Here, we start with an image of a cat:
<div class="imgcap">
<img src="/assets/gm/cat2.jpg" style="border:none;width:40%">
</div>

This is the image after many iterations:
<div class="imgcap">
<img src="/assets/gm/cat.png" style="border:none;width:40%">
</div>


> We turn around a CNN network to generate realistic images through backpropagation by exaggerate certain features.

### Generative models

In a discriminative model, we draw conclusion on something we observe: 

$$ y = f(image) $$

A generative model generates data that we observe:

$$ image = G(z) $$

For example, in a generative model, we ask the model to generate an image that resemble a bedroom. In previous sections, we train a CNN to extract features of our training dataset. Then we iteratively and selectively backpropagate features to generate images. In the following sections, we adopt a more direct approach in generating image.

> z can sometimes realize as the latent variables of an image.

#### DCGAN (Deep Convolutional Generative Adversarial Networks)
In DCGAN, we generate an image directly using a deep network while using a second discriminator network to guide the generation process. Here is the generator network:

Source - Alec Radford, Luke Metz, Soumith Chintala:
<div class="imgcap">
<img src="/assets/gm/gm.png" style="border:none;width:80%">
</div>

$$ image = G(z) $$

The input $$ z $$ to the model is a 100-Dimensional vector (100 random numbers). We randomly select input vectors, says $$ (x_1, x_2, \cdots ,  x_{100}) = (0.1, -0.05, \cdots, 0.02) $$, and create images $$ z_{out} $$ using multiple layers of transpose convolutions  (CONV 1, ... CONV 4).

The first animation is the a convolution of a 3x3 filter on a 4x4 input. The second animation is the corresponding transpose convolution of a 3x3 filter on a 2x2 input.
[Animation source](https://github.com/vdumoulin/conv_arithmetic):
<div class="imgcap">
<img src="/assets/gm/conv.gif" style="border:none;width:30%">
<img src="/assets/gm/transpose.gif" style="border:none;width:30%">
</div>
<div class="imgcap">
</div>

5x5 region to a 5x5 region using a 3x3 filter and padding of 1:
<div class="imgcap">
<img src="/assets/gm/tp.png" style="border:none;width:100%">
</div>

2x2 region to a 5x5 region using a 3x3 filter, padding of 1 and stride 2:
<div class="imgcap">
<img src="/assets/gm/s2.png" style="border:none;width:100%">
</div>

Source: Vincent Dumoulin and Francesco Visin

> Transpose convolutions are sometimes call deconvolution.

At the beginning, $$ z_{out} $$ are just random noisy images. In DCGAN, we use a second network called a discriminator to guide how images are generated. With the training dataset and the generated images from the generator network, we train the discriminator (just another CNN classifier) to classify whether its input image is real or generated. But simultaneously, for generated images, we backpropagation the score in the discimiantor to the generator network.  The purpose is to train the $$W$$ of the generator network so it can generate more realistic images. So the discriminator servers 2 purposes. It determines the fake one from the real one and gives the score of the generated images to the generative model so it can train itself to create more realistic images. By training both networks simultaneously, the discriminator is better in distinguish generated images while the generator tries to narrow the gap between the real image and the generated image. As both improving, the gap between the real and generated one will be diminished. 

The following are some room pictures generated by a generative model: 

Source - Alec Radford, Luke Metz, Soumith Chintala:
<div class="imgcap">
<img src="/assets/gm/room.png" style="border:none;width:60%">
</div>

As we change $$ z $$ gradually, the images will be changed gradually also.

<div class="imgcap">
<img src="/assets/gm/rm.png" style="border:none;width:60%">
</div>

> The following code is written in TensorFlow. 

Here is a discriminator network that looks similar to the usual CNN classification network. It compose of 4 convolution layers. With the exception of the first convolution layer, other convolution layers are linked with a batch normalization layer and then a leaky ReLU. Finally, it is connected to a fully connected layer (linear) with a sigmoid classifier.
```python
def discriminator(image):
    d_bn1 = batch_norm(name='d_bn1')
    d_bn2 = batch_norm(name='d_bn2')
    d_bn3 = batch_norm(name='d_bn3')

    h0 = lrelu(conv2d(image, DIM, name='d_h0'))
    h1 = lrelu(d_bn1(conv2d(h0, DIM * 2, name='d_h1')))
    h2 = lrelu(d_bn2(conv2d(h1, DIM * 4, name='d_h2')))
    h3 = lrelu(d_bn3(conv2d(h2, DIM * 8, name='d_h3')))
    h4 = linear(tf.reshape(h3, [batchsize, -1]), 1, scope='d_h4')
    return tf.nn.sigmoid(h4), h4
```

The generator looks very similar to the reverse of the discriminator except the convolution layer is replaced with the transpose convolution layer.
```python
def generator(z):
    g_bn0 = batch_norm(name='g_bn0')
    g_bn1 = batch_norm(name='g_bn1')
    g_bn2 = batch_norm(name='g_bn2')
    g_bn3 = batch_norm(name='g_bn3')

    z2 = linear(z, DIM * 8 * 4 * 4, scope='g_h0')
    h0 = tf.nn.relu(g_bn0(tf.reshape(z2, [-1, 4, 4, DIM * 8])))
    h1 = tf.nn.relu(g_bn1(conv_transpose(h0, [batchsize, 8, 8, DIM * 4], name="g_h1")))
    h2 = tf.nn.relu(g_bn2(conv_transpose(h1, [batchsize, 16, 16, DIM * 2], name="g_h2")))
    h3 = tf.nn.relu(g_bn3(conv_transpose(h2, [batchsize, 32, 32, DIM * 1], name="g_h3")))
    h4 = conv_transpose(h3, [batchsize, 64, 64, 3], name="g_h4")
    return tf.nn.tanh(h4)
```

We build a placeholder for image to the discriminator, and a placeholder for $$z$$. We build 1 generator and initializes 2 discriminators. But both discriminators share the same trainable parameters so they are actually the same. However, with 2 instances, we can separate the scores (logits) for the real and the generated images by feed real images to 1 discriminator and generated images to another.
```python
images = tf.placeholder(tf.float32, [batchsize, DIM, DIM, 3] , name="real_images")
zin = tf.placeholder(tf.float32, [None, Z_DIM], name="z")

G = generator(zin)
with tf.variable_scope("discriminator") as scope:
    D_prob, D_logit = discriminator(images)
    scope.reuse_variables()
    D_fake_prob, D_fake_logit = discriminator(G)
```

We computed the lost function for the discriminator using the cross entropy for both real and generated images.
```python
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit, labels=tf.ones_like(D_logit)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logit, labels=tf.zeros_like(D_fake_logit)))

d_loss = d_loss_real + d_loss_fake
```

We compute the lost function for the generator by using the logits of the generated images from the discriminator. Then, we backpropagate the gradient to train the $$W$$ such that it can later create more realistic images.
```python
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logit, labels=tf.ones_like(D_fake_logit)))
```

We use 2 separate optimizer to train both network simultaneously:
```python
t_vars = tf.trainable_variables()
d_vars = [var for var in t_vars if 'd_' in var.name]
g_vars = [var for var in t_vars if 'g_' in var.name]

d_optim = tf.train.AdamOptimizer(learningrate, beta1=beta1).minimize(d_loss, var_list=d_vars)
g_optim = tf.train.AdamOptimizer(learningrate, beta1=beta1).minimize(g_loss, var_list=g_vars)
```

The full source code is available [here](https://github.com/jhui/machine_learning/tree/master/generative_adversarial_network).
 
### Variational Autoencoders (VAEs)

Variational autoencoders use gaussian models to generate images.  

#### Gaussian distribution
Before going into the details of VAEs, we discuss the use of gaussian distribution for data modeling. 

In the following diagram, we assume the probability of X equal to a certain value $$x$$, $$p(X=x)$$, follows a gaussian distribution: 
<div class="imgcap">
<img src="/assets/gm/g0.png" style="border:none;width:60%">
</div>

$$
\text{Probability density function (PDF)} = p(X=x) = f(x) = \frac{e^{-(x - \mu)^{2}/(2\sigma^{2}) }} {\sigma\sqrt{2\pi}}
$$

We can sample data using the PDF above. We use the following notation for sample data using a gaussian distribution with mean $$ \mu $$ and standard deviation $$ \sigma $$.

$$
x \sim \mathcal{N}{\left(
\mu 
,
\sigma
\right)}
$$

In the example above, mean: $$ \mu=0 $$, standard deviation: $$ \sigma=0.1$$:

> In many real world examples, the data sample distribution follows a gaussian distribution. 

Now, we generalize it with multiple variables. For example, we want to model the relationship between the body height and the body weight for San Francisco residents. We collect the information from 1000 adult residents and plot the data below with each red dot represents 1 person:

<div class="imgcap">
<img src="/assets/gm/auto.png" style="border:none;width:80%">
</div>

We can plot the corresponding probability density function 

$$ PDF = probability(height=h, weight=w)$$ 

in 3D:

<div class="imgcap">
<img src="/assets/gm/auto2.png" style="border:none;width:60%">
</div>

We can model usch probability density function using a gaussian distribution function.

The PDF with p variables is:

$$
x = \begin{pmatrix}
x_1 \\
\vdots \\
x_p
\end{pmatrix}
$$

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

The notation for sampling x is:

$$
x 
\sim \mathcal{N}{\left(
\mu
,
\sum
\right)}
$$

$$
x =
\begin{pmatrix}
x_1 \\
\vdots \\
x_p
\end{pmatrix}

\sim \mathcal{N}{\left(
\mu
,
\sum
\right)}

= \mathcal{N}{\left(
\begin{pmatrix}
\mu_1 \\
\vdots \\
\mu_p
\end{pmatrix}
,
\sum
\right)}
$$

Let's go back to our weight and height example to illustrate it.

$$
x = \begin{pmatrix}
weight \\
height 
\end{pmatrix}
$$

From the data, we comput the mean weight is 190 lb and mean height is 70 inches:

$$
\mu = \begin{pmatrix}
190 \\
70
\end{pmatrix}
$$

For the covariance matrix $$ \sum $$, here, we illustrate how to compute one of the element $$ E_{21} $$

$$
 E_{21} = E[(x_{2} - \mu_{2})(x_{1} - \mu_{1})] = E[(x_{height} - 70)(x_{weight} - 190)]
$$

which $$E$$ is the expected value. Let say we have only 2 datapoints (200 lb, 80 inches) and (180 lb, 60 inches)

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

$$ E_{21} $$ measures the co-relationship between variables $$x_2$$ and $$x_1$$. Positive values means both are positively related. With not surprise, $$ E_{21} $$ is positive because weight increases with height. If two variables are independent of each other, it should be 0 like:

$$
\sum = \begin{pmatrix}
    100 & 0 \\
    0 & 50 \\
\end{pmatrix}
$$

and we will simplify the gaussian distribution notation here as:
$$
x \sim \mathcal{N}{\left(
\begin{pmatrix}
190 \\
70
\end{pmatrix}
,
\begin{pmatrix}
    100 \\
    50 \\
\end{pmatrix}
\right)}
$$


#### Autoencoders

In an autoencoders, we use a deep network to map the input image (for example 256x256 pixels = 256x256 = 65536 dimension) to a lower dimension **latent variables** (latent vector say 100-D  vector: $$ (x_1, x_2, \cdots x_{100}) $$). We use another deep network to decode the latent variables to restore the image. We train both encoder and decoder network to minimize the difference between the original image and the decoded image. By forcing the image to a lower dimension, we hope the network learns to encode the image by extracting core features.

<div class="imgcap">
<img src="/assets/gm/auto3.jpg" style="border:none;width:100%">
</div>

For example, we enter a 256x256 image to the encoder, we use a CNN to encode the image to 20-D latent variables $$ (x_1, x_1, ... x_{20}) = (0.1, 0, ..., -0.05) $$. We use another network to decode the latent variables into a 256x256 image. We use backpropagation with cost function comparing the decoded and input image to train both encoding and decoding network.

#### VAEs

For VAEs, we replace the middle part with a stochastic model using a gaussian distribution. Let's get into an example to demonstrate the flow:

<div class="imgcap">
<img src="/assets/gm/auto4.jpg" style="border:none;width:100%">
</div>

For a variation autoencoder, we replace the middle part with 2 separate steps. VAE does not generate the latent vector directly. It generates 100 Gaussian distributions each represented by a mean $$ (\mu_i) $$ and a standard deviation $$ (\sigma_i) $$. Then it samples a latent vector, say (0.1, 0.03, ..., -0.01), from these distributions.  For example, if element $$x_i $$ of the latent vector has $$ \mu_i=0.1 $$ and $$ \sigma_i=0.5 $$. We randomly select $$ x_i $$ with probability based on this Gaussian distribution: 

$$
p(X=x_{i}) = \frac{e^{-(x_{i} - \mu_i)^{2}/(2\sigma_i^{2}) }} {\sigma_i\sqrt{2\pi}}
$$

$$
z = 
\begin{pmatrix}
z_1 \\
\vdots \\
z_{20}
\end{pmatrix}
\sim \mathcal{N}{\left(
\begin{pmatrix}
\mu_1 \\
\vdots \\
\mu_{20}
\end{pmatrix}
,
\begin{pmatrix}
\sigma_{1}\\
\vdots\\
\sigma_{20}\\
\end{pmatrix}
\right)}
$$

Say, the encoder generates $$ \mu=(0, -0.01, ..., 0.2) $$ and $$ \sigma=(0.05, 0.01, ..., 0.02) $$ 

We can sample a value from this distribution:

$$
\mathcal{N}{\left(
\begin{pmatrix}
0 \\
-0.01 \\
\vdots \\
0.2
\end{pmatrix}
,
\begin{pmatrix}
0.05 \\
0.01 \\
\vdots \\
0.02
\end{pmatrix}
\right)}
$$

with the latent variables as (say) :

$$
z = 
\begin{pmatrix}
z_1 \\
\vdots \\
z_{20}
\end{pmatrix}
=
\begin{pmatrix}
0.03 \\
-0.015 \\
\vdots \\
0.197
\end{pmatrix}
$$

The autoencoder in the previous section is very hard to train with not much guarantee that the network is generalize enough to make good predictions.(We say the network simply memorize the training data.) 
 In VAEs, we add a constrain to make sure:
1. The latent variable are relative independent of each other, i.e. the 20 variables are relatively independent of each other (not co-related). This maximizes what a 20-D latent vectors can represent. 
1. Latent variables $$z$$ which are similar in values should generate similar looking images. This is a good indication that the network is not trying to memorize individual image.

To achieve this, we want the gaussian distribution model generated by the encoder to be as close to a normal gaussian distribution function. We penalize the cost function if the gaussian function is deviate from a normal distribution. This is very similar to the L2 regularization in a fully connected network in avoiding overfitting.

$$
z \sim \mathcal{N}{\left(
0
,
1
\right)}
= \text{normal distribution}
$$

In a normal gaussian distribution, the covariance $$ E_{ij} $$ is 0 for $$ i \neq j $$. That is the latent variables are independent of each other. If the distribution is normalize, the distance between different $$z$$ will be a good measure of its similarity. With sampling and the gaussian distribution, we encourage the network to have similar value of $$z$$ for similar images. 
  
#### Encoder
 
Here we have a CNN network with 2 convolution layers using leaky ReLU follow by one fully connected layer to generate 20 $$ \mu $$ and another fully connected layer for 20 $$ \sigma $$.
```python
def recognition(self, input_images):
     with tf.variable_scope("recognition"):
         h1 = lrelu(conv2d(input_images, 1, 16, "d_h1"))   # Shape: (?, 28, 28, 1) -> (?, 14, 14, 16)
         h2 = lrelu(conv2d(h1, 16, 32, "d_h2"))            # (?, 7, 7, 32)
         h2_flat = tf.reshape(h2, [self.batchsize, 7 * 7 * 32])  # (100, 1568)

         w_mean = linear(h2_flat, self.n_z, "w_mean")      # (100, 20)
         w_stddev = linear(h2_flat, self.n_z, "w_stddev")  # (100, 20)

     return w_mean, w_stddev
```

#### Decoder
The decoder feeds the 20 latent variables to a fully connected layer followed with 2 transpose convolution layer with ReLU. The output is then feed into a sigmoid layer to generate the image.
```python
def generation(self, z):
    with tf.variable_scope("generation"):
        z_develop = linear(z, 7 * 7 * 32, scope='z_matrix')  # (100, 20) -> (100, 1568)
        z_matrix = tf.nn.relu(tf.reshape(z_develop, [self.batchsize, 7, 7, 32]))  # (100, 7, 7, 32)
        h1 = tf.nn.relu(conv_transpose(z_matrix, [self.batchsize, 14, 14, 16], name="g_h1"))  # (100, 14, 14, 16)
        h2 = conv_transpose(h1, [self.batchsize, 28, 28, 1], name="g_h2")  # (100, 14, 14, 16)
        out = tf.nn.sigmoid(h2)  # (100, 28, 28, 1)

    return out     
```
 
#### Building the VAE

We use the encoder to encode the input image. Use sampling to generate $$ z $$ from the mean and variance of the gaussian distribution and then decode it.
```python
# Encode the image
z_mean, z_stddev = self.recognition(image_matrix)
		
# Sampling z
samples = tf.random_normal([self.batchsize, self.n_z], 0, 1, dtype=tf.float32)
guessed_z = z_mean + (z_stddev * samples)

# Decode the image
self.generated_images = self.generation(guessed_z)
generated_flat = tf.reshape(self.generated_images, [self.batchsize, 28 * 28])
```

#### Cost function & training

We define a generation loss which measure the difference between the original and the decoded message using the mean square error.
The latent loss measure the difference between gaussian function of the image from a normal distribution using KL-Divergence.
```python
self.generation_loss = -tf.reduce_sum(
    self.images * tf.log(1e-8 + generated_flat) + (1 - self.images) * tf.log(1e-8 + 1 - generated_flat), 1)

self.latent_loss = 0.5 * tf.reduce_sum(
    tf.square(z_mean) + tf.square(z_stddev) - tf.log(tf.square(z_stddev)) - 1, 1)

self.cost = tf.reduce_mean(self.generation_loss + self.latent_loss)
```
We use the adam optimizer to train both networks.
```python
self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.cost)
```
 
#### Cost function in detail

KL divergence measures the difference of 2 distributions. By definition, KL divergence is defined as: 

$$
KL\left(q||p\right) = \sum_{x} q(x) \log (q(x)/p(x)) 
$$

$$
KL\left( q||p \right) = E_q[log (q(x))−log (p(x))]
$$

$$ q_\lambda (z∣x) $$ is the distribution of $$ z $$ predicted by our deep network. We want it to match the true distribution $$ p(z∣x) $$. We want the distribution approximate from the deep network has little divergence from the true distribution. i.e. we want to find $$ \lambda $$ with the smallest KL divergence.

$$
KL\left( q_\lambda (z∣x) ∣∣ p(z∣x)\right) = E_q \lbrack \log q_λ (z∣x) \rbrack - E_q \lbrack \log p (z∣x) \rbrack 
$$

Apply:

$$
p(z|x) = \frac{p(x,z)}{p(x)}
$$

$$
KL\left( q_\lambda (z∣x) ∣∣ p(z∣x)\right) = E_q \lbrack \log q_λ (z∣x) \rbrack - E_q \lbrack \log p (x,z) \rbrack + \log (p(x))
$$

Define the term ELBO (Evidence lower bound) as:

$$
ELBO(λ) =   - ( E_q \lbrack \log q_λ (z∣x) \rbrack - E_q \lbrack \log p (x,z) \rbrack )
$$

$$
KL\left( q_\lambda (z∣x) ∣∣ p(z∣x)\right) = - ELBO(λ) + \log (p(x))
$$

To minimize the KL-divergence. we want to find $$ \lambda $$ to maximize $$ ELBO(λ)$$. (Since $$ log (p(x)) $$ is not dependent on $$ \lambda $$, we skip the term $$\log (p(x))$$.)

Back to the equation:

$$
KL\left( q_\lambda (z∣x) ∣∣ p(z∣x)\right) = - ELBO(λ) + \log (p(x))
$$

$$
ELBO(λ) =   \log (p(x)) - KL\left( q_\lambda (z∣x) ∣∣ p(z∣x)\right)
$$

$$
ELBO_i(\lambda) = E_{q_\lambda(z∣x_i) }  \lbrack  \log (p(x_{i}|z))  \rbrack - KL\left( q_\lambda (z∣x_{i}) ∣∣ p(z)\right)
$$

Let $$\theta$$ be the parameter for the encode network and $$\phi$$ for the decoder network:

$$
ELBO_i(\theta, \phi) = E_{q_\theta(z∣x_i) }  \lbrack  \log (p_{\theta}(x_{i}|z))  \rbrack - KL\left( q_\phi (z∣x_{i}) ∣∣ p(z)\right)
$$

The first term measured the probability of output $$x_i$$ in the decoder with $$x_i$$ as input to the encoder.  The second term is the KL divergence.

 
#### Result
The full source code for the VAE is located [here](https://github.com/jhui/machine_learning/tree/master/variational_autoencoder). Here is the digits created by a VAE.
<div class="imgcap">
<img src="/assets/gm/r2.png" style="border:none;width:50%">
</div>


### Credits
Part of the source code for GAN & Variational Autoencoders is originated from https://github.com/kvfrans/generative-adversial and https://github.com/carpedm20/DCGAN-tensorflow.