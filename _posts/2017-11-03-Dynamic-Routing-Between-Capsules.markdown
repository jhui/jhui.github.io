---
layout: post
comments: true
mathjax: true
priority: 410
title: “Understanding Dynamic Routing between Capsules”
excerpt: “A simple tutorial in understanding Capsules, Dynamic routing and CapsNet”
date: 2017-11-03 11:00:00
---

This article covers the technical paper by Sara Sabour, Nicholas Frosst and Geoffrey Hinton on [Dynamic Routing between Capsules](https://arxiv.org/pdf/1710.09829.pdf). The source code implementation is originated from [XifengGuo](https://github.com/XifengGuo/CapsNet-Keras) using Keras with Tensorflow. In this article, we will first describe the basic concept and later apply it in CapsNet to detect digits in MNist.

### Capsule

In deep learning, the activation level of a neuron is often interpreted as the likelihood of detecting a specific feature. 

<div class="imgcap">
<img src="/assets/capsule/fc.jpg" style="border:none;width:70%;">
</div>

A capsule is a group of neurons that not only capture the likelihood but also the parameters of the specific feature.

For example, the first row below indicates the probabilities of detecting the number "7" by a neuron. A 2-D capsule is formed by combining 2 neurons. This capsule outputs a 2-D vector in detecting the number "7". For the first image in the second row, it outputs a vector $$ v = (0, 0.9)$$. The magnitude of the vector $$ \| v \| = \sqrt{ 0^2 + 0.9^2 } = 0.9 $$ corresponds to the probability of detecting "7".

<div class="imgcap">
<img src="/assets/capsule/cap1.jpg" style="border:none;width:60%;">
</div>

In the third row, we rotate the image by 20°. The capsule will generate vectors with the same magnitude but different orientations. Here, the angle of the vector represents the angle of rotation for the number "7". As we can image, we can add 2 more neurons to a capsule to capture the size and stroke width. 

<div class="imgcap">
<img src="/assets/capsule/style.jpg" style="border:none;width:30%;">
</div>

> We call the output vector of a capsule as the **activity vector** with magnitude represents the probability of detecting a feature and its orientation represents its parameters (properties).

### Compute the output of a capsule

Recall a fully connected neural network:

<div class="imgcap">
<img src="/assets/capsule/fc1.jpg" style="border:none;width:35%;">
</div>

The output of each neuron is computed from the output of the neurons from the previous layer:

$$
\begin{split}
z_j &= \sum_i W_{ij} x_i \\
y_j &= ReLU(z_j) \\
\end{split}
$$

which $$W_{ij}, z_j$$ and $$y_i$$ are all scalars. 

For a capsule, the input $$u_i$$ and the output $$v_j$$ of a capsule are vectors. 

<div class="imgcap">
<img src="/assets/capsule/fc2.jpg" style="border:none;width:35%;">
</div>

We apply a transformation matrix $$W_{ij}$$ to the capsule output $$ u_i $$ of the pervious layer. For example, if $$u_i$$ is a k-D vector, we can apply a $$m \times k $$ matrix to transform it to a m-D $$\hat{u}_{j \vert i}$$. Then we compute a weighted sum (with weights $$c_{ij}$$) $$\hat{u}_{j \vert i}$$ of all the capsules from the previous layer.

$$
\begin{split}
\hat{u}_{j|i} &= W_{ij} u_i \\
s_j & = \sum_i c_{ij}  \hat{u}_{j|i} \\
\end{split}
$$

$$c_{ij}$$ are **coupling coefficients** that are trained by the iterative dynamic routing process (discussed next) and $$ \sum_{i} c_{ij}$$ are designed to sum to one.

Instead of applying a ReLU function, we apply a squashing function. 

$$
\begin{split}
v_{j} & = \frac{\| s_{j} \|^2}{ 1 + \| s_{j} \|^2} \frac{s_{j}}{ \| s_{j} \|}  \\
\end{split}
$$

It shrinks small vectors to zero and long vectors to unit vectors.

$$
\begin{split}
v_{j} & \approx \| s_{j} \| s_{j}  \quad & \text{for } s_{j} \text { is short } \\
v_{j} & \approx \frac{s_{j}}{ \| s_{j} \|}  \quad & \text{for } s_{j} \text { is long } \\
\end{split}
$$

### Iterative dynamic Routing

The coupling coefficients $$ c_{ij} $$ determines the relevancy of a capsule in activating another capsule in the next layer. After apply a transformation matrix $$W_{ij}$$ to the previous capsule output $$u_i$$, we compute the **prediction vector** $$\hat{u}_{j \vert i}$$. 

$$
\begin{split}
\hat{u}_{j|i} &= W_{ij} u_i \\
\end{split}
$$

The similarity of the prediction vector with the capsule output $$v_j$$ corresponds to the relevancy of $$u_i$$ in activating $$v_i$$. In short, how much a capsule output should contribute to the next layer capsule output after a transformation of its properties. The higher the similarity, the larger the coupling coefficients $$ c_{ij} $$ should be. Such similarity is measured using the scalar product and we adjust a relevancy score $$ b_{ij} $$ according to the similarity. 

$$
\begin{split}
similarity & = \hat{u}_{j \vert i} \cdot v_j \\
b_{ij} & ←  b_{ij} + similarity \\
\end{split}
$$

The coupling coefficients $$ c_{ij} $$ is finally calculated as:

$$
c_{ij} = \frac{\exp{b_{ij}}} {\sum_k \exp{b_{ik}} }
$$

This dynamic routing mechanism ensure that the output of a capsule gets route to an appropriate capsule in the next layer. A capsule prefers to send its output to the next layer capsules whose activity vectors have a big scalar product (similarity) with the prediction coming from that capsule.

Here is the pseudo code:

<div class="imgcap">
<img src="/assets/capsule/alg.jpg" style="border:none;width:90%;">
</div>

[Source Sara Sabour, Nicholas Frosst, Geoffrey Hinton](https://arxiv.org/pdf/1710.09829.pdf) 

> Routing a capsule to the next layer capsule based on such similarity is called Routing-by-agreement.

In max pool, we only keep the most dominating (max) features. Capsules maintain a weighted sum of features from the previous layer. Hence, it is more suitable in detecting overlapping features. (for example detecting multiple overlapping digits in the handwriting)

### Loss function (Margin loss)

To detect multiple digits in a picture, Capsules use a separate margin loss, $$L_c$$ for each category $$c$$:

$$
L_c = T_c max(0, m^+ − \|vc\|)^2 + λ (1 − T_c) max(0, \|vc\| − m^−)^2
$$

which $$T_c = 1$$ if an object of class $$c$$ is present. $$m^+ = 0.9$$ and $$m^− = 0.1$$. The λ down-weighting (default 0.5) stops the initial learning from shrinking the activity
vectors of all classes. The total loss is just the sum of the losses of all classes.

Computing the margin loss in Keras
```python
def margin_loss(y_true, y_pred):
    """
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))
```

### CapsNet architecture

CapsNet applies capsules to classify the MNist digits. The following is the architecture using CapsNet.

<div class="imgcap">
<img src="/assets/capsule/arch1.jpg" style="border:none;width:70%;">
</div>

Image is feed into the ReLU Conv1 which is a standard convolution layer. It applies 256 9x9 kernels (stride 1, no padding) to generate an output with 256 channels (feature maps). Without padding, the spatial dimension is reduced to 20x20 ( 28-9+1=20) It is then feed into PrimaryCapsules which is a modified convolution layer supporting capsules. In a regular convolution layer, we use k-kernels to create an output with k channels. PrimaryCapsules used 8x32 kernels to generate 32 8-D capsules. (i.e. 8 output neurons are grouped to form a capsule) PrimaryCapsules uses 9x9 kernels (stride 2, no padding) to reduce the spatial dimension from 20x20 to 6x6 ( $$\frac{20-9+1}{2} = 6 $$). In PrimaryCapsules, we have 32x6x6 capsules. We apply a transformation matrix $$W_{ij} $$ with shape 16x8 to convert each capsule (shape: 8x1) to a 16-D capsule (vector) for each class $$j$$ (from 1 to 10).

$$
\begin{split}
\hat{u}_{j|i} &= W_{ij} u_i \\
\end{split}
$$

The final output $$v_j$$ for class $$j$$ is computed as:

$$
\begin{split}
s_j & = \sum_i c_{ij}  \hat{u}_{j|i} \\
v_{j} & = \frac{\| s_{j} \|^2}{ 1 + \| s_{j} \|^2} \frac{s_{j}}{ \| s_{j} \|}  \\
\end{split}
$$

Because there are 10 classes, the shape of DigiCaps will be 10x16 (10 16-D vector.) Each vector $$v_j$$ acts as the capsule for class $$j$$. The probability of the image to be classify as $$j$$ is computed by $$\| v_j \|$$. In our example, $$ v_7$$ is the latent representation of the input image with the true label 7. Using $$v_7$$ and fully connected networks, we can reconstruct the 28x28 image. 

| Layer Name | Apply | Output shape |
| --- | --- | --- | --- |
| Image | Raw image array |  28x28x1|
| ReLU Conv1 | Convolution layer with 9x9 kernels output 256 channels, stride 1, no padding with ReLU  | 20x20x256 |
| PrimaryCapsules | Convolution capsule layer with 9x9 kernel output 32x6x6 8-D capsule, stride 2, no padding  | 6x6x32x8 |
| DigitCaps | Capsule output computed from a $$W_{ij} $$ (16x8 matrix) between $$u_i$$ and $$v_j$$ ($$i$$ from 1 to 32x6x6 and $$j$$ from 1 to 10). | 10x16 |
| FC1 | Fully connected with ReLU | 512 |
| FC2 | Fully connected with ReLU | 1024 |
| Output image | Fully connected with sigmoid | 784 (28x28) | 

Here is the Keras code in creating the CapsNet model:
```python
def CapsNet(input_shape, n_class, num_routing):
    """
    :param input_shape: (None, width, height, channels)
    :param n_class: number of classes
    :param num_routing: number of routing iterations
    :return: A Keras Model with 2 inputs (image, label) and 
             2 outputs (capsule output and reconstruct image)
    """
    # Image
    x = layers.Input(shape=input_shape)

    # ReLU Conv1
    conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, 
	             padding='valid', activation='relu', name='conv1')(x)

    # PrimaryCapsules: Conv2D layer with `squash` activation, 
    # reshape to [None, num_capsule, dim_vector]
    primarycaps = PrimaryCap(conv1, dim_vector=8, n_channels=32, 
	                    kernel_size=9, strides=2, padding='valid')

    # DigitCaps: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_vector=16, 
	        num_routing=num_routing, name='digitcaps')(primarycaps)

    # The length of the capsule's output vector 
    out_caps = Length(name='out_caps')(digitcaps)

    # Decoder network.
    y = layers.Input(shape=(n_class,))

    # The true label is used to extract the corresponding vj
    masked = Mask()([digitcaps, y])  
    x_recon = layers.Dense(512, activation='relu')(masked)
    x_recon = layers.Dense(1024, activation='relu')(x_recon)
    x_recon = layers.Dense(784, activation='sigmoid')(x_recon)
    x_recon = layers.Reshape(target_shape=[28, 28, 1], name='out_recon')(x_recon)

    # two-input-two-output keras Model
    return models.Model([x, y], [out_caps, x_recon])
```

Length of the capsule's output vector $$\| v\| $$ which correspond to the probability of it belong to a class. (For example, $$ \| v_7 \| $$ is the probability of the input image is a 7.)
```python
class Length(layers.Layer):
    def call(self, inputs, **kwargs):
        # L2 length which is the square root 
        # of the sum of square of the capsule element
        return K.sqrt(K.sum(K.square(inputs), -1))
```
			
#### PrimaryCapsules

PrimaryCapsules outputs 32x6x6 8-D capsules.
```python
def PrimaryCap(inputs, dim_vector, n_channels, kernel_size, strides, padding):
    """
    Apply Conv2D `n_channels` times and concatenate all capsules
    :param inputs: 4D tensor, shape=[None, width, height, channels]
    :param dim_vector: the dim of the output vector of capsule
    :param n_channels: the number of types of capsules
    :return: output tensor, shape=[None, num_capsule, dim_vector]
    """
    output = layers.Conv2D(filters=dim_vector*n_channels, kernel_size=kernel_size, strides=strides, padding=padding)(inputs)
    outputs = layers.Reshape(target_shape=[-1, dim_vector])(output)
    return layers.Lambda(squash)(outputs)
```

#### Squash function

Squash function behaves like a sigmoid function to squash a vector between a 0 vector and an unit vector.

$$
\begin{split}
v_{j} & = \frac{\| s_{j} \|^2}{ 1 + \| s_{j} \|^2} \frac{s_{j}}{ \| s_{j} \|}  \\
\end{split}
$$


```python
def squash(vectors, axis=-1):
    """
    The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    :param vectors: some vectors to be squashed, N-dim tensor
    :param axis: the axis to squash
    :return: a Tensor with same shape as input vectors
    """
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm)
    return scale * vectors
```

#### DigitCaps with dynamic routing

Create a capsule layer (DigitCaps) with 10 (n_class) 16-D (dim_vector) capsules:
```python
# num_routing is default to 3
digitcaps = CapsuleLayer(num_capsule=n_class, dim_vector=16, 
                  num_routing=num_routing, name='digitcaps')(primarycaps)
```

CapsuleLayer is just a simple extension of a dense layer. Instead of taking a scalar and output a scalar, it takes a vector and output a vector:

* input shape = (None, input_num_capsule (32), input_dim_vector(8) )
* output shape = (None, num_capsule (10), dim_vector(16) ) 

Here is the CapsuleLayer and we will detail some part of the code for explanation later.
```python
class CapsuleLayer(layers.Layer):
    """
    The capsule layer. 
 	
    :param num_capsule: number of capsules in this layer
    :param dim_vector: dimension of the output vectors of the capsules in this layer
    :param num_routings: number of iterations for the routing algorithm
    """
    def __init__(self, num_capsule, dim_vector, num_routing=3,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule    # 10
        self.dim_vector = dim_vector      # 16
        self.num_routing = num_routing    # 3
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

    def build(self, input_shape):
        "The input Tensor should have shape=[None, input_num_capsule, input_dim_vector]"		
        assert len(input_shape) >= 3, 
        self.input_num_capsule = input_shape[1]
        self.input_dim_vector = input_shape[2]

        # Transform matrix W
        self.W = self.add_weight(shape=[self.input_num_capsule, self.num_capsule, 
                                 self.input_dim_vector, self.dim_vector],
                                 initializer=self.kernel_initializer,
                                 name='W')

        # Coupling coefficient. 
        # The redundant dimensions are just to facilitate subsequent matrix calculation.
        self.bias = self.add_weight(shape=[1, self.input_num_capsule, self.num_capsule, 1, 1],
                                    initializer=self.bias_initializer,
                                    name='bias',
                                    trainable=False)
        self.built = True

    def call(self, inputs, training=None):
        # inputs.shape = (None, input_num_capsule, input_dim_vector)
        # Expand dims to (None, input_num_capsule, 1, 1, input_dim_vector)
        inputs_expand = K.expand_dims(K.expand_dims(inputs, 2), 2)

        # Replicate num_capsule dimension to prepare being multiplied by W
        # Now shape = [None, input_num_capsule, num_capsule, 1, input_dim_vector]
        inputs_tiled = K.tile(inputs_expand, [1, 1, self.num_capsule, 1, 1])

        # Compute `inputs * W` by scanning inputs_tiled on dimension 0. 
        # inputs_hat.shape = [None, input_num_capsule, num_capsule, 1, dim_vector]
        inputs_hat = tf.scan(lambda ac, x: K.batch_dot(x, self.W, [3, 2]),
                             elems=inputs_tiled,
                             initializer=K.zeros([self.input_num_capsule, self.num_capsule, 1, self.dim_vector]))
        # Routing algorithm
        assert self.num_routing > 0, 'The num_routing should be > 0.'
        for i in range(self.num_routing):
            c = tf.nn.softmax(self.bias, dim=2)  # dim=2 is the num_capsule dimension
            # outputs.shape=[None, 1, num_capsule, 1, dim_vector]
            outputs = squash(K.sum(c * inputs_hat, 1, keepdims=True))

            # last iteration needs not compute bias which will not be passed to the graph any more anyway.
            if i != self.num_routing - 1:
                self.bias += K.sum(inputs_hat * outputs, -1, keepdims=True)
        return K.reshape(outputs, [-1, self.num_capsule, self.dim_vector])
```

_build_ declares the self.W parameters representing the transform matrix W and self.bias representing the coupling coefficient. 
```python
    def build(self, input_shape):
        "The input Tensor should have shape=[None, input_num_capsule, input_dim_vector]"		
        assert len(input_shape) >= 3, 
        self.input_num_capsule = input_shape[1]
        self.input_dim_vector = input_shape[2]

        # Transform matrix W
        self.W = self.add_weight(shape=[self.input_num_capsule, self.num_capsule, 
                                 self.input_dim_vector, self.dim_vector],
                                 initializer=self.kernel_initializer,
                                 name='W')

        # Coupling coefficient. 
        # The redundant dimensions are just to facilitate subsequent matrix calculation.
        self.bias = self.add_weight(shape=[1, self.input_num_capsule, self.num_capsule, 1, 1],
                                    initializer=self.bias_initializer,
                                    name='bias',
                                    trainable=False)
        self.built = True
```

To compute:

$$
\begin{split}
\hat{u}_{j|i} &= W_{ij} u_i \\
\end{split}
$$

The code first expand the dimension of $$u_i$$ and then multiple it with $$w$$. Nevertheless, the simple dot product implementation of $$ W_{ij} u_i $$ (commet out below) is replaced by tf.scan for better speed performance.

```python
class CapsuleLayer(layers.Layer):
    ...

    def call(self, inputs, training=None):
        # inputs.shape = (None, input_num_capsule, input_dim_vector)
        # Expand dims to (None, input_num_capsule, 1, 1, input_dim_vector)
        inputs_expand = K.expand_dims(K.expand_dims(inputs, 2), 2)

        # Replicate num_capsule dimension to prepare being multiplied by W
        # Now shape = [None, input_num_capsule, num_capsule, 1, input_dim_vector]
        inputs_tiled = K.tile(inputs_expand, [1, 1, self.num_capsule, 1, 1])

        """  
        # Compute `inputs * W` 
        # By expanding the first dim of W.
        # W has shape (batch_size, input_num_capsule, num_capsule, input_dim_vector, dim_vector)
        w_tiled = K.tile(K.expand_dims(self.W, 0), [self.batch_size, 1, 1, 1, 1])
        
        # Transformed vectors, 
        inputs_hat.shape = (None, input_num_capsule, num_capsule, 1, dim_vector)
        inputs_hat = K.batch_dot(inputs_tiled, w_tiled, [4, 3])
        """
		
        # However, we will implement the same code with a faster implementation using tf.sacn	
        # Compute `inputs * W` by scanning inputs_tiled on dimension 0. 
        # inputs_hat.shape = [None, input_num_capsule, num_capsule, 1, dim_vector]
        inputs_hat = tf.scan(lambda ac, x: K.batch_dot(x, self.W, [3, 2]),
                             elems=inputs_tiled,
                             initializer=K.zeros([self.input_num_capsule, self.num_capsule, 1, self.dim_vector]))
```

Here is the code to implement the following Iterative dynamic Routing pseudo code.

<div class="imgcap">
<img src="/assets/capsule/alg.jpg" style="border:none;width:90%;">
</div>

```python
class CapsuleLayer(layers.Layer):
    ...
    def call(self, inputs, training=None):
        ...
        # Routing algorithm
        assert self.num_routing > 0, 'The num_routing should be > 0.'
		
        for i in range(self.num_routing):  # Default: loop 3 times
            c = tf.nn.softmax(self.bias, dim=2)  # dim=2 is the num_capsule dimension
			
            # outputs.shape=[None, 1, num_capsule, 1, dim_vector]
            outputs = squash(K.sum(c * inputs_hat, 1, keepdims=True))

            # last iteration needs not compute bias which will not be passed to the graph any more anyway.
            if i != self.num_routing - 1:
                self.bias += K.sum(inputs_hat * outputs, -1, keepdims=True)
        return K.reshape(outputs, [-1, self.num_capsule, self.dim_vector])
```

#### Decoder

We use the true label to select $$ v_j $$ to reconstruct the image during training. Then we feed $$v_j$$ through 3 fully connected layers to re-generate the original image. 

Select $$v_j$$ in training with Mask
```python
class Mask(layers.Layer):
    """
    Mask a Tensor with shape=[None, d1, d2] by the max value in axis=1.
    Output shape: [None, d2]
    """
    def call(self, inputs, **kwargs):
        # use true label to select target capsule, shape=[batch_size, num_capsule]
        if type(inputs) is list:  # true label is provided with shape = [batch_size, n_classes], i.e. one-hot code.
            assert len(inputs) == 2
            inputs, mask = inputs
        else:  # if no true label, mask by the max length of vectors of capsules
            x = inputs
            # Enlarge the range of values in x to make max(new_x)=1 and others < 0
            x = (x - K.max(x, 1, True)) / K.epsilon() + 1
            mask = K.clip(x, 0, 1)  # the max value in x clipped to 1 and other to 0

        # masked inputs, shape = [batch_size, dim_vector]
        inputs_masked = K.batch_dot(inputs, mask, [1, 1])
        return inputs_masked
```

#### Reconstruction loss

A reconstruction loss $$ \| image - \text{reconstructed image} \|$$ is added to the loss function. It trains the network to capture the critical properties into the capsule. However, the reconstruction loss is multiple by a regularization factor (0.0005) so it does not dominate over the marginal loss.

### What capsule is learning?

Each capsule in DigiCaps is a 16-D vector. By slightly varying one dimension by holding other constant, we can learn what property for each dimension is capturing. Each row below is the reconstructed image (using the decoder) of changing only one dimension.

<div class="imgcap">
<img src="/assets/capsule/dim.png" style="border:none;width:70%;">
</div>

[Source Sara Sabour, Nicholas Frosst, Geoffrey Hinton](https://arxiv.org/pdf/1710.09829.pdf) 


