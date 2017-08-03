---
layout: post
comments: true
mathjax: true
priority: 180000
title: “Machine learning - PCA, Latent Factor Model, Visualization”
excerpt: “Machine learning - PCA, Latent Factor Model, Visualization”
date: 2017-01-15 12:00:00
---

### Principal Component Analysis (PCA)

PCA is a linear model in mapping d-dimensional input features to k-dimensional latent features (Principal components). 

$$ 
\begin{split}
x_{ij} & = w^T_j z_i \\
\\
x_{ij} & = w_{1j} z_{i1} + w_{2j} z_{i2} + \cdots + w_{kj} z_{ik} \\
\end{split}
$$ 

<div class="imgcap">
<img src="/assets/ml/eqp2.png" style="border:none;width:40%">
</div>

$$
\text{which } w_j \text{ is column } j \text{ of } W. \\
$$


### Example

For example, here is the $$W$$ to match a RGB pixel to one of the 4 color palete:

$$
z = \begin{pmatrix}
0 \\
0 \\
1 \\
0 \\
\end{pmatrix}
W = \begin{pmatrix}
R_1 & G_1 & B_1\\
R_2 & G_2 & B_2\\
R_3 & G_3 & B_3\\
R_4 & G_4 & B_4\\
\end{pmatrix}
$$


$$
\begin{split}
x_{ij} & = w^T_j z_i \\
feature_2 & = W_{12} z_1 + W_{22} z_2 + W_{32} z_3 + W_{42} z_4 \\
\\
G & =  G_{1} z_1 + G_{2} z_2 + G_{3} z_3 + G_{4} z_4 \\
\end{split}
$$

For d = 2 and k = 1, 2D features (purple dots) are projected onto the 1D blue line. PCA selects a projection that can maximize the variance.

<div class="imgcap">
<img src="/assets/ml/pca.png" style="border:none;width:50%">
</div>

#### Matrix factorization
It can also be viewed as an approximation to matrix factorization.

$$
\text{X: N x d} \\
\text{Z: N x k} \\
\text{W: k x d} \\
$$

$$
X \approx Z W \\
$$


<div class="imgcap">
<img src="/assets/ml/x12.png" style="border:none;width:50%">
</div>

### Cost Function

The cost function is:

$$
\begin{split}
J(W, Z) & = \sum^N_{i=1} \sum^d_{j=1} (w^T_j z_i - x_{ij})^2 \\
& = \sum^N_{i=1} \| W^T z_i - Xi \|^2  \\
& =  \| ZW - X \|^2_F \\
\end{split}
$$

which 

$$
\| M \|^2_F = \sum_i \sum_j m_{ij}^2
$$


### Solving W

PCA is an unsupervised learning on latent factors $$W$$ and latent features $$Z$$. We want to find $$W$$ that minimize $$J$$

$$
\begin{split}
J(W, Z) & =  \| ZW - X \|^2_F \\
\end{split}
$$

To solve $$W$$, we can use

* Singular value decomposition (SVD) - non-iterative approach
* Alternating minimization:
	 * Optimize ‘W’ with ‘Z’ fixed
	 * Optimize ‘Z’ with ‘W’ fixed
	 * Keep repeating
	 
$$
\begin{split}
\nabla_W J(W, Z) & = Z^TZW-Z^TX = 0\\
\implies W &= (Z^TZ)^{-1}(Z^TX) \\
\end{split}
$$

$$
\begin{split}
\nabla_Z J(W, Z) & = ZWW^T-XW^T = 0\\
\implies Z & = XW^T(WW^T)^{-1} \\
\end{split}
$$
	 
* Gradient descent

### Predicting Z

Given $$W$$ and testing data $$\hat{X}$$, $$\hat{Z}$$ is computed by first subtract the training mean from the features and then optimize $$Z$$.  

$$
\begin{split}
\hat{x_i} & = \hat{x_i} -\mu \\
\hat{Z} & = \hat{X}W^T(WW^T)^{-1} \\
\end{split}
$$

### Choosing number of latent factors k

PCA is about maximizing variance. To keep 90% of variance, we want to set $$R = 0.1$$.

$$
\begin{split}
J_k & =  \| ZW - X \|^2_F \\
J_0 & =  \| X \|^2_F \\
\\
\frac{J_k}{J_0} & = \frac{\| ZW - X \|^2_F}{\| X \|^2_F} < R \\
R & > \frac{\| ZW - X \|^2_F}{n \cdot var(x_{ij})}
\end{split}
$$

### W uniqueness (orthogonal)

The solutin for W is not unique. But we can impose the magnitude of each row of $$W$$ ($$W_c$$) to 1 and each latent factors are independent of each other (cross product is 0).

$$
\begin{split}
\| W_c \| & = 1 \\
W_{c}^T W_{c'} & = 0 \quad \text{for } c^{'} \neq c \\ 
\end{split}
$$

We can still have the factors $$W_c$$ rotated or label switching. To fix this, we can 

* First set k = 1, and solve W with the constraint above
* Set k = 2 with the first factor set and solve $$W$$ for the second factor $$W_2$$
* Repeat until reaching our target k

The blue line is our first optimized W for $$k = 1$$. The optimal solution $$W$$ for $$k = 2$$ is the red line orthogonal to the blue line.

<div class="imgcap">
<img src="/assets/ml/pca2.png" style="border:none;width:50%">
</div>

$$
\begin{split}
\| w_c  \| & = 1 \\
w_c^T w_{c'} & = \begin{cases}
                        1  \quad \text{ if } c = c^{'} \\
                        0 \quad \text{ if } c \neq c^{'} \\
\end{cases} \\
\implies W W^T & = I \\
\end{split}
$$

Solving Z becomes:

$$
\begin{split}
Z & = XW^T(WW^T)^{-1}
= XW^T
\end{split}
$$

### Eigenfaces

Eigenfaces apply PCA to represent a facial image with latent factors. First we compute the mean image and the top k eigenvectors (Principal components)
 
<div class="imgcap">
<img src="/assets/ml/eign1.png" style="border:none;width:50%">
</div>

The image is encoded with the latent factors:
<div class="imgcap">
<img src="/assets/ml/face2.png" style="border:none;width:80%">
</div>

### Non-negative matrix factorization (NMF):

* W and Z are non-negative instead of orthogonality
* Promote sparsity
	* Avoiding postive & negative elements cancelling each other
	* Brain seem to use sparse representation 
	* Energy efficient
	* Increase the number of concepts that can memorize
* Learning the parts of objects

In NMF. the latent factors are more close to individual facial features.

<div class="imgcap">
<img src="/assets/ml/face4.png" style="border:none;width:60%">
</div>

Credit: Daniel D. Lee: Learning the parts of objects by non-negative matrix factorization

### Sparse Matrix Factorization

Here is the plot of J with an optimized $$w$$ smaller than 0.

<div class="imgcap">
<img src="/assets/ml/L02.png" style="border:none;width:60%">
</div>

With the constraint of $$w>0$$, the optimized cost is now at $$w=0$$ (promote sparsity).

<div class="imgcap">
<img src="/assets/ml/L03.png" style="border:none;width:60%">
</div>

We are going to use Gradient descent to train $$W$$ and reset the value to 0 if it is negative:

$$
w_i = max(0, w_i - \alpha \nabla_{w_i} J)
$$

#### L1-regularization 

We can also use L1 regularization to control sparsity:

$$
\begin{split}
J(W, Z) & =  \| ZW - X \|^2_F + \frac{\lambda_1}{2} \| W \|_1 + \frac{\lambda_2}{2} \| Z \|_1   \\
\end{split}
$$

Here is the visualization of latent factors with different techniques:
 
<div class="imgcap">
<img src="/assets/ml/face11.png" style="border:none;width:100%">
</div>

Credit: Julien Mairal etc... Online Learning for Matrix Factorization and Sparse Coding

### Regularized Matrix Factorization

L2 regularized PCA to replace normalization, orthogonality and sequential-fitting.

$$
\begin{split}
J(W, Z) & =  \| ZW - X \|^2_F + \frac{\lambda_1}{2} \| W \|^2_f + \frac{\lambda_2}{2} \| Z \|^2_f   \\
\end{split}
$$

### Latent Factor Model using logistic loss

We can use logistic loss for our cost function

$$
\begin{split}
J(W, Z) & =  \sum^n_{i=1} \sum^d_{j=1} \log( 1+ e^{ - x_{ij}w^T_jz_i} )
\end{split}
$$

### Robust PCA

We can use L1-norm as the cost function which make it less vulnerable to outliers:

$$
\begin{split}
J(W, Z) & =  \vert ZW - X \vert \\
\end{split}
$$

### Multi-dimensional scaling (MDS)

Multi-dimensional scaling helps us to visualize data in low dimension. PCA map input features from d dimensional feature space to k dimensional latent features. MDS focuses on creating a mapping that will also preserve the relative distance between data. If 2 points are close in the feature space, it should be close in the latent factor space such that the structure of the data is preserved when visualize in the low dimension. Our cost function penalize the model if the relative distances are different in both spaces. In MDS, we optimize $$z$$ directly with the following cost function using the gradient descent.

$$
\begin{split}
J(z) & = \sum^n_{i=1} \sum^n_{j=i+1} (\| z_i - z_j \| - \| x_i - x_j \|)^2
\end{split}
$$

<div class="imgcap">
<img src="/assets/ml/swiss.png" style="border:none;width:40%">
</div>
Source: wiki

The cost function above measure distance by Euclidean distance. In general, the cost function can be generalized with different measurement methods:

$$
J(z) = \sum^n_{i=1} \sum^n_{j=i+1} d3( d2(z_i - z_j), d1(x_i - x_j ))
$$

We can apply L1 norm which make the model less vulnerable to outliers:

$$
J(z) = \sum^n_{i=1} \sum^n_{j=i+1} d3(\vert z_i - z_j \vert,  \vert x_i - x_j \vert)
$$

### Sammon’s mapping

Even though we may want the model to maintain relative space, we may want dense area to have a larger scale than the sparse area. In Sammon's mapping, the relative distance is re-calibrate with the distance in the input feature space such that we can have a finer resolution on dense area. Sammon's mapping:

$$
J(z) = \sum^n_{i=1} \sum^n_{j=i+1} (\frac{ d2(z_i - z_j) - d1(x_i - x_j )}{d1(x_i - x_j )})^2
$$

### IsoMap

In some cases, we do not want to measure distance by Euclidean distance. For example, when we display the structure on the left below with PCA, all the color dots are meshed together even though the 3D image shows a clear spectrum of color on a S curve shape. We will introduce a new MDS method IsoMap which use geodesic to measure distance such that when we project the S curve structure into a 2D space, we can see how color is transit from red to blue. 

<div class="imgcap">
<img src="/assets/ml/sro1.png" style="border:none;width:80%">
</div>

Source: [http://ciera.northwestern.edu/Education/REU/2015/Thorsen/]

IsoMap uses geodesic rather than Euclidian space to measure distance

<div class="imgcap">
<img src="/assets/ml/sro2.png" style="border:none;width:60%">
</div>

Source: [https://bmcbioinformatics.biomedcentral.com/track/pdf/10.1186/1471-2105-13-S7-S3?]

To visualize the "swiss-roll" manifold in 2D, we measure the geodesic distance on the manifold. We represent datapoints as nodes in a weight graph with edges defined as the geodesic distance between 2 points. 

<div class="imgcap">
<img src="/assets/ml/scro4.png" style="border:none;width:60%">
</div>

* Find the neighbors of each point.
* Compute edge weights (distance between neighbors)
* Compute weighted shortest path between all points
* Run MDS using the computed distance above

### t-sne (t-Distributed Stochastic Neighbor Embedding)

t-sne is a MDS with special function for d1, d2 and d3.

The distance d1 between 2 datapoints $$i, j$$ in the input feature space is defined as:

$$
dist_{ij} \approx \frac{\text{Similarity of i and j measured by a Gaussian distribution}}{\text{Similarity of all points measured by a Gaussian distribution}}
$$ 


<div class="imgcap">
<img src="/assets/ml/dprob1.png" style="border:none;width:30%">
</div>

<div class="imgcap">
<img src="/assets/ml/dprob2.png" style="border:none;width:15%">
</div>

The distance d2 between 2 datapoints $$i, j$$ in the latent factor space is defined as:

<div class="imgcap">
<img src="/assets/ml/pro3.png" style="border:none;width:30%">
</div>

It is very similar to d1 with the exception that a Student-t distribution is used instead of the Gaussian distribution. This allows dissimilar objects to be modeled far apart in the map.

Finally, we use the KL divergence as d3.

$$
D_{KL}(p \vert \vert q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}
$$

Image features are extracted in the 4096-dimensional fc7 CNN layer and displayed in 2-D with t-sne. If we look in detail, the 2D display maintains the spatial relationship of the fc7 layer: images of the same type are displayed close to each others.

<div class="imgcap">
<img src="/assets/ml/tsne.jpg" style="border:none;width:50%">
</div>

<div class="imgcap">
<img src="/assets/ml/tsne2.png" style="border:none;width:50%">
</div>

Source: [http://cs.stanford.edu/people/karpathy/cnnembed/]


Note: the challenge in MDS methods is how to space the datapoints. 
* PCA focuses on preserving large distance which results in crowding in short distances.
* Sammon mapping use weighted cost function so large/small distances is more comparable.
* ISOMAP measures distances in geodesic instead of flat plains. This allow us to explore the manifold.
* T-SNE focus on dense area and have gap between groups.

<div class="imgcap">
<img src="/assets/ml/tsne3.png" style="border:none;width:70%">
</div>

Source [http://lvdmaaten.github.io/publications/papers/JMLR_2008.pdf]
