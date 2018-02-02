---
layout: post
comments: true
mathjax: true
priority: 990000
title: “Deep learning - Linear algebra.”
excerpt: “Deep learning - Linear algebra.”
date: 2017-01-05 12:00:00
---


### Basic

Transpose:

$$
(AB)^T = B^TA^T \\
(A + B)^T = A^T + B^T \\
$$

$$
\begin{split}
x^Ty &= y^Tx &\quad \text{for vector} \\
s^T &= s	&\quad \text{for scalar} \\
\end{split}
$$

> We use upper case for matrix $$M$$and lower case for vector $$v$$.

Distributive, associative & communicative:

$$
A(B + C) = AB + AC \\
A(BC) = (AB)C \\
AB \neq BA \\
$$

Element-wise product:

$$
A \odot B
$$

### Norms

Norms measures the size of vectors.

L1-norm (Manhattan distance):

$$
\begin{split}
L_1 & = \| x \|_1 =  \sum^d_{i=0} \vert x_i \vert  \\
\end{split}
$$

L2-norm (Euclidian distance): it is commonly used in deep learning and with notation simplified as $$ \| x\| $$. However, L2-norm may not penalize the near-zero parameters enough to push it to 0. Hence, L1-norm is preferable if the sparsity of the model's parameters is important.

$$
\begin{split}
L_2  & = \| x \|_2 = \| x \| =  \Big(\sum^d_{i=0} x_i^2\Big)^{\frac{1}{2}}  \\
L_2^2  & = x^Tx \\
\end{split}
$$

A **unit vector** is a vector with $$ \| x \| = 1$$.

Lp-norm

$$
\begin{split}
L_p  & = \| x \|_p =  \Big({\sum^d_{i=0} x_i^p}\Big)^{\frac{1}{p}}  \\
\end{split}
$$

$$\text{L}_\infty$$-norm (max norm)

$$
\begin{split}
L_\infty (x) &  =  max(\vert x_i \vert)  \\
\end{split}
$$

Frobenius norm: It measures the size of a matrix:

$$
L_F =  \Big(\sum_{i,j} A_{ij}^2\Big)^{\frac{1}{2}}  \\
$$

Sometimes, we count the number of non-zero element: add 1 for every non-zero elements.

$$
\begin{cases}
                        0 \quad \text{ if } x_i = 0 \\
                        1 \quad \text{otherwise}
\end{cases}
$$

### Diagonal matrix

A diagonal matrix is a matrix with all non-diagonal element being zero. We form a square diagonal matrix by moving vector elements into the diagonal position of the matrix.

$$
M = diag(v)
$$

Providing $$v$$ has no element with zero value, we replace each diagonal element with
$$
\frac{1}{v_{i}}
$$
to form its inverse $$M^{-1}$$.

Machine learning may approximate solutions with diagonal matrices because finding inverse or performing matrix multiplication is easy. Non-square diagonal matrix does not have an inverse matrix.

### Symmetric matrix

A symmetric matrix is a matrix with $$A_{ij} = A_{ji}$$. 

$$
A^T = A
$$

In machine learning, many equations in calculating elements between $$i \leftrightarrow j$$ are not directional ($$f(x,y) = f(y,x)$$). For example, the distance between 2 points is not directional. Hence, many matrices in machine learning is symmetrical. The inverse of a symmetric matrix is also symmetric. Any real $$ n \times n $$ symmetric matrices can be decomposed into $$n$$ eigenvalues and eigenvectors which is very desirable in matrix factorization. Symmetric matrix can easily decompose into orthogonal matrices: $$A=Q \Lambda Q^T$$ which its inverse equals to its transpose.

### Inverse matrix

$$
A A^{-1} = I
$$

Properties:

$$
(AB)^{-1} = B^{-1} A^{-1}\\
(A^T)^{-1} = (A^{-1})^T
$$

Solving linear equation with an inverse matrix:

$$
\begin{split}
Ax &= b \\
x & = A^{-1} b
\end{split}
$$

$$Ax$$ is equivalent to the multiplication of each column vectors $$A_{:, j}$$ with vector element $$x_i$$:

$$
\begin{split}
Ax = b \\
\big[ A_{:, 1},  A_{:, 2}, \cdots , A_{:,n} \big] x = b \\
A_{:, 1} x_1 +  A_{:, 2} x_2 + \cdots + A_{:,n} x_n = b\\
\sum_i x_i A_{:,i}  = b \\
\end{split}
$$

The **span** of a set of vectors is the set of points reachable by linear combination of those vectors. The column vectors of $$A$$ form a column space. A square matrix with linearly dependent columns is known as **singular**. If $$A$$ is singular, its inverse does not exist.

In machine learning, we rarely use inverse matrix to solve linear equation. $$A^{-1}$$ is often not numerical stable: small input errors amplifies output errors. In machine learning, $$A$$ is often a sparse matrix, however, the inverse is not which will take too much memory.

### Orthogonal matrix

A set of vectors are **orthonormal** if and only if the inner product of any two vectors are zero. An **orthogonal matrix** is a _square_ matrix whose rows (columns) are mutually orthonormal. i.e. no dot products of 2 row vectors (column vectors) are 0. 

$$
A^T A = A A^T = I
$$

For an orthogonal matrix, there is one _important_ property. Its inverse is the transpose which is very easy to compute. $$ A A^T = I \Rightarrow A^{-1} A A^T = A^{-1} I \Rightarrow A^T = A^{-1} $$.

$$
A^T = A^{-1}
$$

Also orthogonal matrices $$Q$$ does not amplify errors which is very desirable.

$$
\| Qx \|^2_2 = (Qx)^T Qx = x^TQ^T Qx = x^T x = \| x \|^2_2
$$

The size of the multiplication result $$\|Qx\|$$ has the same size as $$\|x\|$$. If we multiple $$x$$ with orthogonal matrices, the errors present in $$x$$ will not be amplified by the multiplication. i.e. it is more numeric stable.

Decompose matrix into smaller components helps us to solve some problems faster with better numeric stability. Here is the pseudo code to use SVD to decompose matrix into orthogonal matrices and solve the linear equation with the result.

$$
\begin{split}
& Ax =b \\
& (U,D,V)  \leftarrow svd(A) \\
& A^{+}  = V D^{+}U^T \\
& x  = A^{+} y \\
\end{split}
$$

where $$D^{+} $$ takes the reciprocal $$\frac{1}{x_i}$$ of the non-zero elements of $$D$$ and then transpose. 

### Quadric form equation in Matrix

A quadric form equation contains terms $$x^2, y^2$$ and $$xy$$. 

$$
a x^2 + 2 b c x y + c y^2
$$

The matrix form of a quadratic equation:

$$
\begin{bmatrix}
    x  &  y  \\
\end{bmatrix}
\begin{bmatrix}
    a  &  b  \\
    b &  a  \\
\end{bmatrix}
\begin{bmatrix}
    x   \\
    y   \\
\end{bmatrix}
$$

With 3 variables:

$$
\begin{bmatrix}
    x  &  y  & z\\
\end{bmatrix}
\begin{bmatrix}
    a  &  b  & c\\
    b &  d & e  \\
    c &  e & f  \\
\end{bmatrix}
\begin{bmatrix}
    x   \\
    y   \\
    z   \\
\end{bmatrix}
$$

### Eigen vector & eigen value

Scalar $$\lambda$$ and vector $$v$$ are the eigenvalue and eigenvector of $$A$$ respectively if:

$$
Av = \lambda v
$$


Properties:

* A $$n \times n$$ matrix has at most $$n$$ eigenvalues and eigenvectors.
* A matrix is singular iff any eigenvalues are 0. 

Find the eigenvalues and eigenvectors for $$A$$.

$$
\begin{split}
Av & = \lambda v \\
(A - \lambda I) v & = 0 \\
\implies det(A - \lambda I) & = 0 \\
\end{split}
$$

#### Finding the eigenvalues

Find the eigenvalues of:

$$
A =
\begin{bmatrix}
    1 & -3 & 3 \\
    3 & -5 & 3 \\
    6 & -6 & 4  \\
\end{bmatrix}
$$

To solve:

$$
\begin{split}
& det(A - \lambda I) = 0 \\
\\
& \begin{vmatrix}
1 − λ & −3  & 3 \\
3 & −5 − λ & 3 \\
6 & −6 & 4 − λ \\
\end{vmatrix} = 0\\
\\
& (1 -\lambda) 
\begin{vmatrix}
    −5 − λ  & 3\\
    -6 & 4 − λ  \\
\end{vmatrix}
- (-3)
\begin{vmatrix}
   3 & 3 \\
   6 & 4 − λ \\
\end{vmatrix}
+ 3
\begin{vmatrix}
 3 & −5 − λ \\
 6 & −6 \\
\end{vmatrix} = 0 \\
\\
& 16 + 12λ − λ^3 = 0 \\
\\
& λ^3  - 12 λ - 16 = 0 \\
\end{split}
$$

The possible factors for 16 are 1, 2, 4, 8, 16.

when $$λ=4$$, $$λ^3  - 12 λ - 16 = 0$$

So 
$$λ^3  - 12 λ - 16  = (λ − 4)(λ^2 + 4λ + 4) = 0$$

By solving the root, the eigenvalues are -4, 2, -2.

#### Finding the eigenvectors

$$
A − 4I =
\begin{vmatrix}
−3 &−3 &3 \\
3 & −9 & 3 \\
6 & −6 & 0 \\
\end{vmatrix}
$$

Doing row reduction to solve the linear equation $$ (A - \lambda I) v = 0 $$

$$
A − 4I =
\begin{vmatrix}
−3 &−3 &3 \\
3 & −9 & 3 \\
6 & −6 & 0 \\
\end{vmatrix}
$$

Appending 0:

$$
\begin{vmatrix}
−3 &−3 &3 & 0\\
3 & −9 & 3 & 0\\
6 & −6 & 0 & 0 \\
\end{vmatrix}
$$

Perform $$R_1 = - \frac{1}{3} R_1$$

$$
\begin{vmatrix}
1 & 1 & -1 & 0\\
3 & −9 & 3 & 0\\
6 & −6 & 0 & 0 \\
\end{vmatrix}
$$

Perform row subtraction/multiplication:

$$
\begin{vmatrix}
1 & 1 & −1 & 0 \\
0 &−12& 6 &0 \\
0 &−12& 6 &0 \\
\end{vmatrix}
$$

After many more reductions:

$$
\begin{vmatrix}
1 & 0 &−1/2 & 0 \\
0 & 1 &−1/2 & 0 \\
0 & 0 & 0 & 0 \\
\end{vmatrix}
$$

$$
x_1 − \frac{1}{2} x_3 = 0 \\
x_2 − \frac{1}{2} x_3 = 0 \\
$$

So for $$\lambda=4$$, the eigenvector is:

$$
\begin{bmatrix}
1/2 \\
1/2 \\
1 \\
\end{bmatrix}
$$

### Eigendecomposition

Matrix decomposition decompose a matrix into special type of matrices for easy manipulation in solving problems like linear equations. But eigendecomposition is only defined for square matrices. 

Say $$A$$ has $$n$$ eigenvalues $$\lambda_1, \lambda_2, \cdots, \lambda_n$$. We concatenate all values to form a column vector $$\lambda = [\lambda_1, \lambda_2, \cdots, \lambda_n]^T$$. $$A$$ also has $$n$$ eigenvectors $$v_1, v_2, \cdots, v_n$$. We compose a matrix $$V$$ with $$v_i$$ as the column $$i$$ of the matrix. $$V= [v^{(1)}, . . . ,v^{(n)}]$$. The eigen decomposition of A is:

$$
A = V diag(λ)V^{−1}
$$

Not every $$A$$ has eigendecomposition. But in deep learning, we often due with real symmetric metrics. **Real symmetric metrics** are **eigendecomposable** and the equation can be further simplify to:

$$
A = Q \Lambda Q^{T}
$$

which $$Q$$ is an orthogonal matrix composed of eigenvectors of $$A$$. $$\Lambda$$ is a diagonal matrix. The value of diagonal element $$ \Lambda_{ii}$$ is the eigenvalue of the corresponding eigenvector in column $$i$$ of $$Q$$. 
We do not specify the order of eigenvectors. Therefore different order of $$v^{(i)}$$ creates different $$V$$. By convention, we re-arrange the column order $$v^{(i)}$$ in $$v$$ by the descending sorting order of its eigenvalue $$\lambda_i$$. It helps us to decompose $$A$$ in a more deterministic way.

In eigendecomposition, we decompose the matrix $$A$$ into different eigenvectors $$v^{(i)}$$ scaled by the eigenvalue $$\lambda_i$$. Therefore, for any vectors pointing at the same direction of eigenvector $$v^{(i)}$$, $$Ax$$ scales $$x$$ by the corresponding eigenvalue $$\lambda_i$$. 

For a quadratic equation in the form of $$f(x) = x^TAx$$, if $$x$$ is an unit vector equal to $$v^{(i)}$$, $$f(x)$$ will be equal to the eigenvalues $$\lambda_i$$. Therefore, the max and min of $$f(x)$$ is the max and min of the eigenvalues.


### Singular value decomposition (SVD)

SVD factorizes a matrix into singular vectors and singular values. Every real matrix has a SVD but not true for eigendecomposition.

$$
A = U D V^T
$$

* A is a m×n matrix. (Does not need to be a square matrix like eigendecomposition.)
* Left-singular vector: U is a m×m orthogonal matrix (the eigenvectors of $$A A^T$$)
* Reft-singular vector: V is a n×n orthogonal matrix (the eigenvectors of $$A^T A$$)
* Singular values: D is a m×n diagonal matrix (square roots of the eigenvalues of $$A A^T$$ and $$A^T A$$ )

SVD is a powerful but expensive matrix factorization method. In numerical linear algebra, many problems can be solved to represent $$A$$ in this form.

### Positive definite or negative definite matrix

If all eigenvalues of $$A$$ are:

* positive: the matrix is positive definite.
* positive or zero: positive semi-deﬁnite.
* negative: the matrix is negative definite.

If a matrix is positive semi-deﬁnite,  $$ x^TAx \geq 0$$. If a matrix is positive definite and $$ x^TAx = 0$$, it  implies $$x = 0 $$. 

Positive definite or negative definite helps us to solve optimization problem. Quadratic equation $$x^TAx$$ with positive definite matrices $$A$$ are always positive for non-zero $$x$$ and the function is convex. i.e. it guarantees the existences of the global minimum. This allows us to use Hessian matrix to solve the optimization problem. Similar arguments hold true for negative definite.

### Trace

Trace is the sum of all diagonal elements

$$
Tr(A) = \sum_{i} A_{ii}
$$

We can rewrite some operations using Trace to get rid of the summation (like the summation in the Frobenius norm):

$$
\| A \|_F= \sqrt{Tr(AA^T)}
$$

Other properties:

$$
\begin{split}
Tr(A) &= \sum_i \lambda_i \quad \text{sum of eigenvalues.}\\
Tr(A) & = Tr(A^T) \\
Tr(AB) & = Tr(BA) \\
Tr(ABC) & = Tr(CAB) = Tr(BCA)   \quad \text{move first element to the end or vice versa.} \\
\end{split}
$$

### Derivative of matrix

For vector $$x$$:

$$
\begin{split}
\frac{\partial Ax}{\partial x} & = A \\
\frac{\partial Ax}{\partial z} & = A \frac{\partial x}{\partial z}\\
\\
\frac{\partial x^Ta}{\partial x} &  = a \\
\frac{\partial a^T x}{\partial x} & = a \\
\\
\frac{\partial y^TAx}{\partial x} &  = y^TA \\
\frac{\partial y^TAx}{\partial y} & = x^TA^T \\
\\
\frac{\partial x^T x}{\partial x} & = 2 x^T \\
\\
\frac{\partial x^T A x}{\partial x} & =  x^T(A + A^T) \\
\frac{\partial x^TAx}{\partial x} & = 2 x^T A \quad \text{if } A \text{ is symmetric.}\\
\\
\end{split}
$$

For matrix $$X$$:

$$
\begin{split}
\frac{\partial a^TXb}{\partial X} & =  a b^T \\
\\
\frac{\partial a^TX^Tb}{\partial X} & =  b a^T \\
\\
\frac{\partial a^TXa}{\partial X} & =  \frac{\partial a^TX^Ta}{\partial X}  = a a^T \\
\\
\frac{\partial b^TX^TXc}{\partial X} & =  X(b c^T + cb^T) \\
\\
\frac{\partial X^T B X}{\partial X} & =  (B + B^T)X \\
\\
\frac{\partial w^T X^Tx w}{\partial w} & =  ((x^Tx) + (x^Tx)^T)w = (x^Tx + x^Tx)w = 2x^Tx w \\
\\
\end{split}
$$

Example, Optimize mean square error of a linear regression.

Given: 
$$N$$ is the size of the dataset, 
$$\hat{y} , y \in \mathbb{R}^N $$,
$$ w \in \mathbb{R}^k  $$,
$$ X \in \mathbb{R}^{N \times k}  $$.

$$
\begin{split}
J & = \| \hat{y} - y \|^2\\
J & = \| X w - y \|^2  \quad \text{Given} \hat{y} = X w \\  
\nabla_w J  & = \nabla_w \| X w - y \|^2 \\
& = \nabla_w (X w - y )^T (X w - y ) \\
& = \nabla_w (w^T X^T - y^T) (X w - y ) \\
& = \nabla_w (w^T X^T X w - w^T X^T y  - y^TX w + y^T y ) \\
& = \nabla_w (w^T X^T X w - w^T X^T y  - (w^T X^T y)^T + y^T y ) \\
& = \nabla_w (w^T X^T X w - 2 w^T X^T y  + y^T y ) \quad \text{since } w^T X^T y \text{ is scalar which equals to the transpose.}\\
& = \nabla_w (w^T X^T X w - 2 w^T X^T y) \quad \text{remove terms not relative to } w. \\
& = 2 X^T X - 2 X^T y \\
\nabla_w J  & = 2 X^T X w - 2 X^T y = 0 \quad \text{set to 0 for the critical point.}\\
w & = (X^T X)^{-1} X^T y \\
\end{split}
$$

### Principal Component Analysis (PCA)

PCA encodes a n-dimensional input space into an m-dimensional output space with $$n > m$$. We want to minimize the amount of information lost during the reduction and minimize the difference if it is  reconstructed. 

Let's $$f$$ and $$g$$ be the encoder and the decoder:

$$
\begin{split}
c & = f(x) \quad \text{which } x \in \mathbb{R}^{n}\\
x^{'} & = g(c) \\
\end{split}
$$

We apply a linear transformation to decode $$c$$. We constraint $$D$$ to be a matrix $$\mathbb{R}^{n \times m}$$ composed of columns that is orthogonal to each other with unit norm. (i.e. $$D^TD = I_l$$.)

$$
g(c) = Dc
$$

PCA uses L2-norm to minimize the reconstruction error. 

$$
c^∗= \arg \min_c \|x − g(c)\|_2^2
$$

Compute L2-norm:

$$
\begin{split}
(x − g(c))^T(x − g(c)) &= (x^T − g(c)^T)(x − g(c)) \\
&= x^Tx -x^Tg(c) - g(c)^Tx - g(c)^Tg(c) \\ 
&= x^Tx - 2x^Tg(c) - g(c)^Tg(c) \quad \text{since }g(c)^Tx \text{ is a scalar and } s^T=s \\ 
\end{split}
$$

Optimize $$c$$:

$$
\begin{split}
c^∗ &= \arg \min_c x^Tx - 2x^Tg(c) - g(c)^Tg(c) \\
&= \arg \min_c - 2x^Tg(c) - g(c)^Tg(c) \quad \text{the optimal point remains the same without the constant. terms}\\
&= \arg \min_c - 2x^TDc - c^TD^TDc \\
&= \arg \min_c - 2x^TDc - c^TIc \quad \text{since }  D \text{ is orthongonal: } D^TD = I.\\
&= \arg \min_c - 2x^TDc - c^Tc \\
\end{split}
$$

To optimize it:

$$
\begin{split}
\nabla_c (- 2x^TDc - c^Tc) & = 0 \\
− 2D^Tx + 2c &= 0 \\
c &= D^Tx \\
\end{split}
$$

So, the optimize encode and decode scheme is 

$$
c = f(x) = D^Tx \\
x^{'} = g(c) = Dx = D  D^Tx \\
r(x) = D  D^Tx  \quad \text{reconstuction.}\\
$$

To find the optimal transformation $$D^{*}$$, we want to optimize $$D$$ over all datapoint $$x^{(i)}$$:

$$
D^{*} = \arg \min_D \sum_{ij} \| \big( x^{(i)}_j - r(x^{(i)})_j \big)  \|^2 \quad \text{which } D^TD = I_l.
$$ 

Let's consider $$l=1$$, so D is just a vector $$d$$.

$$
\begin{split}
d^{*}  & = \arg \min_D \sum_{ij} \| \big( x^{(i)} - d d^T x^{(i)} \big)  \|^2_2 \quad \text{which } \| d \| = 1.\\
d^{*}  & = \arg \min_D \sum_{ij} \| \big( x^{(i)} - d^T x^{(i)} d \big)  \|^2_2 \quad \text{move the scalar }d^T x^{(i)} \text{ to the front.}\\
d^{*}  & = \arg \min_D \sum_{ij} \|  \big( x^{(i)} - x^{(i)T} d d \big) \|^2_2 \quad \text{transpose the scalar } d^T x^{(i)}.\\
d^{*}  & = \arg \min_D \| \big( X - X d d^T \big)  \|^2_F \quad \text{which } \| d \| = 1. \\
\end{split}
$$ 

where $$X$$ contains all datapoints.
Xdd
$$
\begin{split}
D^∗ &= \arg \min_D \|X - Xdd^T\|^2_F  \\
&= \arg \min_D Tr((X - Xdd^T)^T(X - Xdd^T))  \\
&= \arg \min_D Tr( X^TX −X^TXdd^T− dd^TX^TX + dd^TX^TXdd^T )\\
&= \arg \min_D Tr( −X^TXdd^T− dd^TX^TX + dd^TX^TXdd^T ) \quad \text{ take away terms not depend on D} \\
&= \arg \min_D -Tr(X^TXdd^T) − Tr(dd^TX^TX) + Tr(dd^TX^TXdd^T )  \\
&= \arg \min_D -2Tr(X^TXdd^T)  + Tr(dd^TX^TXdd^T )  \\
&= \arg \min_D -2Tr(X^TXdd^T)  + Tr(X^TXdd^Tdd^T )  \quad \text{cycle the order in Tr.} \\
&= \arg \min_D -2Tr(X^TXdd^T)  + Tr(X^TXdd^T )  \quad \text{apply} \|d^Td\|=1 \\
&= \arg \min_D -Tr(X^TXdd^T) \\
&= \arg \max_D Tr(X^TXdd^T) \\
&= \arg \max_D Tr(d^TX^TXd) \quad \text{subject to} \|d^Td\|=1 \\
\end{split}
$$

This equation can be solved using eigendecomposition. Optimal $$d$$ is the eigenvector of $$X^TX$$ that has the largest eigenvalue. For $$l > 1$$, the matrix D is given by the eigenvectors corresponding to the $$l$$ largest eigenvalues.

### Moore-Penrose Pseudoinverse

For a linear equation:

$$
Ax = y
$$

A^{+} is a pseudo inverse of matrix A. We do not called it $$A^{-1}$$ because inverse matrix is only defined for a square matrix.

$$
x = A^{+} y
$$

$$A^{+}$$ is computed as (if exist):

$$
A^{+} = V D^{+}U^T
$$

which $$U, D \text{ and } V $$ are the SVD of $$A$$. The pseudoinverse $$D^{+}$$ of a diagonal matrix D is computed by taking the reciprocal $$\frac{1}{x}$$ of all non-zero elements then taking the transpose of the resulting matrix.

### Determinant

The determinant of the matrix $$A$$ is the product of all eigenvalues. If the absolute value is greater than 1, $$Ax$$ expands the output space. If it is between 0 and 1, it shrinks the space.


