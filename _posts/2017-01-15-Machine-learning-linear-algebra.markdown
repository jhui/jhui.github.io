---
layout: post
comments: true
mathjax: true
priority: 422000
title: “Machine learning - Linear algebra.”
excerpt: “Machine learning - Linear algebra.”
date: 2017-01-15 12:00:00
---

### Operation

#### Element-wise product:

$$
A \odot B
$$

#### Distributive and associative:

$$
A(B + C) = AB + AC
$$

$$
A(BC) = (AB)C
$$

#### Transpose

$$
(AB)^T = B^TA^T
$$

#### Commutative:

For matrix:

$$
AB \neq BA
$$

Dot product for 2 vectors is communicative:

$$
x^Ty = y^Tx
$$

Proof:

$$
\text{For scalar } s=s^T  
$$

$$
\begin{split}
x^Ty &= (x^Ty)^T \quad \\
 &= y^Tx \\
\end{split}
$$

#### Matrix inverse 

$$
A A^{-1} = I
$$

#### Singular matrix

A matrix without an inverse. i.e. the determinant is 0. One or more of its rows (columns) is a linear combination of some other rows (columns).

#### Linear equation

$$
\begin{split}
A x & = b \\
x &= A^{-1} b \\
\end{split}
$$

In practice, we rarely solve $$x$$ by finding $$A^{-1}$$. We lost precision in computing $$A^{-1}$$. In machine learning, $$A$$ can be sparse but $$A^{-1}$$ is dense which requires too much memory to perform the computation. 

#### Symmetry matrix

$$
A^T = A
$$

#### Orthogonal matrix

An orthogonal matrix is a _square_ matrix whose rows (columns) are mutually orthonormal. i.e. no dot products of 2 row vectors (column vectors) are 0.

$$
A^T A = A A^T = I
$$

For an orthogonal matrix, there is one important property:

$$
A^T = A^{-1}
$$

Orthogonal matrices are particular interesting because the inverse is easy to find.

Also orthogonal matrices $$Q$$ does not amplify errors which is very desirable:

$$
\| Qx \|^2_2 = (Qx)^T Qx = x^TQ^T Qx = x^T x = \| x \|^2_2
$$

So if we multiple the input with orthogonal matrices, the errors present in $$x$$ will not be amplified by the multiplication. We can decompose matrices into orthogonal matrices (like SVD) in solving linear algebra problems. Also, symmetry matrix can be decomposed into orthogonal matrices: $$A=Q \Lambda Q^T$$.

#### Quadric form

Quadric form equation contrains terms of $$x^2, y^2$$ and $$xy$$. 

$$
a x^2 + 2 b c x y + c y^2
$$

In matrix form:

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

$$
Av = \lambda v
$$

which $$\lambda$$ is a scalar and $$v$$ is a vector.

Find the eigenvalues and eigenvectors for

$$
\begin{bmatrix}
    1 & -3 & 3 \\
    3 & -5 & 3 \\
    6 & -6 & 4  \\
\end{bmatrix}
$$

$$
\begin{split}
Av & = \lambda v \\
(A - \lambda I) v & = 0 \\
\implies & det(A - \lambda I) = 0
\end{split}
$$

* A matrix is singular iff any eigenvalues are 0. 
* To optimize quadratic form equations, $$f(x) = x^TAx$$ given $$ \| x\| = 1$$
	* If x is the eigenvector of $$A$$, $$f(x)$$ equals to the corresponding eigenvalues
	* The max (min) of $$f(x)$$ is the max (min) of the eigenvalues

#### Finding the eigenvalues

$$
\begin{split}
det(A - \lambda I) & = 0 \\
\begin{vmatrix}
1 − λ & −3  & 3 \\
3 & −5 − λ & 3 \\
6 & −6 & 4 − λ \\
\end{vmatrix} &= 0\\
(1 -\lambda) 
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
\end{vmatrix} & = 0 \\
 16 + 12λ − λ^3 &= 0 \\
  λ^3  - 12 λ - 16 &= 0 \\
\end{split}
$$

Consider possible factor for 16: 1, 2, 4, 8, 16

when $$λ=4$$, $$λ^3  - 12 λ - 16 = 0$$

So 
$$λ^3  - 12 λ - 16  = (λ − 4)(λ^2 + 4λ + 4) = 0$$

The other eigenvalues are -2, -2.

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


#### Eigendecomposition

For a matrix $$V$$ with one eigenvector per column  $$V= [v^{(1)}, . . . ,v^{(n)}]$$ and a vector $$ \lambda= [\lambda_1, . . . ,\lambda_n]^T$$. The eigen decomposition of A is 

The eigendecomposition of $$A$$ is

$$
A = V diag(λ)V^{−1}
$$

If $$A$$ is real and **symmetry**,

$$
A = Q \Lambda Q^{T}
$$

which $$Q$$ is an orthogonal matrix composed of eigenvectors of $$A$$. The eigenvalue $$ \Lambda_{ii}$$ is associated with the eigenvector in column $$ Q_{:,i} $$. This is important because we often deal with symmetrical matrices.

Eigendecomposition requires $$A$$ to be a square matrix. Not every squared matrix can have eigendecomposition.

### Poor Conditioning

$$
A y = x 
$$

Poorly conditioned matrices amplify pre-existing errors. Consider the ratio:

$$
\max_{i, j} \vert \frac{\lambda_i}{\lambda_j} \vert
$$

which $$\lambda_i, \lambda_j$$ is the largest and smallest eigenvalue of $$A$$.

If it is high, its inversion will multiple the errors in $$x$$.

### Jacobian matrix

$$
J = \begin{pmatrix}
    \frac{\partial f_1}{ \partial x_1} & \frac{\partial f_1}{ \partial x_2} & \dots  & \frac{\partial f_1}{ \partial x_n} \\
    \frac{\partial f_2}{ \partial x_1} & \frac{\partial f_2}{ \partial x_2} & \dots  & \frac{\partial f_2}{ \partial x_n} \\
    \vdots & \vdots & \ddots & \vdots \\
    \frac{\partial f_m}{ \partial x_1} & \frac{\partial f_m}{ \partial x_2} & \dots  & \frac{\partial f_m}{ \partial x_n} \\
\end{pmatrix}
$$

### Hessian matrix

$$
H = \nabla^2 f = \begin{pmatrix}
    \frac{\partial^2 f_1}{ \partial x_1^2} &  \frac{\partial^2 f_1}{ \partial x_1 \partial x_2} & \dots  & \frac{\partial^2 f_1}{ \partial x_1 \partial x_n} \\
    \frac{\partial^2 f_2}{ \partial x_1^2} &  \frac{\partial^2 f_2}{ \partial x_1 \partial x_2} & \dots  & \frac{\partial^2 f_2}{ \partial x_1 \partial x_n} \\
    \vdots & \vdots & \ddots & \vdots \\
    \frac{\partial^2 f_m}{ \partial x_1^2} &  \frac{\partial^2 f_m}{ \partial x_1 \partial x_2} & \dots  & \frac{\partial^2 f_m}{ \partial x_1 \partial x_n} \\
\end{pmatrix}
$$

### Positive definite/negative definite

If all eigenvalues of $$A$$ are:

* positive: the matrix is positive definite
* positive or zero: positive semi-deﬁnite
* negative: the matrix is negative definite

Properties of positive definite:

* $$ x^TAx \geq 0$$ if positive semi-deﬁnite and
* $$ x^TAx = 0 \implies x = 0 $$ for positive definite

Positive definite or negative definite helps us to solve optimization problem. Quadratic forms on positive definite matrices
$$x^TAx$$ are always positive for non-zero $$x$$ and are convex. It guarantees the existences of global minima. It allows us to use Hessian matrix to optimize multivariate functions. Similar arguments hold true for negative definite.

### Taylor series in 2nd order

$$
\begin{split}
f(x) & ≈ f(x^0) + (x − x^0)^Tg +\frac{1}{2}(x − x^0)^TH(x − x^0) \\
f(x^0− \epsilon g) &≈ f(x^0) − \epsilon g^Tg +\frac{1}{2} \epsilon^2g^THg \\
\end{split}
$$

which $$H$$ is the Hessian matrix.

If $$g^THg$$ is negative or 0, $$f(x)$$ decreases as $$\epsilon$$ increases. However, we cannot drop $$\epsilon$$ too far as the accuracy of the Taylor series drops as $$\epsilon$$ increases. If $$g^THg$$ is positive, the optimal step will be

$$
\epsilon^{∗}= \frac{g^Tg}{g^THg}
$$

### Newton method

The critical point of 2nd order taylor equation is

$$
x^{∗}= x^0 − H(f)(x^0))^{−1} ∇_xf(x^0)
$$



### Singular value decomposition (SVD)

SVD factorizes a matrix into singular vectors and singular values. Every real matrix has a SVD but not true for eigendecomposition. (E.g. Eigendecomposition requires a square matrix.)

$$
A = U D V^T
$$

* A is a m×n matrix
* Left-singular vector: U is m×m orthogonal matrix (the eigenvectors of $$A A^T$$)
* Singular values: D is m×n diagonal matrix (square roots of the eigenvalues of $$A A^T$$ and $$A^T A$$ )
* Reft-singular vector: V is n×n orthogonal matrix (the eigenvectors of $$A^T A$$)

SVD is a powerful but expensive matrix factorization method. In numerical linear algebra, many problems can be solved to represent $$A$$ in this form.

### Solving linear equation with SVD

$$
\begin{split}
Ax = y \\
x = A^{+} y \\
\end{split}
$$

which $$A^{+}$$ is

$$
A^{+}= V D^{+}U^T
$$

$$V, U$$ is from SVD and $$D^{+}$$ is the reciprocal of the non-zeroelements. Then take the transpose.

### Norms
L0-norm (0 if x is 0, otherwise it is 1)

$$
L_0 = \begin{cases}
                        0 \quad \text{ if } x_i = 0 \\
                        1 \quad \text{otherwise}
                    \end{cases}
$$

L1-norm (Manhattan distance)

$$
\begin{split}
L_1 & = \| x \|_1 =  \sum^d_{i=0} \vert x_i \vert  \\
\| x - y \|_1 & =  \sum^d_{i=0} \vert x_i - y_i \vert  \\
\end{split}
$$

L2-norm (Euclidian distance)

$$
\begin{split}
L_2  & = \| x \|_2 = \| x \| =  \Big(\sum^d_{i=0} x_i^2\Big)^{\frac{1}{2}}  \\
L_2^2  & =  \sum^d_{i=0} x_i^2 = x^Tx  \\
\| x - y \| & =  \Big(\sum^d_{i=0} (x_i - y_i)^2\Big)^{\frac{1}{2}}  \\
\end{split}
$$

Lp-norm

$$
\begin{split}
L_p  & = \| x \|_p =  \Big({\sum^d_{i=0} x_i^p}\Big)^{\frac{1}{p}}  \\
\end{split}
$$

$$\text{L}_\infty$$-norm

$$
\begin{split}
L_\infty (x) &  =  max(\vert x_i \vert)  \\
\end{split}
$$

Frobenius norm

It measures the size of a matrix.

$$
L_F =  \Big(\sum_{i,j} A_{ij}^2\Big)^{\frac{1}{2}}  \\
$$

Other properties:

$$
Tr(A) = Tr(A^T) \\
Tr(AB) = Tr(BA) \\
Tr(ABC) = Tr(CAB) = Tr(BCA) \\
$$

### Determinant

The determinant is the product of all eigenvalues. If the absolute value is greater than 1, it expand the space. If it is between 0 and 1, it shrinks the space.

### Trace

Trace is the sum of all diagonal elements

$$
Tr(A) = \sum_{i} A_{ii}
$$

We can rewrite some operations using Trace to get rid of the summation:

$$
\| A \|_F= \sqrt{Tr(AA^T)}
$$



### PCA

$$
c = f(x) \\
x^{'} = g(c) \\
$$

Using a matrix for transformation:
$$
g(c) = Dc
$$

PCA constrains the columns of $$D$$ to be orthogonal to each other with magnitude 1.

PCA minimize the 

$$
c^∗= \arg \min_c \|x − g(c)\|_2^2
$$

$$
\begin{split}
(x − g(c))^T(x − g(c)) &= (x^T − g(c)^T)(x − g(c)) \\
&= x^Tx -x^Tg(c) - g(c)^Tx - g(c)^Tg(c) \\
&= x^Tx - 2x^Tg(c) - g(c)^Tg(c) \\
\end{split}
$$

$$
\begin{split}
c^∗ &= \arg \min_c x^Tx - 2x^Tg(c) - g(c)^Tg(c) \\
&= \arg \min_c - 2x^Tg(c) - g(c)^Tg(c) \\
&= \arg \min_c - 2x^TDc - c^TD^TDc \\
&= \arg \min_c - 2x^TDc - c^TIc \quad \text{apply the constraint of D by PCA}\\
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
c = D^Tx \\
x^{'} = Dc
x^{'} = D D^Tx
$$

Let's assume the constraint $$D D^T=I $$, and since $$DD^Tx^i_j$$ is scalar, we can just take a transpose to $$XDD^T$$ and we assume $$c$$ is 1-D.

$$
\begin{split}
D^∗ &= \arg \min_D \sum_{ij} \|x^i_j - DD^Tx^i_j)\|^2_2  \\
&= \arg \min_D \|X - XDD^T\|^2_2  \\
&= \arg \min_D Tr((X - XDD^T)^T(X - XDD^T))  \\
&= \arg \min_D Tr( X^TX −X^TXDD^T− DD^TX^TX + DD^TX^TXDD^T )\\
&= \arg \min_D Tr( −X^TXDD^T− DD^TX^TX + DD^TX^TXDD^T ) \quad \text{ take away terms not depend on D}\\
&= \arg \min_D Tr( −2X^TXDD^T  + DD^TX^TXDD^T ) \\
&= \arg \min_D Tr( −2X^TXDD^T  + X^TXDD^TDD^T ) \\
&= \arg \min_D Tr( −2X^TXDD^T  + X^TXDD^T ) \quad \text{ apply } D D^T=I \\
&= \arg \min_D -Tr( X^TXDD^T ) \\
&= \arg \max_D Tr( X^TXDD^T ) \\
&= \arg \max_D Tr( D^TX^TXD ) \\
\end{split}
$$

By induction, we can expand $$c$$ to higher dimension.

Optimal $$D$$ is given by the eigenvector of $$X^TX$$ corresponding to the largest eigenvalue.

### Broadcasting

The vector $$b$$ is added to each row of $$C$$. The implicit replication of elements for an operation with a higher dimensional tensor is called broadcasting.

$$
C_{ij}=A_{ij}+b_j
$$

### Karush–Kuhn–Tucker

> This section is in-complete

Optimize

$$
\begin{split}
\min_x \hspace{3 mm} & f(x) \\
\text{subject to  }& h_i(x) ≤ 0, i = 1, . . . m \\
&l_j(x) = 0, j = 1, . . . r \\
\end{split}
$$

Lagrangian:

$$
L(x, u, v) = f(x) +\sum^m_{i=1} u_ih_i(x) +\sum^r_{j=1} v_j l_j (x)
$$

which $$ {u_i, v_j}$$ are scalars.

### Reference
The Matrix Cookbook by Kaare Petersen, Michael Pedersen.
Linear Algebra by Georgi Shilov.

	




  