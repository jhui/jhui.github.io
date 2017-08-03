---
layout: post
comments: true
mathjax: true
priority: 130000
title: “Machine learning - Recommendation”
excerpt: “Machine learning - Recommendation”
date: 2017-01-15 12:00:00
---

### Association rules

**Support**:

$$
P(S) = P(S1, S2, ... S_k)
$$

**Confidence**: (How offen $$T$$ happens when $$S$$ happens)

$$
P(T \vert S)
$$

In association, we are looking for high support and high confidence:

$$
\begin{split}
S & \implies T \\
P(S) & \geq s \\
P(T \vert S) & \geq c \\
\end{split}
$$

For computation reason, we will restrict the number of variables to consider in S and T. ($$ \vert S \vert $$ and $$ \vert T \vert $$)

####  Priori algorithm 

Support Set Pruning: To determine the probability, we will prune the tree to eliminate combination that is not promising:

<div class="imgcap">
<img src="/assets/ml/prun.png" style="border:none;width:70%">
</div>

* Generate list of all sets ‘S’ with k = 1.
* Prune candidates where $$ p(S) < s $$
* Go to k = k + 1 and generate the list again without the pruned items

#### Generating rules

If $$S = {A,B,C}$$, candidate rules are 

$$
\begin{split}
& A \implies B, C \quad B \implies A, C, \quad C \implies A, B, \quad \\
& A, B \implies C, \quad A, C \implies B, \quad B, C \implies A.
\end{split}
$$

Once again, we use pruning to reduce the number of evaluation.

<div class="imgcap">
<img src="/assets/ml/prun2.png" style="border:none;width:70%">
</div>

Note: 

* With large amount of data, it is possible that we generate a rule with high probability but yet $$A$$ and $$B$$ has no special co-relationship. 
* The rule $$ A \implies B $$ can be misleading even $$P(B \vert A) $$ has high confidence when $$P(B \vert A) < P(B) $$. This happens often when $$P(B) $$ is very high which $$P(B)$$ is more likely than P(B \vert A). So $$A$$ makes $$B$$ less likely.
 
One alternative to confidence is lift:

$$
Lift(S \implies T) = \frac{P(T \vert S)}{P(T)}
$$

which reduce the rule likeliness when $$P(T)$$ is high.

#### Amazon recommendation algorithm

Use bag of customer to represent who brought $$item_i$$

| | User 1 | User 2 | ... | User N |
| $$X_i$$ Item i | 1 | 0 | ... | 1 |
| $$X_j$$ Item j | 0 | 0 | ... | 1 |

To measure the similarity of 2 items:

$$
\cos(X_i, X_j) = \frac{X^T_i X_j}{\| X_i \| \| X_j \|}
$$

### Clustering

We build a matrix $$X$$ containing information on what products users purchase.

Rows in $$X$$ contains what a user purchase:

$$
X = \begin{bmatrix}
    -- \text{ Products purchased by user 1  }-- \\
    -- \text{ Products purchased by user 2  }-- \\
	\cdots \\
    -- \text{ Products purchased by user N } -- \\
\end{bmatrix}
$$

Columns in $$X$$ contains who purchase a product:

$$
X^T = \begin{bmatrix}
    -- \text{ Users purchase product 1  }-- \\
    -- \text{ Users purchase product 2  }-- \\
	\cdots \\
    -- \text{ Users purchase product N  }-- \\
\end{bmatrix}
$$

$$
X_{ij} = \begin{cases}
        1 \text{ User i purchase product j}
        \\
        0 \text{ otherwise}
        \end{cases}
$$

We can perform clustering by columns to identify products that attract similar people.

An association rule can be formed by finding $$P(T \vert S) > s$$

<div class="imgcap">
<img src="/assets/ml/re1.png" style="border:none;width:40%">
</div>

Note: The columns here is the same as the bags of customer in the Amazon recommendation algorithm.

### Content-based filtering

Content-based filtering is a supervised learning which it extract features $$x_i$$ of users and products and building model to predict rating $$y_i$$ given $$x_i$$. Later, we make predictions based on this model.

### Collaborative Filtering 

Collaborative filtering is an unsupervised learning which we provide the product ratings from users. We make predictions based on available ratings. Here, each column represents a movie and each row represents a user.

$$
Y = \begin{bmatrix}
    ? & 1  & 3 & 1 & 5 & ? & \dots  & 4 & 1 \\
    2 & 4  & ? & 1 & ? & 3 & \dots  & 1 & 2\\
    2 & 3  & 3 & ? & 5 & 1 & \dots  & ? & 1\\
    & & \vdots & \vdots & \ddots & \vdots & \vdots & \\
    ? & 1  & 5 & ? & 4 & ? & \dots  & 2 & 1 \\
\end{bmatrix}
$$

With our latent factor model:

$$
y_{ij} \approx w^T_j z_i \\
$$

which $$z_i$$ is the latent features for user $$i$$, $$w_j$$ is the latent feature for movie $$j$$ and $$y_{ij}$$ is the ratings for movie $$j$$ from user $$i$$.

$$
\begin{split}
J(W, Z) & = \sum_i \sum_j (w^T_j z_i - y_{ij})^2 + \frac{\lambda_1}{2} \| W \|^2_f + \frac{\lambda_2}{2} \| Z \|^2_f   \\
\end{split}
$$

#### Bias

We do not need to assume $$ y_ij $$ is zero centered. We can add general bias, user bias and product bias in the calculation:

$$
y_{ij} \approx w^T_j z_i + b + b_i + b_j\\
$$

#### Hybrid approach (SVDfeature)

We can also add content based filtering 

$$
y_{ij} \approx w^T_j z_i + w^Tx_{ij} + b + b_i + b_j\\
$$

which $$x_{ij} $$ based on user/product features.

### Other recommendation consideration

* New content
* May want to suggest something new or different
* How long should we show the recommendation if user show or do not show interests
* Give editorial recommendation
* Get recommendation from friends or communities
