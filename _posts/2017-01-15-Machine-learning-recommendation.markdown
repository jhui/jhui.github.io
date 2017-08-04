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

**Support**: Probability of events $$S_1, S_2, \cdots S_k = S$$ are happening. 

$$
P(S) = P(S_1, S_2, ... S_k)
$$

**Confidence**: How offen $$T$$ happens when $$S$$ happens.

$$
P(T \vert S)
$$

In association, we are looking for high support and high confidence:

$$
\begin{split}
S & \implies T \\
\text{i.e. } \quad P(S) & \geq s \\
P(T \vert S) & \geq c \\
\end{split}
$$

For computation reason, we will restrict the maximum number of variables to consider in S and T. ($$ \vert S \vert  \leq k $$ and $$ \vert T \vert \leq k $$)

####  Priori algorithm 

Support Set Pruning: To reduce un-necessary $$P(S)$$ computation, we will prune the support set tree for children that with ancestor's  $$P(S) \lt s$$:

<div class="imgcap">
<img src="/assets/ml/prun.png" style="border:none;width:80%">
</div>

* Generate list of all sets ‘S’ with k = 1.
* Prune candidates where $$ p(S) \lt s $$
* Go to k = k + 1 and generate the list again without the pruned items

#### Generating rules

If $$S = {A,B,C}$$, we generate candidate rules with all possible sub-sets:

$$
\begin{split}
& A \implies B, C \quad B \implies A, C, \quad C \implies A, B, \quad \\
& A, B \implies C, \quad A, C \implies B, \quad B, C \implies A.
\end{split}
$$

Once again, we use pruning to reduce the number of evaluation.

<div class="imgcap">
<img src="/assets/ml/prun2.png" style="border:none;width:80%">
</div>

Note: 

With large amount of data, it is possible that we generate a rule with high probability$$ P(S \vert T)$$ but yet $$S$$ and $$T$$ has no special co-relationship. For example, if $$P(T)$$ is very high, the corresponding $$P(S \vert T)$$ can be pulled above the confidence threshold just because how common $$T$$ is.

#### Lift
 
One alternative to confidence is lift:

$$
Lift(S \implies T) = \frac{P(T \vert S)}{P(T)}
$$

which reduce the rule likeliness when $$P(T)$$ is high.


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

The column $$j$$ in $$X$$ is the list of users that purchase product $$j$$. We can apply clustering to group products (columns) together with the assumption that products are similar if they are purchased by similar users.

<div class="imgcap">
<img src="/assets/ml/CC2.png" style="border:none;width:40%">
</div>

We can apply the association rules to make product recommendations.

$$P(T \vert S) > s$$

<div class="imgcap">
<img src="/assets/ml/re1.png" style="border:none;width:40%">
</div>

Note: The columns here is the same as the bags of customer in the Amazon recommendation algorithm.

### Amazon recommendation algorithm

We need a scalable solution to handle large amount of users and products. First, we can limit the size of $$S$$ and $$T$$ to 1. Then we compute the similarity of 2 products by:

* Use bag of customer to represent who brought $$item_i$$

| | User 1 | User 2 | ... | User N |
| $$X_i$$ Item i | 1 | 0 | ... | 1 |
| $$X_j$$ Item j | 0 | 0 | ... | 1 |

* Measure the similarity of 2 items by:

$$
\cos(X_i, X_j) = \frac{X^T_i X_j}{\| X_i \| \| X_j \|}
$$

Then we can recommend products by soving the k-nearest neighbors using the cosine similarity as distance measurement.

### Content-based filtering

Content-based filtering is a supervised learning which it extracts features $$x_i$$ for a person or a product. For example, the following is the features describing what type of movie may be.

$$ x = (romance, action, scifi) $$

We can collect the marketing material of a movie and use a classifier with supervised learning to classify a pure romance movie as:

$$x = (1, 0, 0) $$

Then we collect how a user ranks movies. Based on these information, we can create the corresponding features of a person and determine what the person likes. For example, a person may have a 0.9 chance of give high ranking to an action or a SciFi movie but no chance for romance movie. The preference $$x_i$$ for this person will be:

$$x = (0, 0.9, 0.9)$$

Content-based filtering builds a model to predict rating or recommendation $$y$$ given $$x_i$$ of a person. In making recommendation, we suggest movies that is closet to the user's features. For ratings, we can make predictions based on.

$$
y = w^T x \\
$$

which $$w$$ and $$x$$ is the features for a movie and a user respectively.

Nevertheless, content-based filtering requires labels for the training data which can be expensive to collect. 

### Collaborative Filtering 

Collaborative filtering is an unsupervised learning which we make predictions from ratings supplied by people. Each rows represents the ratings of movies from a person and each column indicates the ratings of a movie.

$$
Y = \begin{bmatrix}
    ? & 1  & 3 & 1 & 5 & ? & \dots  & 4 & 1 \\
    2 & 4  & ? & 1 & ? & 3 & \dots  & 1 & 2\\
    2 & 3  & 3 & ? & 5 & 1 & \dots  & ? & 1\\
    & & \vdots & \vdots & \ddots & \vdots & \vdots & \\
    ? & 1  & 5 & ? & 4 & ? & \dots  & 2 & 1 \\
\end{bmatrix}
$$

In content-based filtering, we define the feature set already. $$ x = (romance, action, scifi) $$ and we recall that the rating can be computed as:

$$
y = w^T x \\
$$

In Collaborative Filtering, we do not know the feature set before hands. Instead, we try to learn those. Just like the handwritten digit recognition MNist, we do not know what features to extract at the beginning but eventually the program learns those latent features (edge. corner, circle) itself. 

So let say the latent features that a program learn for a person $$i$$ is $$z_i$$ and that for a movie $$j$$ is $$w_j$$. The rating $$ y_{ij}$$ for the movie will be

$$
y_{ij} = w^T_j z_i \\
$$

#### Low rank matrix factorization

We are given a matrix $$Y$$ which the rows represent people and the columns represent products. Low rank matrix factorization means we decompose the matrix $$Y$$ into 2 lower rank matrices: one representing the latent features of a person and the second represent the latent features of a product.

$$
y \approx w^T_j z_i \\
$$

In collaborative filtering

* We decide the number of latent features to learn. i.e. the dimension of $$w$$ and $$z$$. More latent features helps us to build a more complex model but will be harder to train.
* Normalize the rating to be zero centered

$$
Y_{ij} = Y_{ij} - \mu_j 
$$

* Define the cost function as mean square error MSE with L2-regularization

$$
\begin{split}
J(W, Z) & = \sum_i \sum_j (w^T_j z_i - y_{ij})^2 + \frac{\lambda_1}{2} \| W \|^2_f + \frac{\lambda_2}{2} \| Z \|^2_f   \\
\end{split}
$$

* Optimize $$W$$ and $$Z$$ using Gradient descent

To find similar products, we can find the similarity between the latent features of two products. $$W$$. for example:

$$
\| W_i - W_j \|^2
$$

#### Bias

We do not need to assume the rating $$ y_{ij} $$ is zero centered. We can add general bias, bias for the user $$i$$ and bias  for the product $$j$$ in the calculation:

$$
\hat{y_{ij}} \approx w^T_j z_i + b + b_i + b_j\\
$$

#### Hybrid approach (SVDfeature)

We can also adopt a hybrid approach combing both Collaborative filtering and Content-based filtering. The rating is calculated by combining the ratings from both methods:

$$
\hat{y_{ij}} = w^T_j z_i + w_{j}^Tx_{i} + b + b_i + b_j\\
$$

### Other recommendation consideration

* New content
* May want to suggest something new or different
* How long should we show the recommendation if user show or do not show interests
* Give editorial recommendation
* Get recommendation from friends or communities
