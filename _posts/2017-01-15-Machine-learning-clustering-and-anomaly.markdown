---
layout: post
comments: true
mathjax: true
priority: 129000
title: “Machine learning - Clustering and Anomaly detection”
excerpt: “Machine learning - Clustering and Anomaly detection”
date: 2017-01-15 12:00:00
---

### K nearest neighbor (KNN)

<div class="imgcap">
<img src="/assets/ml/knn.png" style="border:none;width:50%">
</div>

* Find k (say k=5) training data $$x_i$$ that are similar to x (the black dot)
* Classify using the result of $$y_i$$ (2 pink, 3 blue)
	* No training phase
	* Predictions are expensive O(nd)
	* Measure similarity
		* L2 distance
		* L1 distance
		* Jaccard similarity:  $$ \frac{count(x_i, x_j)}{count(x_i \text{ or } x_j)} $$
		* Cosine similarity
		* Distance after dimensionality reduction
* Curse of dimension - large space need more training data
* Features need similar scale
* Can argument data by translate/rotate/transform images
 


### K-means clustering

K-means clustering groups datapoints into K clusters. Datapoints are assigned to the closet centroid of a cluster. (the black dot)
<div class="imgcap">
<img src="/assets/ml/kmean.png" style="border:none;width:50%">
</div>

* Pick K random points as centroids
* Form clusters by grouping points to their nearest centroid
	* Distance is calculated as the L2 norm

$$
dist = \sqrt{\sum^d_{j=1} (x^i_{j} - c^i_{j})^2 }
$$

* For points in the same cluster, compute the new centroid

$$
c^i_j = \frac{1}{n_{c^i}} \sum_{m \in c^i} x^m_j
$$

* Re-cluster all datapoints based on the new centroid location
* Repeat the process in computing the new centroids and re-clustering
* Finish when no points switch to another cluster

Corresponding cost function:

$$
J = \sum^N_{i=1} \sum^d_{j=1}  (x^i_j - c^i_j )^2
$$

In image **vector quantization**, we use K-means clustering to map a RGB pixel into one of the cluster's centroid. For example, with a 8-means clustering, we have $$2^8=256$$ clusters. We map a 24 bits RGB pixel into one of the cluster's centroid RGB value.

### K-median clustering

The center of the cluster is based on medium. Compute the medium separately for each dimension.

$$
c^i_j = median_j (x^i_j)  
$$

which $$ c_i $$ is the center of the cluster $$i$$.

Cluster assignment: Assign datapoints to the closest cluster with distance measured with the L1 norm 

$$
dist^i = \sum^d_{j=1} \vert x^i_j - c^i_j \vert  
$$

<div class="imgcap">
<img src="/assets/ml/med.png" style="border:none;width:60%">
</div>

Less vulnerable to outliners: The green outliner on the top left will be grouped into the green cluster in a 3-median cluster instead of having itself as a separate cluster and merge the green and blue cluster together.
Corresponding cost function:

$$
J = \sum^N_{i=1} \sum^d_{j=1} \vert x^i_j - c^i_j \vert  
$$

 
### K-means++ clustering

In K-means clustering, instead of initializing K means randomly at the beginning, we random select one mean at a time. For the next mean, we still select it randomly but with higher preference on datapoints further away from the first mean. For example, we select $$W_1$$ as the first mean and then random select the second mean with higher preference of points further away from $$W_1$$. We select $$W_2$$ and repeat the iteration with more preference of points far away from $$W_1$$ and $$W_2$$.

<div class="imgcap">
<img src="/assets/ml/plus.png" style="border:none;width:50%">
</div>

* Randomly select a point as the first-mean $$w^1$$
* For all datapoints $$ x^i $$, compute the distance from all the existing centroids

$$
dist^i_c = \| x^i - w^c \|_2
$$

* Find the minimum

$$
dist^i = \min \| x^i - w^c \|_2
$$

* We randomly pick $$x^i$$ as the next centroids with the probability

$$
P(x_i) = \frac{dist_i^2}{\sum^N_{j=1} (dist^j)^2}
$$

* Repeating the process until we have k-means

### Density based clustering (DBSCAN) 

As shown below, a distance based cluster like K-means will have problem to cluster concave shape cluster:
<div class="imgcap">
<img src="/assets/ml/den2.png" style="border:none;width:40%">
</div>

Density based clustering connects neighboring high density points together to form a cluster. A datapoint is a core point if within radius $$r$$, there are $$m$$ reachable points. 
<div class="imgcap">
<img src="/assets/ml/dde2.png" style="border:none;width:40%">
</div>

A cluster is form by connecting core points that are reachable from the others. The green cluster is formed by 
* located all the core points (dark green) 
* Join all the core points that are within $$r$$
* Join all points that are within $$r$$ from all those core points (shown as green)
<div class="imgcap">
<img src="/assets/ml/dde3.png" style="border:none;width:60%">
</div>

> Unlike other clustering, a datapoint may not belong to any cluster.

Compute the distances for one datapoints to others are expensive if we have a lot of datapoints. Instead, datapoints are partitioned into regions. We use a grid system with grid size $$r$$. We only connect points that are in the same or adjacent regions. Since datapoints can be sparse, we can use a hash to store the datapoints that belong to a grid. But in high dimension, the number of neighboring regions can still be large.

### Ensemble Clustering (UBClustering)

We can run K-means many times with different initialization to produce models. We use those models to group datapoints $$ x^i $$ and $$ x^j $$ that voted by the models to be stayed together.


* Run K-means M times with different initialization to produce M models
* For datapoints $$x_i$$ and $$x_j$$, if a simple majority of M models agrees they belongs to the same cluster
	* If both are already assigned to clusters, merge both clusters
	* If none are assigned, form a new cluster
	* If only one is assigned, assign the other one into the same cluster


### Density-Based Hierarchical Clustering

In Density based clustering (DBSCAN), radius $$r$$ acts as a threshold to connect datapoints. If $$r$$ is small, we will create 2 smaller cluster C1 and C2 below. 
<div class="imgcap">
<img src="/assets/ml/hier2.png" style="border:none;width:40%">
</div>

If we increase $$r$$, 2 new clusters are formed.
<div class="imgcap">
<img src="/assets/ml/hier3.png" style="border:none;width:40%">
</div>

Hierarchical clustering create an hierarchy of cluster using different granularity $$r$$.
<div class="imgcap">
<img src="/assets/ml/hier4.png" style="border:none;width:50%">
</div>

#### Agglomerative clustering (Bottom-Up)
* Starts with each datapoint as its own cluster
* Merge the two closest clusters
	* Average-link: Merge cluster with smallest average distance between datapoints
	* Single-link: Minimum distance between datapoints
	* Complete-link: Maximum distance between datapoints
	* Ward's method: Minimize variance
* Stop when only one cluster left
* More common

#### Bottom-down

* Starts with k-means at the top level
* Continue running k-means for each children cluster

### Anomaly detection

#### Model-based outlier detection

Z-score measures the probability of $$x_i$$:

$$
Z_i = \frac{X_i - \mu}{\sigma}
$$

<div class="imgcap">
<img src="/assets/ml/g0.png" style="border:none;width:60%">
</div>

Issue: The outlines contributes to the value of $$ \mu $$ and $$\sigma$$ and assume it is uni-modal. (The probability distribution function has only 1 peak.)

#### Graphical outlier detection

Plot data to locate outliner visually:

* Use Box plot for 1 variable at a time
* Scattering plot for 2 variables at a time
* Scattering array to look at multiple combination at a time. But still ploting 2 variables at a time
* Scattering plot of 2-D PCA

#### Cluster-Based outlier detection

* Cluster the data
* Find points that do not belong to a cluster, (density based clustering) or
<div class="imgcap">
<img src="/assets/ml/dde4.png" style="border:none;width:35%">
</div>
* Far away from the center of the cluster, (K-means) or
<div class="imgcap">
<img src="/assets/ml/med.png" style="border:none;width:40%">
</div>
* Only join a hierarchy clustering at the coarse grain level


#### Global distance-based outlier detection: KNN

We can measure the distance of a datapoint from its neighbors to detect outliner.

* Calculate the average distance for its K-neighbors
* Choose the biggest values as outliners
* Good for locate global outlier

<div class="imgcap">
<img src="/assets/ml/knn2.png" style="border:none;width:35%">
</div>

#### Outlier-ness

Nevertheless, some datapoints may be close to a cluster in the global sense but should not be consider as part of it after consider the average distance among the cluster's members. Members in the Cluster C2 is closer together than Cluster C1. So even the blue dot is only $$r$$ away from the green dot, it is not consider as part of the cluster C2 while the red dot will be consider as part of C1.

<div class="imgcap">
<img src="/assets/ml/hier6.png" style="border:none;width:50%">
</div>


The average distance for $$x_i$$ from its k neighbors is:

$$
\begin{split}
D^k(x^i) & = \frac{1}{k} \sum_{j \in N_i} \| x^i - x^j \| \\
\end{split}
$$

To consider whether it is outlier, we need to consider how close $$x_i$$ to its neighbors $$ N_i $$ and how close those neighbors $$ x_l$$ are with their neighbors $$ N_l $$. 

$$
\begin{split}
O^k(x^i) & = \frac{D^k(x^i)}{\frac{1}{k} \sum_{l \in N_j} D^k(x^l)}
\end{split}
$$

If $$ O^k(x^i) > 1$$, the datapoint between itself and the cluster is greater than the average distance among the cluster members.

#### Influenced outlierness

However, we may still have problems for clusters that are very close. The circled green dot has high $$O^k(x^i) $$ even though it is part of the green cluster because it counts the red dots as its closest neighbors.

<div class="imgcap">
<img src="/assets/ml/out3.png" style="border:none;width:30%">
</div>

In influenced outlierness, we are not finding the average distance of its neighbors' neighbors. Instead, we find the average distance of its neighbors that consider $$x_i$$ as its neighbors.

<div class="imgcap">
<img src="/assets/ml/out4.png" style="border:none;width:30%">
</div>

Then we replace the denominator with the average distance of those tick dots.

$$
\begin{split}
O^k(x^i) & = \frac{D^k(x^i)}{\frac{1}{k} \sum_{i \in N_l} D^k(x^l)}
\end{split}
$$



#### Supervised outlier detection

We can use supervising learning to determine whether a datapoint is an outlier. This method can detect complex rule but will require the labeling of the training data.




