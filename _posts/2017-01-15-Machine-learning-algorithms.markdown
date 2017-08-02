---
layout: post
comments: true
mathjax: true
priority: 130000
title: “Machine learning - Algorithms”
excerpt: “Machine learning - Algorithms”
date: 2017-01-15 12:00:00
---

### Decision Tree

A decision tree is a binary tree:

<div class="imgcap">
<img src="/assets/ml/dtr.png" style="border:none;width:70%">
</div>

* Each node partitions data according to a splitting rule. (For example, GPA>3)
* Leaf node returns a label for the classification. (Engineer)

Decision stumps:

* Splitting rule based on thresholding one feature (GPA>3)
	* Select the rule with the highest accuracy in predicting a label, or
	* Select the rule that reduce the entropy the most
* Greedy recursive splitting makes 1 stump at a time
	* Can re-use features to split a tree (GPA>2.5)
* Risk overfitting as the depth of the tree increases

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
 
### Ensemble methods

We start with a shallow decision tree and gradually increase its depth to improve the model accuracy. However, at some point, we start overfitting the model and hurting the generalization. Finding a balance between overfitting and model complexity can be hard. Alternatively, we can adopt a simpler model and push the accuracy by not taking decision from just one model but many models. Even each model makes some mistakes, the chance that all make the same mistakes is small. Hence we ensemble simpler models to avoid overfitting and later combining all results to make a much accurate prediction.

**Averaging**

* Make predictions by multiple models
	* Models built with different initial parameters or
	* Collect models at different iterations of the same training or
	* Models from different methods (KNN, decision trees, naive bayes ... )
* Use simple majority vote, averaging probability (for scholastic model)	or weighted average

<div class="imgcap">
<img src="/assets/ml/avg.png" style="border:none;width:70%">
</div>

**Stacking**

* Models can be built with different methods (KNN, decision trees, naive bayes ... )
* The predictions of each model are used as the input feature of another model

<div class="imgcap">
<img src="/assets/ml/stack.png" style="border:none;width:80%">
</div>


**Boosting**

To avoid overfitting, we ensemble many simpler models to make predictions. We start with a training dataset to build the first model. For the next model, we sample the training datasets giving more weight on those mislabeled datapoints. This force the next iteration to address what we miss in our first model. After many iterations, we build many different rules that when combined becomes very powerful.

<div class="imgcap">
<img src="/assets/ml/boosting.png" style="border:none;width:70%">
</div>


* Use a simple model that it is hard to overfit (For example, a shallow decision tree)
* Fit the model with the training data
* For datapoints that are mis-labeled by $$Model_i$$, we assign a higher weight to them.
* Sample the next training datapoints based on the assigned weight
* Fit a new model with the new training dataset
* Continue the process when enough models are built
* Make prediction using ensemble methods

### Random forest

Random forest is an ensemble methods by building many decision trees using bootstrap sample and random tree.

**Bootstrap sample**

* Build many training datasets with sampling from the original training dataset with replacement
	* Since this is sampling with replacement, the new dataset will have duplicate and missing datapoints
	* About 63% of original data will be included in each new dataset
* Fit multiple models using different datasets above
* Averaging the prediction 	

**Random tree**

* In selecting a splitting rule at each node, we do not try out all possible features
* Instead, we randomly select a sub-set of features for consideration
* Different nodes may have a different sub-set of features to determine the splitting rule
* By dropping features, we can build a deeper tree before overfitting hurts us
* Averaging the predictions to increase accuracy





