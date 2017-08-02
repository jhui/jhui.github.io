---
layout: post
comments: true
mathjax: true
priority: 120000
title: “Machine learning - Deep learning project approach and resources”
excerpt: “Machine learning - Deep learning project approach and resources.”
date: 2017-01-15 12:00:00
---

### Project approach

* Define task (Object detection, Colorization of line arts)
* Collect dataset (MS Coco, Public web sites)
	* Search for academic datasets and baselines
	* Build your own (From Twitter, News, Website ...)
* Define the metrics 
	* Search for established metrics
* Clean and preprocess the data
	* Select features and transform data
		* One-hot vector, bag of words, bucketize
		* Logarithm scale, spectrogram
	* Remove noise or outliers (Based on algorithms)
	* Remove invalid and duplicate data	
	* Scaling or whiten data
* Split datasets for training, validation and testing
	* Visualize and summarize dataset
	* Validate dataset
* Establish a baseline
	* Implement a simple model
	* Compute metrics as a baseline
	* Analyze errors for area of improvements
* Select network structure
	* CNN, LSTM...
* Implement a deep network
	* Code debugging and validation
	* Parameter initialization
	* Compute metrics
	* Choose hyper-parameters
	* Visualize, validate and summarize result
	* Analyze errors
	* Refine
		* Add layers and nodes
		* Optimization tricks
* Hyper-parameters fine tunings
* Try our model variants

### Deep learning resources
 
Listing of major papers: [https://github.com/kjw0612/awesome-deep-vision] \\
Machine learning competition: [https://www.kaggle.com/]

#### Research paper publication:

CVPR: IEEE Conference on Computer Vision and Pattern Recognition \\
ICCV: International Conference on Computer Vision \\
ECCV: European Conference on Computer Vision \\
NIPS: Neural Information Processing Systems \\
ICLR: International Conference on Learning Representations \\

#### Dataset listing:
[http://www.cvpapers.com/datasets.html] \\
[http://riemenschneider.hayko.at/vision/dataset/] \\
[https://en.wikipedia.org/wiki/List_of_datasets_for_machine_learning_research]

