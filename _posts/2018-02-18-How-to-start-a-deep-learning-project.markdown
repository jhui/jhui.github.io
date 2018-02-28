---
layout: post
comments: false
mathjax: true
priority: 846
title: “How to start and finish a deep learning project?”
excerpt: “How to start and finish a deep learning project?”
date: 2018-02-11 14:00:00
---

<div class="imgcap">
<img src="/assets/dl/gaa0.png" style="border:none; width:80%;">
</div>

(Images [generated from PaintsChainer.](https://paintschainer.preferred.tech/index_en.html))

### Part 1: Start a deep learning project

### What project to pick?

Many AI projects are not that serious and pretty fun. In early 2017, as part of my research on the topic of Generative Adversaries Network (GAN), I started a project to colorize Japanese Manga. The problem is difficult and pushes me to think harder. It was fascinating in particular I cannot draw!  In looking for projects, look beyond incremental improvements. Make a product that is marketable. Create a new model design. Have fun and being innovative. 

In this article, I will explain the whole journey from starting to finishing a deep learning (DL) project. The key emphasis is on topics that are critical to the success. I will illustrate the core ideas with real projects to make the discussion more concrete. Many topics in this article can spawn off into a mini-article. I will concentrate on the core concepts. Many detail implementation steps can be found with simple Google search.

### Big picture

#### Debug Deep Network (DN) is tough

DL training composes of million iterations to build a model. Bugs are hard to locate. Break the problem down and move in small steps. Model optimization tasks like regularization can always wait after the code is more proven. Visualize your predictions and model metrics frequently. Make something works first so you have a reference. Build incrementally. It will be much more fun to see progress.

>  Start small, move small.

#### Measure and learn

Most personal projects last from two to four months for the first public release. It is pretty short since research, debugging and experiments take time. We make design changes and schedule the complex experiments to run overnight. By early morning, we want enough information to plot our next move. As a good rule of thumb, those experiments should not run longer than 12 hours in the early development phase. To achieve that, we narrow the scope to single female figures. As a flashback, we should further reduce the scope. We have many designs to test and the turn around time of our experiments is not fast enough. Educated decisions are hard to make if we cannot measure and learn fast.

> Build, measure and learn. 

#### Research v.s. product

When we start the Manga project in the Spring of 2017, Kevin Frans has a Deepcolor project to colorize Manga with spatial color hints using GAN.  

<div class="imgcap">
<img src="/assets/dl/kev.jpg" style="border:none; width:40%;">
</div>

Many AI fields are pretty competitive. When defining the goal, you want to push hard enough so the project is still relevant when it is done. GAN model is pretty complex and the quality is usually not product-ready in early 2017. Nevertheless, if you smartly constraint what the product can handle, you may push the quality high enough as a commercial product. To achieve that, select your training samples carefully, and we will detail it in the dataset section later. For any DL projects, we often strick a good balance among model generalization (capability), capacity and accuracy. 

#### Cost

Models trained on a GPU is 20 to 100 times faster than a CPU. The lowest price Amazon GPU p2.xlarge spot instance is about $7.5/day. The next 8 units GPU jumps to $75/day. Google can spend a whole week to train NLP models using one thousand servers. GAN is more complex than the typical classification problem. In our Manga project, some experiments took over 2 days. We spend an average of $150/week. For faster development iterations, the more expensive instance may ring up the bill to $1500/week. A standalone machine with the Nvidia GeForce GTX 1080 TI costs about $2200 in Feb 2018. It is about 5x faster than a p2.xlarge instance in training a fine tunned VGG model.

### Time line

We define our development in four phases with the last 3 phases executed in multiple iterations.

* Project research
* Design
* Implementation and debugging
* Experiment and tuning

### Project research

We do research in current offerings on both AI and non-AI approaches to explore what is missing. For many GAN type solutions, they utilized spatial color hints. The drawings are a little bit wash out or muddy. The colors sometimes bleed. We set a 2-month timeframe for our project with 2 top priorities: regenerate image without hints and improve color fidelity. And our first project definition is:

> Color a grayscale Manga drawing without hints on single female figures.

#### Standing on the shoulders of giants 

Then we study related research and open source projects. Spend a good amount of time in doing research. Gain intuitions on where existing models are flawed or performed well. Many people go through at least a few dozen papers and projects before starting their implementations. For example, when we deep down into GANs, there are over a dozen new GAN models: DRAGAN, cGAN, LSGAN etc... Reading research papers can be painful. Skim through the paper quickly to grab the core ideas. Pay more attention to the figures. After knowing what is important, read the paper again. Now, you know better what to skip and what to pay attention.


Deep learning (DL) codes are condensed but difficult to troubleshoot. Research papers often miss details in reproducing the models. Typically, many implementations start off with repurpose code that shows successes for similar problems. Search hard using Google. Try a few options. We locate code implementations on different GAN variants. We trace the code and give them a few test drives. We replace the generative network of one implementation with an image encoder and a decoder. As a special bonus, we find the hyperparameters in place are pretty decent. Otherwise, searching for the initial hyperparameters can be tedious when the code is still buggy.

### Part 2: Build a Deep Learning dataset

### Dataset

> Garbage in and garbage out. Good data trains good models.

For research projects, search for academic datasets. Collect samples for custom datasets may not be difficult. However, building custom datasets require careful data cleaning and require significant resource to produce quality dataset. Bad quality data creates bad models. Academic datasets are pre-filtered and cleaner for training. There are published model performance that you can reference for your progress. If you have more than one options, select the one with the highest quality samples for your target category. 

For real life problems, we need samples originated from the problem domains. Smaller projects rarely collect as many samples comparing with academic datasets. Apply transfer learning if needed. Many projects collect samples by clawing data from web sites. The quality of the samples varies. Some samples are not relevant. Even we collect Manga tagged as single females, we filter out about 10% of our crawled data.

I re-visit the progress on the Manga colorization when I write this article. Deepcolor was inspired by PaintsChainer. We did not spend much time on PaintsChainer when we started the project. But I am glad to play them a visit. (Feb. 2018).

The left is an image provided by the application and the right is the drawing colored by the machine. Definitely, this is product ready quality.

<div class="imgcap">
<img src="/assets/dl/gaa1.png" style="border:none; width:80%;">
</div>

We decide to test it with some of our training samples. It is less impressive. Fewer colors are applied and the style is not correct.

<div class="imgcap">
<img src="/assets/dl/gaa2.png" style="border:none; width:50%;">
</div>

<div class="imgcap">
<img src="/assets/dl/gaa3.png" style="border:none; width:50%;">
</div>

Since we trained our model for a while, we knew what drawings will perform badly. As expected, it has hard time for drawings with entangled structures.

<div class="imgcap">
<img src="/assets/dl/gaa4.png" style="border:none; width:50%;">
</div>
  
This illustrates a very important point: choose your dataset well. As a product offering, PaintsChainer makes a smart move on focusing the type of scratches that they excel. To proof that, I use a clean lineart picked from the internet. The result is impressive again.

<div class="imgcap">
<img src="/assets/dl/gaag.png" style="border:none; width:50%;">
</div>

There are a few lessons learned here. There is no bad data, just the data is not solving your needs. Focus on what your product wants to offer. As samples' categories increased, it is much harder to train and to maintain output quality. Many Anime drawings are well tagged. Crawl the web site and store the samples under different categories. Train with one category before expanding them for what the model can handle. For example, start the training with Anime portraits first. If your samples are not tagged, use a DN classifier. If the Anime needs to categorize in finder details, for example hair color, use tool like Illustration2Vec to estimate tags by extracting semantic feature vectors. 
 
Verify the quality of the data and the accuracy of the labels. Before crawling data, look for any other possible web sites. Even some web sites are popular for mining deep learning samples, a better source of samples can put your over others. Filter out weird samples or samples that out of the scope. In early development, we realize some drawings are too entangled. Without significant increasing the model capacity, those drawings produce little values in training and better be left off. It just makes the training much inefficient.

>  Trimming dataset can be beneficial. It will result in a more focused model for your purposes.

For classification problem, verify the amount of samples for each class is relatively the same.

### Part 3: Designs

#### Simple and smart

Start your design simple and small. In the study phase, people are flooded with many cool ideas. We tend to code all the bots and nuts in one shoot. This will not work. Try to beat the state-of-the-art too early is not practical. Ironically, try to beat the odd of random guessing first. Design with few layers and less customizations first. Avoid solutions that require un-necessary hyperparameters tuning at the early phase. Verify the loss is dropping and check whether the model shows signs of "intelligence". Do not waste time training the model with too many iterations or too large batch size. After a short debugging, our model produces pretty un-impressive results after 5000 iterations. But colors start confined to regions. There is hope that skin tone is showing up. 

<div class="imgcap">
<img src="/assets/dl/gaaw.jpg" style="border:none; width:40%;">
</div>

This gives us valuable feedbacks on whether the model starts coloring. Do not start with something big. You will spend most of your time debugging this or wondering do we just need another hour to train the model.

<div class="imgcap">
<img src="/assets/dl/gaan.jpg" style="border:none; width:40%;">
</div>

Nevertheless, this is easier to say than doing it. We jump steps. But you are warned!

#### Priority & incremental driven design

To create simple designs first, we need to sort out the top priorities. For a complex problem, we may break in down into smaller problems and solve them in steps. Our first design involves convolutions, transposed convolutions and a discriminative network. This may be too complex as a first design. But since we already fully test our public code, it is still manageable. To maximize the chance of success, we analyse what is our top issue. GAN is hard to train. From our previous experience, we know the gradient diminishing problem will be an issue for us. So we research a couple skip connections to mitigate the problem. 

Everyone has a plan 'till they get punched in the mouth. (Mike Tyson) The right strategy in a DL project is to maneuver quickly from what you see. Before jumping to a model using no hints, we start with one with spatial color hints. We do not move to a "no hint" design in one step. We first move to a model with color hints but dropping the hints' spatial information. The color quality drops significantly. We know our priority is shifted. We refine our deep learning (DN) model first before making the next big jump. Instead of making a long term plan that keeps changing, adopt a shorter lean process driven by priorities. Model changes usually have many surprises, bugs and new issues. Use shorter and smaller design iterations to make it manageable.

#### Transfer learning with pre-trained models

Training from scratch takes a long time. As stated from the VGG paper in 2014: the VGG model was originally trained with four NVIDIA Titan Black GPUs, training a single net took 2–3 weeks depending on the architecture. Training a complex CNN model like VGG19 on a small dataset will overfit the model. We can implement a much smaller capacity model to improve generalization. For other problem, accuracy strongly depends on how well the CNN extracts latent factors. Training from scratch is not practical. Instead, we fine tune pre-trained model like VGG19 or ResNet using our sample data. We can repurpose the network in shorter time. Usually, the first few convolution layers perform common tasks like detecting edges and color. We can freeze those layers from updates. We can have further speedup the training at the cost of accuracy if we only retrain the last few fully connected layers. Because we start with pretty good model parameters, we often use smaller learning rate to retrain the mode. Pre-trained models can be used across multiple disciplines. Indeed, we can model the Chinese language with a pre-trained English model.

#### Constraints 

From the Boltzmann machine to the Restricted Boltzmann machine, we apply constraints to the network design to make training more effective. In CNN, nodes are only connected to their neighbors in the previous layer. For Variational Autoencoders, the latent factors are suppose to be Normal distributed. In Batch normalization, the node output is normalized. Building deep learning is not only putting layers together. Adding good constraints make learning more efficient or more "intelligent". In our first design, we apply denoising to restrict the amount of information that the spatial color hints may provide. We purposely soften those hints or corrupt it by zeroing a large fraction of hints. Ironically, it generalizes the model better and less sensitive to variants.

>  DL is more than adding layers.

### Design details

#### Cost function and Metrics

Not all cost functions are created equally. It impacts how easy to train the model. Some cost functions are pretty standard, but some problem domains need some careful thoughts. 

| Problems   |      Cost functions     | 
|----------|:------------|
| Classification |  Cross entropy, SVM | 
| Regression | Mean square error (MSE) |
| Object detection or segmentation | Intersection over Union (IoU) | 
| Policy optimization | Kullback–Leibler divergence |
| Word embedding | Noise Contrastive Estimation (NCE) |
| Word vectors | Cosine similarity |

Cost functions looking good in the theoretical analysis may not perform well in practice. For example, the cost function for the discriminator network in GAN adopts a more practical and empirical approach than the theoretical one. In some problem domains, the cost functions can be part guessing and part experimental. It can be a combinations of a few cost functions. In our project, we start with the standard GAN cost functions. We also add a reconstruction cost using MSE and other regularization costs. However, our brain does not judge styling by MSE. One of the un-resolved areas in our project is to finding a better reconstruction cost function. We believe it will have significant impact on the color fidelity.

> Finding good cost functions becomes more important when we move into less familiar problem domains.

Good metrics help you to compare and tune models better. Search for established metrics for your type of problem. Unfortunately, for our project, it is very hard to define a precise accuracy formula for artistics rendering.

#### Regularization

> L1 and L2 regularization are both common but L2 regularization is more popular in DL. 

L1 regularization promotes sparsity in parameters. Sparsity courage representations that disentangle the underlying representation. Since each non-zero parameter adds penalty to the cost, it prefers more zero parameters. i.e. it prefers many zero and a slightly larger parameter than many tiny parameters like L2. L1 regularization makes filters cleaner and therefore easier to interpret. Since one way to train larger models for less computation and energy is to introduce sparsity, L1 is more suitable for mobile device. L1 is also less vulnerable to outliners and works better if the data is less clean. L2 regularization can be used for feature selection.

Dropout can be applied to the fully connected layers as another regularization technique. Like many advance techniques, the benefit of combining dropout with L2 regularization can be domain specific. Usually we apply those techniques in fine tuning and only the collected empirical data can verify whether it is beneficial. Generally, use a dropout value between 20% and 50% of nodes. Dropouts make noticeable difference for a bigger network.

#### Gradient descent

Always monitor gradient closely for diminishing or exploding gradients.  Gradient descent problems have many possible causes which are very hard to verify. Do not jump into the learning rate tuning or model design changes too quickly. In early debugging, the small gradients may simply cause by programming bugs. For example, the input data is not scaled properly, or the weights are all initialized to zero. Tuning takes time. It will have better returns if we have verify other causes that is easier to check or more likely to be happened in early development.

If other possible causes are eliminated, apply gradient clipping (in particular for NLP) when gradient explode. Skip connections are common technique to mitigate gradient diminishing problem. In ResNet, a residual layer allows the input to be bypassing the current block directly to the next block. Effectively, it reduces the depth of the network in early training to make training easier.

#### Activation functions

In DL, ReLU is the most popular activation function to introduce non-linearity to the model. If the learning rate is too high, many node can be dead and stay dead. If changing the learning rate does not help, we can try leaky ReLU or PReLU.  In a leaky ReLU, instead of outputting zero when $$x < 0$$, it has a small predefined downward slope (say 0.01 or set by a hyperparameter). Parameter ReLU (PReLU) pushes a step further. Each node will have a trainable slope.

#### Scaling

We often scale input features using a mean and a variance computed from the training dataset. When we scale the validation or testing data, we reuse the same mean and the variance from the training dataset. We do not compute the mean and variance from the validation or testing data.

#### Batch Normalization & Layer Normalization

Apply batch normalization (BN) to CNN. DN learns faster and better if inputs are properly normalized. In BN, we compute the means and the variances for each location from each batch of training data. For example, with a batch size of 16 and a feature map of 10x10 spatial dimension, we compute 100 means and 100 variances (one per location). The mean at each location is the average of the corresponding locations from the 16 samples. We use the means and the variance to renormalize the node output at each location. BN improves accuracy and reduces training time. As a side bonus, we can increase the learning rate further to make training faster.

However, BN is not effective for RNN. We use Layer normalization instead. In RNN, the means and variances from the BN are not suitable to renormalize the output of RNN cells. It is likely because of the recurrent nature of the RNN and the sharing parameters. In Layer normalization, the output is renormalize by the mean and the variance calculated by the layer's output of the current sample. A 100 elements layer have only one mean and one variance from the current sample.

#### Baseline 

Setting a baseline helps us to compare models and debugging. Research projects often require an established model as a baseline to compare models. For example, we can have a VGG19 model as the baseline for classification problems. Alternatively, we can extend some established and simple models to solve our problem first. This helps us to understand the problem better and establish a performance baseline for comparison. This baseline implementation can also be used to validate data processing codes. By keeping the first implementation simple, we can use it as a baseline. In our project, we modify an established GAN implementation and redesign the generative network as our baseline. 

#### Checkpoints

We save models' output and metrics periodically for comparison. Sometimes, we want to reproduce results for a model or reload a model to train it further. Checkpoints allow us to save models to be reloaded later. However, if a model has changed, all old checkpoints cannot be loaded. Even there is no automated process to solve this, we use Git tagging to trace multiple models and reload the correct model for a specific checkpoint. Checkpoints in TensorFlow is huge. our early designs already take 4 GB per checkpoint. Working in a cloud environment, configure enough storages accordingly. We start and terminate Amazon cloud instances frequently. Hence, we store all the files in the Amazon EBS so it can be reattached easily.


#### Split dataset

To test the real performance, we split our data into three parts: 70% for training, 20% for validation and 10% for testing. Make sure samples are randomized probably in each dataset and each batch of training samples. During training, we use the training dataset to build models with different hyperparameters. We run those models with the validation dataset and pick the one with the highest accuracy. This strategy works if the validation dataset is similar to what we want to predict. But, as a last safeguard, we use the 10% testing data for a final insanity check. This testing data is for a final verification, but not for model selection. If your testing result is dramatically different from the validation result, the data should be randomized more, or more data should be collected.

#### DL framework

In just six months after the release of TensorFlow from Google in Nov 2015, it became the most popular deep learning framework. While it seems implausible for any challengers soon, PyTorch was released by Facebook a year later and get a lot of traction from the research community. As of 2018, there are many choices of deep learning platform including TensorFlow, PyTorch, Caffe, Caffe2, MXNet, CNTK etc ... In TensorFlow, you build a computation graph and submit it to a session later for computation. In PyTorch, every computation is executed immediately. TensorFlow's approach creates a black box that makes debugging in-convenient. That is why tf.print become so dominant in debugging TensorFlow. The TensorFlow approach should be easier to optimize but yet we have not seen the speedup as it may indicates. Nevertheless, the competition is so fierce that any feature comparison will be obsolete in months. For example, Google already released an alpha version of eager execution in v1.5 to address this issue. But there is one key factor triggers the defection of some researchers to PyTorch. PyTorch design is end user focus. The API is simple and intuitive. The error messages make sense and the documentation is well organized. The pre-trained model and data pre-processing in PyTorch is very popular. TensorFlow does a great job in matching features. But so far, it adopts a bottom-up approach that make things un-necessary complex. It may provide flexibility but many of them is rarely used. The design thinking in TensorFlow can be chaotic. For example, it has about half a dozen set of APIs to build models: result of many consolidations and matching competitor offerings.

By no mistakes, TensorFlow is still dominating as of Feb 2018. The developer community is still much bigger. This may be the only factor matters. If you want to train the model with hundred machines or deploy the inference engine onto a mobile phone, TensorFlow is the only choice. Nevertheless, if other platforms prove themselves to provide what end user needs, we will foresee more deflection for small to mid size projects.


#### Custom layers

Built-in layers from DL software packages are better tested and optimized. Nevertheless, if custom layers are needed:

* Unit test the forward pass and back propagation code.
* At the beginning, test with non-random data.
* Compare the backpropagation result with the naive gradient check.
* Add tiny ϵ for divison or log computation to avoid NaN.

#### Randomization

One of the challenge in DL is reproducibility.

During debugging, if the input or the initial model parameters are changed from the last session, we need to recompute what the result should be. Hence, we explicitly initialize the seed of the randomizer. We often use more than one randomizer. In our project, we initialize the seeds for python, the NumPy and the TensorFlow.

For final tuning, we may want to turn off those explicitly seed initialization so we can explore different models for each run. To reproduce the result of a model, we need need to save (checkpoint) a model and reload the model later.

### Optimizer

Adam optimizer is one of the most popular optimizer in DL, if not the most popular. It suites for many problems including models with sparse or noisy gradients. It achieves good results fast with the greatest benefit of easy to tune. Default configuration parameters often do well.

It combines the advantages of AdaGrad and RMSProp. Instead of one single learning rate for all parameters, Adam internally maintains a learning rate for each parameter and separately adapted them as learning unfolds. Adam is momentum based with a running record of the gradients. Therefore, the gradient descent runs smoother and it dampens the parameter oscillation problem due to large gradient and learning rate.

A less popular option is the SGD using Nesterov Momentum.

#### Adam optimizer tuning

Adam has 4 configurable parameters. 

* The learning rate (default 0.001)
* $$\beta_1$$ is the exponential decay rate for the first moment estimates (default 0.9). 
* $$\beta_2$$ is the exponential decay rate for the second-moment estimates (default 0.999). This value should be set close to 1.0 on problems with a sparse gradient. 
* $$\epsilon$$ (default $$1e^{-8}$$) is a small value add to mathematical operation to avoid illegal operations like divide by zero. However, in the TensorFlow documentation, it states "The default value of 1e-8 for epsilon might not be a good default in general. For example, when training an Inception network on ImageNet a current good choice is 1.0 or 0.1."

β temporally smooth out the stochastic gradient descent by accumulate information on previous descent to smooth the changes by the current data sample. 

The default configuration, including $$\epsilon$$, works well for early development usually. The most likely tunned parameter is the learning rate. 

### Part 4: Visualize a Deep network model and matrices

### Instrumentation

> Never shot in the dark. Make educated guess.

We fall into the hype of DL by exaggerating how easy it is by counting the lines of code. This is deceptive because tracing DL problems are extremely hard. It is important to track every move and to examine results at each step. We often ignore the instrumentation needs and end up trouble shooting problems in the dark. Nevertheless, with the help of pre-built package, integrating instrumentation code is not hard and the rewards are almost instantaneously. 

#### Data visualization (input, output)

Verifying the input and the output of the model is important. Before feeding data into a model, add code to save some training and validation samples for later visual verification. Apply steps to undo some of the pre-processing including rescaling the pixel value back to the range between 0 and 255. Repeat the process for a few batches to verify the training samples are probably randomized. At each epoch, save some model outputs on the validation set for verification and monitoring. The left side images below are some training samples and the right is a validation sample.

<div class="imgcap">
<img src="/assets/dl/base00.png" style="border:none; width:80%;">
</div>

Sometimes, it is good to verify the input data is zero centered in the proper range. (using TensorBoard) 

<div class="imgcap">
<img src="/assets/dl/inp1.png" style="border:none; width:40%;">
</div>
(Source Sung Kim)


We also save the model's output regularly for verification and future reference.

<div class="imgcap">
<img src="/assets/dl/base01.png" style="border:none; width:80%;">
</div>

#### Metrics (Loss & accuracy)

Beside displaying the loss and the accuracy to the stdout regularly, we record and plot them to analysis its long term trend. The diagram below is the accuracy and the cross entropy loss displayed by the TensorBoard. 

<div class="imgcap">
<img src="/assets/dl/mm1.png" style="border:none; width:80%;">
</div>

Plotting the cost helps us to tune the learning rate. Any prolonged jump of cost indicate the learning rate is too high. As indicated below, if it is too low, we learn slowly. If it is too high, we cannot reach the lowest cost.

<div class="imgcap">
<img src="/assets/dl/mont1.png" style="border:none; width:50%;">
</div>

Here is another real example when the learning rate is too high.

<div class="imgcap">
<img src="/assets/dl/wallner.png" style="border:none; width:80%;">
</div>

(Image source Emil Wallner)

Below we monitor the accuracy. If there a major difference between the validation accuracy and the training accuracy, we know the model is overfit. We can increase the regularization factor to reduce overfit. We can use this technique to verify the coding in computing regularization.

<div class="imgcap">
<img src="/assets/dl/mont2.png" style="border:none; width:50%;">
</div>

#### History summary 

**Weight & bias**: We want to monitor the weights and the biases closely. Here is the Layer 1's weights and biases distributions at different training iterations. Seeing large (postive or negative) weights or bias are abnormal. A Normal distribution for weight is a good sign that the training is going well (but not absolutely necessary).

<div class="imgcap">
<img src="/assets/dl/lss1.png" style="border:none; width:100%;">
</div>

<div class="imgcap">
<img src="/assets/dl/actt.png" style="border:none; width:90%;">
</div>

**Gradients**: Monitor gradients at different layers are important in identifying one of the most serious problem in DL: gradient diminishing or exploding problems. If gradient diminish quickly from the right most layers to the left most layers, we have a gradient diminishing problem.

<div class="imgcap">
<img src="/assets/dl/lgrad.png" style="border:none; width:50%;">
</div>

Even not common done, we can visualize the filters $$W$$s for CNN in the first couple layers. This helps us to identify what type of features that the model is extracting. The first layer should extract simple structures like edge and color.

<div class="imgcap">
<img src="/assets/cnn/cnnfilter.png" style="border:none; width:50%;">
</div>

**Activation**: For gradient descent to perform the best, the nodes' outputs before the activation function should be normal distributed. For example, we should apply a batch normalization to correct the non-zero centered pre-activation output below. The right side plot is the activation output of a ReLU function. It will be bad if there are too many dead or highly activated nodes. 

For CNN based networks, we can visualize what a feature map is learning. In the following picture, it captures the top 9 pictures (right picture) having the highest activation in a particular map. It aslo applies a deconvolution network to reconstruct the spatial image (left picture) from the feature map.

<div class="imgcap">
<img src="/assets/dl/viss.png" style="border:none; width:50%;">
</div>

(Source from Visualizing and Understanding Convolutional Networks, Matthew D Zeiler et al.)

The image reconstruction is rarely done. But in a generative model, we often vary just one factor while holding other constant to verify whether the model is learning anything smart. 

<div class="imgcap">
<img src="/assets/capsule/dim.png" style="border:none; width:60%;">
</div>

(Source Dynamic Routing Between Capsules, Sara Sabour, Nicholas Frosst, Geoffrey E Hinton)

### Part 5: Debug a deep learning network

In the early development, we are fighting multiple battles at the same time. Implement only the core part of the model with minimum customizations: use out of the box layers and cost functions. Other designs to optimize the model can wait. Focus on verifying the model is functioning first. 

* Set the regularization factors to zero.
* No other regularization including dropouts.
* Use the Adam optimizer with default settings.
* Use ReLU.
* No data augmentation. 
* Fewer DN layers.
* Scale your input data but no un-necessary pre-processing.
* Don't waste time in long training iterations or large batch size.

Overfitting the model with a small amount of training data is the best way for debugging. If loss does not drop within a few thousand iterations, debug the code further. Achieve your first milestone by beating the odd of guessing. Then make incremental modifications to the model: add more layers and customization. Train it with the full training dataset. Add regularization to control the overfit by monitor the accuracy gap between the training and validation dataset. 

> If stuck, take out all bells and whistles and make your problem smaller.

#### Initial hyperparameters

Many hyperparameters are not relevant until model optimization. Turn them off or use default values. Use Adam optimizer. It is fast, efficient and the default learning rate does well. Early problems are mostly from bugs rather from the model design or tuning problems. Go through the checklist in the next section before any tunings. It is more common and easier to verify. If loss still does not drop, tune the learning rate. If the loss drops too slow, increase the learning rate by 10. If the loss goes up or the gradient explodes, decrease the learning rate by 10. Repeat the process until the loss drop gradually and nicely. Typical learning rates are between 1 and $$1e^{-7}$$.

<div class="imgcap">
<img src="/assets/dl/mont1.png" style="border:none; width:40%;">
</div>

#### Checklist

Data:

* Visualize and verify the input data (after data pre-processing and before feeding to the model). 
* Verify the accuracy of the input labels (with or without data shuffle).
* Do not feed the same batch of data over and over.
* Scale your input properly (likely between -1 and 1 and zero centered). 
* Verify the range of your output (e.g. beween -1 and 1).
* Always use the mean/variance from the training dataset to rescale the validation/testing dataset.
* All input data to the model has the same dimensions.
* Access the overall quality of the dataset. (Are there too many outliners or bad samples?)

Model:

* The model parameters are initialized correctly. The weights are not set to all 0.
* Debug layers that the activations or gradients diminish/explode. (from right most to left most layers)
* Debug layers that weights are mostly zero or too large.
* Verify and test your loss function.
* For pre-trained model, your input data range matches the range used in the model.
* Dropout in inference and testing should be always off.

Weight should be initialized with Gaussian distribution with $$\mu =0$$ and $$\sigma = \min\big( 5e^{-2},  \sqrt{\frac{2}{\text{# input nodes}}} \big)$$. Verify and test the correctness of your loss function. The loss of your model must be lower than the one from the random guessing. For a classification problem with 10 classes, the cross entropy loss for random guessing is $$ -\ln \frac{1}{10}$$. 

#### Analysis errors

Your design iterations need constant review of what is doing badly (errors) and improve it. Visualize your errors. In our project, the model performs badly for images with highly entangled structure. Identify the model weakness to make changes. For example, add more convolution layers with smaller filters to disentangle small features. Augment data if necessary, or collect more similar samples to train the model better. In some situtation, you may want to remove those samples and constraint yourself to a more focus model.

<div class="imgcap">
<img src="/assets/dl/badd.jpg" style="border:none; width:50%;">
</div>

#### Regularization tuning

> Turn off regularization until the model code makes reasonable predictions for the training data.

Once the model code is working, the next tuning parameters are the regularization factors. We increase the regularizations to narrow the gap between the validation and training accuracy. Do not overdo it as we often want a slightly overfit model to work with. Monitor both data and regularization cost closely, regularization loss should not dominate the data loss over prolonged periods. If we significant increase the regularization rate but yet the gap does not narrow, we need to debug the regularization code or use a different type of regularization. 

We start with the most important regularization factor. We increase the volume of our training data and then increase the regularizations to narrow the gap. Similar to the learning rate, we change testing values in the logarithmic scale. (for example, change by a factor of 10 at the beginning) After this is done, we move onto the next regularization factor. Beware that each regularization factor can be in a total different order of magnitude.

<div class="imgcap">
<img src="/assets/dl/mont2.png" style="border:none; width:50%;">
</div>

#### Multiple cost functions

For the first implementations, avoid using multiple cost functions. If more than one cost functions are used later. Tune their corresponding weights accordingly. Notice that each weight can be in different order of magnitude.

#### Frozen variables

When we use pre-trained models, we may freeze those model parameters in certain layers to speed up computation. Double check no variables are frozen in-correctly. 

### Unit testing the code

As less often talked, we should unit test some core modules so the implementation is less vulnerable to code changes. We can mock the input data with Tensors with fixed values (not random values). For each modules (layers), We can verify 

* the shape of the output in both training and inference.
* the number of trainable variables (not the number of parameters).

#### Dimension mismatch

Always keep track of the shape of the Tensor (matrix) and document it inside the code. For a Tensor with shape \[N, channel, W, H \], if the dimension of $$W$$ (width) and $$H$$ (height) are the same, the code will not generate any errors if we swap the $$W$$ and $$H$$ channel by mistake. Therefore, we should unit test our code with a non-symmetrical shape. For example, we unit test the code with a \[4, 3\] Tensor instead of a \[4, 4\] Tensor.

### Part 6: Improve Deep Learning model performance & tuning

Once we have our models debugged, we can focus on increasing the model capacity and hyperparameters tuning.

### Increase model capacity

We add layers, feature maps and hidden nodes to a DN gradually to increase its capacity. Deeper models produce more complex models. For convolutional layers, the key factors are:

* Number of convolution layers
* Number of output feature maps (channels)
* Filter sizes

We increase the number of convolution layers with smaller filter size. Smaller filters (3x3 or 5x5) usually performs better than larger filters. Each convolution layer will reduce or retain the spatial dimension while widen the depth of the channels. (Or vice versa in upsampling transposed convolution.)

<div class="imgcap">
<img src="/assets/cnn/cnn3d2.png" style="border:none; width:65%;">
</div>

The tuning process is more empirical than theoretical. An easier approach is to locate proven model designs for related problems and use their values as a starting tuning point. If none found, we start from a simple model and add layers and feature maps gradually to increase training accuracy and to overfit the model. This is manageable since we can tone down the overfitting with regularizations. We repeat such iterations until we have diminishing validation accuracy improvements to justify the drop in training and computation performance.

In the fully connected layer, we gradually decrease the hidden nodes to get closer to the output dimension. For example, the last FC layer below is reduced to 256 to classify objects into 10 classes. For my intuition, I often ask how much features (channels or hidden nodes) the machine needs at each layer to solve the problem? And this number is usually higher than you thought. However, GPUs do not page out memory. As of early 2018, the high end NVIDIA GeForce GTX 1080 TI has 11GB memory. The number of hidden nodes between two layers are in particular restricted by the memory size.

<div class="imgcap">
<img src="/assets/cnn/cnn3d5.png" style="border:none; width:65%;">
</div>

For very deep networks, the gradient diminishing problem is serious. We often add skip connection design (like the residual connections in ResNet) to mitigate the problem.

For fully connected layers, there are no filters but the tuning process is very similar.

* Number of fully connected layers
* Number of hidden units 

If the input features are lower than a hundred features, we may gradually increase the number of hidden nodes in the following layers before dropping it closer to the output dimension.

### Model & dataset design changes

Here is the checklist for the model design to improve performance:

* Analyze errors (bad predictions) in the validation dataset.
* Monitor the activation. Consider batch or layer normalization if it is not zero centered.
* Monitor the percentage of dead nodes.
* Apply gradient clipping (in particular NLP) to control exploding gradients.
* Shuffle dataset (manually or programmatically).
* Un-balance dataset (Different amount of datapoints for each class).

If the DN has a huge amount of dead nodes, we should trace the problem further. It can be a problem in the code, weight initializations or diminishing gradients. If none of them is the cause, experiment some advance ReLU functions like leaky ReLU.

### Dataset collection & cleanup

One of the most effective way to avoid overfitting and to improve performance is collecting more training data. We analyze the errors in our project. We find images with highly entangled structure perform badly. We can make change to the model like adding more convolution layers with smaller filters. We can also collect more entangled samples for further training. Alternatively, we can limit what the model can handle. Remove those samples (outliners) so the training can be more focus. Custom dataset requires much cleanup work than standard dataset.

### Data augmentation

For the labeled data in the supervising learning, collect new data can be expensive. For images, we can apply data augmentation with simple techniques like rotation, clipping, shifting, shear and flipping to create more samples from existing data. If low contrast images are not performing, we can augment existing images to include different level of contrast. 

<div class="imgcap">
<img src="/assets/dl/darg.png" style="border:none; width:70%;">
</div>

We can also supplement data with unsupervised pre-training. We use our model to classified non-labeled data. For samples with high confidence prediction, we add them to the dataset with the corresponding label predictions.


### Tuning

#### Learning rate tuning

Let's have a short recap on tuning the learning rate. In early development, we turn off or set them to zero for any un-critical hyperparameters including regularizations. With the Adam optimizer, the default learning rate usually works well. If we are confidence about the code but yet the loss does not drop, start testing the learning rate. Typical learning rate is from $$1$$ to $$1e^{-7}$$. Drop the rate each time by a factor of 10. Test it in short iterations. Monitor the loss closely. If it goes up consistently, the learning rate is too high. If it does not go down, the learning rate is too low. Increase it until it prematurely flattens.

<div class="imgcap">
<img src="/assets/dl/mont1.png" style="border:none; width:50%;">
</div>

This is a real example (source Emil Wallner) showing the learning rate is too high and cause a sudden surge in cost with the Adam optimizer:

<div class="imgcap">
<img src="/assets/dl/wallner.png" style="border:none; width:70%;">
</div>

A less often used practice, people monitor the updates to $$W$$ ratio:

$$
\frac{\| \alpha \cdot dw \|}{\| W \|} \quad \text{which } \alpha \text{ is the learning rate and } dw \text{ is the gradient.} 
$$

* If the ratio is $$ > 1e-3 $$, consider lower the learning rate.
* If the ratio is $$ < 1e-3 $$, consider increase the learning rate.

#### Hyperparameters tuning

Once the model design is stabled, we can tune the model further. The most tuned hyperparameters are:

* Mini-batch size
* Learning rate
* Regularization factors
* Layer specific hyperparameters (like dropout)

#### Mini-batch size

We use our first implementation as a baseline for comparison. You can use proven models as a baseline instead. Typical batch size is either 8, 16, 32 or 64. If the batch size is too small, the gradient descent will not be smooth. The model is slow to learn and the he loss may oscillates. If the batch size is too high, the time to complete one training iteration (one round of update) will be long with relatively small returns. For our project, we drop the batch size lower because each training iteration takes too long. We monitor the overall learning speed and the loss closely. If it oscillates too much, we know we are going too far. Batch size impacts hyperparameters like regularization factors. Once we determine the batch size, we usually lock the value.

#### Learning rate & regularization factors
 
We can tune our learning rate and regularization factors further with the approach mentioned before. We monitor the loss to control the learning rate and the gap between the validation and the training accuracy to adjust the regularization factors. Instead of changing the value by a factor of 10, we change that by a factor of 3 (or even smaller in the fine tuning). 

<div class="imgcap">
<img src="/assets/dl/mont2.png" style="border:none; width:50%;">
</div>

Tuning is not a linear process. Hyperparameters are related, and we will come back and forth in tuning hyperparameters. Learning rate and regularization factors are highly related and may need to tune them together sometimes. Do not waste time in fine tuning too early. Design changes easily void such efforts.

#### Dropout

The dropout rate is typically from 20% to 50%. We should start from 20%. Monitor the accuracy gap again until the model is just slightly overfit.

#### Other tuning

* Sparsity
* Activation functions

Model parameters' sparsity reduces power consumption in particular for mobile devices, and the computation is easier to optimize. If needed, we may replace (or add) the regularizations with the L1 regularization. ReLU is the most popular activation function. For some deep learning competitions, people experiment more advance variants of ReLU to move the bar slightly higher. It also reduce dead nodes in some scenarios.

#### Advance tuning

There are more advanced fine tunings. 

* Learning rate decay schedule
* Momentum
* Early stopping

Smaller projects may simply use the default values and ignore early stopping. 

Instead of a fixed learning rate, we can decay the learning rate regularly. The hyperparameters include how often and how much it drops. For example, you can have a 0.95 decay factor for every Epoch. To tune these parameters, we monitor the cost to verify it is dropping faster but not pre-maturely flatten.

Advance optimizers use momentum to smooth out the gradient descent. In the Adam optimizer, there are two momentum settings controlling the first order (default 0.9) and the second order (default 0.999) momentum. For problem domains with steep gradients like NLP, we may increase the value slightly to check whether cost drops faster.

Overfitting can be reduced by stopping the training when the validation errors increases persistently with increasing iterations of training. 

<div class="imgcap">
<img src="/assets/dl/ears.png" style="border:none; width:35%;">
</div>

[(source)](http://page.mi.fu-berlin.de/prechelt/Biblio/stop_tricks1997.pdf)

This is only a visualization of the concept. The real time error may go up temporarily and then drop again. We can checkpoint models regularly with the corresponding validation errors and select the correct model later.

#### Grid search for hyperparameters

Some hyperparameters are strongly related. We should tune them together with a mesh of possible combinations in a logarithmic scale. For example, for 2 hyperparameters $$\lambda$$ and $$\gamma$$, we start from the corresponding initial value and drop it by a factor of 10 in each step:

* ($$e^{-1}, e^{-2}, … $$ and $$e^{-8}$$) and,
* ($$e^{-3}, e^{-4}, ... $$ and $$e^{-6}$$). 

The corresponding mesh will be $$[(e^{-1}, e^{-3}), (e^{-1}, e^{-4}), \dots , (e^{-8}, e^{-5}) $$ and $$(e^{-8}, e^{-6})]$$. 

Instead of using the exact cross points, we randomly shift those points slightly. This randomness may lead us to some surprises that otherwise hidden. If the optimal point lays in the border of the mesh (the blue dot), we retest it further in the boarder region.

<div class="imgcap">
<img src="/assets/dl/grid.png" style="border:none;width:30%">
</div>

Grid search is computational intense. For smaller projects, this is used sporadically. We can start tuning parameters in coarse grain using fewer training iterations. To fine tune the result, we use longer iterations and we drop each value by a factor of 3 (or even smaller).

### Model ensembles

In machine learning, we can take votes from a number of decision trees to make predictions. It works because mistakes are often localized: there is a smaller chance for two models making the same mistakes. In DL, we start training with random guesses (providing random seeds are not explicitly set) and the optimized models are not unique. We pick the best models after many runs using the validation dataset. We take votes from those models to make final predictions. This method requires running multiple sessions, and can be prohibitively expensive. Alternatively, we run the training once and checkpoints multiple models. We pick the best models from the checkpoints. With ensemble models, the predictions can based on:

* one vote per model, 
* weighted votes based on the confidence level of it's prediction.

Model ensembles are very effective in pushing the accuracy up a few percentage points in some problems and very common in some DL competitions.

### Model improvement

Instead of fine tuning a model, we can try out different model variants to leap frog the model performance. For example, we have consider replacing the color generator partially or completely with a LSTM based design. This concept is not completely foreign: we draw pictures in steps. 

[Image source](http://webneel.com/how-draw-faces-drawings)

<div class="imgcap">
<img src="/assets/gm/face.png" style="border:none; width:30%;">
</div>

Intuitively, there are merits in introducing a time sequence method in image generation. This method has proven with some success in DRAW: A Recurrent Neural Network For Image Generation.


#### Fine tune vs model improvement

Major breakthroughs require model design changes. However, some studies indicate fine tuning a model can be more beneficial than making incremental model changes. The final verdicts are likely based on your own benchmarking results.

### Kaggle

You may have a simple question like should I use Leak ReLU. It sounds so simple but you will never get a straight answer anywhere. Some research papers may show empirical proof that leaky ReLU is superior, but yet some projects see no improvement. There are too many variables and many projects do not have the resources to benchmark even a resonable portions of the possibilities. [Kaggle](https://www.kaggle.com) is an online platform for data science competitions including deep learning. Dig through some of the competitions. You can find what are the most common performance metrics. Some teams publish their code (called kernels) and with some patience, it is a great source of information.


### Experiment framework

DL requires many experiments and tuning hyperparameters is tedious. Creating an experiment framework can expedite the process. For example, some people develop code to externalize the model definitions into a string for easy modification. Those efforts are usually counter productive for a small team. I personally find the drop in code simplicity and traceability is far worsen than the benefit. Those coding makes simple code modification harder that it should. Easy to read code has less bugs and more flexible. Many AI cloud offerings start providing automatic hyperparameters tuning. It is still in early development but should be a general trend that we do not code the framework ourself. Stay tuned for any development!






