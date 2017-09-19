---
layout: post
comments: true
mathjax: true
priority: 790
title: “Install CUDA, CuDNN & TensorFlow in AWS EC2 P2”
excerpt: “Install CUDA, CuDNN & TensorFlow in AWS EC2 P2”
date: 2017-03-07 14:00:00
---

### EC2 P2 spot instance with Deep Learning AMI 
Amazon EC2 instances with GPU can be very expensive. Therefore, we pick the spot instances to reduce the price. Nevertheless, spot instances can be terminated when the bidding price falls below the current price. To mitigate that, we can create and attach another EBS volume to the instance. We use this additional volume to store training data, results and checkpoints so the files are not deleted when the instance is terminated. 

> Different AWS regions have different pricing. Usually N. Virginia, Ohio, Oregon have cheaper price. Terminate the instance when it is not needed.

We select an Amazon AMI for deep learning and the cheapest instance with GPU for deep learning is **p2.xlarge**. Currently, p2.xlarge costs about $0.2 per hour.

#### p2.xlarge and Deep Learning AMI

In the AWS console, we create a new p2.xlarge spot instance.

<div class="imgcap">
<img src="/assets/tensorflow/s1.png" style="border:none;width:120%">
</div>

The pre-built deep learning AMI can be located [here]( https://aws.amazon.com/marketplace/search/results?x=12&y=20&searchTerms=Deep+Learning+AMI&page=1&ref_=nav_search_box). To select the AMI, we click on "Select for AMI" and enter the AMI ID found in the link above.

<div class="imgcap">
<img src="/assets/tensorflow/s2.png" style="border:none;width:80%">
</div>

#### Network setting

We keep the default for the network.

<div class="imgcap">
<img src="/assets/tensorflow/s3.png" style="border:none;width:100%">
</div>

#### Storage

In the next page, we can keep the setting in the storage and instance details. A 50 GiB EBS volume is created.

<div class="imgcap">
<img src="/assets/tensorflow/s4.png" style="border:none;width:120%">
</div>

> We can attach another EBS volume to store permanent data later.

#### Keypair and security

We will need to configure a key and a security group for the new instance:
<div class="imgcap">
<img src="/assets/tensorflow/s5.png" style="border:none;width:120%">
</div>

Create a new key pair:

<div class="imgcap">
<img src="/assets/tensorflow/key.png" style="border:none;width:80%">
</div>

Create a security group:

<div class="imgcap">
<img src="/assets/tensorflow/s6.png" style="border:none;width:80%">
</div>

#### Review

The final configuration should look similar to:

<div class="imgcap">
<img src="/assets/tensorflow/s7.png" style="border:none;width:80%">
</div>

<div class="imgcap">
<img src="/assets/tensorflow/s8.png" style="border:none;width:80%">
</div>

#### ssh

We copy our private key _tf.pem_ into our home directory and configure the ssh access:

```sh
cd ~/.ssh
mv ../tf.pem .
chmod og-r tf.pem
```

Edit the ssh configuration
```sh
vi config
```
 
Replace the hostname with your public IP address:
 
```
Host tf
    Hostname xx.xx.xx.xx
    Port 22
    User ec2-user
    ServerAliveInterval 60
    IdentityFile ~/.ssh/tf.pem
```

To access the spot instance:
```
ssh tf
```

### Testing

We should see K80 as the GPU when we run nvidia-smi.

```sh
$ nvidia-smi
Tue Sep 19 18:36:16 2017
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 375.66                 Driver Version: 375.66                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla K80           On   | 0000:00:1E.0     Off |                    0 |
| N/A   64C    P8    30W / 149W |      0MiB / 11439MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID  Type  Process name                               Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

Run a TensorFlow application in the TensorFlow distribution:

```sh
git clone https://github.com/tensorflow/tensorflow.git
```

Run the MNist application which will take a little bit time (a couple minute) for the application to download the MNIst dataset.
```sh
python3 tensorflow/tensorflow/examples/tutorials/mnist/fully_connected_feed.py
Successfully downloaded train-images-idx3-ubyte.gz 9912422 bytes.
Extracting /tmp/tensorflow/mnist/input_data/train-images-idx3-ubyte.gz
Successfully downloaded train-labels-idx1-ubyte.gz 28881 bytes.
Extracting /tmp/tensorflow/mnist/input_data/train-labels-idx1-ubyte.gz
Successfully downloaded t10k-images-idx3-ubyte.gz 1648877 bytes.
Extracting /tmp/tensorflow/mnist/input_data/t10k-images-idx3-ubyte.gz
Successfully downloaded t10k-labels-idx1-ubyte.gz 4542 bytes.
Extracting /tmp/tensorflow/mnist/input_data/t10k-labels-idx1-ubyte.gz
2017-09-19 18:39:57.043575: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2017-09-19 18:39:57.043605: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2017-09-19 18:39:57.043611: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
2017-09-19 18:39:57.043615: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
2017-09-19 18:39:57.043619: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
2017-09-19 18:39:58.214521: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:893] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2017-09-19 18:39:58.215033: I tensorflow/core/common_runtime/gpu/gpu_device.cc:955] Found device 0 with properties:
name: Tesla K80
major: 3 minor: 7 memoryClockRate (GHz) 0.8235
pciBusID 0000:00:1e.0
Total memory: 11.17GiB
Free memory: 11.11GiB
2017-09-19 18:39:58.215056: I tensorflow/core/common_runtime/gpu/gpu_device.cc:976] DMA: 0
2017-09-19 18:39:58.215062: I tensorflow/core/common_runtime/gpu/gpu_device.cc:986] 0:   Y
2017-09-19 18:39:58.215069: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1045] Creating TensorFlow device (/gpu:0) -> (device: 0, name: Tesla K80, pci bus id: 0000:00:1e.0)
Step 0: loss = 2.32 (7.246 sec)
Step 100: loss = 2.18 (0.002 sec)
Step 200: loss = 1.95 (0.002 sec)
Step 300: loss = 1.75 (0.002 sec)
Step 400: loss = 1.29 (0.002 sec)
Step 500: loss = 1.09 (0.002 sec)
Step 600: loss = 0.87 (0.001 sec)
Step 700: loss = 0.77 (0.001 sec)
Step 800: loss = 0.67 (0.002 sec)
Step 900: loss = 0.54 (0.002 sec)
Training Data Eval:
  Num examples: 55000  Num correct: 46762  Precision @ 1: 0.8502
Validation Data Eval:
  Num examples: 5000  Num correct: 4274  Precision @ 1: 0.8548
Test Data Eval:
  Num examples: 10000  Num correct: 8560  Precision @ 1: 0.8560
Step 1000: loss = 0.45 (0.007 sec)
Step 1100: loss = 0.50 (0.101 sec)
Step 1200: loss = 0.48 (0.002 sec)
Step 1300: loss = 0.46 (0.001 sec)
Step 1400: loss = 0.46 (0.001 sec)
Step 1500: loss = 0.53 (0.002 sec)
Step 1600: loss = 0.54 (0.002 sec)
Step 1700: loss = 0.39 (0.001 sec)
Step 1800: loss = 0.50 (0.002 sec)
Step 1900: loss = 0.45 (0.001 sec)
Training Data Eval:
  Num examples: 55000  Num correct: 49169  Precision @ 1: 0.8940
Validation Data Eval:
  Num examples: 5000  Num correct: 4526  Precision @ 1: 0.9052
Test Data Eval:
  Num examples: 10000  Num correct: 9013  Precision @ 1: 0.9013
```