---
layout: post
comments: true
mathjax: true
title: “Memory network”
excerpt: “Use a memory network to make a 'question & Answer' session more conversational.”
date: 2017-03-15 11:00:00
---
**This is work in progress...**

### Memory network

Virtual assistances are pretty good at answering a one-line query, but hardly carry out a conversation. Here is a verbal exchange demonstrating the challenge ahead of us:

Me: Can you find me some restaurants?
Assistance: I found a few places within 0.25 mile. The first one is Caffé Opera on Lincoln Street. The second one is ...
Me: Can you make a reservation at the first restaurant? 
Assistance: Ok. Let's make a reservation for the Sushi Tom restaurant on the First Street.

Why can't the virtual assistance follow my instruction to book the Caffé Opera? Actually, the virtual assistance does not remember our conversation and simply response my second one-line query with its best effort. Therefore, it finds me a restaurant that matches the word "first". Memory networks address the issue using the computer's memory to record information in a conversation.

A memory network consists of a memory m to store information and 4 components I, G, O and R as follows:
* Input feature map (I)
* Generalization (G)
* Output feature map (O)
* Response (R)

> The information for memory networks in this article is from 2 technical papers [Memory networks, Jason Weston etc.](https://arxiv.org/pdf/1410.3916.pdf) and [End-To-End Memory Networks, Sainbayar Sukhbaatar etc.](https://arxiv.org/abs/1503.08895)

### Input feature map (I)



