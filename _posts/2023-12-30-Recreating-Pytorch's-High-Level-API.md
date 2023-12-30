---
title: "BadTorch Part 1: Motivating Recreating PyTorch's High-Level API "
tags:
    - PyTorch
    - BadTorch
    - Deep Learning
---
This post will outline the motivation for my "BadTorch" series of blog posts. It is also my first blog post (!!!) after procrastinating on starting a blog for literal years.

### Background

A few weeks ago, I was experimenting with training  basic LSTMs for speech recognition using PyTorch. Now, I have been using PyTorch for ~2 years and thought that I was pretty decent at it. After all, I thought, I had used PyTorch for tasks ranging from fine-tuning foundation models to playing with adversarial training - surely I would be able to use PyTorch to implement any useful ideas I came across in the literature? This ended up being completely wrong as I found out when I investigated the use of Dropout in recurrent models. 

Without going in to much detail about this now (I'm planning to explain this further in a follow-up post) PyTorch's Dropout implementation leaves much to be wanting when applied to reccurent models. For one, PyTorch's implementation of LSTMS only allows the user to apply Dropout to the outputs of recurrent layers, not to recurrent-to-reccurent updates. Additionally, PyTorch applies a different Dropout mask to the output at each timestep which seems somewhat at odds with Dropout's theoretical justification.  

This proved problematic as, for the life of me, I could not figure out how to implement a Dropout that fixed both of these problems (at least one that didn't make models prohibitively slow to train). Experiencing this made me realise that my understanding of PyTorch left much to be desired, and motivated me to fix this. BadTorch is, therefore my re-creation of PyTorch's high-level API (modules, optimisers etc...) using only Pytorch tensors. The goal of BadTorch is educational such that I avoid some of PyTorch's tools for improving efficiency. This leads to code that would be less-than-perfect for industrial use (hence the name) but that is easily sufficient for experimenting with additions such as adding the forms of Dropout described above.

### Going Forward

Overall, this has been an incredibly useful exercise so far. In this series of "BadTorch" blog posts I will walk through what the process of finishing this exercise, including some pointers to things that I found particular useful or intersting in case any one else wants to have a go at doing this. All code can be found in the repo [here](https://github.com/tuphs28/BadTorch).
