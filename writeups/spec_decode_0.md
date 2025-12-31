---
title: On Chain of Draft Speculative Decoding and how to speed up your LLMs - 0
author: Hassan Sarwat
pubDatetime: 2025-12-28T20:12:32Z
slug: spec_decode_0
featured: true
draft: false
tags:
  - LLMs
  - Technical
  - Speculative Decoding
  - Chain of Draft
description:
  "The first part of a series explaining speculative decoding and my plan on applying a chain of draft instead of chain of thought when training reasoning models."
---

# Introduction

### Why I started this series

I have recently finished my master's degree at TUM, and seeing that I'm unemployed (at the time of writing this blog, if this is not crossed out or deleted feel free to send me opportunities) and with more free time on my hand I've decided to explore and enhance my knowledge, and also satiate my curiosity and desire to continously know things so I can pretend to be better than everyone else by throwing around esoteric terminologies.

Recently I was reading up on fine-tuning models and their deployments, and I realized that while I do know a decent amount about fine-tuning LLMs and the different shortcuts you can take, from few shot prompting to parameter efficient fine tuning methods such as quantization or low rank adapters, I don't really know a lot about inference. 

I was working on a chatbot with friends and when we considered deploying our own fine-tuned models I became acutely aware of how expensive maintaining it would be, not just that, when building a chatbot something like latency is very crucial to user experience. Given that there are already tens of thousands of AI applications and services, we knew that in order to survive our responses needed to be fast, really fast, faster than your response time to someone that doesn't like you back. While the problem might seem solvable by throwing more GPUs at it, we didn't want to burn through our limited funds then hope that someone will give us more money that we can then burn through as well.


<figure style="text-align: center;">
  <img src="/assets/images/gpu_cost_soldier_image.jpg" alt="GPU Cost Soldier" width="50%">
</figure>

My thought process was as follows, *if people have sped up and improved LLM training, then surely something exists for LLM inference.* And I was correct. Having spent the last 3 years mostly working on pure research during my master's, inference speed was not a metric I cared for a lot. Very rarely was it mentioned in most papers and if it was it was written in the appendix, people usually cared more about being correct than being fast.

So I searched, and checked what methods are there for me to employ and how can I speed up inference, this is where I found speculative decoding[^1], and the further improvements such as Medusa[^2] and EAGLE[^3], with EAGLE-3[^4] currently being SOTA for inference improvements. 

Nevertheless I decided to go with speculative decoding, mainly due to the model agnosticism and freedom provided versus the being limited to a selection of models if I decided to go with the other methods, maybe later in the series I'll talk about them and their implementation but for now we start with Speculative Decoding. So.

### What is Speculative Decoding?

And how can it make my inference faster? Speculative decoding and other methods follow a very similar way of thinking. Why have the very smart but slow and deliberate Professor do the work when you can train a faster but dumber Student to act like it, and then the Professor just needs to approve and/or correct the work.

The way standard speculative decoding does it is by using a smaller draft model. For example, in this code tutorial we fine-tune a Qwen2.5-14B-Instruct as our target model (Professor) and use a smaller Qwen2.5-0.5B-Instruct model as our draft model (Student). Medusa on the other hand works by training multiple heads and attaching them to the same model. So if we used Medusa our Qwen2.5-14B-Instruct would have the main head to predict token N, and additional K heads that will be used to predict tokens from N+1 to N+K+1 ([^2] recommends K=5). Where standard speculative decoding is trained on the output of the target model, EAGLE is a lightweight transformer that is trained on the second to top layer embeddings of the target model before the output, learning semantic context that wouldn't be possible otherwise. 

Here's a figure from the original EAGLE [^3] paper showing the comparisons 

<figure style="text-align: center;">
  <img src="/assets/images/eagle_paper_image_comparisons.png" alt="Comparisons" width="100%">
  <figcaption>A visualization of how different methods perform speculative decoding</figcaption>
</figure>


That was a very brief literature review section and we can now resume with the speculative decoding, and the math basics. We can think of our target model, the 14B parameter, as a Professor, extremely smart and accurate but a lot of thought and intent behind every word. Then we have our 0.5B model, we can think of it as a bachelor's student, they type a lot and don't really put much thought behind each word.

In standard decoding you will have the Professor write up the entire response, word by word, taking forever. In Speculative decoding you have the student write the report, and the professor can have a look at N words at a time and then either accept, or reject and suggest edits. 

**The Mathematics of speculative decoding**

Like any good maths explanation, we first have to start by defining what our variables are, we have:

* **$T_{target}$**: The time taken for 1 forward pass from the target model
* **$T_{draft}$**: The time taken for 1 forward pass from the draft model
* **$K$**: The number of tokens the draft model guesses ahead (e.g. 5)
* **$\alpha (Alpha)$**: The acceptance rate

Standard decoding generates 1 token in time $T_{target}$. Speculative decoding takes a gamble step which costs

$$
\begin{aligned}
Cost_{step}=(K\cdot T_{draft}) + T_{target}
\end{aligned}
$$

Or in English, time to draft K tokens + time to verify them, this single step is expected to generate

$$ 
\begin{aligned}
E[\text{tokens}] = 1 + K\alpha
\end{aligned}
$$

Basically the first token, then the number of tokens generated by the draft multiplied by acceptance rate

Therefore the expected time per token in speculative decoding is the cost per step/expected tokens.

$$
\begin{aligned}
Time_{spec} = \frac{(K\cdot T_{draft}) + T_{target}}{1 + K \alpha}
\end{aligned}
$$

So the best way to minimize the $Time_{spec}$ is to minimize the $Cost_{step}$, which can be achieved by having smaller draft models and to maximize the $E[\text{tokens}]$, which can be achieved by either higher acceptance rate and/or predicting more tokens

This was just a brief introduction to the math of speculative decoding, if you are interested in reading more and maths is your thing, I've attached the references at the bottom, this is probably the deepest I'll dive into the maths in this series but I am also open to requests, just contact and ask and I'll be happy to give an explanation.

### What do we want to do

For this project I wanted to apply speculative decoding as a learning exercise, and also experiment a bit. I've been reading up on reasoning models and realized that many of them are trained on Chain of Thoughts, which can sometimes ramble on a bit too much. 

My hypothesis is as follows, can we train a reasoning model on a summarized chain of thought, also called chain of draft, and teach a model to reason with less steps without affecting accuracy? 

As always when starting a project you need to check if someone had a similar idea, and yes, There's already a paper published earlier this year in February 2025 called **Chain of Draft: Thinking Faster by Writing Less**[^5], however their strategy involves prompting whereas ours involves fine-tuning. Our project also investigates the effect of chain of draft on speculative decoding.

**So for this project we have the following hypotheses:**
* Fine-tuning the model on reasoning will improve its accuracy
* Speculative decoding will reduce our inference time without affecting accuracy
* Chain of draft will reduce our total tokens with only minimal effect on accuracy

The first two hypotheses are self explanatory, the third hypothesis needs a bit more clarification. How do we define chain of draft, a summary is not exact enough for our very scientific brains. In this experiment our definition will be as follows. **A chain of draft must be 3-5 steps where each step has a maximum of 5 words.**

Why did we pick this definition, what is the reason behind it? First, chain of thought includes a lot of rambling, more rambling means more tokens and more tokens mean slower speed. The second reason is because when doing speculative decoding, we want the target model to accept the tokens of the draft model, and the longer the sequences of tokens the higher chance of divergence and therefore more rejections from the main model.


<figure style="text-align: center;">
  <img src="/assets/images/why-waste-time-when-few-word-do-trick.gif" alt="Comparisons" width="75%">
</figure>

For now, we will purely focus on the implementation of speculative decoding. In the later parts of the series we will do the tests and analysis for chain of draft, it's improvement over chain of thought, how much tokens were saved, was the accuracy different, how much faster is it? etc...

So now that we have our problem, what are we going to do? In the rest of this blog I will explain the design of the experiments and also spoilers for the rest of the series. If it's a link know that it's ready to be viewed and implemented

### The Plan

1. Dataset generation & Analysis
2. Target & Draft Model Training
3. Inference & Evaluation

**Dataset Generation**

In this part of the series I'll walk you through which datasets to be picked for our hypotheses, how to generate your own chain of draft/thought datasets using Batch API from gemini, how to analyze and make sure the datasets are clean, and how to analyze the datasets

**Target & Draft Model Training**

This part will involve deciding which models to train on which datasets, as this is a hobby project and the GPU/training cost is from my own money, I'll stick to the bare minimum to prove the hypotheses mentioned above, but as mentioned above we will be training a Qwen2.5-14B-Instruct (target) on both chain of draft and chain of thought reasoning, and finally we will be distilling the reasoning of those models and using it to train a Qwen2.5-0.5B-Instruct (draft) models. As a list, it will be as follows, we will train the following:

1. Chain of Thought Qwen2.5-14B-Instruct (Target CoT)
2. Chain of Draft Qwen2.5-14B-Instruct (Target CoD)
3. Chain of Thought Qwen2.5-0.5B-Instruct (Draft CoT)
4. Chain of Draft Qwen2.5-0.5B-Instruct (Draft CoD)

**Inference and Evaluation**

In the final blog post of the series we will compare our trained models against each other and against untrained models. We will talk about which metrics to use and why, and also how to measure said metrics. Finally after evaluation we will identify whether our hypotheses were correct or not, and if not, why? What went wrong and what can be improved.


## References
[^1]: [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/pdf/2211.17192)
[^2]: [Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/pdf/2401.10774)
[^3]:[EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/pdf/2401.15077)
[^4]:[EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/pdf/2503.01840)
[^5]:[Chain of Draft: Thinking Faster by Writing Less](https://arxiv.org/pdf/2502.18600)
