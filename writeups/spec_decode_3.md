---
title: On Chain of Draft Speculative Decoding and how to speed up your LLMs - 3
author: Hassan Sarwat
pubDatetime: 2026-01-05T04:12:32Z
slug: spec_decode_3
featured: true
draft: false
tags:
  - LLMs
  - Technical
  - Speculative Decoding
  - Chain of Draft
  - Fine-tuning LLMs
  - Knowledge Distillation
  - Unsloth
  - vLLM
description:
  "The third part of the series explaining speculative decoding and my plan on applying a chain of draft instead of chain of thought when training reasoning models. This part dives into target and draft model training"
---

# Model Training

Welcome to the third post of the chain of draft speculative decoding series, where we discuss model training. The other parts of the blog can be found here

1. [Part 1: Introduction](/slug_decode_1)
2. [Part 2: Dataset Generation and Evaluation](/slug_decode_2)


The post will have the following structure
 
0. Connecting to Online GPUs
1. Environment set up
2. Target Model Training
3. Knowledge Distillation
4. Draft Model Training

Note: the machine used for this blog had an Nvidia L4 GPU (24GB VRAM) and 4 vCPUs with 16GB memory.

## Connecting to online GPUs

This is a small walkthrough for google cloud platform and deployment for model training. If you are using other services like [Runpod](runpod.io) or [Vaast.ai](vaast.ai) or even AWS then this section probably won't be as meaningful to you. 

This will be a quick section, if you remember in the last post I mentioned the free credits from google, we will continue to utilize those and deploy our machines on google cloud platform.

First, if you open the console, and type in deep learning VM, click on that, then click launch, you will be directed to this fun menu where you can select a region, and after that selection you can select GPU and CPU types. For this experiment I've used an `Nvidia L4` with 24GB of VRAM and a `g2-standard-4` machine with 4 vCPUs and 16GB Memory, and finally 200gb of drive space. Mainly because I wouldn't need to keep the machine running for more than a few days and didn't want to worry about having to delete models.

If you want to check which regions have the above mentioned specifications feel free to run this command in the google command line interface

```bash
gcloud compute accelerator-types list   --filter="name=nvidia-l4 AND zone:(europe-west*)"
```
Note: Edit europe-west to be whatever region you want

After this process is done and you've given a name to your machine, you are welcome to click on launch and then once again pray to our machine overlords that google doesn't say there are no machines available. I've had to keep trying around 5 different regions before one of them finally deployed, but as the saying goes, beggars can't be choosers. I am using the free credits so I'll stop my complaining here.

Now that you're lucky and the machine has deployed, it's time to ssh into it.

## Environment set up

For this project we use UV, a python package manager that uses rust. It's a lot faster than pip or conda and less of a hassle than poetry. I thought about using and deploying docker machines but a lot of online hosted GPU services are actual docker machines, and Docker-in-Docker is just pure ass, so I've decided to use uv and conveniently set up a bash script for you that will install both environments needed.

That's right, BOTH, there are two of them. The first one is Unsloth which we need for our training, 


