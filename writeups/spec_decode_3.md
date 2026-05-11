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
 
0. Introduction
1. Training the Target Model
2. Generating Distilled Data
3. Training the draft model
4. Evaluation Design
5. Results
6. Conclusion

## Introduction

In this section I'll talk about the training part and the evaluation, mainly what was used and why was it used in comparison to other methods. But first

**Environment setup**

The GPU used in this experiment was an A40 (40gb VRAM), the models selected were Qwen3-14B for target and Qwen3-0.6B for draft. Nevertheless this is a model agnostic method, and the model can be adjusted as is suitable for your case. 

Regarding libraries the specific versions used can be found in the `requirements.txt` file.

To set up the environment we use UV, a python package manager that is a bit faster than most. It's fine if you don't have it just simply run the script

```bash
bash scripts/uv_setup_envs.sh
```

This will detect if uv package manager is installed in your system or not, then it will install the environment and in a virtual environment called env. This was a personal choice for comfort, not a 

**General Process**

Our process is not that complicated. We first train the target models using the datasets generated in our previous post ^[1]. After fine tuning the model we generate a distilled dataset using our target model. We then fine-tune our draft model on the distilled dataset before finally running our tests and evaluations.

You might ask why would we train the draft models on the distilled datasets instead of the original datasets that were used to train the target model, the reason is for the draft model we mostly care about alignment. If we train it on the original dataset it might develop a different thinking process than the target model, causing the target model to reject it's output and slowing down our speculative decoding.

So remember that for speculative decoding, we care about our target model being correct, and our draft model thinking like our target model, even if that makes it less accurate than if it was trained on the original dataset. Alignment is key here.

## Training the Target Model

Our method of improving speculative decoding is unique in the sense that previous research (at the time of writing) focuses more on the draft model itself, whereas our experiment hypothesize that an "easier to understand" target model improves acceptance in speculative decoding. 

That is why we fine tune, we use Qwen3-14B as our target model, while there are newer models in the Qwen family the experiments done here can translate to any model family. We just need to make sure the target and draft model match on the same vocabulary (tokenizer).

As this is a hobby project and we are using an A40 GPU (40gb VRAM) to train the model. we apply LoRA,(Low Rank Adaption) a parameter efficient fine-tuning method. It's also possible to use QLoRA (Quantized LoRA) however as quantization can have effects on the target model performance we didn't want it to affect the integrity of the experiment and stuck with traditional LoRA. We also used the Unsloth[^2] Library to train our model, utilizing it's lower VRAM requirements and training speedup.

**Model Parameters**







## References
[^1]:[Chain of Draft Speculative Decoding 2: Dataset Generation and Evaluation](/slug_decode_2)
[^2]:[Unsloth](https://unsloth.ai/docs)
[^3]:[vLLM](https://docs.vllm.ai/en/stable/)

