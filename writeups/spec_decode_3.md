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

This will detect if uv package manager is installed in your system or not, then it will install the environment and in a virtual environment called env. 

**General Process**

Our process is not that complicated. We first train the target models using the datasets generated in our previous post ^[1]. After fine tuning the model we generate a distilled dataset using our target model. We then fine-tune our draft model on the distilled dataset before finally running our tests and evaluations.

You might ask why would we train the draft models on the distilled datasets instead of the original datasets that were used to train the target model, the reason is for the draft model we mostly care about alignment. If we train it on the original dataset it might develop a different thinking process than the target model, causing the target model to reject it's output and slowing down our speculative decoding.

So remember that for speculative decoding, we care about our target model being correct, and our draft model thinking like our target model, even if that makes it less accurate than if it was trained on the original dataset. Alignment is key here.

## Training the Target Model

Our method of improving speculative decoding is unique in the sense that previous research (at the time of writing) focuses more on the draft model itself, whereas our experiments hypothesize that an "easier to understand" target model improves acceptance in speculative decoding. 

That is why we fine tune, we use Qwen3-14B as our target model, while there are newer models in the Qwen family the experiments done here can translate to any model family. We just need to make sure the target and draft model match on the same vocabulary (tokenizer).

We are using an A40 GPU (40gb VRAM) to train the model. We also use LoRA (Low Rank Adaptation), a parameter efficient fine-tuning method that updates a low rank approximation of the weight matrix instead of the full weight matrix, reducing our VRAM requirement for a full fine-tune from approximately 100gb for a 14B model to 32-35gb.

We are also using Unsloth in Python, which uses custom kernels for attention and a chunked cross entropy loss computation to lower VRAM requirements and improve training speedup. The latter is specifically helpful for Qwen3's large vocabulary, where standard loss materializes a tensor too large to fit comfortably alongside model weights.

*NOTE*: It's possible to reduce VRAM requirements further by using QLoRA (Quantized LoRA), which applies 4 bit quantization to the base model using NF4 before applying LoRA, however as the A40 has enough VRAM we just use normal LoRA. If you think compute overhead is worth lowering VRAM requirements go for it, however be advised that the quantization method needs to work with vLLM otherwise it won't be able to read the fine-tuned model. 

**Model Parameters**

For our model parameters, we set the following values, which can be found in `configs` folder. Our target model parameters are in the `config\target_14b.yaml` and has the following main settings

* `max_seq_length`: This decides the max sequence output and input of the model. We use 2048 to avoid OOM, but also we unfortunately truncate 2 samples from the Chain of Thought Hard Scenario (0.4%)
* `lora_target_modules`: As we want to change the way the model reasons, we change all attention layers and feed-forward MLP layers, giving our LoRA more surface to work with
* `lora_r`: LoRA Rank, how expressive the adaptation is. We set it to 16, as we worry 32 and 64 might cause overfitting
* `lora_alpha`: LoRA Alpha, the magnitude of the update applied to the frozen weight. As we want higher magnitude we set it to 32, amplifying the adapter's influence as we want to apply significant style change.
* `num_train_epochs`: 3, As mentioned in [^4], small curated datasets benefit  from few epochs and show diminishing returns beyond that.
* `per_device_train_batch_size`: 2 & `gradient_accumulation_step`: 4, making the effective batch size 2 * 4 = 8, memory management decision
* `optim`: adamw_8bit, another memory optimization. AdamW's adaptive learning rates suit transformers, and the 8 bit version cuts optimizer state memory by 4x with negligible accuracy impact.

**Training Curves and Memory**

I've used weights and biases to monitor model training, here are the loss curves for each model as well as gpu memory usage

### Generating Distilled Dataset







## References
[^1]:[Chain of Draft Speculative Decoding 2: Dataset Generation and Evaluation](/slug_decode_2)
[^2]:[Unsloth](https://unsloth.ai/docs)
[^3]:[vLLM](https://docs.vllm.ai/en/stable/)
[^4]:[LIMA: Less is More for Alignment](https://arxiv.org/pdf/2305.11206)
