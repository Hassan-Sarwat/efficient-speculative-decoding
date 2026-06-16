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

# Model Training and Evaluation

Welcome to the third and final post of the chain of draft speculative decoding series. The other parts of the blog can be found here

1. [Part 1: Introduction](/slug_decode_1)
2. [Part 2: Dataset Generation and Evaluation](/slug_decode_2)


The post will have the following structure
 
0. Introduction
1. Training the Target Model
2. Generating Distilled Data
3. Training the draft model
4. Evaluation 
5. Conclusion

## Introduction

In this section I'll talk about the training part and the evaluation, mainly what was used and why in comparison to other methods. But first

**Environment setup**

The GPU used in this experiment was an A40 (40gb VRAM), the models selected were Qwen3-14B for target and Qwen3-0.6B for draft. Nevertheless this is a model agnostic method, and the model can be adjusted as is suitable for your case. 

Regarding libraries the specific versions used can be found in the `requirements.txt` file.

To set up the environment we use UV, a python package manager that is a bit faster than most. It's fine if you don't have it just simply run the script

```bash
bash scripts/uv_setup_envs.sh
```

This will detect if uv package manager is installed in your system or not, then it will install the dependenciesinto a virtual environment called env. 

**General Process**

Our process is not that complicated. We first train the target models using the datasets generated in our previous post ^[1]. After fine tuning the model we generate a distilled dataset using our target model. We then fine-tune our draft model on the distilled dataset before finally running our tests and evaluations.

You might ask why would we train the draft models on the distilled datasets instead of the original datasets that were used to train the target model, the reason is for the draft model we mostly care about alignment. If we train it on the original dataset it might develop a different thinking process than the target model, causing the target model to reject its output and slowing down our speculative decoding.

So remember that for speculative decoding, we care about our target model being correct, and our draft model thinking like our target model, even if that makes it less accurate than if it was trained on the original dataset. Alignment is key here.

## Training the Target Model

Our method of improving speculative decoding is unique in the sense that previous research (at the time of writing) focuses more on the draft model itself, whereas our experiments hypothesize that an "easier to understand" target model improves acceptance in speculative decoding. 

That is why we fine tune, we use Qwen3-14B as our target model, while there are newer models in the Qwen family, the experiments done here can translate to any model family. We just need to make sure the target and draft model match on the same vocabulary (tokenizer).

We are using an A40 GPU (40gb VRAM) to train the model. We also use LoRA (Low Rank Adaptation), a parameter efficient fine-tuning method that updates a low rank approximation of the weight matrix instead of the full weight matrix, reducing our VRAM requirement for a full fine-tune from approximately 100gb for a 14B model to 32-35gb.

We are also using Unsloth[^2] in Python, which uses custom kernels for attention and a chunked cross entropy loss computation to lower VRAM requirements and improve training speedup. The latter is specifically helpful for Qwen3's large vocabulary, where standard loss materializes a tensor too large to fit comfortably alongside model weights.

*NOTE*: It's possible to reduce VRAM requirements further by using QLoRA (Quantized LoRA), which applies 4 bit quantization to the base model using NF4 before applying LoRA, however as the A40 has enough VRAM we just use normal LoRA. If you'd rather trade some compute overhead for lower VRAM requirements, go for it, however be advised that the quantization method needs to work with vLLM otherwise it won't be able to read the fine-tuned model. 

**Model Parameters**

For our model parameters, we set the following values, which can be found in `configs` folder. Our target model parameters are in the `config\target_14b.yaml` and has the following main settings

* `max_seq_length`: This decides the max sequence output and input of the model. We use 2048 to avoid OOM, but also we unfortunately truncate 2 samples from the Chain of Thought Hard Scenario (0.4%)
* `lora_target_modules`: As we want to change the way the model reasons, we change all attention layers and feed-forward MLP layers, giving our LoRA more surface to work with
* `lora_r`: LoRA Rank, how expressive the adaptation is. We set it to 16, as we worry 32 and 64 might cause overfitting
* `lora_alpha`: LoRA Alpha, the magnitude of the update applied to the frozen weight. As we want higher magnitude we set it to 32, amplifying the adapter's influence as we want to apply significant style change.
* `num_train_epochs`: 3, As mentioned in [^4], small curated datasets benefit  from few epochs and show diminishing returns beyond that.
* `per_device_train_batch_size`: 2 & `gradient_accumulation_step`: 4, making the effective batch size 2 * 4 = 8, memory management decision
* `optim`: adamw_8bit, another memory optimization. AdamW's adaptive learning rates suit transformers, and the 8 bit version cuts optimizer state memory by 4x with negligible accuracy impact.

**Memory Usage and Training Loss**

![Target GPU Usage Graph](../images/training_gpu_target.png)

From this chart we notice a few things, first the training with chain of draft peaked at around 33gb vs chain of thought which peaked at around 36gb, we also notice that there was an approximately 10 minute difference between training of chain of draft and training with chain of thought. 

Which makes sense, if you refer back to the previous blog chain of draft dataset answers generally had 50% of the tokens compared to chain of thought.

![Target Training Loss Graph](../images/training_loss_target.png)

The second part is the training loss, we mostly observe the training loss to make sure training worked as intended and model trained correctly, otherwise if we use a model that didn't train correctly to generate a distilled dataset the output won't be as fun.

The code for training both target and draft models can be found in `src/train.py`


### Generating Distilled Dataset

Now that we've trained our target models, we need to generate a distilled dataset. As mentioned above, the reason for using a distilled dataset of the original dataset is that we want to focus on alignment of draft models with target models, not necessarily maximizing their accuracy.

For inference, we use another popular library called vLLM[^3], which is heavily optimized towards inference. Mainly using 3 different methods

* Paged Attention: Instead of pre-allocating a full KV-cache slot per sequence (), vLLM allocates pages on demand, treating it like a virtual memory system. In simpler terms, when LLMs generate an output they have a maximum sequence length, instead of occupying the entire length, vLLM divides it into blocks and allocates as needed, so if the maximum sequence length was 1024 tokens, but the actual output was 100 tokens, vLLM uses memory as needed instead of occupying spaces for the entire sequence.


* Continuous batching: Instead of waiting for the entire batch to finish, when one sequence finishes vLLM hands over the free GPU compute to the next prompt, allowing for faster processing and data generation.

* Native LoRA serving: Since we don't need to update the LoRA weights during inference, vLLM writes them directly to the GPU kernel, so the GPU only sees one set of calculation, allowing for much faster inference.

The code for generating the models can be found in `src/distill_data.py`, however a few design choices about the code

* **Resumability**: The code checks for previous generated samples and continues from there, we only have ~1000 samples in these experiments, so it's not a huge difference but it's a good habit to have

* **Temperature=0**: The temperature parameter is used to control the randomness of the output, since we want the dataset to reflect the target's modal behavior, not a random sample from its distribution, we set it to 0. It's also good for reproducability.

* **Two Layer Validation**:
  * First we validate that target model learned the output format we need, by making sure most answers produce a #### or \boxed{} for the answer
  * We check the answer correctness, basically how accurate is the target model 

### Training the Draft Model

When it comes to selection of draft models there are two things to note down, first you can't pick a model from a different family, as they have different tokenizers which will cause a lot of alignment issues. Second point is the size of the draft model, we pick the Qwen3-0.6B model, which is roughly a reduction of 23x. We wanted a model that is not too big the latency of it's forward pass is not much of an improvement, but we also don't want a model that's too small the acceptance rate gets too low and basically has no impact.

Now that we have our distilled dataset, and picked our draft model, the next step is to train the different versions. Same as the previous step except with some modified parameters. The first being dataset, where we train it on the distilled dataset we generated in the previous step. 

The second being the gradient accumulation step, with draft models being smaller we can increase the _per_device_train_batch_size_ parameter from 2 to 8 while reducing gradient accumulation step from 4 to 2. As well as reducing eval_steps from 50 to 20. The reasoning behind all these changes is that since we are training a smaller model we have more gpu and can optimize towards faster training.

Everything else is identical, even the LoRA parameters, you could play around with it and do an ablation study, but as we're testing the improvement of speculative decoding, as long as the draft model of both CoD and CoT had similar and sensible parameters, the experiment remains valid.

**Memory Usage and Training Loss**

Just because it feels like everything will be good doesn't mean it has to be, we need to check. So we once again look at the loss curves and make sure they learned correctly

![Draft Training Loss Graph](../images/training_loss_draft.png)

From the graphs we can see that our loss function has been steadily decreasing for all the different draft models, 

![Draft GPU Usage Graph](../images/training_GPU_draft.png)

In this graph as well we notice that it follows the trend of the previous one, with Chain of Draft models using noticeably less memory, ~3-4GB compared to ~5-6GB for Chain of Thought.

**Running the pipeline**

Instead of running all the different files with their different parameters. You can adjust the config files in the `config` folder and run the entire pipeline using the command

```bash
bash scripts/train_pipeline.sh -t <type> -s <scenario>
```
where type can be 'cod|cot' and scenario can be 'easy|medium|hard'

If you want to run the entire experiment, use the command

```bash
bash scripts/run_queue.sh
```

### Evaluation

Now that we've finished training our model, it's time to do the predictions so we can evaluate, but first, what exactly do we want to evaluate.

This is both an experiment on speculative decoding, but it also discusses the hypothesis that chain of draft models will be faster than chain of thought models. As in, it's easier for a draft model to align with a target model if the target model is more concise.

So for our hypothesis, what we will need to keep track off:

* Acceptance Rate: This is the most important metric, any increase between the two methods shows that our hypothesis has merit
* Average Tokens per Response: As mentioned in [^1], we  want to see if our fine-tuned models also learn the behavior from the different thought types. 

Now that we've decide on the main metrics, we also need some validation metrics. Basically make sure that the change is not due to other factors. So we will also need to measure

* Accuracy: how many answers are correct
* Answer Validation: How many answers were inside a separator or a \boxed{} format
* GPU overhead

It is also important to note that when training and evaluating, we used the system prompts in `data_generation/prompts.py`, just to maintain unity of prompts across the experiment.


## References
[^1]:[Chain of Draft Speculative Decoding 2: Dataset Generation and Evaluation](/slug_decode_2)
[^2]:[Unsloth](https://unsloth.ai/docs)
[^3]:[vLLM](https://docs.vllm.ai/en/stable/)
[^4]:[LIMA: Less is More for Alignment](https://arxiv.org/pdf/2305.11206)
