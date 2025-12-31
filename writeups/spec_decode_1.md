---
title: On Chain of Draft Speculative Decoding and how to speed up your LLMs - 1
author: Hassan Sarwat
pubDatetime: 2025-12-28T20:12:32Z
slug: spec_decode_1
featured: true
draft: false
tags:
  - LLMs
  - Technical
  - Speculative Decoding
  - Chain of Draft
description:
  "The second part of the series explaining speculative decoding and my plan on applying a chain of draft instead of chain of thought when training reasoning models. This part dives into dataset generation and selection"
---

# Dataset Generation and Evaluation

### Datasets

This is the second part of the series on Chain of Draft Speculative Decoding, first part can be found [here](/spec_decode_0). In this part we will talk about which dataset to pick and how to generate them. The code for this script can be found [here]()

We want to apply the logic from the paper from Magister et al[^1], where they teach small models to reason. For that we will fine-tune them on Chain of Thought Data and Chain of Draft, and compare the results. 

Let's have a quick reminder of our hypotheses:
* Fine-tuning the model on reasoning will improve its accuracy
* Speculative decoding will reduce our inference time without affecting accuracy
* Chain of draft will reduce our total tokens with only minimal effect on accuracy

Looking at the requirements we notice that we will need to stress test our ideas under 3 conditions, an easy condition where our untrained 14B model can solve, and we mainly observe the impact on inference speed and accuracy when using speculative decoding and reasoning, this should be the case where you ask yourself "should I train the model with reasoning or not". 

The second test will be a medium difficulty setting where our untrained target model can solve some questions, but fine-tuning will notably improve the accuracy. In this one we note the speedup using speculative decoding and also compare the accuracy of chain of draft vs chain of thought, can our small draft models keep up where even our target models have difficulty.

The final case will be the robustness test, a dataset that our untrained target model can't solve and only when it reasons can it begin to solve, we want to see can chain of draft keep up with chain of thought under difficult conditions, and how does it impact our acceptance rate.

Now that we've identified the requirements, we need to identify the pipeline that is scenario agnostic and fair

**Based on these conditions we decide on the following datasets**

1. **GSM8K[^2]**: This is our easy dataset, it consists of grade school math questions and even our untrained model shouldn't have much difficulty solving these questions. Here's how a sample question and answer look like (new lines inserted for readability):
```json
{
  "Question":"Weng earns $12 an hour for babysitting.
    Yesterday, she just did 50 minutes of babysitting. 
    How much did she earn?",
  "Answer":"Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute. 
    Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10. 
    #### 10"
}
```
 

2. **MATH (Level 1-2, Algebra)[^3]**: Our medium difficulty dataset, this is a small step above our easy difficulty that the standard 14B model will still be able to solve some questions but we should notice an increase in accuracy from the trained models. The reason we picked Algebra is because it's procedural and doesn't need visual intuition versus something like geometry. This is where we expect our speculative decoding to really shine and give us a significant speed up. Here's a sample from the dataset.
```json
{
  "Question":"Suppose $d\not=0$. 
    We can write $\left(12d+13+14d^2\right)+\left(2d+1\right)$, 
    in the form $ad+b+cd^2$, where $a$, $b$, 
    and $c$ are integers. Find $a+b+c$.",
  "Answer":"Adding the $d$ terms gives us $14d$. 
    Adding the constant terms gives us $14$. 
    Adding the $d^2$ terms gives us $14d^2$. 
    Adding the terms together gives us ${14d+14+14d^2}$, 
    so $a+b+c = \boxed{42}$."
}
```

3. **MATH (Level 3-4, Intermediate Algebra)[^3]**: Our final difficulty, this is where we stress test the chain of draft logic and see if it holds up in significantly long and complex scenarios. We also stress test the speculative decoding models as 0.5B draft models will have a significantly more difficult time solving this. 
```json
{
  "Question":"The increasing sequence of positive integers $a_1,$ $a_2,$ $a_3,$ $\dots$
    has the property that \[a_{n + 2} = a_{n + 1} + a_n\]for all $n \ge 1.$ If $a_7 = 120,$ then find $a_8.$",
  "Answer":"Let $a_1 = a$ and $a_2 = b.$ Then
\begin{align*}
a_3 &= a + b, \\
a_4 &= a + 2b, \\
a_5 &= 2a + 3b, \\
a_6 &= 3a + 5b, \\
a_7 &= 5a + 8b, \\
a_8 &= 8a + 13b.
\end{align*}Hence, $5a + 8b = 120.$ Then $5a = 120 - 8b = 8(15 - b).$ Since 5 is relatively prime to 8, $a$ is divisible by 8.

If $a = 8,$ then $b = 10.$ If $a = 16,$ then $b = 5,$ which does not work, because the sequence is increasing, so $b > a.$ Note that higher values of $b$ return lower values of $a,$ so the only possible value of $a$ is 8. Then $b = 10,$ so $a_8 = 8a + 13b = \boxed{194}.$"
}
```

**Pipeline**

Next step would be to plan the experiments, how do we test and evaluate each scenario.

So our pipeline would look as follows 

1. Generate a chain of thought (CoT) dataset for the answers
2. Summarize chain of thought to chain of draft (CoD)
3. Analyze datasets and make sure everything looks good 
3. Train CoT target model
4. Train CoD target model
5. Use output of CoT target model to generate distilled CoT answers
6. Use output of CoD target model to generate distilled CoD answers
7. Train CoT Draft model
8. Train CoD draft model
9. Evaluate metrics for:
    * Untrained target model
    * Trained CoT target model
    * Trained CoD target model
    * Trained CoT target model with speculative decoding using CoT draft model
    * Trained CoD target model with speculative decoding using CoD draft model

And in this blog post we are going to be doing the first three steps, I'll walk you through (mostly logic with some code) how to  

### Generating Chain of Thought (CoT) datasets

The first question you will probably ask would be something like, why would you do this? You already have the answers from the dataset. This is a good question, but as an agnostic experiment I want to unify the datasets, in MATH it uses latex, in GSM8K the answers use the `<< >>` calculator tag, our draft model will have difficulty with expressions like this significantly impacting our acceptance rate. We want our answers to be in a format that both our target and draft model easily understand and output to not cause misalignment.

Naturally your second question will be, but isn't it very expensive generating our own chain of thoughts dataset, seeing we have to use expensive reasoning models. The answer is yes, but the magic of it is that we don't need to generate the entire dataset, referencing the LIMA[^4] paper from Meta, we really only need 1000 samples for each scenario to align the models, the second point is that if you add a credit card to your google api, they give you 300 (I only got 257.50 though, not that I'm complaining) USD in api credit that you can use for some stuff, like training and generating datasets. Technically this entire project has been funded by Google. Thank you Google.

I hope this reasoning is good enough, so now that we know what we want to do we can go ahead and write some code. As programmers with a bit of self respect, every code we write has to be abstracted and modularized. Normally at this part someone will tell you that we need to install some libraries and set up environments, but I'll leave that for the next tutorial where I cry about dependency hell. For this part we have just one script, and we only need one library, `google-genai`, and we go to the [gemini api documentation](https://ai.google.dev/gemini-api/docs/quickstart) and we get the installation command. 

Also keep in mind that we won't need this for any other part of the tutorial, google-genai is not included in the training or inference environment. I don't assume it will clash with either so install in global, install with train, install with inference, up to you it doesn't matter. I give you the freedom of choice.

For dataset generation, we can make use of something called Batch API, this is where we send our requests all at once and we check in every now again to see if it finished. This saves us the hassle of having to make a single prediction


## References
[^1]:[Teaching Small Language Models to Reason](https://aclanthology.org/2023.acl-short.151.pdf)
[^2]:[Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168)
[^3]:[Measuring Mathematical Problem Solving with MATH Dataset](https://arxiv.org/pdf/2103.03874)
[^4]:[LIMA: Less is More for Alignment](https://arxiv.org/pdf/2305.11206)
[^5]:[PAL: Program-aided Language Models](https://arxiv.org/pdf/2211.10435)
