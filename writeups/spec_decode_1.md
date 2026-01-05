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

This is the second part of the series on Chain of Draft Speculative Decoding, first part can be found [here](/spec_decode_0). In this part we will talk about which dataset to pick and how to generate them. The code for this script can be found [here](https://github.com/Hassan-Sarwat/efficient-speculative-decoding/tree/master/data_generation)


Before we start let's  have a quick reminder of our hypotheses:
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
 

2. **MATH {Levels: (1,2), Types: (Algebra, Intermediate Algebra, Precalculus)}[^3]**: Our medium difficulty dataset, this is a small step above our easy difficulty that the standard 14B model will still be able to solve some questions but we should notice an increase in accuracy from the trained models. The reason we picked these types is because it's procedural and doesn't need visual intuition versus something like geometry. This is where we expect our speculative decoding to really shine and give us a significant speed up. Here's a sample from the dataset.
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

3. **MATH {Levels: (1,2), Types: (Algebra, Intermediate Algebra, Precalculus)}[^3]**: Our final difficulty, this is where we stress test the chain of draft logic and see if it holds up in significantly long and complex scenarios. We also stress test the speculative decoding models as 0.5B draft models will have a significantly more difficult time solving this. 
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
You might have noticed that we are generally using math datasets, the reason I'm not using common sense or visual reasoning or information datasets is because math is both a good proxy for reasoning, it's procedural and you can see the thinking, and finally we can easily test the results of our models on math datasets.

**Pipeline**

Next step would be to plan the experiments, how do we test and evaluate each scenario.

So our pipeline would look as follows 

1. Generate a chain of thought (CoT) and chain of draft (CoD) datasets for the answers
2. Analyze datasets and make sure everything looks good 
3. Train CoT/CoD target models  
4. Use output of CoT/CoD target models to generate distilled answers
5. Train Cot/CoD draft models on distilled answers
6. Evaluate metrics and performance

And in this blog post we are going to be doing the first three steps, I'll walk you through the logic with addition of some code, mainly focusing on the why as with the rise of AI it's more important to understand the reasoning behind things than just how to implement it.  

**Generating our own chain of thought**

You might be asking yourself *Why would we generate our own chain of thought instead of using the answers provided by the dataset?*, there are a few reasons for this. First is reproducability, if you have a dataset on huggingface you can run this code and do your own tests just fine. The second is unification, different datasets will have different logic or trains of thought, for example GSM8K uses the '<< >>' (calculator) tag to represent calculations, our 14B model will understand this but our 0.5B draft model will not, same applies to MATH dataset which uses latex. We want to filter out these operations and create a dataset that both our draft and target model can understand. 

You might also be asking, *Isn't it expensive to generate our own train of thought?* And normally you'd be right, but there was a paper published in 2023 called LIMA: Less is More for Alignment[^4] which shows that what you really need is around 1000 samples to fine-tune a model for reasoning, and more than that you get diminishing returns. The second point and one of the reasons I'm using gemini for this experiment is that if you add your credit card information, you get 257$ in credits, which is more than enough for this experiment. 

## Implementation

**Setting up the environment**

I'm considering the data_generation part of the code it's own pipeline, for it you need the following packages:

```bash
pip install google-genai==1.56.0 datasets tqdm python-dotenv
```
The only unique one here is the `google-genai==1.56.0` package, the rest you can find in the train environment. If you want you can add it to that environment.

**Downloading the dataset**

I'll start with a top level overview of the code, the first file you will run will be the `launch_generation.py` file. It takes the following parameters

| Argument | Description | Default |
|---|---|---|
| `--chain` | Type of chain: `thought` or `draft` | `thought` |
| `--dataset` | Hugging Face dataset name | `None` |
| `--filter` | Filter string (e.g., `level=Level 1,Level 2`) | `None` |
| `--file_suffix` | Output suffix (`cot_{suffix}.jsonl`) | `None` |
| `--limit` | Max number of samples to process | `None` |
| `--dry-run` | Prepare batch file but do not submit | `False` |
| `--auto_fill` | Auto-select "Fill Gap" if existing < limit | `False` |
| `--auto_extend` | Auto-select "Extend" if existing data found | `False` |


The first thing it does is to download the dataset using the `dataset_loader.py` file, first it checks if the dataset is already downloaded, if not it downloads it. After it downloads it, it filters based on the `--filter` parameter and for each entry it applies the filter, it also raises a value error if the filter doesn't exist to notify us that something went wrong with the application of the filter

```python
for key, valid_values in filters.items():
    # Validate filter key exists
    if key not in available_columns:
        raise ValueError(
            f"Filter key '{key}' not found in dataset.\n"
            f"Available columns: {available_columns}"
        )
```
After the filter is applied we use the `--limit` paramater to slice the number of samples. We also log the final dataset saved size to confirm that the filtration wasn't too strict.

```python
logger.info(f"Final dataset size: {len(dataset)}")
```

After the dataset has been loaded, filtered, and sliced, we save it to a jsonl file with the safe name where we replace any '/' with '_' and if `--suffix` parameter is provided we append it to the file name. For example if run the command `python data_generation/launch_generation.py --dataset gsm8k --file_suffix easy --limit 1000` it will download the gsm8k dataset, limit it to 1000 samples, and save it to a jsonl file named `gsm8k_easy.jsonl`

The `auto_fill` and `auto_extend` parameters are there in case you are running the code again and either want to fill in the gap for some corrupted samples, or extend the dataset to a larger size.

**Submitting Batch Job**

Perfect, now that we have our dataset the next step would be to generate our chain of thought dataset. Given that we have a 1000 samples we want to run, it doesn't make to stream as this might take a long time and requires our constant supervision, plus any interruption might causes the job to fail and we will have to restart. It's also cheaper to use batch jobs which is always a plus.

For this, we will look at the [gemini batch api](https://ai.google.dev/gemini-api/docs/batch-api?batch=file) and we notice that it requires from us to submit a jsonl file with key as ID and a request object that contains the questions, and other configurations such as system instructions. 

We therefore create a jsonl file stored in a tmp directory with the following format:

```json
{"key": "req_7d1341a6677b", "request": {"contents": [{"parts": [{"text": "question_from_dataset"}]}], "systemInstruction": {"parts": [{"text": "system_instruction_from_prompts"}]}}}
```

Where `question_from_dataset` is the question from the dataset and `system_instruction_from_prompts` is the system instruction from the `prompts.py` file. We use a modified version of the prompts from [^6], the reason is their prompts were too vague. We therefore created a more specific prompt that is more aligned with our task and allows for easier analysis. Feel free to adjust them as is necessary for your case.  

After `launch_generation.py` creates the batch file it will submit a batch request. For our requests, we are using the `gemini-3-pro-preview` model in our experiments, it's not best practicies to use a preview model as it might not be available in the future, but as of this moment the gemini-3 model has no available non preview alternatives in the [website](https://ai.google.dev/gemini-api/docs/models/). The code for submission and batch management with gemini can be found in `batch_client.py`.

So to submit our requests for the experiments, we would need to run the following commands:

```bash
# For the easy datasets
python data_generation/launch_generation.py --chain thought --dataset gsm8k --file_suffix easy --limit 1000

python data_generation/launch_generation.py --chain draft --dataset gsm8k --file_suffix easy --limit 1000

# For the medium datasets
python data_generation/launch_generation.py --chain thought --dataset "qwedsacf/competition_math" --filter "level=Level 1,Level 2" --filter "type=Algebra,Intermediate Algebra,Precalculus" --file_suffix "medium" --limit 1000

python data_generation/launch_generation.py --chain draft --dataset "qwedsacf/competition_math" --filter "level=Level 1,Level 2" --filter "type=Algebra,Intermediate Algebra,Precalculus" --file_suffix "medium" --limit 1000

# For the hard datasets
python data_generation/launch_generation.py --chain thought --dataset "qwedsacf/competition_math" --filter "level=Level 3,Level 4" --filter "type=Algebra,Intermediate Algebra,Precalculus" --file_suffix "hard" --limit 1000

python data_generation/launch_generation.py --chain draft --dataset "qwedsacf/competition_math" --filter "level=Level 3,Level 4" --filter "type=Algebra,Intermediate Algebra,Precalculus" --file_suffix "hard" --limit 1000
```

Note: It is generally good practice to submit a small batch job (--limit 10) or to do a dry run (--dry-run) to test that the code works, however as I've already done the testing I'm skipping this step in the blog. I'd also generally advise against submitting all batches at once as you might get rate limited. 

**Processing the results**

Now that we've submitted our batch jobs, we need to wait for them and then download them. that's why we have `process_results.py` file. This file reads the tmp file we generate in `launch_generation.py` and checks the status of the corresponding batch job, whether it's finished or not. If it's done it downloads result as a jsonl and stores it in the data folder. The commands for processing results are as follows, They all need to be run individually and to check status of the jobs.

For CoT Easy Difficulty
```bash
python data_generation/process_results.py --chain thought --dataset gsm8k --file_suffix easy
```

For CoD Easy Difficulty
```bash
python data_generation/process_results.py --chain draft --dataset gsm8k --file_suffix easy
```

For CoT Medium Difficulty
```bash
python data_generation/process_results.py --chain thought --dataset "qwedsacf/competition_math" --file_suffix "medium"
```

For CoD Medium Difficulty
```bash
python data_generation/process_results.py --chain draft --dataset "qwedsacf/competition_math" --file_suffix "medium"
```

For CoT Hard Difficulty
```bash
python data_generation/process_results.py --chain thought --dataset "qwedsacf/competition_math" --file_suffix "hard"
```

For CoD Hard Difficulty
```bash
python data_generation/process_results.py --chain draft --dataset "qwedsacf/competition_math" --file_suffix "hard"
```

Finally after all jobs are completed we can analyze the data and verify that the format is correct and the results are as expected.


## Analyzing the data

Now that we have our data, as data scientist or researchers we need to follow murphy's law, which states that "Anything that can go wrong will go wrong" and make sure at no point in the pipeline did our data get corrupted or lost.

The code for the jupyter notebook for the analysis can be found [here](https://github.com/Hassan-Sarwat/efficient-speculative-decoding/blob/master/data_generation/analysis.ipynb)

The Analysis process consists of 3 parts. The first part will be validating the raw datasets, which we've already done before submitting by raising errors in the code but I'm paranoid and I like to double check. 

The second part will be analyzing the output of the LLM and our answers, mainly if they follow the correct structure or not and if they are accurate. If not, we will clean the data because we don't want to give our models garbage answers.

The final part will be comparison between Chain of Draft and Chain of thought in terms of metrics.

**1. Raw Data Validation**

This step is simple, we just need to make sure that all raw datasets contain 1000 unique samples, and that in the scenarios where filters were applied that the filters stuck.

They did. You can have a look at the notebook. Maybe there's a lesson in there that I should trust my code more but I am paranoid.

**2. LLM Output Validation**

This part is a big bigger. We need to make 2 validations, the first being that they are all structurally compliant and the second being that they are all correct.

Regarding the structure compliance this is easy to extract, we just check for an answer separator '####' and a step divider '->'.

The answer being correct is more difficult, as this is a math dataset we will need to do some extraction, for example in gsm8k the answer shows up after a '####' separator but in MATH dataset the answer appears in \\boxed{}, so we will need to accomodate this. We also need to take into account small differences such as 990.00 and 990.

After running a quick analysis we get the following result:

| Scenario | Dataset | Total Raw | Joined Count | Count (####) | Count (->) | Count (Correct) | No Steps & Incorrect |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| "easy" | "cot" | 1000 | 1000 | 1000 | 987 | 964 | 0 |
| "easy" | "cod" | 1000 | 1000 | 1000 | 997 | 955 | 1 |
| "medium" | "cot" | 1000 | 1000 | 1000 | 802 | 918 | 28 |
| "medium" | "cod" | 1000 | 1000 | 997 | 964 | 923 | 6 |
| "hard" | "cot" | 1000 | 1000 | 1000 | 798 | 896 | 20 |
| "hard" | "cod" | 1000 | 1000 | 998 | 950 | 898 | 7 |


**Data Cleaning**
Now these results are unfortunately not the best and we will need to do some cleaning. Note that besides cleaning there are other steps you can take. 

1. Extend to 1500 samples instead of 1000 samples, and use remainder
2. Use LLM as a judge to make sure answers are semantically correct as their could be problems with extraction
3. Pass non structurally compliant samples to LLM again to correct and make them structurally compliant.

In this blog we just assume that further usage of LLMs is limited and go about it the classical way, but just wanted to let you know that's an option.

First we look at the ones with no `->` and investigate why they happen. The Chain of Draft data looks fine so we will only focus on Chain of Thought. After looking at some of the samples we notice that instead of `->` they use `\n\n` to separate steps in chain of Thought, while not 100% it should help us filter out. So we apply the filter and the results change to the following.

| Scenario | Dataset | Total Raw | Joined Count | Count (####) | Count (->) | Count (Correct) | No Steps & Incorrect |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| "easy" | "cot" | 1000 | 1000 | 1000 | 987 | 964 | 0 |
| "easy" | "cod" | 1000 | 1000 | 1000 | 997 | 955 | 1 |
| "medium" | "cot" | 1000 | 1000 | 1000 | 999 | 918 | 0 |
| "medium" | "cod" | 1000 | 1000 | 997 | 964 | 923 | 6 |
| "hard" | "cot" | 1000 | 1000 | 1000 | 1000 | 896 | 0 |
| "hard" | "cod" | 1000 | 1000 | 998 | 950 | 898 | 7 |

With this we notice a significant improvement, but we still have problem in the accuracy of the answers. Again we dive a bit deeper and we notice that some of the answers we have are not resolving due to a syntax difference, such as `\\frac{1}{2}` instead of `1/2` or `0.5`. We therefore create a helper function to resolve these issues our table changes to

| Scenario | Dataset | Total Raw | Joined Count | Count (####) | Count (->) | Count (Correct) | No Steps & Incorrect |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| "easy" | "cot" | 1000 | 1000 | 1000 | 987 | 964 | 0 |
| "easy" | "cod" | 1000 | 1000 | 1000 | 997 | 955 | 1 |
| "medium" | "cot" | 1000 | 1000 | 1000 | 999 | 933 | 0 |
| "medium" | "cod" | 1000 | 1000 | 997 | 964 | 932 | 6 |
| "hard" | "cot" | 1000 | 1000 | 1000 | 1000 | 935 | 0 |
| "hard" | "cod" | 1000 | 1000 | 998 | 950 | 932 | 8 |

As you can observe, we noticed a significant bump in accuracy in the hard scenario. We can try looking at more samples and getting better accuracy, however I'll stop here for now.

Our next step will to be remove the samples that aren't compliant with the structure and are incorrect. After removing them we will slice down the bigger dataset in each scenario so that both Chain of Draft and Chain of Thought are same in size.





## References

[^1]:[Teaching Small Language Models to Reason](https://aclanthology.org/2023.acl-short.151.pdf)
[^2]:[Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168)
[^3]:[Measuring Mathematical Problem Solving with MATH Dataset](https://arxiv.org/pdf/2103.03874)
[^4]:[LIMA: Less is More for Alignment](https://arxiv.org/pdf/2305.11206)
[^5]:[PAL: Program-aided Language Models](https://arxiv.org/pdf/2211.10435)
[^6]:[Chain of Draft: Thinking Faster by Writing Less](https://arxiv.org/pdf/2502.18600)

