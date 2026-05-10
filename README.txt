# Bird Species Knowledge Evaluation Using Local and API-Based LLMs

## Project Overview

This project evaluates how well different Large Language Models can answer bird-species knowledge questions. The task focuses on predicting the **common name** and selected **visual attributes** of bird species using scientific names, sex labels, and optional contextual information from web-sourced bird descriptions.

The project compares both local open-source models and API-based models under different experimental conditions. The main goal is to understand how model accuracy changes when:

- Context is provided or removed
- The number of simultaneous bird queries increases
- Thinking-based models are compared with non-thinking models
- Smaller local models are compared with larger API-based models

The project uses `birdNames.txt`, downloaded bird article context, local Qwen models, OpenAI API models, evaluation scripts, generated plots, and a final `Analysis.pdf` report.

---

## Objectives

- Build a dataset of male and female bird species using scientific names.
- Download bird description context from online resources.
- Query multiple LLMs for bird common names and visual attributes.
- Compare local open-source models against API-based models.
- Measure common-name prediction error.
- Measure difference from a thinking-model reference key.
- Study the effect of context level on accuracy.
- Study the effect of workload size on accuracy.
- Generate plots and a final analysis report.

---

## Tools and Technologies Used

- Python 3
- Pandas
- Matplotlib
- ReportLab
- OpenAI API
- Hugging Face Transformers
- PyTorch
- Qwen/Qwen2.5 local instruct models
- GPT-4.1-mini
- o4-mini
- Wikipedia API
- CSV / JSONL experiment logging
- Google Colab workflow

---

## Project Files

| File | Purpose |
|---|---|
| `birdNames.txt` | Input list of bird sex labels and scientific names |
| `birdDescriptionDownloader.py` | Downloads bird article descriptions and creates `birdArticles.csv` |
| `testBirdSpeciesNameKnowledge-local.py` | Runs local Qwen model experiments |
| `testBirdSpeciesNameKnowledge-API.py` | Runs API model experiments |
| `comparisonEvaluation.py` | Computes evaluation metrics |
| `make_analysis_report.py` | Generates plots and `Analysis.pdf` |
| `attributeConfigurations.csv` | Defines visual attribute dictionaries |
| `birdArticles.csv` | Stores downloaded bird descriptions |
| `birdKnowledgeTests.csv` | Stores model outputs and evaluation scores |
| `experiment_manifest.jsonl` | Tracks queried birds for each experiment row |
| `Analysis.pdf` | Final analysis report with questions and plots |
| `plot_common_context.png` | Common-name error rate vs context level |
| `plot_common_workload.png` | Common-name error rate vs workload |
| `plot_diff_context.png` | Difference-to-thinking-key vs context level |
| `plot_diff_workload.png` | Difference-to-thinking-key vs workload |
| `chatURLs-*.txt` | Shared chat links for thinking/context experiments |

---

## Dataset Description

The input file `birdNames.txt` contains bird entries in the following format:

```text
male Cardinalis cardinalis
female Cardinalis cardinalis
male Haemorhous mexicanus
female Haemorhous mexicanus

Each row includes:

sex + scientific species name

These entries are parsed and used as query inputs for model testing.

Methodology and Project Execution

The project was completed through a structured data collection, model testing, and evaluation workflow.

1. Bird Description Collection

The first step was to create a context dataset for each bird species.

The script birdDescriptionDownloader.py reads birdNames.txt, extracts the scientific names and sex labels, searches Wikipedia, and saves article descriptions into birdArticles.csv.

The downloader includes support for:

UTF-8, UTF-8-SIG, and Latin-1 encoded files
Male/female bird labels
Binomial and trinomial scientific names
Hybrid notation such as ×
Wikipedia exact-title search
Redirect handling
Search fallback
Known difficult taxonomy cases

The output file contains:

speciesName, sex, sourceURL, extractedTextDescription
2. Attribute Configuration Setup

The project uses predefined visual attribute dictionaries stored in attributeConfigurations.csv.

Each model is asked to return:

Scientific name
Common name
Attribute values from a fixed categorical dictionary

The model output is required to be valid JSON.

Example expected structure:

[
  {
    "scientificName": "Cardinalis cardinalis",
    "commonName": "Northern cardinal",
    "bodyColor": "red",
    "wingColor": "red",
    "beakShape": "cone-shaped"
  }
]

The exact attributes depend on the selected attribute configuration.

3. Local Model Testing

The script testBirdSpeciesNameKnowledge-local.py evaluates local open-source instruct models.

Default local models:

Qwen/Qwen2.5-0.5B-Instruct
Qwen/Qwen2.5-1.5B-Instruct
Qwen/Qwen2.5-3B-Instruct

The script:

Loads bird entries from birdNames.txt
Loads article context from birdArticles.csv
Loads attribute dictionaries
Builds prompts with or without context
Runs model inference locally
Saves JSON outputs to birdKnowledgeTests.csv
4. API-Based Model Testing

The script testBirdSpeciesNameKnowledge-API.py evaluates API-based models.

API models used:

gpt-4.1-mini
o4-mini

The script runs the same bird-identification task under controlled experimental settings.

It varies:

contextLevel = 0, 1, 4, 8
numSimultaneousQueries = 1, 8, 16, 32

The API script sends prompts to OpenAI models, parses JSON responses, records token usage, and appends results to birdKnowledgeTests.csv.

5. Experimental Conditions

The experiments compare model performance across two main dimensions.

Context Level

Context level controls how much supporting bird-description text is provided to the model.

contextLevel = 0
No supporting article context

contextLevel = 1
Relevant context for the queried bird

contextLevel = 4
Relevant context plus extra distractor context

contextLevel = 8
More context and more distractor information
Workload

Workload controls how many bird queries are submitted at the same time.

numSimultaneousQueries = 1
numSimultaneousQueries = 8
numSimultaneousQueries = 16
numSimultaneousQueries = 32

This tests whether models perform better when answering fewer questions per prompt or when handling larger batches.

6. Evaluation Metrics

The script comparisonEvaluation.py computes two main metrics.

Common Name Error Rate

This metric checks whether the model predicted the correct common name for each scientific name.

commonNameErrorRate = 0.0 means correct
commonNameErrorRate = 1.0 means incorrect
Difference to Thinking Key

This metric compares each model’s output against a reference answer produced by a thinking model under a controlled setting.

The reference key is defined as:

thinking model
contextLevel = 1
numSimultaneousQueries = 1

The script evaluates differences in common name and attribute values, then saves the updated metrics back into birdKnowledgeTests.csv.

7. Analysis Report

The final analysis report is saved as:

Analysis.pdf

The report answers three main evaluation questions:

Without provided context from relevant articles, how accurate are the responses from the various models?
Did providing web resources with information about the bird species improve accuracy for common names and visual attributes?
Does submitting fewer queries, each with a larger amount of work, improve or reduce performance?

The report includes four major plots:

commonNameErrorRate vs contextLevel
differenceToThinkingKey vs contextLevel
commonNameErrorRate vs numSimultaneousQueries
differenceToThinkingKey vs numSimultaneousQueries
Analysis Plot Summary
1. Common Name Error Rate vs Context Level

This plot compares how common-name prediction changes when context increases from 0 to 8.

Main observation:

Most models stayed near high error-rate values, but some showed small improvements or instability depending on context level.

gpt-4.1-mini showed a visible drop in error at context level 1, while o4-mini remained flat across all context levels. Some local Qwen models also showed unstable behavior across different context settings.

2. Difference to Thinking Key vs Context Level

This plot measures how close each model stayed to the thinking-model reference answer.

Main observation:

o4-mini stayed perfectly aligned with the thinking-key reference, while gpt-4.1-mini and smaller Qwen models showed more variation.

The graph showed that gpt-4.1-mini had the largest deviation at context level 1, while Qwen/Qwen2.5-1.5B-Instruct gradually moved farther from the reference as context level increased.

3. Common Name Error Rate vs Workload

This plot compares common-name prediction accuracy when the model answers 1, 8, 16, or 32 bird queries at once.

Main observation:

Higher workload did not consistently improve performance and sometimes introduced instability.

For example, Qwen/Qwen2.5-0.5B-Instruct showed a major drop around workload 16, while gpt-4.1-mini varied across workload levels. o4-mini stayed flat.

4. Difference to Thinking Key vs Workload

This plot evaluates whether batch size affects agreement with the thinking-model reference.

Main observation:

Large workloads can reduce agreement with the reference answer because the model must maintain structured JSON and attribute consistency across many birds.

o4-mini remained stable, while gpt-4.1-mini and Qwen/Qwen2.5-1.5B-Instruct showed variation as workload increased.

Key Findings
Question	Finding
Does context improve bird-name accuracy?	Context helped some models slightly, but not consistently across all models
Does more context always help?	No. Extra context can introduce distractors and does not guarantee better performance
Do larger workloads improve accuracy?	Not consistently. Larger batches can reduce output consistency
Are thinking models more stable?	Yes. o4-mini remained the most stable across plots
Can small local models handle the task?	Yes, but they showed more instability and weaker consistency
What is the hardest part?	Maintaining accurate common names and structured attributes across many birds
Results and Outcomes

The project successfully produced a complete LLM evaluation workflow.

Major outcomes included:

Parsed a bird-species dataset from birdNames.txt
Downloaded article-based bird descriptions
Built a context-augmented bird knowledge dataset
Tested local Qwen models
Tested API-based OpenAI models
Compared thinking and non-thinking model behavior
Evaluated common-name prediction accuracy
Evaluated agreement with a thinking-model reference key
Generated four performance plots
Created a final Analysis.pdf report
Documented chat URLs for different context and thinking conditions
Project Workflow
Prepare birdNames.txt
        ↓
Download bird article descriptions
        ↓
Create birdArticles.csv
        ↓
Load attribute configurations
        ↓
Run local Qwen model experiments
        ↓
Run API model experiments
        ↓
Store results in birdKnowledgeTests.csv
        ↓
Evaluate common name and attribute errors
        ↓
Generate plots
        ↓
Create Analysis.pdf
How to Run the Project
1. Install Required Packages
pip install pandas matplotlib reportlab requests openai torch transformers accelerate
2. Download Bird Descriptions
python birdDescriptionDownloader.py

Expected output:

birdArticles.csv
3. Run Local Model Experiments
python testBirdSpeciesNameKnowledge-local.py \
  --base_dir /content/drive/MyDrive/at3684-lab3

Optional custom models:

python testBirdSpeciesNameKnowledge-local.py \
  --base_dir /content/drive/MyDrive/at3684-lab3 \
  --models Qwen/Qwen2.5-0.5B-Instruct,Qwen/Qwen2.5-1.5B-Instruct,Qwen/Qwen2.5-3B-Instruct
4. Run API-Based Experiments

Set the OpenAI API key:

export OPENAI_API_KEY="your_api_key_here"

Run the API experiment script:

python testBirdSpeciesNameKnowledge-API.py \
  --base_dir /content/drive/MyDrive/at3684-lab3

Optional custom models:

python testBirdSpeciesNameKnowledge-API.py \
  --base_dir /content/drive/MyDrive/at3684-lab3 \
  --models gpt-4.1-mini,o4-mini
5. Evaluate Results
python comparisonEvaluation.py --thinking-model-name o4-mini

Expected updated file:

birdKnowledgeTests.csv
6. Generate Analysis Report
python make_analysis_report.py

Expected outputs:

plot_common_context.png
plot_common_workload.png
plot_diff_context.png
plot_diff_workload.png
Analysis.pdf
Chat Transcript Links

The project includes shared chat links for different experiment settings.

File	Description
chatURLs-noThinking-noContext.txt	Non-thinking, no-context chat
chatURLs-noThinking-withContext.txt	Non-thinking, with-context chat
chatURLs-withThinking-noContext.txt	Thinking, no-context chat
chatURLs-withThinking-withContext.txt	Thinking, with-context chat

These files document the shared conversation links used for the experiment variants.

Repository Structure
.
├── README.md
├── README.txt
├── birdNames.txt
├── birdDescriptionDownloader.py
├── birdArticles.csv
├── attributeConfigurations.csv
├── testBirdSpeciesNameKnowledge-local.py
├── testBirdSpeciesNameKnowledge-API.py
├── comparisonEvaluation.py
├── make_analysis_report.py
├── birdKnowledgeTests.csv
├── experiment_manifest.jsonl
├── gatheredWebResources.zip
├── Analysis.pdf
├── chatURLs-noThinking-noContext.txt
├── chatURLs-noThinking-withContext.txt
├── chatURLs-withThinking-noContext.txt
├── chatURLs-withThinking-withContext.txt
├── plot_common_context.png
├── plot_common_workload.png
├── plot_diff_context.png
└── plot_diff_workload.png
Major Learnings

Through this project, I gained practical experience with:

LLM evaluation workflows
Prompt design for structured JSON output
Local model inference using Hugging Face Transformers
API-based model testing
Context-augmented question answering
Scientific-name to common-name mapping
Dataset creation from web resources
Evaluation metric design
Error-rate analysis
Plot generation and reporting
Comparing thinking and non-thinking model behavior
Studying context and workload effects on LLM accuracy
Conclusion

This project evaluated how different LLMs perform on bird-species knowledge tasks under changing context and workload conditions.

The pipeline collected bird descriptions, generated controlled prompts, tested local and API-based models, computed error metrics, and produced analysis plots. The final Analysis.pdf showed that model performance depends not only on model size, but also on context relevance, workload size, structured-output reliability, and whether the model uses thinking-style reasoning.

Overall, stronger thinking-based models were more stable, while smaller local models were more sensitive to workload and context changes. The project demonstrates a complete workflow for evaluating LLM knowledge, structured JSON reliability, and context-based reasoning performance.
