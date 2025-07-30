# Dissertation
Enhancing regulatory Compliance When sending data to LLMs

**Background/Motivation-** 

Large Language Models (LLMs) are increasingly being deployed across industries to automate and enhance email processing tasks such as classification, summarization, sentiment analysis, and intent recognition. For instance, in a modern data center or corporate IT environment, hundreds of emails may be received daily—ranging from support requests and incident reports to policy escalations and operational updates. Manually sorting and responding to these emails is not only time-consuming but also prone to human error. LLMs are now being used to categorize the type of email and generate automated responses, improving both efficiency and consistency. 

However, these emails often contain sensitive information, such as names, email addresses, system credentials, IP addresses, and confidential project details. When such emails are processed by LLMs—especially via external APIs—this sensitive content is also exposed, introducing significant privacy risks. One way to mitigate this is to mask or encrypt the sensitive content prior to sending it to the LLM. Unfortunately, traditional masking or encryption techniques often distort the format and structure of the data, which can lead to loss of context, broken sentence flow, or failure to maintain logical linkages between entities—ultimately degrading LLM performance. 
To address this challenge, this project proposes the use of Format-Preserving Encryption (FPE) as a privacy-preserving mechanism. FPE maintains the original format, structure, and length of sensitive values while still rendering them unreadable. This allows LLMs to operate on realistic-looking surrogates, enabling downstream tasks to function effectively without exposing actual private data. 

**Overview-** 
This project explores how Format-Preserving Encryption (FPE) can be used to protect sensitive information in unstructured text—specifically emails—while maintaining the utility of Large Language Models (LLMs) for downstream tasks. With the increasing adoption of LLMs in industry for tasks such as email classification, summarization, intent detection, and entity recognition, the privacy risks associated with exposing raw, sensitive content to third-party models have become a critical concern. 
To investigate this, the project uses the publicly available Enron email dataset, which mimics a real-world corporate communication environment containing various types of sensitive information. The study applies FPE-based anonymization to emails and evaluates the impact on LLM performance across multiple tasks, comparing it with traditional tokenization and full redaction methods. 
The goal is to assess whether FPE strikes a balance between privacy protection and functional accuracy, enabling organizations to use LLMs safely without significant degradation in model outputs or context understanding. By doing so, the project contributes to the broader field of privacy-preserving natural language processing (PP-NLP) and provides practical insights for securely integrating LLMs into real-world workflows. 

**Main Steps in the Project** 

1. Data Acquisition & Preprocessing 

Load and clean the Enron email dataset. 

Extract relevant metadata (e.g., sender, recipients, subject, message body). 

Filter and group emails into conversation threads based on subject and participants. 

Select a meaningful subset (e.g., longer threads with project or status-related keywords) for experimentation. 

 

2. Entity Detection & Anonymization 

Use Named Entity Recognition (NER) models and regex to identify Personally Identifiable Information (PII) such as: 

Names 

Email addresses 

Card numbers 


Apply three anonymization strategies: 

Raw (no anonymization) 

Tokenization (e.g., <NAME_001>) 

Format-Preserving Encryption (FPE) 

 

3. LLM-Based Task Execution 

For each anonymization version of the emails, send prompts to an LLM (local) for various tasks: 


Classification (in which category the mail falls into e.g., request, update, complaint) 

Named Entity Recognition : Does this mail have a credit card number. If yes then who is the issuer

Relationship Extraction (e.g., "Who reports to whom?") 

Sentiment analysis- Positive, Negative, Neutral tone of email

 

4. Evaluation of Output Quality 

Evaluate the LLM outputs across the three versions using: 

Automatic metrics: ROUGE, BLEU, accuracy, precision/recall 

Human judgment (if feasible): fluency, coherence, utility 

Analyze how well FPE preserves task performance compared to raw and tokenized inputs. 

 

5. Privacy Impact Discussion [optional]

Qualitatively compare the privacy guarantees of FPE vs. tokenization. 

Discuss risks of memorization, type leakage, and referential integrity. 

 

6. Conclusions & Recommendations 

Summarize findings on the tradeoff between privacy and utility. 

Recommend best practices for organizations considering LLMs on sensitive text. 

Suggest future directions, e.g., hybrid models, use of different LLMs, or larger corpora. 

 


Hypothesis: Using former preserving encryption or tokenization will be closer to the performance of no anonymisation than replacing with hexadecimal 

Assumption: llm's are not just looking at these as arbitrary tokens, but they know whether something is a name or e-mail address. 


Enron dataset can be downloaded from: https://www.kaggle.com/datasets/wcukierski/enron-email-dataset.
This is a 1.43 GB email database with 2 columns-file,messages
