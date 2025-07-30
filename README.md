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




Enron dataset can be downloaded from: https://www.kaggle.com/datasets/wcukierski/enron-email-dataset
