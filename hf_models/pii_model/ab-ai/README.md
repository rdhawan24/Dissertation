---
license: apache-2.0
base_model: bert-base-cased
tags:
- PII
- NER
- Bert
- Token Classification
datasets:
- generator
metrics:
- precision
- recall
- f1
- accuracy
model-index:
- name: pii_model
  results:
  - task:
      name: Token Classification
      type: token-classification
    dataset:
      name: generator
      type: generator
      config: default
      split: train
      args: default
    metrics:
    - name: Precision
      type: precision
      value: 0.954751
    - name: Recall
      type: recall
      value: 0.965233
    - name: F1
      type: f1
      value: 0.959964
    - name: Accuracy
      type: accuracy
      value: 0.991199
pipeline_tag: token-classification
language:
- en
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# Personal Identifiable Information (PII Model)

This model is a fine-tuned version of [bert-base-cased](https://huggingface.co/bert-base-cased) on the generator dataset.
It achieves the following results:

- Training Loss: 0.003900
- Validation Loss: 0.051071
- Precision: 95.53%
- Recall: 96.60%
- F1: 96%
- Accuracy:99.11%

## Model description

Meet our digital safeguard, a savvy token classification model with a knack for spotting personally identifiable information (PII) entities. Trained on the illustrious Bert architecture and fine-tuned on a custom dataset, this model is like a superhero for privacy, swiftly detecting names, addresses, dates of birth, and more. With each token it encounters, it acts as a vigilant guardian, ensuring that sensitive information remains shielded from prying eyes, making the digital realm a safer and more secure place to explore.

## Model can Detect Following Entity Group

- ACCOUNTNUMBER
- FIRSTNAME
- ACCOUNTNAME
- PHONENUMBER
- CREDITCARDCVV
- CREDITCARDISSUER
- PREFIX
- LASTNAME
- AMOUNT
- DATE
- DOB
- COMPANYNAME
- BUILDINGNUMBER
- STREET
- SECONDARYADDRESS
- STATE
- EMAIL
- CITY
- CREDITCARDNUMBER
- SSN
- URL
- USERNAME
- PASSWORD
- COUNTY
- PIN
- MIDDLENAME
- IBAN
- GENDER
- AGE
- ZIPCODE
- SEX




### Training hyperparameters
The following hyperparameters were used during training:

| Hyperparameter               | Value         |
|------------------------------|---------------|
| Learning Rate                | 5e-5          |
| Train Batch Size             | 16            |
| Eval Batch Size              | 16            |
| Number of Training Epochs    | 7             |
| Weight Decay                 | 0.01          |
| Save Strategy                | Epoch         |
| Load Best Model at End       | True          |
| Metric for Best Model        | F1            |
| Push to Hub                  | True          |
| Evaluation Strategy          | Epoch         |
| Early Stopping Patience      | 3             |


### Training results

| Epoch | Training Loss | Validation Loss | Precision (%) | Recall (%) | F1 Score (%) | Accuracy (%) |
|-------|---------------|-----------------|---------------|------------|--------------|--------------|
| 1     | 0.0443        | 0.038108        | 91.88         | 95.17      | 93.50        | 98.80        |
| 2     | 0.0318        | 0.035728        | 94.13         | 96.15      | 95.13        | 98.90        |
| 3     | 0.0209        | 0.032016        | 94.81         | 96.42      | 95.61        | 99.01        |
| 4     | 0.0154        | 0.040221        | 93.87         | 95.80      | 94.82        | 98.88        |
| 5     | 0.0084        | 0.048183        | 94.21         | 96.06      | 95.13        | 98.93        |
| 6     | 0.0037        | 0.052281        | 94.49         | 96.60      | 95.53        | 99.07        |






### Author
abhijeet__@outlook.com

### Framework versions

- Transformers 4.38.2
- Pytorch 2.1.0+cu121
- Datasets 2.18.0
- Tokenizers 0.15.2