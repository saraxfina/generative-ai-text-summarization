# Project Planning

## Problem Statement  
In academic, corporate, and legal environments, there is a need for efficient comprenhension and decision-making from lengthy documents and articles. However, the time and effort these tasks demand pose a significant challenge. To address this, the SummarAIze model sets out to generate concise summaries while enriching them with contextually relavant details such as background information, definitions, and explantions. 


## Project Scope and Objectives  

* **Model Development**: Develop a summarization model capable of generating concise summaries from lengthy documents.  
* **Contextual Enrichment**: Implement method to enrich the summaries with contextually relevant information to enhance comprehension.  
* **Performance Optimization**: Optimize the model's performance to ensure efficient processing of large volumes of text while maintaining accuracy and coherence in the summaries.  
* **User Interface**: Ensure easy, user-friendly input of documents/articles and retrieval of enriched summaries.  
* **Evaluation**: Conduct thorough evaluation and validation of the model's effectiveness in generating accurate summaries and providing valuable contextual information.  
* **Deployment**: Deploy the model for demo usage.

## Key Deliverables
* Data dictionary and data sources  
* EDA notebooks  
* Model development scripts  
* API code and documentation  
* Presentation materials and project report

## Timeline  
Estimated duration: 36 hours

**Monday**: Setup; Planning; Data Access & Preparation; and start EDA  
**Tuesday**: EDA; Model Development; start Model Training & Validation  
**Wednesday**: Model Training & Validation; Model Auditing   
**Thursday**: Model Serving & Containerization; Project Documentation and Demo Preparation


## Dataset

### Summary & Usage

The CNN/DailyMail dataset is a collection of over 300,000 English-language news articles sourced from CNN and the Daily Mail. This dataset is often leveraged to train models for both extractive and abstractive summarization. Model evaluation entails comparing generated summaries against the author-defined highlights using ROUGE scores.   

### How to access

The CNN/DailyMail dataset is publicly available. Please refer to the [link](https://huggingface.co/datasets/cnn_dailymail).  

### Data Dictionary

| Name          | Description                        | Data Type | Required? |
| ------------- | ---------------------------------- | --------- | --------- |
| Article       | Body of the news article           | String    | Yes       |
| Highlights    | Author-chosen article highlights   | String    | Yes       |
| ID            | SHA1 hash of article URL           | String    | Yes       |

