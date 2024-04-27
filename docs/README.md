# Table of Contents
1. [Setup](#phase1)  
2. [Project Planning](#phase2)  
3. [Data](#phase3)  
4. [Results](#phase4)  


## 1. Setup <a name="phase1"></a>
[Instructions](../config/README.md)  


## 2. Project Planning <a name="phase2"></a>

### Problem Statement  
In academic, corporate, and legal environments, there is a need for efficient comprenhension and decision-making from lengthy documents and articles. However, the time and effort these tasks demand pose a significant challenge. To address this, the SummarAIze model sets out to generate concise summaries while enriching them with contextually relavant details such as background information, definitions, and explanations.   

### Project Objectives  

* **Model Development**: Develop a summarization model capable of generating concise summaries from documents.  
* **Contextual Enrichment**: Enrich the summaries with contextually relevant information to enhance comprehension.  
* **Performance Optimization**: Optimize the model's performance to ensure efficient processing of large volumes of text while maintaining accuracy and coherence in the summaries.  
* **Evaluation**: Conduct thorough evaluation and validation of the model's effectiveness in generating accurate summaries and providing valuable contextual information.  
* **Deployment**: Deploy the model for demo usage.

### Key Deliverables
* Data dictionary and data sources  
* EDA notebooks  
* Model development scripts  
* API code and documentation  
* Presentation materials and project report

### Timeline  
Estimated duration: 36 hours

**Monday**: Setup; Planning; Data Access & Preparation; start EDA  
**Tuesday**: EDA; Model Development; start Model Training & Validation  
**Wednesday**: Model Training & Validation; Model Auditing   
**Thursday**: Model Serving & Containerization; Project Documentation & Demo Preparation


## 3. Data <a name="phase3"></a>

### Datasets

[**CNN/DailyMail**](../data/README.md): A collection of over 300,000 English-language news articles sourced from CNN and the Daily Mail; used for training summarization models.  
[**News API**](../data/README.md): A comprehensive collection of real-time news articles from global sources.  

### Data Dictionary

| Name          | Description                        | Data Type | Required? |
| ------------- | ---------------------------------- | --------- | --------- |
| Article       | Body of the news article           | String    | Yes       |
| Highlights    | Author-chosen article highlights   | String    | Yes       |
| ID            | SHA1 hash of article URL           | String    | No        |

Both datasets contain these three columns, in which the 'Article' column will be the documents and 'Highlights' the corresponding summaries. The 'ID' column is not used for this project.  

### Preprocessing steps

1. Remove missing values  
2. Convert all text to lowercase  
3. Remove special characters  
4. Split into test, train, and val sets


## 4. Results <a name="phase4"></a>

### Highlights

| **Reference Summary**                                         | **AI-Generated Summary**                                       |
|---------------------------------------------------------------|----------------------------------------------------------------|
| Follow live text and BBC Radio 5 Sports Extra commentary at   | Red Bull team principal Christian Horner told Sky Sports after |
| the Chinese Grand Prix.                                       | the race that Verstappen is on another planet at the moment.   |
|                                                               | On comparing his form to the rest of the field he added.       |
|---------------------------------------------------------------|----------------------------------------------------------------|
| Olympic champion Peres Jepchirchir storms to victory in a     | Kenya's Alexander Mutiso Munyao held off distance running      |
| women's only world record time of two hours 16 minutes and 16 | great Kenenisa Buckle to win the men's race in 2024.           |
| seconds at the London Marathon.                               |                                                                |
|---------------------------------------------------------------|----------------------------------------------------------------| 


### Evaluation

|               | **ROUGE-1** | **ROUGE-2** | **ROUGE-L** |
|---------------|-------------|-------------|-------------|
| **Precision** | 0.34        | 0.12        | 0.26        |
|---------------|-------------|-------------|-------------|
| **Recall**    | 0.24        | 0.09        | 0.18        |
|---------------|-------------|-------------|-------------|
| **F1**        | 0.27        | 0.09        | 0.20        |
|---------------|-------------|-------------|-------------|




