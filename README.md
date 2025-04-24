# Project-4: Job Fraud Detection Using BERT & Logistic Regression

## Project Overview
This project focuses on detecting fraudulent job postings using natural language processing (NLP) techniques. Leveraging the power of DistilBERT embeddings and a logistic regression classifier, we built a robust pipeline to analyze job description text and classify listings as either fraudulent or non-fraudulent.The main goal of this project is to identify fake job listings based solely on text data—helping protect users from scams and misleading offers.

## Dataset Source
Due file size limitations, we were unable to upload a .csv file into our repo. Thus, we have provided the direct link to the dataset below:
https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction

## Workflow Summary
### 1. **Data Handling & Preprocessing**
- Imported and explored the dataset  
- Cleaned data: Removed blank or zero-value rows and cells  
- Text preprocessing with NLTK:  
  - Stopword removal  
  - Lemmatization (reducing words to root forms)  
  - General text cleaning  

### 2. **Text Embedding**
- Loaded DistilBERT and tokenizer for efficient and accurate embeddings  
- Created a custom `BertEmbedder` class to:  
  - Tokenize job description text  
  - Process text in batches  
  - Generate contextual BERT embeddings  
- Transformed job descriptions into meaningful numerical vectors  

### 3. **Feature Extraction & Engineering**
- Applied TF-IDF vectorization to capture important term patterns  
- Used chi-squared feature selection to identify high-signal words linked to fraud  
- Added a new column with BERT-generated features for each job post  

### 4. **Modeling & Evaluation**
- Split data into training and test sets  
- Built a pipeline combining embedding, feature processing, and classification  
- Trained a logistic regression model to detect fraudulent listings  
- Evaluated model using classification report (precision, recall, F1-score)  
- Extracted top 10 keywords for interpretability from the first 3 test samples 

### 5. **Output & Serialization**
- Saved the following as `.pkl` files for reuse and deployment:  
  - Trained model  
  - TF-IDF vectorizer  
  - Processed datasets  
  - Feature selection results 

### 6. **Tech Stack & Libraries Used**
- Transformers (HuggingFace) — DistilBERT model & tokenizer  
- PyTorch — for model inference  
- NLTK — text preprocessing  
- Scikit-learn — modeling, vectorization, evaluation  
- Pandas — data handling  

### 7. **Key Takeaways**
- Contextual embeddings (like BERT) significantly boost NLP model performance  
- Simple models (e.g., logistic regression) perform well with strong features  
- Interpretable ML (TF-IDF + chi-squared) explains predictions and highlights fraud indicators

## Repo Structure
Due to file size limitations, we were unable to upload the output of each Trial's .ipynb file. Below is a brief description of the repo structure, including each Trial's .ipynb file and their outputs:

"Pickle_Coding": Contains code to load and open all pickle files.

"Pipeline + Bert_Plain TF_IDF Output.xlsx": A .csv file with two tables showcasing classification reports and logistic regression results across all 4 trials.

"Project 4 Slides_FINAL.pptx": Final presentation slides.

"Trial 1_Fake Job_100 Row Test_FINAL": Ran on 100 rows of data, includes logistic regression and top 10 keywords for the first 3 test samples.

"Trial 2_Fake Job_1000 Row Test_FINAL": Ran on 1000 rows, includes tests from Trial 1 and a classification report.

"Trial 3_Fake Job_All Row Test_FINAL": Ran on the entire dataset with a batch size of 32, including all tests from Trial 2.

"Trial 4_Fake Job_All Row Test_FINAL": Similar procedure to Trial 3, with the addition of a chi-square test.

"Trial 4_red_flag_words_highlighted.xlsm": Contains the highlighted red-flag words.

"Visualization notebook.ipynb": Includes visualizations of the results.

Output Files Not Included
Output Coding: Trials 2-4 .ipynb files contain code for generating and auto-saving "short_df.csv" and Pickle files locally and to Google Drive.
## References
- https://www.analyticsvidhya.com/blog/2020/10/overcoming-class-imbalance-using-smote-techniques/-
-  https://aws.amazon.com/what-is/nlp/#:~:text=Natural%20language%20processing%20(NLP)%20is%20critical%20to%20fully%20and%20efficiently,day%2Dto%2Dday%20conversations
- https://www.cbsnews.com/news/fake-job-listing-ghost-jobs-cbs-news-explains/
- https://www.datacamp.com/tutorial/stemming-lemmatization-python
- https://www.geeksforgeeks.org/understanding-tf-idf-term-frequency-inverse-document-frequency/
- https://github.com/resources/articles/ai/natural-language-processing
- https://www.indeed.com/career-advice/finding-a-job/how-to-know-if-a-job-is-a-scam
- https://www.learndatasci.com/glossary/tf-idf-term-frequency-inverse-document-frequency/
- https://medium.com/codex/properly-pickle-out-to-a-path-in-python-when-using-google-colab-741f0905e68b
-  https://stackoverflow.com/questions/49206488/accessing-pickle-file-in-google-colab
