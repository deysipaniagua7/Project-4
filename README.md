# Project-4: Job Fraud Detection Using BERT & Logistic Regression
## Project Overview
This project focuses on detecting fraudulent job postings using natural language processing (NLP) techniques. Leveraging the power of DistilBERT embeddings and a logistic regression classifier, we built a robust pipeline to analyze job description text and classify listings as either fraudulent or non-fraudulent.

## Dataset Source
Due file size limitations, we were unable to upload a .csv file into our repo. Thus, we have provided the direct link to the dataset below:
https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction

## Workflow Summary
1. Data Handling & Preprocessing
  Imported and explored dataset
  Cleaned data: removed blank or zero-value rows and cells
  Text pre-processing with nltk:
  Stopword removal
  Lemmatization (reducing words to their root forms)
  General text cleaning

2. Text Embedding
  Loaded DistilBERT and its tokenizer for efficiency and accuracy
  Created a custom BertEmbedder class to:
  Tokenize job description text
  Process text in batches
  Generate contextual BERT embeddings
  Transformed job descriptions into meaningful numerical vectors

3. Feature Extraction & Engineering
  TF-IDF Vectorization to capture important term patterns
  Chi-squared feature selection to identify high-signal (red flag) words associated with fraud
  Added a new column containing BERT-generated features for each job post

4. Modeling & Evaluation
  Split data into training and test sets
  Built a pipeline that integrates embedding, feature processing, and classification
  Trained a logistic regression model to classify postings
  Evaluated the model using classification reports (precision, recall, F1-score)
  Identified top 10 keywords for the first three test samples for interpretability

5. Output & Serialization
  Saved: Trained model
  TF-IDF vectorizer
  Processed datasets
  Feature selection results
  as .pkl files for future use and deployment

6. Goal
  The main goal of this project is to identify fraudulent job listings based solely on text data—helping protect users from scams and misleading offers.
  Tech Stack & Libraries
  Transformers (HuggingFace) — DistilBERT model & tokenizer
  PyTorch — for model inference
  NLTK — text pre-processing
  Scikit-learn — modeling, vectorization, evaluation
  Pandas — data handling

7. Key Takeaways
  Contextual embeddings (like BERT) greatly enhance model performance for NLP tasks.
  Simple models like logistic regression can still perform well when paired with strong features.
  Interpretable ML (TF-IDF + chi-squared) helps us explain model decisions and identify fraud indicators.

## Repo Structure
Due file size limitations, we were unable to upload each Trial's .ipynb file output. Therefore, below is a brief description of our repo including each Trial's .ipynb file and their output:
1. "Pickle_Coding": This file includes coding which allows users to load and open all pickle files.
3. "Pipeline + Bert_Plain TF_IDF Output.xlsx": This .csv file contains 2 tables showcasing a classification report and logistic regression results across all 4 trials.
4. "Project 4 Slides_FINAL.pptx":
5. "Trial 1_Fake Job_100 Row Test_FINAL": This file was run on 100 rows of data. It included a logistic regression and a list of top 10 keywords for the first 3 test samples.
6. "Trial 2_Fake Job_1000 Row Test_FINAL": This file was run on 1000 rows of data. It included the two tests from Trial 1 and a classifcation report.
7. "Trial 3_Fake Job_All Row Test_FINAL": This file was run on the entire dataset. It included a batch_size of 32 and all tests from Trial 2.
8. "Trial 4_Fake Job_All Row Test_FINAL" This file followed the same procedure used in Trial 3 and it included a chi-square test.
9. "Trial 4_red_flag_words_highlighted.xlsm":
10. "Visualization notebook.ipynb":

**Output Files not Included in Our Repo*
12. Output Coding: The .ipynb files for Trials 2-4 contain coding which allows users to generate and auto-save each model's "short_df.csv" and Pickle files locally and onto their Google drive.

## References
https://www.analyticsvidhya.com/blog/2020/10/overcoming-class-imbalance-using-smote-techniques/
https://aws.amazon.com/what-is/nlp/#:~:text=Natural%20language%20processing%20(NLP)%20is%20critical%20to%20fully%20and%20efficiently,day%2Dto%2Dday%20conversations
https://www.cbsnews.com/news/fake-job-listing-ghost-jobs-cbs-news-explains/
https://www.datacamp.com/tutorial/stemming-lemmatization-python
https://www.geeksforgeeks.org/understanding-tf-idf-term-frequency-inverse-document-frequency/
https://github.com/resources/articles/ai/natural-language-processing
https://www.indeed.com/career-advice/finding-a-job/how-to-know-if-a-job-is-a-scam
https://www.learndatasci.com/glossary/tf-idf-term-frequency-inverse-document-frequency/
https://medium.com/codex/properly-pickle-out-to-a-path-in-python-when-using-google-colab-741f0905e68b
https://stackoverflow.com/questions/49206488/accessing-pickle-file-in-google-colab
