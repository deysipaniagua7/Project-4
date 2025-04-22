# Project-4: Job Fraud Detection Using BERT & Logistic Regression
## Project Overview
This project focuses on detecting fraudulent job postings using natural language processing (NLP) techniques. Leveraging the power of DistilBERT embeddings and a logistic regression classifier, we built a robust pipeline to analyze job description text and classify listings as either fraudulent or non-fraudulent.

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
