# Twitter Sentiment Classification Using Machine Learning

This project focuses on predicting the sentiment of Twitter posts, classifying them into categories such as **Positive**, **Negative**, **Neutral**, and **Irrelevant**. It uses several machine learning models and **Natural Language Processing (NLP)** techniques to analyze the sentiment of tweets, based on the language used in the text.

## Objective

The goal of this project is to apply machine learning algorithms and NLP techniques to classify the sentiment of Twitter posts, which can help in understanding public opinions, monitoring brand health, and analyzing user feedback. The project involves data cleaning, text preprocessing, feature extraction, and the use of classification models like **Naive Bayes** and **Random Forest**. The performance of these models is evaluated based on various metrics like accuracy, precision, recall, and F1-score.

## Technologies Used

- **Python**: The primary programming language used for data processing and model implementation.
- **Scikit-learn**: A machine learning library used for model building, training, and evaluation.
- **Pandas**: For data manipulation and cleaning.
- **Matplotlib**: For data visualization, including sentiment distribution charts.
- **NLTK**: For text preprocessing, including tokenization, stopword removal, and punctuation elimination.
- **TfidfVectorizer**: For transforming text data into a format suitable for machine learning models.
- **Natural Language Processing (NLP)**: Used for various text processing tasks like tokenization, removing stopwords, stemming, and feature extraction from the text.

## Dataset

The dataset used for this project is the **Twitter Entity Sentiment Analysis** dataset, available on [Kaggle](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis). This dataset provides entity-level sentiment analysis on multilingual tweets, which is ideal for training and evaluating sentiment classification models.

## Steps Involved

### 1. **Data Preprocessing**:
   - Loading and cleaning the dataset.
   - Removing missing values and irrelevant data.
   - Text processing includes tokenizing, removing stopwords, and punctuation using NLP techniques.

### 2. **Exploratory Data Analysis (EDA)**:
   - Visualizing the distribution of sentiment labels in the dataset using pie charts.
   - Understanding the vocabulary size and the sparsity of the dataset.
   - Using NLP to extract relevant features from the text for model training.

### 3. **Modeling**:
   - Splitting the dataset into training and testing sets.
   - Training machine learning models, including **Naive Bayes** and **Random Forest** classifiers.
   - Evaluating the models' performance using metrics like accuracy, precision, recall, and F1-score.

### 4. **Overfitting and Correct Evaluation**:
   - Identifying overfitting in manual model evaluation when the training set is used for both fitting and prediction.
   - Ensuring correct model evaluation by splitting data into training and testing sets using a pipeline approach.

### 5. **Model Comparison**:
   - Comparing the performance of **Naive Bayes** and **Random Forest** classifiers to determine the most accurate model for sentiment classification.

## Results

- The **Random Forest Classifier** achieved the best performance, with high accuracy and recall, particularly in predicting **Positive** and **Negative** sentiments.
- The **Naive Bayes** model showed good results but suffered from overfitting when evaluated on the training data.
- The final recommendation is to use the **Random Forest Classifier** for its superior performance in classifying Twitter sentiment, based on accuracy and generalization ability.

