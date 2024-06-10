# Amazon-headset_reviews-analysis

This project analyzes Amazon product reviews focusing on various sentiment analysis techniques. The goal is to understand customer sentiment, identify patterns, and build models to predict the sentiment of reviews. The analysis is divided into four main parts:

Pre-processing

Exploratory Data Analysis (EDA)

Sentiment Analysis with Classical Methods

Sentiment Analysis with Word2Vec

# Table of Contents

Project 

Datasets

Installation

Usage

Notebooks

Results


Conclusion

# Project Overview

Amazon product reviews are a rich source of customer feedback and insights. This project focuses on analyzing these reviews to extract sentiment, understand customer preferences, and predict future sentiments. The analysis includes data preprocessing, exploratory data analysis, and building sentiment analysis models using various techniques.

# Datasets

The datasets used in this project include 2M Amazon phones and accessories reviews and metadata. The specific focus is on reviews of electronics and headsets (https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/)

# Installation

To run the notebooks and reproduce the analysis, you'll need to have Python and the following libraries installed:

pandas

numpy

matplotlib

seaborn

nltk

scikit-learn

keras

gensim


# 1. Pre-processing
Notebook: preprocessing.ipynb

This notebook covers the initial data cleaning and preprocessing steps, including:

Loading and merging review and metadata datasets

Handling missing values

Extracting relevant features (e.g., product titles containing specific keywords)

Converting rating classes into binary sentiment classes ('good' and 'bad')

# 2. Exploratory Data Analysis (EDA)

Notebook: eda.ipynb

This notebook provides insights into the dataset through various visualizations and statistical analysis:

Distribution of reviews across different brands and products

Analysis of review ratings

Visualization of top and bottom reviewed products

# 3. Sentiment Analysis with Classical Methods

Notebook: sentiment_analysis_classical.ipynb

This notebook implements sentiment analysis using classical machine learning methods:

Feature extraction using CountVectorizer, TF-IDF, and HashingVectorizer

Dimensionality reduction using TruncatedSVD and IncrementalPCA

Building and evaluating models using Logistic Regression, Random Forest, and gaussian Bayes

Visualization of model performance through confusion matrices and classification reports

# 4. Sentiment Analysis with Word2Vec

Notebook: sentiment_analysis_word2vec.ipynb

This notebook implements sentiment analysis using Word2Vec embeddings and a neural network:

Tokenization and sequence padding of text data

Training a Word2Vec model

Creating an embedding matrix

Building and training a neural network model using Keras

Evaluating model performance and visualizing training history

# Results

The analysis revealed key insights into customer sentiments and preferences in Amazon reviews. The models built using classical methods and Word2Vec embeddings provided accurate predictions of review sentiments. The visualizations and statistical analysis highlighted important trends and patterns in the data.

# Conclusion

This project demonstrates the power of natural language processing and machine learning techniques in analyzing and predicting customer sentiment from product reviews. The insights gained can help businesses improve their products and services based on customer feedback.
