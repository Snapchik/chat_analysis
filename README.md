# Chat Analysis Project

This project analyzes text messages from different chat sources using various Natural Language Processing (NLP) techniques and machine learning models.

## Project Overview

The project implements multiple approaches to analyze and classify chat messages:

1. **Logistic Regression Model**
   - Custom implementation of logistic regression from scratch
   - Feature extraction from text data
   - Model evaluation and error analysis
   - PyTorch implementation for comparison

2. **Naive Bayes Classification**
   - Implementation of Naive Bayes algorithm for text classification
   - Calculation of prior and likelihood probabilities
   - Word frequency analysis
   - Model accuracy evaluation

3. **Cosine Similarity Analysis**
   - Using Gensim for text embedding
   - Creation of TF-IDF and LSI models
   - Document similarity comparisons
   - Vector space modeling

## Technical Implementation

### Data Processing
- Text preprocessing including tokenization and cleaning
- Feature extraction for machine learning models
- Vocabulary building and frequency analysis

### Models
- Custom Logistic Regression implementation
- Naive Bayes classifier
- Gensim-based similarity models
- PyTorch neural network implementation

### Tools and Libraries
- NumPy for numerical computations
- Pandas for data manipulation
- NLTK for natural language processing
- Gensim for text similarity analysis
- PyTorch for deep learning implementation
- Matplotlib for visualization

## Results

The project achieves high accuracy in classification tasks:
- Logistic Regression accuracy: ~99.2%
- Naive Bayes accuracy: ~99.7%

The cosine similarity analysis successfully identifies semantically similar messages across different chat sources.

## Project Structure

- `Logistic_regression_which_chat.ipynb`: Logistic regression implementation and analysis
- `Naive_Bayes_which_chat.ipynb`: Naive Bayes classifier implementation
- `Cosine_similarity.ipynb`: Text similarity analysis using Gensim
- `utils.py`: Helper functions for text processing and model implementation

## Usage

The project is implemented in Jupyter notebooks. To run the analysis:

1. Install required dependencies
2. Load the chat data
3. Run the notebooks in sequence
4. Examine results and visualizations

## Dependencies

- Python 3.x
- NumPy
- Pandas
- NLTK
- Gensim
- PyTorch
- Matplotlib