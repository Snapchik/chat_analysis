import re
import string
import numpy as np
import nltk
import torch

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


#This code is for linear regression model

#preprocessing words in a cleared text

def extract_hashtags(messages):
    try:
        return re.findall(r'#(\w+)', str(messages))[0]
    except IndexError as err:
        return np.nan

def extract_text(row):
  """Extracts text from a list nested with dictionary.

  Args:
    row: a row of the dataframe.

  Returns:
    A string of the text.
  """
  lst = []
  for i in row[1]:
    try:
      lst.extend(i.values())
    except AttributeError as err:
      lst.extend(i)
  lst = ''.join(lst)
  #getting rid of typetext/n
  typetext = '\w+'
  re.sub(typetext,'', lst)
  return lst

def process_text(text):
    """Process text function.
    Input:
        text: a string containing a text
    Output:
        texts_clean: a list of words containing the processed text

    """
    #Adding stopwords and stemmer a separate cell for reusability
    stopwords_english = stopwords.words('english')
    stemmer = PorterStemmer()
    text_tokens = nltk.word_tokenize(text)

    texts_clean = []
    for word in text_tokens:
        if (word not in stopwords_english and  # remove stopwords
                word not in string.punctuation):  # remove punctuation
            stem_word = stemmer.stem(word)  # stemming word
            texts_clean.append(stem_word)

    return texts_clean


#building frequency dictionary for words in each dataset - hammas and idf

def build_freqs(text_list, ys):
    """Build frequencies.
    Input:
        text_list: a list of texts
        ys: an m x 1 array with the sentiment label of each text
            (either 0 or 1)
    Output:
        freqs: a dictionary mapping each (word, sentiment) pair to its
        frequency
    """
    # Convert np array to list since zip needs an iterable.
    # The squeeze is necessary or the list ends up with one element.
    # Also note that this is just a NOP if ys is already a list.
    yslist = np.squeeze(ys).tolist()

    # Start with an empty dictionary and populate it by looping over all texts
    # and over all processed words in each text.
    freqs = {}
    for y, text in zip(yslist, text_list):
        for word in process_text(text):
          pair = (word, y)
          if pair in freqs:
              freqs[pair] += 1
          else:
              freqs[pair] = 1
    return freqs


#Using most popular activation function
def sigmoid(z):
    '''
    Input:
        z: is the input (can be a scalar or an array)
    Output:
        h: the sigmoid of z
    '''


    # calculate the sigmoid of z
    h = 1 / (1 + np.exp(-z))


    return h

#Here training is happening

def gradientDescent(x, y, theta, alpha, num_iters):
    '''
    Input:
        x: matrix of features which is (m,n+1)
        y: corresponding labels of the input matrix x, dimensions (m,1)
        theta: weight vector of dimension (n+1,1)
        alpha: learning rate
        num_iters: number of iterations you want to train your model for
    Output:
        J: the final cost
        theta: your final weight vector
        J_history: list of costs over iterations
    '''
    m = x.shape[0]
    eps = 1e-10
    J_history = []  # Initialize a list to store the cost at each iteration

    for i in range(num_iters):
        # get z, the dot product of x and theta
        z = np.dot(x, theta)

        # get the sigmoid of z
        h = sigmoid(z)

        # calculate the cost function
        J = - (1/m) * (np.dot(y.T, np.log(h + eps)) + np.dot((1 - y.T), np.log(1 - h + eps)))

        # Save the cost J in every iteration
        J_history.append(float(J))

        # update the weights theta
        theta = theta - (alpha/m) * np.dot(x.T, (h - y))

    J = float(J)
    return J, theta, J_history


def extract_features(text, freqs):
    '''
    Input:
        text: a list of words for one text
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
    Output:
        x: a feature vector of dimension (1,3)
    '''
    # process_text tokenizes, stems, and removes stopwords
    word_l = process_text(text)

    # 3 elements in the form of a 1 x 3 vector
    x = np.zeros((1, 3))

    #bias term is set to 1
    x[0,0] = 1

    # loop through each word in the list of words
    for word in word_l:

        # increment the word count for the hamas label 1
        x[0,1] += freqs.get((word, 1), 0)

        # increment the word count for the idf label 0
        x[0,2] += freqs.get((word, 0), 0)


    assert(x.shape == (1, 3))
    return x

def extract_features_t(text, freqs):
    '''
    Input:
        text: a list of words for one text
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
    Output:
        x: a feature vector of dimension (1,3)
    '''
    # process_text tokenizes, stems, and removes stopwords
    word_l = process_text(text)


    # 3 elements in the form of a 1 x 3 vector
    x = torch.zeros((1, 3))

    #bias term is set to 1
    x[0,0] = 1

    # loop through each word in the list of words
    for word in word_l:

        # increment the word count for the hamas label 1
        x[0,1] += freqs.get((word, 1), 0)

        # increment the word count for the idf label 0
        x[0,2] += freqs.get((word, 0), 0)


    assert(x.shape == (1, 3))
    return x

def predict_text(text, freqs, theta):
    '''
    Input:
        text: a string
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
        theta: (3,1) vector of weights
    Output:
        y_pred: the probability of a text being hamass or idf
    '''

    # extract the features of the text and store it into x
    x = extract_features(text, freqs)

    # make the prediction using x and theta
    y_pred = sigmoid(np.dot(x, theta))

    return y_pred

def test_logistic_regression(test_x, test_y, freqs, theta, predict_text=predict_text):
    """
    Input: 
        test_x: a list of texts
        test_y: (m, 1) vector with the corresponding labels for the list of texts
        freqs: a dictionary with the frequency of each pair (or tuple)
        theta: weight vector of dimension (3, 1)
    Output: 
        accuracy: (# of texts classified correctly) / (total # of texts)
    """
    
    
    # the list for storing predictions
    y_hat = []
    
    for text in test_x:
        # get the label prediction for the text
        y_pred = predict_text(text, freqs, theta)
        
        if y_pred > 0.5:
            # append 1.0 to the list
            y_hat.append(1.0)
        else:
            # append 0 to the list
            y_hat.append(0.0)

    # With the above implementation, y_hat is a list, but test_y is (m,1) array
    # convert both to one-dimensional arrays in order to compare them using the '==' operator
    accuracy = np.mean(((np.asarray(y_hat)) == np.squeeze(test_y)))
    print()
    
    return accuracy


def count_texts(result, texts, ys):
    '''
    Input:
        result: a dictionary that will be used to map each pair to its frequency
        texts: a list of text
        ys: a list corresponding to the sentiment of each text (either 0 or 1)
    Output:
        result: a dictionary mapping each pair to its frequency
    '''
    for y,text in zip(ys, texts):
        for word in process_text(text):
            # define the key, which is the word and label tuple
            pair = tuple(([word, y]))
            
            # if the key exists in the dictionary, increment the count
            if pair in result:
                result[pair] += 1

            # else, if the key is new, add it to the dictionary and set the count to 1
            else:
                result[pair] = 1

    return result


def train_naive_bayes(freqs, train_x, train_y):
    '''
    Input:
        freqs: dictionary from (word, label) to how often the word appears
        train_x: a list of texts
        train_y: a list of labels correponding to the texts (0,1)
    Output:
        logprior: the log prior. (equation 3 above)
        loglikelihood: the log likelihood of you Naive bayes equation. (equation 6 above)
    '''
    loglikelihood = {}
    logprior = 0


    # calculate V, the number of unique words in the vocabulary
    vocab = set(pair[0] for pair in freqs.keys())
    V = len(vocab)    

    # calculate N_pos, N_neg, V_pos, V_neg
    N_pos = N_neg = 0
    for pair in freqs.keys():
        # if the label is positive (greater than zero)
        if pair[1] > 0:

            # Increment the number of positive words by the count for this (word, label) pair
            N_pos += freqs[pair]

        # else, the label is negative
        else:

            # increment the number of negative words by the count for this (word,label) pair
            N_neg += freqs[pair]
    
    # Calculate D, the number of documents
    D = len(train_y)

    # Calculate D_pos, the number of positive documents
    D_pos = np.count_nonzero(train_y == 1)

    # Calculate D_neg, the number of negative documents
    D_neg = np.count_nonzero(train_y == 0)

    # Calculate logprior
    logprior = np.log(D_pos) - np.log(D_neg)
    
    # For each word in the vocabulary...
    for word in vocab:
        # get the positive and negative frequency of the word
        freq_pos = freqs.get((word, 1),0)
        freq_neg = freqs.get((word, 0),0)

        # calculate the probability that each word is positive, and negative
        p_w_pos = (freq_pos + 1) / (N_pos + V)
        p_w_neg = (freq_neg + 1) / (N_neg + V)

        # calculate the log likelihood of the word
        loglikelihood[word] = np.log((p_w_pos/p_w_neg))


    return logprior, loglikelihood


def naive_bayes_predict(text, logprior, loglikelihood):
    '''
    Input:
        text: a string
        logprior: a number
        loglikelihood: a dictionary of words mapping to numbers
    Output:
        p: the sum of all the logliklihoods of each word in the text (if found in the dictionary) + logprior (a number)

    '''
    # process the text to get a list of words
    word_l = process_text(text)

    # initialize probability to zero
    p = 0

    # add the logprior
    p += logprior

    for word in word_l:

        # check if the word exists in the loglikelihood dictionary
        if word in loglikelihood:
            # add the log likelihood of that word to the probability
            p += loglikelihood[word]

    return p



def test_naive_bayes(test_x, test_y, logprior, loglikelihood, naive_bayes_predict=naive_bayes_predict):
    """
    Input:
        test_x: A list of texts
        test_y: the corresponding labels for the list of texts
        logprior: the logprior
        loglikelihood: a dictionary with the loglikelihoods for each word
    Output:
        accuracy: (# of texts classified correctly)/(total # of texts)
    """
    accuracy = 0  # return this properly

    y_hats = []
    for text in test_x:
        # if the prediction is > 0
        if naive_bayes_predict(text, logprior, loglikelihood) > 0:
            # the predicted class is 1
            y_hat_i = 1
        else:
            # otherwise the predicted class is 0
            y_hat_i = 0

        # append the predicted class to the list y_hats
        y_hats.append(y_hat_i)

    # error is the average of the absolute values of the differences between y_hats and test_y
    error = np.mean(np.abs(y_hats - test_y))

    # Accuracy is 1 minus the error
    accuracy = 1 - error


    return accuracy


def lookup(freqs, word, label):
    '''
    Input:
        freqs: a dictionary with the frequency of each pair (or tuple)
        word: the word to look up
        label: the label corresponding to the word
    Output:
        n: the number of times the word with its corresponding label appears.
    '''
    n = 0  # freqs.get((word, label), 0)

    pair = (word, label)
    if (pair in freqs):
        n = freqs[pair]

    return n


def get_ratio(freqs, word):
    '''
    Input:
        freqs: dictionary containing the words

    Output: a dictionary with keys 'hammas', 'idf', and 'ratio'.
        Example: {'hammas': 10, 'idf': 20, 'ratio': 0.5}
    '''
    hammas_idf_ratio = {'hammas': 0, 'idf': 0, 'ratio': 0.0}

    # use lookup() to find positive counts for the word (denoted by the integer 1)
    hammas_idf_ratio['hammas'] = lookup(freqs, word, 1)
    
    # use lookup() to find negative counts for the word (denoted by integer 0)
    hammas_idf_ratio['idf'] = lookup(freqs, word, 0)
    
    # calculate the ratio of positive to negative counts for the word
    hammas_idf_ratio['ratio'] = (hammas_idf_ratio['hammas']+1) / (hammas_idf_ratio['idf']+1)

    return hammas_idf_ratio


def get_words_by_threshold(freqs, label, threshold, get_ratio=get_ratio):
    '''
    Input:
        freqs: dictionary of words
        label: 1 for positive, 0 for negative
        threshold: ratio that will be used as the cutoff for including a word in the returned dictionary
    Output:
        word_list: dictionary containing the word and information on its hammas count, idf count, and ratio of hammas to idf counts.
        example of a key value pair:
        {'happi':
            {'hammas': 10, 'idf': 20, 'ratio': 0.5}
        }
    '''
    word_list = {}

    for key in freqs.keys():
        word, _ = key

        # get the positive/negative ratio for a word
        hammas_idf_ratio = get_ratio(freqs, word)

        # if the label is 1 and the ratio is greater than or equal to the threshold...
        if label == 1 and hammas_idf_ratio['ratio'] >= threshold:
        
            # Add the pos_neg_ratio to the dictionary
            word_list[word] = hammas_idf_ratio

        # If the label is 0 and the pos_neg_ratio is less than or equal to the threshold...
        elif label == 0 and hammas_idf_ratio['ratio'] <= threshold:
        
            # Add the pos_neg_ratio to the dictionary
            word_list[word] = hammas_idf_ratio

        # otherwise, do not include this word in the list (do nothing)

    return word_list
