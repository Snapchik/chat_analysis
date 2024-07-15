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
    """Process tweet function.
    Input:
        text: a string containing a tweet
    Output:
        texts_clean: a list of words containing the processed tweet

    """
    #Adding stopwords and stemmer a separate cell for reusability
    stopwords_english = stopwords.words('english')
    stemmer = PorterStemmer()
    text_tokens = nltk.word_tokenize(text)

    texts_clean = []
    for word in text_tokens:
        if (word not in stopwords_english and  # remove stopwords
                word not in string.punctuation):  # remove punctuation
            # tweets_clean.append(word)
            stem_word = stemmer.stem(word)  # stemming word
            texts_clean.append(stem_word)

    return texts_clean


#building frequency dictionary for words in each dataset - hammas and idf

def build_freqs(text_list, ys):
    """Build frequencies.
    Input:
        text_list: a list of texts
        ys: an m x 1 array with the sentiment label of each tweet
            (either 0 or 1)
    Output:
        freqs: a dictionary mapping each (word, sentiment) pair to its
        frequency
    """
    # Convert np array to list since zip needs an iterable.
    # The squeeze is necessary or the list ends up with one element.
    # Also note that this is just a NOP if ys is already a list.
    yslist = np.squeeze(ys).tolist()

    # Start with an empty dictionary and populate it by looping over all tweets
    # and over all processed words in each tweet.
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
    Hint: you might want to print the cost to make sure that it is going down.
    '''

    # get 'm', the number of rows in matrix x
    m = x.shape[0]

    for i in range(0, num_iters):

        # get z, the dot product of x and theta
        z = np.dot(x, theta)

        # get the sigmoid of z
        h = sigmoid(z)

        # calculate the cost function
        J = - (1/m) * (np.dot(y.T, np.log(h)) + np.dot((1 - y.T), np.log(1 - h)))

        # update the weights theta
        theta = theta - (alpha/m) * np.dot(x.T, (h - y))


    J = float(J)
    return J, theta


def extract_features(text, freqs):
    '''
    Input:
        text: a list of words for one text
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
    Output:
        x: a feature vector of dimension (1,3)
    '''
    # process_tweet tokenizes, stems, and removes stopwords
    word_l = process_text(text)

    # 3 elements in the form of a 1 x 3 vector
    x = np.zeros((1, 3))

    #bias term is set to 1
    x[0,0] = 1

    # loop through each word in the list of words
    for word in text:

        # increment the word count for the hamas label 1
        x[0,1] += freqs.get((word, 1), 0)

        # increment the word count for the idf label 0
        x[0,2] += freqs.get((word, 0), 0)


    assert(x.shape == (1, 3))
    return x

def extract_features_t(text, freqs):
    '''
    Input:
        tweet: a list of words for one tweet
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
    Output:
        x: a feature vector of dimension (1,3)
    '''

    # 3 elements in the form of a 1 x 3 vector
    x = torch.zeros((1, 3))

    #bias term is set to 1
    x[0,0] = 1



    # loop through each word in the list of words
    for word in text:

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

    # extract the features of the tweet and store it into x
    x = extract_features(text, freqs)

    # make the prediction using x and theta
    y_pred = sigmoid(np.dot(x, theta))

    return y_pred






# def process_df(df):
#     """Process dataframe function.
#     Input:
#         df: a dataframe containing a json export
#     Output:
#         text_clean: a list of words containing the dataframe tweet

#     """
#     stemmer = PorterStemmer()
#     stopwords_english = stopwords.words('english')
#     # remove stock market tickers like $GE
#     tweet = re.sub(r'\$\w*', '', tweet)
#     # remove old style retweet text "RT"
#     tweet = re.sub(r'^RT[\s]+', '', tweet)
#     # remove hyperlinks    
#     tweet = re.sub(r'https?://[^\s\n\r]+', '', tweet)
#     # remove hashtags
#     # only removing the hash # sign from the word
#     tweet = re.sub(r'#', '', tweet)
#     # tokenize tweets
#     tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
#                                reduce_len=True)
#     tweet_tokens = tokenizer.tokenize(tweet)

#     tweets_clean = []
#     for word in tweet_tokens:
#         if (word not in stopwords_english and  # remove stopwords
#                 word not in string.punctuation):  # remove punctuation
#             # tweets_clean.append(word)
#             stem_word = stemmer.stem(word)  # stemming word
#             tweets_clean.append(stem_word)

#     return tweets_clean


# def build_freqs(tweets, ys):
#     """Build frequencies.
#     Input:
#         tweets: a list of tweets
#         ys: an m x 1 array with the sentiment label of each tweet
#             (either 0 or 1)
#     Output:
#         freqs: a dictionary mapping each (word, sentiment) pair to its
#         frequency
#     """
#     # Convert np array to list since zip needs an iterable.
#     # The squeeze is necessary or the list ends up with one element.
#     # Also note that this is just a NOP if ys is already a list.
#     yslist = np.squeeze(ys).tolist()

#     # Start with an empty dictionary and populate it by looping over all tweets
#     # and over all processed words in each tweet.
#     freqs = {}
#     for y, tweet in zip(yslist, tweets):
#         for word in process_tweet(tweet):
#             pair = (word, y)
#             if pair in freqs:
#                 freqs[pair] += 1
#             else:
#                 freqs[pair] = 1

#     return freqs
