import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
nltk.download('stopwords')
nltk.download('punkt')

def process_review(review):
    '''
    Input:
        review: a string containing a review
    Output:
        tweets_clean: a list of words containing the processed review

    '''
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    # remove stock market tickers like $GE
    review = re.sub(r'\$\w*', '', review)
    # remove old style retweet text "RT"
    review = re.sub(r'^RT[\s]+', '', review)
    # remove hyperlinks
    #review = re.sub(r'https?:\/\/.*[\r\n]*', '', review)
    review = re.sub(r'https?://[^\s\n\r]+', '', review)
    # remove hashtags
    # only removing the hash # sign from the word
    review = re.sub(r'#', '', review)
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    review_tokens = tokenizer.tokenize(review)

    reviews_clean = []
    for word in review_tokens:
        if (word not in stopwords_english and  # remove stopwords
            word not in string.punctuation):  # remove punctuation
            # reviews_clean.append(word)
            stem_word = stemmer.stem(word)  # stemming word
            reviews_clean.append(stem_word)

    return reviews_clean



def lookup(freqs, word, label):
    '''
    Input:
        freqs: a dictionary with the frequency of each pair (or tuple)
        word: the word to look up
        label: the label corresponding to the word
    Output:
        n: the number of times the word with its corresponding label appears.
    '''
    n = 0  

    pair = (word, label)
    if (pair in freqs):
        n = freqs[pair]

    return n

