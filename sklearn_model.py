"""
sklearn_model.py
Created by Jacob Benz on 10/31/22
"""
# sklearn model - Text summarization using sklearn and 

import nltk
import random
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import warnings
from newspaper import Article
from nltk.corpus import stopwords 

# sklearn reference --> https://medium.com/saturdays-ai/building-a-text-summarizer-in-python-using-nltk-and-scikit-learn-class-tfidfvectorizer-2207c4235548

# Higher handicap = more strict summarization (shorter)
HANDICAP = 0.95

def remove_punctuation_marks(text) :
    punctuation_marks = dict((ord(punctuation_mark), None) for punctuation_mark in string.punctuation)
    print(punctuation_marks)
    return text.translate(punctuation_marks)

def get_lemmatized_tokens(text) :
    normalized_tokens = nltk.word_tokenize(text.lower())
    return [nltk.stem.WordNetLemmatizer().lemmatize(normalized_token) for normalized_token in normalized_tokens]

def get_average(values) :
    greater_than_zero_count = total = 0
    for value in values :
        if value != 0 :
            greater_than_zero_count += 1
            total += value 
    return total / greater_than_zero_count

def get_threshold(tfidf_results) :
    i = total = 0
    while i < (tfidf_results.shape[0]) :
        total += get_average(tfidf_results[i, :].toarray()[0])
        i += 1
    return total / tfidf_results.shape[0]

def get_summary(documents, tfidf_results) :
    summary = ""
    i = 0
    while i < (tfidf_results.shape[0]) :
        if (get_average(tfidf_results[i, :].toarray()[0])) >= get_threshold(tfidf_results) * HANDICAP :
                summary += ' ' + documents[i]
        i += 1
    return summary

if __name__ == "__main__" :
    warnings.filterwarnings("ignore")

    # uncomment these when first running if you don't have them
    #nltk.download('stopwords') 
    #nltk.download('punkt')
    #nltk.download('wordnet')
    

    # same article as spacy for a better performance comparison
    url = 'https://www.bleedinggreennation.com/2022/10/17/23408309/eagles-vs-cowboys-good-bad-ugly-cj-gardner-johnson-darius-slay-jalen-hurts-cooper-rush-results-recap'
    article = Article(url)
    article.download()
    article.parse()
    article.nlp()

    text = article.text
 
    print("Length of unfiltered article text --> ", len(text))

    # This method tokenizes then lemmatizes before analyzing
    documents = nltk.sent_tokenize(text)
    tfidf_results = TfidfVectorizer(tokenizer = get_lemmatized_tokens, stop_words = stopwords.words('english')).fit_transform(documents)

    print (get_summary(documents, tfidf_results))