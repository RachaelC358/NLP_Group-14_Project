"""

Models.py
Created by Zachary Smith on 10/19/22

"""
# First model - Abstractive summarization using SpaCy

import spacy

from newspaper import Article
from heapq import nlargest
from string import punctuation
from spacy.lang.en.stop_words import STOP_WORDS

# spacy.cli.download("en_core_web_sm")

# nltk.download('stopwords')

# import tensorflow as tf

# import re

# import matplotlib.pyplot as plt
# nltk.download('punkt')

# stop_words = set(stopwords.words('english'))


# def text_clean(url_text):
#     tokens = url_text.lower()  # lowercase
#     tokens = [w for w in tokens.split() if w not in stop_words]  # removes nltk stopwords
#     tokens = list(set(tokens))  # removes duplicates
#     return tokens


# Spacy implementation

# Reference: https://www.numpyninja.com/post/text-summarization-through-use-of-spacy-library

def summarize(text, per):
    nlp = spacy.load('en_core_web_sm')  # load spacy model
    doc = nlp(text)
    word_frequencies = {}
    for word in doc:  # create frequencies dictionary
        if word.text.lower() not in list(STOP_WORDS):
            if word.text.lower() not in punctuation:
                if word.text not in word_frequencies.keys():  # laplace smoothing
                    word_frequencies[word.text] = 1
                else:
                    word_frequencies[word.text] += 1
    max_frequency = max(word_frequencies.values())

    for word in word_frequencies.keys():  # divide each word by the max frequency
        word_frequencies[word] = word_frequencies[word] / max_frequency
    sentence_tokens = [sent for sent in doc.sents]
    sentence_scores = {}
    for sent in sentence_tokens:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word.text.lower()]
                else:
                    sentence_scores[sent] += word_frequencies[word.text.lower()]

    select_length = int(len(sentence_tokens) * per)
    summary = nlargest(select_length, sentence_scores, key=sentence_scores.get)
    final_summary = [word.text for word in summary]
    summary = ''.join(final_summary)
    return summary


def main():
    # url = input("Enter sports recap URL: ")
    url = 'https://www.bleedinggreennation.com/2022/10/17/23408309/eagles-vs-cowboys-good-bad-ugly-cj-gardner-johnson-darius-slay-jalen-hurts-cooper-rush-results-recap'
    article = Article(url)
    article.download()
    article.parse()
    article.nlp()

    text = article.text
    # data = text_clean(text)
    # print(data)

    print(summarize(text, 0.3))


if __name__ == '__main__':
    main()
