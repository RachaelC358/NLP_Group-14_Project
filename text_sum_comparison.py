"""

Models.py
Created by Zachary Smith on 10/19/22

"""
# First model - Abstractive summarization using SpaCy
from nltk.corpus import stopwords
import warnings
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import random
import sumy
from sumy.utils import get_stop_words
from sumy.nlp.stemmers import Stemmer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.parsers.html import HtmlParser
import nltk
import spacy
from rouge import Rouge

from newspaper import Article
from heapq import nlargest
from string import punctuation
from spacy.lang.en.stop_words import STOP_WORDS

spacy.cli.download("en_core_web_sm")
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Spacy implementation

# Reference: https://www.numpyninja.com/post/text-summarization-through-use-of-spacy-library


def spacy_summarize(url, per):
    article = Article(url)
    article.download()
    article.parse()
    article.nlp()

    text = article.text

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


# Sumy implementation

LANGUAGE = "english"
SENTENCES_COUNT = 10


def sumy_summarize(url):
    parser = HtmlParser.from_url(url, Tokenizer(LANGUAGE))
    # Use plain text files instead of a URL
    # parser = PlaintextParser.from_file("document.txt", Tokenizer(LANGUAGE))
    # parser = PlaintextParser.from_string("Check this out.", Tokenizer(LANGUAGE))
    stemmer = Stemmer(LANGUAGE)

    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)

    text = ""
    for sentence in summarizer(parser.document, SENTENCES_COUNT):
        sentence = str(sentence)
        text = text + sentence + " "

    return text


# Scikit-learn implementation
"""
sklearn_model.py
Created by Jacob Benz on 10/31/22
"""
# sklearn model - Text summarization using sklearn and


# sklearn reference --> https://medium.com/saturdays-ai/building-a-text-summarizer-in-python-using-nltk-and-scikit-learn-class-tfidfvectorizer-2207c4235548

# Higher handicap = more strict summarization (shorter)
HANDICAP = 0.95


def remove_punctuation_marks(text):
    punctuation_marks = dict((ord(punctuation_mark), None)
                             for punctuation_mark in string.punctuation)
    print(punctuation_marks)
    return text.translate(punctuation_marks)


def get_lemmatized_tokens(text):
    normalized_tokens = nltk.word_tokenize(text.lower())
    return [nltk.stem.WordNetLemmatizer().lemmatize(normalized_token) for normalized_token in normalized_tokens]


def get_average(values):
    greater_than_zero_count = total = 0
    for value in values:
        if value != 0:
            greater_than_zero_count += 1
            total += value
    return total / greater_than_zero_count


def get_threshold(tfidf_results):
    i = total = 0
    while i < (tfidf_results.shape[0]):
        total += get_average(tfidf_results[i, :].toarray()[0])
        i += 1
    return total / tfidf_results.shape[0]


def sklearn_summarize(url):
    article = Article(url)
    article.download()
    article.parse()
    article.nlp()

    text = article.text

    print("Length of unfiltered article text --> ", len(text))

    # This method tokenizes then lemmatizes before analyzing
    documents = nltk.sent_tokenize(text)
    tfidf_results = TfidfVectorizer(tokenizer=get_lemmatized_tokens, stop_words=stopwords.words(
        'english')).fit_transform(documents)

    summary = ""
    i = 0
    while i < (tfidf_results.shape[0]):
        if (get_average(tfidf_results[i, :].toarray()[0])) >= get_threshold(tfidf_results) * HANDICAP:
            summary += ' ' + documents[i]
        i += 1
    return summary


def main():
    warnings.filterwarnings("ignore")

    # url = input("Enter sports recap URL: ")
    url1 = 'https://www.bleedinggreennation.com/2022/10/17/23408309/eagles-vs-cowboys-good-bad-ugly-cj-gardner-johnson-darius-slay-jalen-hurts-cooper-rush-results-recap'

    print("++++ SPACY SUMMARIZATION ++++")
    spacy_sum_text = spacy_summarize(url1, 0.2)
    print(spacy_sum_text)

    print("++++ SUMY SUMMARIZATION ++++")
    sumy_sum_text = sumy_summarize(url1)
    print(sumy_sum_text)

    print("++++ SKLEARN SUMMARIZATION ++++")
    sklearn_sum_text = sklearn_summarize(url1)
    print(sklearn_sum_text)

    url1_ref_summary = "The Eagles' defense threw some heat at Cowboys' backup quarterback Cooper Rush, who completed 18 of 38 for 181 yards and a touchdown, while throwing three interceptions, two by safety C.J. Gardner-Johnson and one by cornerback Darius Slay.The Eagles won without Pro Bowl right tackle Lane Johnson out for the second half under concussion protocol, after Johnson did a sound job corralling Dallas outside linebacker Micah Parsons.\
                        On second-and-nine at the Dallas 29, Hurts finding Brown for a 22-yard gain and a first down at the Dallas seven with 7:32 to play.\
                        On third-and-nine at the Dallas 26, cornerback Darius Slay stepped in front of a Rush pass intended for Michael Gallup with 5:14 left in the second quarter, which eventually led to Jake Elliott's second field goal.\
                        Hurts finding Brown on third-and-three at the Dallas 15 with his second TD reception of the season and a 14-0 Eagles' lead with 4:07 left in the half.\
                        Bradberry came back to deflect another Rush pass with 9:24 left in the half, on a critical fourth-and-one at the Dallas 34.\
                        The defended pass led to an Elliott 51-yard field goal and a commanding 17-0 Eagles' lead.\
                        On the Eagles' second drive, on fourth-and-three at the Dallas 38 with 3:38 left in the first quarter.\
                        Linebacker Haason Reddick not being able to get off the block of Cowboys' right tackle Terence Steele on the first play of the game, when Lamb took an end around for eight yards.\
                        Gardner-Johnson and Epps both whiffing on Dallas tight end Jake Ferguson on the Cowboys' seven-yard, fourth-quarter touchdown reception with 14:39 to play, putting Dallas within 20-17.\
                        The play led to Ezekiel Elliott's 14-yard touchdown run with 8:19 left in the third quarter.\
                        It left the Eagles with a third-and-18 at the Dallas 28, forcing them to settle for an Elliott 34-yard field goal and a 20-0 lead."

    rouge = Rouge()
    print("***** SPACY ROUGE SCORE 1 *****")
    spacy_rouge_score1 = rouge.get_scores(spacy_sum_text, url1_ref_summary)
    print(spacy_rouge_score1)
    print(type(spacy_rouge_score1))
    print(spacy_rouge_score1[0]["rouge-1"]["r"])
    print(type(spacy_rouge_score1[0]["rouge-1"]["r"]))

    print("***** SUMY ROUGE SCORE 1 *****")
    print(rouge.get_scores(sumy_sum_text, url1_ref_summary))

    print("***** SKLEARN ROUGE SCORE 1 *****")
    print(rouge.get_scores(sklearn_sum_text, url1_ref_summary))


if __name__ == '__main__':
    main()
