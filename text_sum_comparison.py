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
    url2 = 'https://www.espn.com/soccer/fifa-world-cup/story/4822368/2022-world-cup-japan-rally-past-spain-as-both-sides-reach-round-of-16'
    url3 = 'https://www.foxsports.com/stories/soccer/more-to-come-u-s-projecting-confidence-heading-into-netherlands-match'

    # # Cowboys vs. Eagles article
    # print("++++ SPACY SUMMARIZATION ++++")
    # spacy_sum_text = spacy_summarize(url1, 0.2)
    # print(spacy_sum_text)

    # print("++++ SUMY SUMMARIZATION ++++")
    # sumy_sum_text = sumy_summarize(url1)
    # print(sumy_sum_text)

    # print("++++ SKLEARN SUMMARIZATION ++++")
    # sklearn_sum_text = sklearn_summarize(url1)
    # print(sklearn_sum_text)

    # url1_ref_summary = "The Eagles' defense threw some heat at Cowboys' backup quarterback Cooper Rush, who completed 18 of 38 for 181 yards and a touchdown, while throwing three interceptions, two by safety C.J. Gardner-Johnson and one by cornerback Darius Slay.The Eagles won without Pro Bowl right tackle Lane Johnson out for the second half under concussion protocol, after Johnson did a sound job corralling Dallas outside linebacker Micah Parsons.\
    #                     On second-and-nine at the Dallas 29, Hurts finding Brown for a 22-yard gain and a first down at the Dallas seven with 7:32 to play.\
    #                     On third-and-nine at the Dallas 26, cornerback Darius Slay stepped in front of a Rush pass intended for Michael Gallup with 5:14 left in the second quarter, which eventually led to Jake Elliott's second field goal.\
    #                     Hurts finding Brown on third-and-three at the Dallas 15 with his second TD reception of the season and a 14-0 Eagles' lead with 4:07 left in the half.\
    #                     Bradberry came back to deflect another Rush pass with 9:24 left in the half, on a critical fourth-and-one at the Dallas 34.\
    #                     The defended pass led to an Elliott 51-yard field goal and a commanding 17-0 Eagles' lead.\
    #                     On the Eagles' second drive, on fourth-and-three at the Dallas 38 with 3:38 left in the first quarter.\
    #                     Linebacker Haason Reddick not being able to get off the block of Cowboys' right tackle Terence Steele on the first play of the game, when Lamb took an end around for eight yards.\
    #                     Gardner-Johnson and Epps both whiffing on Dallas tight end Jake Ferguson on the Cowboys' seven-yard, fourth-quarter touchdown reception with 14:39 to play, putting Dallas within 20-17.\
    #                     The play led to Ezekiel Elliott's 14-yard touchdown run with 8:19 left in the third quarter.\
    #                     It left the Eagles with a third-and-18 at the Dallas 28, forcing them to settle for an Elliott 34-yard field goal and a 20-0 lead."

    # rouge = Rouge()
    # print("***** SPACY ROUGE SCORE 1 *****")
    # spacy_rouge_score1 = rouge.get_scores(spacy_sum_text, url1_ref_summary)
    # print(spacy_rouge_score1)
    # print(type(spacy_rouge_score1))
    # print(spacy_rouge_score1[0]["rouge-1"]["r"])
    # print(type(spacy_rouge_score1[0]["rouge-1"]["r"]))

    # print("***** SUMY ROUGE SCORE 1 *****")
    # print(rouge.get_scores(sumy_sum_text, url1_ref_summary))

    # print("***** SKLEARN ROUGE SCORE 1 *****")
    # print(rouge.get_scores(sklearn_sum_text, url1_ref_summary))

    # # Japan, Spain, and Germany World Cup article
    # print("++++ SPACY SUMMARIZATION ++++")
    # spacy_sum_text = spacy_summarize(url2, 0.2)
    # print(spacy_sum_text)

    # print("++++ SUMY SUMMARIZATION ++++")
    # sumy_sum_text = sumy_summarize(url2)
    # print(sumy_sum_text)

    # print("++++ SKLEARN SUMMARIZATION ++++")
    # sklearn_sum_text = sklearn_summarize(url2)
    # print(sklearn_sum_text)

    # url2_ref_summary = "Spain led through Alvaro Morata's early goal, but Japan, who couldn't manage a shot on target in the first half, turned the game around shortly after the break with two goals during a frantic three-minute spell thanks to halftime substitute Ritsu Doan and Ao Tanaka.\
    #                     Tanaka's goal was initially ruled out after the ball was judged to have run out of play before Kaoru Mitoma's cut-back, but VAR overturned the decision and gave Japan the three points they needed to be sure of a place in the knockout rounds.\
    #                     JUMP TO: Player ratings | Best/worst performers | Highlights and notable moments | Postmatch quotes | Key stats | Upcoming fixtures\
    #                     After demolishing Costa Rica 7-0 in the group stage opener, looking comfortable for large spells against Germany and then taking an early lead against Japan, it's almost unbelievable that for a period in the second half here Spain were going out.\
    #                     Japan's rally with two goals in three minutes early in the second half -- combined with Costa Rica taking an unlikely lead against Germany -- meant Spain dropped down to third in Group E and facing an early exit.\
    #                     Spain manager Luis Enrique has Kai Havertz to thank for turning the game against Costa Rica in Germany's favour, but it will be a worry that when La Roja were desperate for a goal, they looked toothless.\
    #                     Winning the group would have likely meant a quarterfinal matchup with Brazil, but now Spain's route includes Morocco in the second round, possibly Portugal in the quarterfinals and perhaps France or England in the semifinals.\
    #                     In the first half, both Rodri (115) and Pau Torres (109) completed more passes than Japan did as an entire team (89) but as is sometimes the case with Spain, it didn't really mean anything, and for spells of the game it almost looked like they wanted to finish second in the group.\
    #                     Japan manager Hajime Moriyasu, on the win: \"We played against Spain, one of the best teams in the world and we knew before the game that this was going to be very tough and difficult, and indeed it was,\" said Moriyasu whose side lost to Costa Rica in their second match in Qatar.\
    #                     Japan's 18% possession against Spain is the lowest by any team at a World Cup game since 1966.\
    #                     Japan are the second team to defeat both Spain and Germany at a single World Cup, joining Austria's 1978 squad.\
    #                     It is the second straight World Cup that the two geographical neighbors will face each other, with La Roja and the Atlas Lions playing to a 2-2 draw in 2018 in the group stage."

    # rouge = Rouge()
    # print("***** SPACY ROUGE SCORE 1 *****")
    # spacy_rouge_score2 = rouge.get_scores(spacy_sum_text, url2_ref_summary)
    # print(spacy_rouge_score2)
    # print(type(spacy_rouge_score2))
    # print(spacy_rouge_score2[0]["rouge-1"]["r"])
    # print(type(spacy_rouge_score2[0]["rouge-1"]["r"]))

    # print("***** SUMY ROUGE SCORE 1 *****")
    # print(rouge.get_scores(sumy_sum_text, url2_ref_summary))

    # print("***** SKLEARN ROUGE SCORE 1 *****")
    # print(rouge.get_scores(sklearn_sum_text, url2_ref_summary))

    # USA and Netherlands World Cup article
    print("++++ SPACY SUMMARIZATION ++++")
    spacy_sum_text = spacy_summarize(url3, 0.2)
    print(spacy_sum_text)

    print("++++ SUMY SUMMARIZATION ++++")
    sumy_sum_text = sumy_summarize(url3)
    print(sumy_sum_text)

    print("++++ SKLEARN SUMMARIZATION ++++")
    sklearn_sum_text = sklearn_summarize(url3)
    print(sklearn_sum_text)

    url3_ref_summary = "FIFA World Cup 2022 'More to come': U.S. projecting confidence heading into Netherlands match 10 hours ago share facebook twitter reddit link\
                        Two days before the United States plays the Netherlands on Saturday (10 a.m. ET, FOX and the FOX Sports app) in a colossal round of 16 contest at the 2022 World Cup in Qatar, U.S. stars Christian Pulisic and Tim Weah told reporters during a Thursday news conference that they're confident about getting past the powerful Dutch.\
                        USMNT feeling good Christian Pulisic and Tim Weah talk about the matchup with the Netherlands in the Round of 16 on Saturday.\
                        They're FIFA's No. 8-ranked team (the Americans are 16th) and finished first in Group A with wins over Senegal and the host nation.\
                        \"We all know the Netherlands is a big team, a lot of quality players,\" Weah said.\
                        The team didn't qualify for the 2018 World Cup, either.\
                        A smiling Pulisic, a safe bet to overcome the pelvic injury he suffered in scoring the Americans game-winning goal against Iran even if U.S. Soccer is listing his status as day-to-day, looked more relaxed than he ever has in front of the media Thursday as he laughed and joked with reporters.\
                        Get more from FIFA World Cup 2022 Follow your favorites to get information about games, news and more"

    rouge = Rouge()
    print("***** SPACY ROUGE SCORE 1 *****")
    spacy_rouge_score3 = rouge.get_scores(spacy_sum_text, url3_ref_summary)
    print(spacy_rouge_score3)
    print(type(spacy_rouge_score3))
    print(spacy_rouge_score3[0]["rouge-1"]["r"])
    print(type(spacy_rouge_score3[0]["rouge-1"]["r"]))

    print("***** SUMY ROUGE SCORE 1 *****")
    print(rouge.get_scores(sumy_sum_text, url3_ref_summary))

    print("***** SKLEARN ROUGE SCORE 1 *****")
    print(rouge.get_scores(sklearn_sum_text, url3_ref_summary))


if __name__ == '__main__':
    main()
