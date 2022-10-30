# Rachael Carpenter 10/30/22
#
# Text Summarization using Gensim textRank algorithm
# 
# Extractive text summarization technique where words
# that occur more often are ranked as more important.
# Sentences that contain freqent words are extracted.

!pip install newspaper3k
import nltk

from newspaper import Article
from string import punctuation
nltk.download('punkt')

# Import gensim package library with summarizer
import gensim
from gensim.summarization import summarize


def main():
    # url = input("Enter sports recap URL: ")
    url = 'https://www.davyjoneslockerroom.com/2022/10/29/23428708/seattle-kraken-pittsburgh-penguins-preview-game-time-channel'
    article = Article(url)
    article.download()
    article.parse()

    text = article.text
  
    summary_by_ratio=summarize(text,ratio=0.3)
    print(summary_by_ratio)

if __name__ == '__main__':
    main()
