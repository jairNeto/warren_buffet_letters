""" Utils module """

from urllib.request import urlopen
from pathlib import Path
import re
from bs4 import BeautifulSoup
from tika import parser
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger',
               'stopwords', 'vader_lexicon'])


def get_text_from_html(url, tags_to_ignore=["script", "style"]):
    """Extract the text from a webpage

    Parameters
    ----------
    url: String
        The url
    tags_to_ignore: List
        List with the tags to skip when getting the text

    Returns
    -------
    String
        A string file with the text from the webpage
    """
    text = ''
    try:
        html = urlopen(url).read()
        soup = BeautifulSoup(html, features="html.parser")
        for script in soup(tags_to_ignore):
            script.extract()  # rip it out

        # get text
        text = soup.get_text()

        # break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # break multi-headlines into a line each
        chunks = (phrase.strip()
                  for line in lines for phrase in line.split("  "))
        # drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)
    except:
        print(f'Could not open the url {url}')

    return text


def get_text_from_pdf(path):
    """Extract the text from a pdf

    Parameters
    ----------
    path: String
        Path to a pdf file

    Returns
    -------
    String
        A string file with the text from the pdf file
    """
    text = ''
    try:
        raw = parser.from_file(path)
        text = raw['content']
    except:
        print(f'Could not open the path {path}')

    return text


def get_letters_corpus_dict(letters_pdf_path, init_year=1977, end_year=2020):
    """Build the dict where the keys are the years and the values are
    the text from the Warren Buffet letters

    Parameters
    ----------
    letters_pdf_path: String
        Path to the directory containing the pdf letters
    init_year: int
        The initial year to start getting the letters
    end_year: int
        The finial year to start getting the letters

    Returns
    -------
    Dictionary
        Dict where the keys are the years and the values are
    the text from the Warren Buffet letters
    """
    if init_year < 1977 or end_year > 2020:
        print('The range supported is between 1977 and 2020')
        return {}

    letters_dict = dict()
    letters_years = [year for year in range(init_year, end_year + 1)]
    for year in letters_years:
        if year >= 2000:
            filename = f'{year}ltr.pdf'
            path = Path(letters_pdf_path).joinpath(filename)
            letter_corpus = get_text_from_pdf(str(path))
        else:
            if year > 1997:
                url = f'https://www.berkshirehathaway.com/letters/{year}htm.html'
            else:
                url = f'https://www.berkshirehathaway.com/letters/{year}.html'
            letter_corpus = get_text_from_html(url)

        letters_dict[year] = letter_corpus

    return letters_dict


def draw_heatmap(df, figsize=(15, 6), cmap='YlOrBr', ylabel='', xlabel='', title=''):
    """Draw a heatmap using seaborn

    Parameters
    ----------
    df: Pandas Dataframe
        Pandas Dataframe with the data to show at the heatmap
    figsize: Tuple
        The plot figure size
    cmap: matplotlib colormap name or object, or list of colors, optional
        The mapping from data values to color space. If not provided,
        the default will depend on whether center is set.
    ylabel: String
        The y label of the plot
    xlabel: String
        The x label of the plot
    title: String
        The title of the plot
    """
    _, ax = plt.subplots(figsize=figsize)
    sns.heatmap(df, cmap=cmap, annot=False)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_title(title, fontsize=20, weight='bold')
    plt.show()


def tokenize(text, freq_words=[]):
    """Tokenize the text

    Parameters
    ----------
    text: String
        The message to be tokenized
    freq_words: List
        List with words that appears frequent at the text

    Returns
    -------
    List
        List with the clean tokens
    """
    text = text.lower()
    text = re.sub("[^a-zA-Z]", " ", text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stopwords.words('english')]
    tokens = [w for w in tokens if w not in freq_words]

    lemmatizer = WordNetLemmatizer()

    clean_tokens_list = []
    for tok in tokens:
        lemmatizer_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens_list.append(lemmatizer_tok)

    return clean_tokens_list


def get_most_frequent_combinatation(tokens, freq=10, num_word_combination=-1):
    """Get a dict with the most frequent onegram, bigram, trigram and quadgrams

    Parameters
    ----------
    text: List
        List with the tokens
    freq: Int
        How many combination to return
    num_word_combination: Int
        1 to onegram
        2 to bigram
        3 to trigram
        4 to quadgrams
        -1 All

    Returns
    -------
    Dict
        Dict with the frequencies
    """
    if num_word_combination < -1 or num_word_combination > 4:
        raise Exception(
            f'The num_word_combination shall be greater than -2 and lesser than 5 the values passes was {num_word_combination}')

    freq_dict = {}
    if num_word_combination in [1, -1]:
        freq_dist = nltk.FreqDist(tokens)
        freq_dict['FreqDist_onegram'] = freq_dist.most_common(freq)

    if num_word_combination in [2, -1]:
        bigrams = nltk.collocations.BigramCollocationFinder.from_words(tokens)
        freq_dict['FreqDist_bigram'] = bigrams.ngram_fd.most_common(freq)

    if num_word_combination in [3, -1]:
        trigram = nltk.collocations.TrigramCollocationFinder.from_words(tokens)
        freq_dict['FreqDist_trigram'] = trigram.ngram_fd.most_common(freq)

    if num_word_combination in [4, -1]:
        quadgrams = nltk.collocations.QuadgramCollocationFinder.from_words(
            tokens)
        freq_dict['FreqDist_quadgrams'] = quadgrams.ngram_fd.most_common(freq)

    return freq_dict


def drawn_wordcloud(corpus, save_path, figsize=(15, 6)):
    """Get a dict with the most frequent onegram, bigram, trigram and quadgrams

    Parameters
    ----------
    corpus: List
        List with the words
    save_path: String
        Path to the file where the wordcloud will be saved at
    figsize: Tuple
        The figsize
    """
    _, _ = plt.subplots(figsize=figsize)
    combined_text = " ".join(text for text in corpus)
    wordcloud = WordCloud().generate(combined_text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig(save_path)
    plt.show()


def tokenize_sent(text):
    """Tokenize the sentence

    Parameters
    ----------
    text: String
        The text to be tokenized

    Returns
    -------
    List
        List with the tokenized sentences
    """
    sentence_list = nltk.tokenize.sent_tokenize(text)
    tokenized_list = []
    for sentence in sentence_list:
        sentence_after_regex = re.sub("[^a-z0-9A-Z]", " ", sentence)
        # Remove sentences where there was only numbers
        if len(re.sub("[^a-zA-Z]", "", sentence_after_regex)) > 6:
            tokenized_list.append(sentence_after_regex)

    return tokenized_list


def calculate_text_sentiment_using_transform(sentence_list):
    """Calculate the test sentiment using transforms

    Parameters
    ----------
    sentence_list: List
        List with the tokenizes sentences

    Returns
    -------
    Dictonary
        Dict with the cumulative sentiment of all the sentences at the list
    """
    sentiment_dict = {'POSITIVE': 0, 'NEGATIVE': 0}
    classifier = pipeline('sentiment-analysis')
    for sentence in sentence_list:
        sentiment_result = classifier(sentence)
        sentiment_dict[sentiment_result[0]['label']
                       ] += sentiment_result[0]['score']

    return sentiment_dict


def calculate_text_sia(sentence_list):
    """Calculate the test sentiment using Sentiment Intensity Analyzer

    Parameters
    ----------
    sentence_list: List
        List with the tokenizes sentences

    Returns
    -------
    Dictonary
        Dict with the cumulative sentiment of all the sentences at the list
    """
    sentiment_dict = {'neg': 0, 'neu': 0, 'pos': 0, 'compound': 0}
    sia = SentimentIntensityAnalyzer()
    for sentence in sentence_list:
        sentiment_result = sia.polarity_scores(sentence)
        for k in sentiment_result:
            sentiment_dict[k] += sentiment_result[k]

    return sentiment_dict


def get_sentiment_analysis_df(letters_dict,
                              calculate_text_sentiment,
                              tokenize_sent,
                              normalized=True):
    """Get the DataFrame with the sentiment of each Warren letter

    Parameters
    ----------
    letters_dict: Dictonary
        Dict with the letters text
    calculate_text_sentiment: function
        Function used to calculate the sentiment of the text
    tokenize_sent: function
        Function to tokenize the text into a list of sentences
    normalized: bool
        If the values of the df will be normalized or not

    Returns
    -------
    Pandas DataFrame
        Pandas DataFrame with the sentiment analysis for each letter
    """
    sentiment_analysis_dict = {}
    for k in letters_dict:
        sentiment_analysis_dict[k] = calculate_text_sentiment(
            tokenize_sent(letters_dict[k]))
    sentiment_analysis_df = pd.DataFrame(sentiment_analysis_dict)
    if normalized:
        return sentiment_analysis_df / sentiment_analysis_df.sum(axis=0)
    return sentiment_analysis_df


def get_answer_using_qa(nlp, question, context):
    """Get answer using a classifier trained with the QA technique

    Parameters
    ----------
    nlp: Pipeline
        Trained QA Pipeline
    question: String
        Question that the model will answer
    context: String
        The Context of the question

    Returns
    -------
    Tuple
        The answer, the score, the start position of the answer at the text and the final
        position of the answer at the text
    """
    result = nlp(question=question, context=context)

    return result['answer'], round(result['score'], 4), result['start'], result['end']


def format_spines(ax, right_border=True):
    """
    This function sets up borders from an axis and personalize colors

    Parameters
    ----------
        Axis: Matplotlib axis
            The plot axis
        right_border: Boolean
            Whether to plot or not the right border
    """
    # Setting up colors
    ax.spines['bottom'].set_color('#CCCCCC')
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['top'].set_visible(False)
    if right_border:
        ax.spines['right'].set_color('#CCCCCC')
    else:
        ax.spines['right'].set_color('#FFFFFF')
    ax.patch.set_facecolor('#FFFFFF')


def get_ngram_plot_data(df, type, sentiment):
    """Format the data to the ngram plot

    Parameters
    ----------
    df: Pandas DataFrame
        Pandas dataframe with the ngrams sentiment data
    type: String
        Type of the ngram to filter
    sentiment: String
        POSITIVE or NEGATIVE

    Returns
    -------
    Pandas Dataframe
        The dataframe filtered
    """
    return df.query("type == @type and sentiment == @sentiment").sort_values('score', ascending=False)
