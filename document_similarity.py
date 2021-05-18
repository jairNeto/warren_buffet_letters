import pickle
import nltk
import re
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import euclidean_distances
from gensim.models.word2vec import Word2Vec
import constants as const
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sentence_transformers import SentenceTransformer
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])


def tokenize(text):
    """Tokenize the text

    Parameters
    ----------
    text: String
        The message to be tokenized

    Returns
    -------
    List
        List with the clean tokens
    """
    text = text.lower()
    text = re.sub("[^a-zA-Z0-9]", " ", text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stopwords.words(const.ENGLISH)]

    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()

    clean_tokens_list = []
    for tok in tokens:
        lemmatizer_tok = lemmatizer.lemmatize(tok).strip()
        clean_tok = stemmer.stem(lemmatizer_tok)
        clean_tokens_list.append(clean_tok)

    return clean_tokens_list


def build_model():
    """Build the model

    Returns
    -------
    sklearn.pipeline.Pipeline
        The model
    """
    pipeline = Pipeline([
        (const.FEATURES, FeatureUnion([

            (const.TEXT_PIPELINE, Pipeline([
                (const.VECT, CountVectorizer(tokenizer=tokenize)),
                (const.TFIDF, TfidfTransformer())
            ]))
    ]))])

    return pipeline


def get_avg_document_vector(model, df, year):
    """Get the a vector representation of a document using word2vec

    Parameters
    ----------
    model: Word2Vec
        Trained Word2Vec model
    df: pandas DataFrame
        Pandas DataFrame with a columns with the tokens
    year: int
        The target year

    Returns
    -------
    Tuple
        The vector representation of the document, the number os words that are not at the model
        vocabulary
    """
    word_vecs = []
    count = 0
    for word in df[const.TOKENIZED].loc[year]:
        try:
            vector = model[word]
            word_vecs.append(vector)
        except KeyError:
            count += 1
            pass
    vector_avg = np.mean(word_vecs, axis=0)

    return vector_avg, count


def get_letters_df(letters_dict_pickle):
    """Get the letters Pandas Dataframe

    Parameters
    ----------
    letters_dict_pickle: string
        Path to the dict with the letters text

    Returns
    -------
    Pandas DataFrame
        Pandas DataFrame with a columns with the tokens
    """
    with open(letters_dict_pickle, 'rb') as handle:
        letters_dict = pickle.load(handle)

    letters_df = pd.DataFrame(letters_dict, index=[const.LETTER_TEXT]).T
    letters_df[const.TOKENIZED] = letters_df[const.LETTER_TEXT].apply(tokenize)

    return letters_df


def get_most_similar_docs(pairwise_similarities, letter_year, distance_method, transformers=False, initial_year=1977):
    """Get the most similar letters to a target one

    Parameters
    ----------
    pairwise_similarities: np.array
        Numpy array of the pairwise similarities
    letter_year: int
        The target letter year
    distance_method: string
        Euclidean or cosine
    transformers: boolean
        True if you are calling from transformers or False otherwise
    initial_year: int
        The initial letter year

    Returns
    -------
    List
        List with the letter year sorted descending by similarity
    """
    letter_i = letter_year - initial_year
    if distance_method == const.COSINE:
        if transformers:
            similarity_index = np.array(np.argsort(-pairwise_similarities[letter_i]))
        else:
            similarity_index = np.array(np.argsort(-pairwise_similarities[letter_i].todense()))[0]
    else:
        similarity_index = np.argsort(pairwise_similarities[letter_i])

    similar_docs_sorted = []
    for index in similarity_index:
        if index == letter_i:
            continue
        similar_docs_sorted.append(index + initial_year)

    return similar_docs_sorted


def get_pipe_vector(letters_df):
    """Get the tfidf vector

    Parameters
    ----------
    letters_df: pandas DataFrame
        The pandas Dataframe with text from the letters

    Returns
    -------
    Np.array
        The tfidf vector representation of the text
    """
    pipeline = build_model()
    pipeline.fit(letters_df[const.LETTER_TEXT])
    vectors = pipeline.transform(letters_df[const.LETTER_TEXT])

    return vectors


def get_tfidf(letters_df, year, n, distance):
    """Get the tfidf most similar years

    Parameters
    ----------
    letters_df: pandas DataFrame
        The pandas Dataframe with text from the letters
    year: int
        The target letter year
    n: int
        The number of letters to return
    distance: string
        Euclidean or cosine

    Returns
    -------
    List
        List with the letter year sorted descending by similarity
    """
    vectors = get_pipe_vector(letters_df)
    if distance == const.COSINE:
        pairwise_dis = vectors @ vectors.T
    else:
        pairwise_dis = euclidean_distances(vectors)

    return get_most_similar_docs(pairwise_dis, year, distance)[:n]


def get_most_similar_docs_docs2vec(letter_year, model, corpus, n, initial_year=1977):
    """Get the docs2vec most similar years

    Parameters
    ----------
    letter_year: int
        The target letter year
    model: docs2vec
        The trained Docs2vec model
    corpus: List
        TaggedDocument list
    n: int
        The number of letters to return
    initial_year: int
        The initial letter year

    Returns
    -------
    List
        List with the letter year sorted descending by similarity
    """
    doc_id = letter_year - initial_year
    inferred_vector = model.infer_vector(corpus[doc_id].words)
    sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
    sims = [index + initial_year for index, _ in sims]

    return sims[1:n + 1]


def get_doc2vec(letters_df, year, n):
    """Get the doc2vec most similar years

    Parameters
    ----------
    letters_df: pandas DataFrame
        The pandas Dataframe with text from the letters
    year: int
        The target letter year
    n: int
        The number of letters to return

    Returns
    -------
    List
        List with the letter year sorted descending by similarity
    """
    EPOCHS = 40
    doc2_model = Doc2Vec(min_count=2)
    corpus = [TaggedDocument(tokens, [i]) for i, tokens in enumerate(list(letters_df[const.TOKENIZED]))]
    doc2_model.build_vocab(corpus)
    doc2_model.train(corpus, total_examples=doc2_model.corpus_count, epochs=EPOCHS)
    return get_most_similar_docs_docs2vec(year, doc2_model, corpus, n)


def get_word2vec(letters_df, year, n, distance):
    """Get the word2vec most similar years

    Parameters
    ----------
    letters_df: pandas DataFrame
        The pandas Dataframe with text from the letters
    year: int
        The target letter year
    n: int
        The number of letters to return
    distance: string
        Euclidean or cosine

    Returns
    -------
    List
        List with the letter year sorted descending by similarity
    """
    model = Word2Vec(letters_df[const.TOKENIZED])
    target, _ = get_avg_document_vector(model, letters_df, year)
    distances = []
    for y in list(letters_df.index):
        if y != year:
            vector_year, _ = get_avg_document_vector(model, letters_df, y)
            if distance == const.COSINE:
                distances.append(target @ vector_year.T / np.linalg.norm(target) / np.linalg.norm(vector_year))
            else:
                distances.append(np.linalg.norm(target - vector_year))

    distances = np.array(distances)
    if distance == const.COSINE:
        return letters_df.index[(-distances).argsort()][:n]
    else:
        return letters_df.index[distances.argsort()][:n]


def get_transformers(pre_trained_model, letters_df, year, n, distance):
    """Get the word2vec most similar years

    Parameters
    ----------
    pre_trained_model: string
        The name of the pre trained transform
    letters_df: pandas DataFrame
        The pandas Dataframe with text from the letters
    year: int
        The target letter year
    n: int
        The number of letters to return
    distance: string
        Euclidean or cosine

    Returns
    -------
    List
        List with the letter year sorted descending by similarity
    """
    model = SentenceTransformer(pre_trained_model)
    embeddings = model.encode(letters_df[const.TOKENIZED].values)
    if distance == const.COSINE:
        pairwise = embeddings @ embeddings.T / np.linalg.norm(embeddings) / np.linalg.norm(embeddings)
        return get_most_similar_docs(pairwise, year, const.COSINE, transformers=True)[:n]
    else:
        euclidean = euclidean_distances(embeddings)
        return get_most_similar_docs(euclidean, year, const.EUCLIDEAN)[:n]
