import argparse
from document_similarity import get_letters_df, get_tfidf, get_word2vec, get_doc2vec, get_transformers
import constants as const
import time


def main(algorithm, distance, letters_dict_pickle, year, n, pre_trained_model=''):
    """Main function

    Parameters
    ----------
    algorithm: string
        The chosen algorithm to compute the similarity/distance.
    distance: string
        euclidean or cosine.
    letters_dict_pickle: string
        Pickle path to the letters dict.
    year: int
        The target letter year.
    n: int
        The number of letters to return.
    pre_trained_model: string
        The pretrained model to use in transformers.

    Returns
    -------
    Np.array
        The tfidf vector representation of the text
    """
    letters_df = get_letters_df(letters_dict_pickle)
    if algorithm == const.TFIDF:
        return get_tfidf(letters_df, year, n, distance)
    elif algorithm == const.WORD2VEC:
        return get_word2vec(letters_df, year, n, distance)
    elif algorithm == const.DOC2VECT:
        return get_doc2vec(letters_df, year, n)
    else:
        return get_transformers(pre_trained_model, letters_df, year, n, distance)


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description="Execute the distance similarity")
    PARSER.add_argument("-alg", "--algorithm", help="The chosen algorithm to"
                                                    " compute the similarity/distance.")
    PARSER.add_argument("-dist", "--distance", help="euclidean or cosine.")
    PARSER.add_argument(
        "-p", "--path", help="Pickle path to the letters dict.")
    PARSER.add_argument(
        "-t", "--target", help="The target letter year.")
    PARSER.add_argument(
        "-n", "--number", help="The number of letters to return.")
    PARSER.add_argument(
        "-pre", "--pretrained", help="The pretrained model to use in transformers.")

    ARGS = PARSER.parse_args()
    start = time.time()
    print(main(ARGS.algorithm, ARGS.distance, ARGS.path, int(ARGS.target), int(ARGS.number), ARGS.pretrained))
    print(f'Execution time = {time.time() - start} seconds.')
