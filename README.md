# Warren Buffet letters Analysis

### Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Running](#running)
4. [Final Considerations](#considerations)

## Overview <a name="overview"></a>

The Goal of this project is to use NLP techniques such as Question and Answering,
Sentiment Analysis, WordCloud, document similarity and others to extract meaningful insights about
Warren Buffet annual letters to the Berkshire Hathaway shareholders.

## Installation <a name="installation"></a>

Create a virtual environment named **ibm_venv**.

```
$ python3 -m venv warren_venv -- for Linux and macOS
$ python -m venv warren_venv -- for Windows
```

After that, activate the python virtual environment

```
$ source warren_venv/bin/activate -- for Linux and macOS
$ warren_venv\Scripts\activate -- for Windows
```

Install the requirements

```
$ pip install -r requirements.txt
```

## Running <a name="running"></a>

### Running the QA notebook

To run it you have to download the letters after 2000 at 
https://www.berkshirehathaway.com/letters/letters.html. After that you need to
change the parameters from the function get_letters_corpus_dict to the directory
containing the letters, after that you only need to run the desired cells of
the notebook

### Running the document similarity

You can get the most similar documents to a specific letter year by running the
doc_sim_main.py.

```
python doc_sim_main.py --algorithm <algorithm> --distance <distance> --path <path> --target <target> --number <number> --pretrained <pretrained>
```

Where:
* algorithm: Could be tfidf, word2vec, doc2vect and transformer
* distance: Could be cosine or euclidean
* path: Pickle path to the letters dict
* target: The target letter year
* number: The number of letters to return
* pretrained: The pretrained model to use in transformers

## Final Considerations and acknowledgments <a name="considerations"></a>

To see the full analysis of this code, access my medium post at:
