import nltk
import math
import os
import string
import sys

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    files = {}

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        with open(file_path, encoding="utf-8") as f:
            files[filename] = f.read()

    return files


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    punctuation = set(string.punctuation)
    stopwords = set(nltk.corpus.stopwords.words("english"))
    words = nltk.tokenize.word_tokenize(document.lower())

    return [word for word in words if word not in punctuation and word not in stopwords]


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    # {word: number of documents that contain word}
    word_count = {}

    for words in documents.values():
        for word in set(words):
            word_count[word] = word_count.get(word, 0) + 1

    return {
        word: math.log(len(documents) / count) for word, count in word_count.items()
    }


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """

    def tf_idf_sum(filename: str):
        return sum(files[filename].count(word) * idfs[word] for word in query)

    return sorted(files, key=tf_idf_sum, reverse=True)[:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """

    def sentence_rank(sentence: str):
        words = sentences[sentence]

        idf_sum = sum(idfs[word] for word in query.intersection(words))
        term_density = sum(word in query for word in words) / len(words)

        return idf_sum, term_density

    return sorted(sentences, key=sentence_rank, reverse=True)[:n]


if __name__ == "__main__":
    main()
