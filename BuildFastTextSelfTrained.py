from pathlib import Path

#uncomment if have not downloaded yet
#import nltk
#nltk.download('punkt')
from nltk.tokenize import word_tokenize
from collections import Counter
from gensim.models import FastText
import numpy as np
import logging


class BuildFastTextSelfTrained:
    def __init__(self, route):
        """
        Assigns by grabbing the input cleaning objects
        """
        self.logger = logging.getLogger(__name__)
        self.route = route

    def get_data(self):
        """
        Create corpus from train.words.txt and test.words.txt
        :return corpus: list of sentences
        """
        # Load Data
        sentences_train = list()
        sentences_test = list()
        with Path(self.route + '/' + 'train.words.txt').open(encoding="utf8") as f:
            for line in f:
                line = line.strip()
                sentences_train.append(line)

        with Path(self.route + '/' + 'test.words.txt').open(encoding="utf8") as f:
            for line in f:
                line = line.strip()
                sentences_test.append(line)
        corpus = sentences_train + sentences_test
        return corpus

    def tokenize_sentence(self, corpus):
        """
        Convert sentences to words or tokens using nltk.tokenize library
        :param corpus: list of sentences to be tokenized
        :return tokenized_sentences : list of lists of tokens
        """
        # Create token
        tokenized_sentences = [word_tokenize(i) for i in corpus]
        # Vocab size
        count_token = list()
        for tokenized_sentence in tokenized_sentences:
            x = Counter(''.join(map(str, tokenized_sentence)))
            count_token.append(len(x.keys()))
        token_size = sum(count_token)
        print('Found {} unique tokens in corpus'.format(token_size))
        return tokenized_sentences

    def build_model(self, tokenized_sentences):
        """
        Build fasttext embedding model. There are some parameters to be set for training. The detail explanation can be
        found at https://radimrehurek.com/gensim/models/fasttext.html#gensim.models.fasttext.FastText
        The parameters used below are considered as important by the code author.
        However, it's open for modification for experiments.
        :param tokenized_sentences: data that will be used for creating embedding model
        """
        # train model
        model = FastText(sentences=tokenized_sentences, size=300, window=3, min_count=3, sg=1, iter=10, word_ngrams=1)
        # the model
        print(model)
        # save model
        model.save(self.route+'/'+'fasttext_selftrained_03.model')

    def get_embeddings(self):
        """
        Get word vectors of vocab words based on the trained embedding model
        """
        # Load vocab
        with Path(self.route + '/' + 'vocab.words.txt').open(encoding="utf8") as f:
            words = {line.strip(): idx for idx, line in enumerate(f)}
        size_vocab = len(words)
        # Load Model
        model = FastText.load(self.route+'/'+'fasttext_selftrained_03.model')
        embeddings = {}
        word_vectors = list()
        for word in words:
            try:
                vec = model.wv[word]
                embeddings[word] = vec
                word_vectors.append(embeddings[word])
            except KeyError:  # note that not every word may be in the vocabulary
                pass
        print('- done. Found {} vectors for {} words'.format(len(embeddings), size_vocab))

        # Save np.array to file
        np.savez_compressed(self.route+'/'+'fasttext_selftrained_03.npz', embeddings=word_vectors)


if __name__ == "__main__":
    build_fasttext = BuildFastTextSelfTrained('./data')
    docs = build_fasttext.get_data()
    token_corpus = build_fasttext.tokenize_sentence(docs)
    build_fasttext.build_model(token_corpus)
    build_fasttext.get_embeddings()
