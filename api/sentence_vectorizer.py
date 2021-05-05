import string
import pickle
import numpy as np

class SentenceVectorizer():
    def __init__(self, filename=None, dim=0, eps=1e-9):
        self.dim = dim
        self.eps = eps
        self.word2vec = {}

        if filename is not None:
            self.load(filename)

    def load(self, filename):
        with open(filename, 'rb') as f:
            self.word2vec = pickle.load(f)
            self.dim = len(next(iter(self.word2vec.values())))

    def encode(self, text):
        '''
        Sum the word vectors for each token in the sentence. Does not take order nor syntax into account, but it is very fast.
        Inspired from this answer on stackoverflow:
        https://stackoverflow.com/questions/30795944/how-can-a-sentence-or-a-document-be-converted-to-a-vector
        '''
        assert type(text) == str, "Wrong data type, text must be str"
        text = self._tokenize(text)
        vec = np.zeros(self.dim)
        for word in text:
            try:
                vec = np.add(vec, self.word2vec[word])
            except:
                pass  # Out of vocabulary (OOV)
        vec /= (np.sqrt(vec.dot(vec)) + self.eps)
        return vec

    def _tokenize(self, text):
        return text.lower()\
            .strip(string.punctuation)\
            .split(' ')
