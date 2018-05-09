import os
from gensim.models import KeyedVectors

GLOVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'glove.twitter.27B.25d.word2vec.p')
DEFAULT_LIMIT = 10
MAX_LIMIT = 100
MIN_LIMIT = 1
MAX_WORD_COUNT = 50
SCORE_PRECISION = 2


class ModelInterface(object):
    def __init__(self, vectors=None):
        self.vectors = KeyedVectors.load(GLOVE_PATH) if vectors is None else vectors  # very slow

    @staticmethod
    def is_list_of_strs(input):
        return isinstance(input, list) and len(input) > 0 and all(isinstance(el, str) for el in input)

    def prediction(self, input):
        if 'words' not in input:
            raise KeyError("Expected key named 'words' in input.")
        if not self.is_list_of_strs(input['words']):
            raise ValueError("'words' should be a list of strings.")
        if len(input['words']) > MAX_WORD_COUNT:
            raise ValueError("Number of words can not exceed {}.".format(MAX_WORD_COUNT))
        if 'limit' in input and not isinstance(input['limit'], int):
            raise ValueError("'limit' must be an integer.")

        limit = max(MIN_LIMIT, min(input['limit'], MAX_LIMIT)) if 'limit' in input else DEFAULT_LIMIT
        results = {}
        for word in [word.lower() for word in input['words']]:
            if word in self.vectors:
                results[word] = [{'label': similar_word, 'score': round(score, ndigits=SCORE_PRECISION)}
                                 for similar_word, score in self.vectors.most_similar(word, topn=limit)]
        return {'words': results}


if __name__ == '__main__':
    interface = ModelInterface()
    print(interface.prediction({"words": ["blue", "twitter", "fkdsjfsa"], "limit": 5}))
