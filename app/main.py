from gensim.models import KeyedVectors

GLOVE_PATH = 'app/glove.twitter.27B.25d.word2vec.p'
DEFAULT_LIMIT = 10
MAX_LIMIT = 100
MIN_LIMIT = 1


class ModelInterface(object):
    def __init__(self, vectors=None):
        self.vectors = KeyedVectors.load(GLOVE_PATH) if vectors is None else vectors  # very slow

    @staticmethod
    def is_list_of_strs(input):
        return isinstance(input, list) and len(input) > 0 and all(isinstance(el, str) for el in input)

    def prediction(self, input):
        if 'words' not in input:
            raise KeyError("No key named 'words' in input.")
        if not self.is_list_of_strs(input['words']):
            raise ValueError("'words' should be a list of strings.")

        limit = DEFAULT_LIMIT
        if 'limit' in input:
            try:
                limit = int(str(input['limit']))
                limit = max(MIN_LIMIT, min(limit, MAX_LIMIT))
            except ValueError:
                raise ValueError("'limit' must be an integer in the range 1-100.")

        results = {}
        for word in [word.lower() for word in input['words']]:
            if word in self.vectors:
                results[word] = [{'label': similar_word, 'score': round(score, ndigits=2)}
                                 for similar_word, score in self.vectors.most_similar(word, topn=limit)]
        return {'words': results}


if __name__ == '__main__':
    interface = ModelInterface()
    print(interface.prediction({"words": ["blue", "twitter", "fkdsjfsa"], "limit": 5}))
