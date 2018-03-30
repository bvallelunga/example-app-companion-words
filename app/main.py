from gensim.models import KeyedVectors

GLOVE_PATH = 'app/glove.twitter.27B.25d.word2vec.p'
DEFAULT_LIMIT = 10
MAX_LIMIT = 100


class ModelInterface(object):
    def __init__(self):
        self.vectors = KeyedVectors.load(GLOVE_PATH)  # very slow

    @staticmethod
    def is_list_of_strs(input):
        return isinstance(input, list) and input and all(isinstance(el, str) for el in input)

    def prediction(self, input):
        if 'words' not in input:
            raise KeyError("No key named 'words' in input.")
        if not self.is_list_of_strs(input['words']):
            raise ValueError("'words' should be a list of strings.")
        if 'limit' in input and (not isinstance(input['limit'], int) or input['limit'] > MAX_LIMIT):
                raise ValueError("'limit' must be an integer in the range 1-100.")

        limit = input['limit'] if 'limit' in input else DEFAULT_LIMIT
        similar_words = {}
        for word in [word.lower() for word in input['words']]:
            if word in self.vectors:
                similar_words[word] = [{'label': similar_word, 'score': round(score, ndigits=2)}
                                       for similar_word, score in self.vectors.most_similar(word, topn=limit)]
        return {'similar_words': similar_words}


if __name__ == '__main__':
    interface = ModelInterface()
    print(interface.prediction({"words": ["blue", "twitter", "fkdsjfsa"], "limit": 5}))
