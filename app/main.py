from gensim.models import KeyedVectors

GLOVE_PATH = 'app/glove.twitter.27B.25d.word2vec.p'


class ModelInterface(object):
    def __init__(self):
        self.vectors = KeyedVectors.load(GLOVE_PATH)  # very slow

    @staticmethod
    def is_list_of_strs(input):
        return isinstance(input, list) and all(isinstance(el, str) for el in input)

    def prediction(self, input):
        if 'words' not in input:
            raise KeyError("No key named 'words' in input.")
        if not self.is_list_of_strs(input['words']):
            raise ValueError("'words' should be a list of strings.")

        similar_words = {}
        for word in [word.lower() for word in input['words']]:
            if word in self.vectors:
                similar_words[word] = [similar_word for similar_word, _ in self.vectors.most_similar(word)]
        return {'similar_words': similar_words}


