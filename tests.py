import unittest

from gensim.models import KeyedVectors

from app.main import ModelInterface

GLOVE_PATH = 'app/glove.twitter.27B.25d.word2vec.p'
VECTORS = KeyedVectors.load(GLOVE_PATH)
INTERFACE = ModelInterface()


class TestModelInterface(unittest.TestCase):
    def test_missing_input_key(self):
        with self.assertRaises(KeyError):
            INTERFACE.prediction({})

    def test_value_invalid_type(self):
        with self.assertRaises(ValueError):
            INTERFACE.prediction({'words': set()})

    def test_token_not_in_vocab(self):
        word = 'hdja'
        similar_words = INTERFACE.prediction({'words': [word]})['similar_words']
        self.assertFalse(word in similar_words)

    def test_valid_input(self):
        words = ['cat', 'dog']
        actual = INTERFACE.prediction({'words': words})['similar_words']
        expected = {word: [similar_word for similar_word, _ in VECTORS.most_similar(word)] for word in words}
        self.assertEqual(actual, expected)

    def test_case_insensitivity(self):
        uncased = INTERFACE.prediction({'words': ['cat']})
        cased = INTERFACE.prediction({'words': ['CAT']})
        self.assertEqual(uncased, cased)


if __name__ == '__main__':
    unittest.main()
