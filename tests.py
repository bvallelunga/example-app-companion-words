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
        expected = {word: [{'label': similar_word, 'score': round(score, ndigits=2)}
                           for similar_word, score in VECTORS.most_similar(word)] for word in words}
        self.assertEqual(actual, expected)

    def test_case_insensitivity(self):
        uncased = INTERFACE.prediction({'words': ['cat']})
        cased = INTERFACE.prediction({'words': ['CAT']})
        self.assertEqual(uncased, cased)

    def test_limit_exceeds_max(self):
        with self.assertRaises(ValueError):
            INTERFACE.prediction({'words': ['cat'], 'limit': 101})

    def test_limit_not_int(self):
        with self.assertRaises(ValueError):
            INTERFACE.prediction({'words': ['cat'], 'limit': 2.2})

    def test_empty_list_of_words(self):
        with self.assertRaises(ValueError):
            INTERFACE.prediction({'words': []})


if __name__ == '__main__':
    unittest.main()
