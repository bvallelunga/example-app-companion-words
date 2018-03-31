import unittest

from gensim.models import KeyedVectors

from app.main import DEFAULT_LIMIT
from app.main import MAX_LIMIT
from app.main import MIN_LIMIT
from app.main import ModelInterface

GLOVE_PATH = 'app/glove.twitter.27B.25d.word2vec.p'
VECTORS = KeyedVectors.load(GLOVE_PATH)


class TestModelInterface(unittest.TestCase):
    def setUp(self):
        self.interface = ModelInterface(VECTORS)

    def tearDown(self):
        self.interface = None

    def test_missing_words_key(self):
        with self.assertRaises(KeyError):
            self.interface.prediction({})

    def test_empty_list_of_words(self):
        with self.assertRaises(ValueError):
            self.interface.prediction({'words': []})

    def test_words_not_strings(self):
        with self.assertRaises(ValueError):
            self.interface.prediction({'words': set()})

    def test_word_not_in_vocab(self):
        word = 'hdja'
        similar_words = self.interface.prediction({'words': [word]})['words']
        self.assertFalse(word in similar_words)

    def test_missing_limit_key(self):
        similar_words = self.interface.prediction({'words': ['cat']})['words']
        self.assertEqual(len(similar_words['cat']), DEFAULT_LIMIT)

    def test_limit_is_string(self):
        with self.assertRaises(ValueError):
            self.interface.prediction({'words': ['cat'], 'limit': '2.2'})

    def test_limit_is_float(self):
        with self.assertRaises(ValueError):
            self.interface.prediction({'words': ['cat'], 'limit': 2.2})

    def test_limit_above_max(self):
        similar_words = self.interface.prediction({'words': ['cat'], 'limit': 101})['words']
        self.assertEqual(len(similar_words['cat']), MAX_LIMIT)

    def test_limit_below_min(self):
        similar_words = self.interface.prediction({'words': ['cat'], 'limit': 0})['words']
        self.assertEqual(len(similar_words['cat']), MIN_LIMIT)

    def test_output_format(self):
        words = ['cat', 'dog']
        actual = self.interface.prediction({'words': words})['words']
        expected = {word: [{'label': similar_word, 'score': round(score, ndigits=2)}
                           for similar_word, score in VECTORS.most_similar(word)] for word in words}
        self.assertEqual(actual, expected)

    def test_case_insensitivity(self):
        uncased = self.interface.prediction({'words': ['cat']})
        cased = self.interface.prediction({'words': ['CAT']})
        self.assertEqual(uncased, cased)


if __name__ == '__main__':
    unittest.main()
