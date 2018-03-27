import unittest

from gensim.models import KeyedVectors

from app.main import ModelInterface

GLOVE_PATH = 'app/glove.twitter.27B.25d.word2vec.p'
VECTORS = KeyedVectors.load(GLOVE_PATH)
INTERFACE = ModelInterface()


class TestModelInterface(unittest.TestCase):
    def test_missing_key(self):
        with self.assertRaises(KeyError):
            INTERFACE.prediction({})

    def test_value_not_valid(self):
        with self.assertRaises(ValueError):
            INTERFACE.prediction({'words': set()})

    def test_token_not_in_vocab(self):
        vectors = INTERFACE.prediction({'words': ['hdja4']})['vectors']
        self.assertTrue(None in vectors)

    def test_valid_input(self):
        words = ['cat', 'dog']
        actual = INTERFACE.prediction({'words': words})['vectors']
        expected = [VECTORS[word].tolist() for word in words]
        self.assertEqual(actual, expected)

    def test_case_insensitivity(self):
        uncased = INTERFACE.prediction({'words': ['cat']})
        cased = INTERFACE.prediction({'words': ['Cat']})
        self.assertEqual(uncased, cased)


if __name__ == '__main__':
    unittest.main()
