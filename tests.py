import unittest

from gensim.models import KeyedVectors

from app.main import DEFAULT_LIMIT
from app.main import MAX_LIMIT
from app.main import MAX_WORD_COUNT
from app.main import MIN_LIMIT
from app.main import ModelInterface
from app.main import SCORE_PRECISION

GLOVE_PATH = 'app/glove.twitter.27B.25d.word2vec.p'
VECTORS = KeyedVectors.load(GLOVE_PATH)


class TestModelInterface(unittest.TestCase):
    def setUp(self):
        self.interface = ModelInterface(VECTORS)

    def tearDown(self):
        self.interface = None

    def test_input_is_missing_words_key(self):
        """'words' is a required key; raise KeyError if 'words' not present."""
        with self.assertRaises(KeyError):
            self.interface.prediction({})

    def test_empty_list_of_words(self):
        """'words' argument can not be empty; raise ValueError if 'words' is an empty list."""
        with self.assertRaises(ValueError):
            self.interface.prediction({'words': []})

    def test_number_of_words_above_max(self):
        """Word count can not exceed the max; raise ValueError if 'words' exceeds max word count."""
        with self.assertRaises(ValueError):
            self.interface.prediction({'words': ['cat'] * (MAX_WORD_COUNT + 1)})

    def test_words_is_not_list(self):
        """'words' must be a list; raise Value error if 'words' is not a list."""
        with self.assertRaises(ValueError):
            self.interface.prediction({'words': set()})

    def test_words_are_not_strings(self):
        """Words, in 'words', must be strings; raise ValueError if words in 'words' are not strings."""
        with self.assertRaises(ValueError):
            self.interface.prediction({'words': ['cat', 1, 2.2]})

    def test_word_is_not_in_vocab(self):
        """Words that are not in the app's vocabulary should not appear in the output."""
        word = 'hdja'
        similar_words = self.interface.prediction({'words': [word]})['words']
        self.assertFalse(word in similar_words)

    def test_input_is_missing_limit_key(self):
        """'limit' is an optional key; use the default limit if no limit is provided."""
        similar_words = self.interface.prediction({'words': ['cat']})['words']
        self.assertEqual(len(similar_words['cat']), DEFAULT_LIMIT)

    def test_limit_is_string(self):
        """'limit' must be an integer; raise ValueError if 'limit' is a string."""
        with self.assertRaises(ValueError):
            self.interface.prediction({'words': ['cat'], 'limit': '2.2'})

    def test_limit_is_float(self):
        """'limit' must be an integer; raise ValueError if 'limit' is a float."""
        with self.assertRaises(ValueError):
            self.interface.prediction({'words': ['cat'], 'limit': 2.2})

    def test_limit_above_max(self):
        """'limit' can not exceed the max; set limit to max if limit exceeds max."""
        similar_words = self.interface.prediction({'words': ['cat'], 'limit': MAX_LIMIT + 1})['words']
        self.assertEqual(len(similar_words['cat']), MAX_LIMIT)

    def test_limit_below_min(self):
        """'limit' can not be less than the min; set limit to min if limit is less than min. """
        similar_words = self.interface.prediction({'words': ['cat'], 'limit': MIN_LIMIT - 1})['words']
        self.assertEqual(len(similar_words['cat']), MIN_LIMIT)

    def test_words_key_in_output(self):
        """The 'words' key must be in the output."""
        self.assertTrue('words' in self.interface.prediction({'words': ['cat']}))

    def test_input_words_are_in_output(self):
        """All words, found in the vocab, must be in the output."""
        input_words = ['cat', 'youtube', 'hdjk']
        output_words = self.interface.prediction({'words': input_words})['words']
        self.assertTrue(all(word in output_words for word in input_words if word in self.interface.vectors))

    def test_words_in_output_are_strings(self):
        """All words, in the output, must be strings."""
        words = self.interface.prediction({'words': ['cat']})['words']
        self.assertTrue(all(isinstance(word, str) for word in words))

    def test_similar_words_have_label_key(self):
        """All similar words must have a label key."""
        words = self.interface.prediction({'words': ['cat']})['words']
        similar_words = []
        for similar_words_list in words.values():
            similar_words.extend(similar_words_list)
        self.assertTrue(all('label' in word for word in similar_words))

    def test_labels_are_strings(self):
        """All labels must be strings."""
        words = self.interface.prediction({'words': ['cat']})['words']
        similar_words = []
        for similar_words_list in words.values():
            similar_words.extend(similar_words_list)
        self.assertTrue(all(isinstance(word['label'], str) for word in similar_words))

    def test_similar_words_have_score_key(self):
        """All similar words must have a score key."""
        words = self.interface.prediction({'words': ['cat']})['words']
        similar_words = []
        for similar_words_list in words.values():
            similar_words.extend(similar_words_list)
        self.assertTrue(all('score' in word for word in similar_words))

    def test_scores_are_floats(self):
        """All scores must be floats."""
        words = self.interface.prediction({'words': ['cat']})['words']
        similar_words = []
        for similar_words_list in words.values():
            similar_words.extend(similar_words_list)
        self.assertTrue(all(isinstance(word['score'], float) for word in similar_words))

    def test_scores_have_correct_precision(self):
        """All scores must have the correct precision."""
        words = self.interface.prediction({'words': ['cat']})['words']
        similar_words = []
        for similar_words_list in words.values():
            similar_words.extend(similar_words_list)
        self.assertTrue(all(word['score'] == round(word['score'], ndigits=SCORE_PRECISION) for word in similar_words))

    def test_number_of_similar_words_matches_limit(self):
        """The number of similar words should match the limit"""
        words = self.interface.prediction({'words': ['cat']})['words']
        similar_words = []
        for similar_words_list in words.values():
            similar_words.extend(similar_words_list)
        self.assertTrue(all(isinstance(word['score'], float) for word in similar_words))

    def test_case_insensitivity(self):
        """App must be case insensitive i.e. 'cat' and 'CAT' will have the same result."""
        uncased = self.interface.prediction({'words': ['cat']})
        cased = self.interface.prediction({'words': ['CAT']})
        self.assertEqual(uncased, cased)


if __name__ == '__main__':
    unittest.main()
