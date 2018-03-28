# Similar Words
This app uses word vectors, obtained from a pretrained GloVe model, to find similar words.


## Preprocessing
To create the binary files used in this app, the word vectors were converted to the word2vec format and then pickled.
```
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors


# convert the original word vectors to the word2vec format
glove2word2vec(original_file, reformatted_file)

# pickle the reformatted word vectors
model = KeyedVectors.load_word2vec_format(reformatted_file)
model.save(binary_file)
```


## Resources
  * [GloVe homepage](https://nlp.stanford.edu/projects/glove/)
  * [GloVe pretrained word vectors](http://nlp.stanford.edu/data/glove.twitter.27B.zip)
  * [Gensim KeyedVectors doc](https://radimrehurek.com/gensim/models/keyedvectors.html)
  * [Gensim glove2word2vec doc](https://radimrehurek.com/gensim/scripts/glove2word2vec.html)

