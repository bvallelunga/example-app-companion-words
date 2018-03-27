# GloVe Embeddings

Stanford's GloVe is an algorithm which learns vector representations for each word in a 
given dataset. These word vectors can be used to explore relationships between words or 
as inputs to another algorithm/model.


## Input and Output Schemes

The input should be an array of words. Note, this implementation is case-insensitive i.e. 
`embedding['cat'] == embedding['CATS]` is always `True` for any word in our vocabulary.
```
input = {
    'words': ['cat', 'king', 'car', 'fkdsjfsa']
}
```

The output will be an array of vectors and `None` objects. If a word is in the vocabulary 
then it's 25-dimensional word vector will be returned. Words not in the vocabulary will 
return `None`. 
 
```
output = {
    'vectors': [
        [-0.9641900062561035, -0.6097800135612488, ...],
        [-0.7450100183486938, -0.11992000043392181, ...],
        [-0.5586100220680237, -0.1958400011062622, ...],
        None 
    ]
}
```


# Training

The vector representations were extracted from a dataset of 2 billion uncased tweets 
(27 billion tokens with 1.2 million distinct words).


# Benchmarks

Not applicable.


# References

[GloVe website](https://nlp.stanford.edu/projects/glove/)

