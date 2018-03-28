# Similar Words
This app can tell you which words are most similar to each other.


## Input and Output Schemes
The input should be an array of words. Note, the app is case-insensitive i.e. 
`'cat'` and `'CAT'` will have the same output.
```
input = {
    'words': ['blue', 'twitter', 'fkdsjfsa']
}
```

The output will map each input word to it's 10 most similar words. Notice how 
`fkdsjfsa` is missing in the output, this is because input words which are 
not found in the app's vocabulary are skipped.
 
```
output = {
    'similar_words': {
        blue': [
            'yellow', 'red', 'green', 'purple', 'black', 
            'white', 'pink', 'gold', 'grey', 'diamond'
            ],
        'twitter': [
            'facebook', 'tweet', 'fb', 'instagram', 'chat', 
            'hashtag', 'tweets', 'tl', 'link', 'internet'
            ]
        ]
    }
}
```


## Training
The model was trained on a dataset of 2 billion uncased tweets (27 billion tokens with 
1.2 million distinct words).


## Benchmarks
Not applicable.
