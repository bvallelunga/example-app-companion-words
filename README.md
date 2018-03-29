# Similar Words
Discover which words are most similar to each other. Trained on billions of tweets
with over 1 million unique words.

**Possible Use Cases**
  * discover the name for your next iPhone app
  * suggest relevant words in a messenger or email client
  * serve relevant ads based on a user's interests


## Input Scheme
The input should be an array of words. Note, the app is case-insensitive i.e. 
`"cat"` and `"CAT"` will have the same output.
```json
{
  "words": ["blue", "twitter", "fkdsjfsa"]
}
```

## Output Scheme
The output will map each input word to it"s 10 most similar words. Notice how 
`"fkdsjfsa"` is missing in the output, this is because input words which are 
not found in the app"s vocabulary are skipped.
 
```json
{
  "similar_words": 
  {
    "blue": [
      "yellow", "red", "green", "purple", "black", 
      "white", "pink", "gold", "grey", "diamond"
    ],
    "twitter": [
      "facebook", "tweet", "fb", "instagram", "chat", 
      "hashtag", "tweets", "tl", "link", "internet"
    ]
  }
}
```


## Training
The model was trained by [Stanford's NLP group][1] on a dataset of 2 billion, uncased tweets 
(27 billion tokens with 1.2 million distinct words).


[1]: https://nlp.stanford.edu/projects/glove/
