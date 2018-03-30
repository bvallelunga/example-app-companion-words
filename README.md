# Similar Words
Discover which words are most similar to each other. Trained on billions of tweets
with over 1 million unique words.

**Possible Use Cases**
  * discover the name for your next iPhone app
  * suggest relevant words in a messenger or email client
  * serve relevant ads based on a user"s interests


## Input Scheme
The input should be an array of words. We can change the number of similar words to return, the default is 10, 
by setting a limit. Note, the app is case-insensitive i.e. `"cat"` and `"CAT"` will have the same output.
```json
{
  "words": ["blue", "twitter", "fkdsjfsa"],
  "limit": 5
}
```

## Output Scheme
The output will map each input word to an array of similar words. Notice how `"fkdsjfsa"` is missing in the output; 
input words which are not found in the app's vocabulary are skipped.
 
```json
{
  "words": 
    { 
      "blue": [
        {"label": "yellow", "score": 0.96}, 
        {"label": "red", "score": 0.95}, 
        {"label": "green", "score": 0.95}, 
        {"label": "purple", "score": 0.95}, 
        {"label": "black", "score": 0.94}
      ], 
      "twitter": [
        {"label": "facebook", "score": 0.95}, 
        {"label": "tweet", "score": 0.94}, 
        {"label": "fb", "score": 0.93}, 
        {"label": "instagram", "score": 0.91}, 
        {"label": "chat", "score": 0.9}
      ]
    }
}
```


## Training
The model was trained by [Stanford's NLP group][1] on a dataset of 2 billion, uncased tweets 
(27 billion tokens with 1.2 million distinct words).


[1]: https://nlp.stanford.edu/projects/glove/
