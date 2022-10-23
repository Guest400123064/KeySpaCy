# __KeySpaCy__

KeySpaCy is a spaCy integration of [KeyBERT](https://github.com/MaartenGr/KeyBERT) by rewriting the embedding backend. It is designed to be a component of __TRANSFORMER-BASED__ spacy pipeline, e.g, the `en_core_web_trf` pre-trained pipeline. Unlike KeyBERT, we only use pre-computed token contextualized embeddings to measure doc-substring similarities (KeyBERT compute document and keyphrase embeddings independently, resulting in recomputing overhead).

## __Table of Contents__


## __1. About the Project__


## __2. Getting Started__

### __2.1 Installation__

TBD

### __2.2 Usage__

The most minimal example can be seen below for the extraction of keywords:

```python
import spacy
import keyspacy

text = """
    Supervised learning is the machine learning task of learning a function that
    maps an input to an output based on example input-output pairs. It infers a
    function from labeled training data consisting of a set of training examples.
    In supervised learning, each example is a pair consisting of an input object
    (typically a vector) and a desired output value (also called the supervisory signal).
    A supervised learning algorithm analyzes the training data and produces an inferred function,
    which can be used for mapping new examples. An optimal scenario will allow for the
    algorithm to correctly determine the class labels for unseen instances. This requires
    the learning algorithm to generalize from the training data to unseen situations in a
    'reasonable' way (see inductive bias).
"""

# Transformer-based spacy pipeline is required
nlp = spacy.load('en_core_web_trf')
nlp.add_pipe(keyspacy.component_name)

# Add extraction extension by piping through `nlp`
doc = nlp(text)
keywords, vocab = doc._.extract_keywords()
```

You can set `keyphrase_ngram_range` to set the length of the resulting keywords/keyphrases:

```python
>>> keywords, vocab = doc._.extract_keywords(keyphrase_ngram_range=(1, 1))
>>> keywords
[('learning', 0.7793),
 ('input', 0.7528),
 ('output', 0.7306),
 ('example', 0.6977),
 ('algorithm', 0.6884)]
```
