#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""KeySpaCy is a spaCy integration of KeyBERT by rewriting the embedding backend.
    It is designed to be a component of TRANSFORMER-BASED spacy pipeline, e.g., 
    `en_core_web_trf` pre-trained pipeline. Unlike KeyBERT, we only use pre-computed
    token contextualized embeddings to measure doc-substring similarities (KeyBERT 
    compute document and keyphrase embeddings independently, resulting in recomputing overhead)."""

__license__ = "MIT"
__version__ = "0.0.1"

import functools
import itertools

import warnings

from collections import defaultdict
from operator import itemgetter

from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from spacy.language import Language
from spacy.tokens import Doc, Span, Token


component_name = "keyword_extractor"


def mmr(
    doc_embedding: np.ndarray,
    word_embeddings: np.ndarray,
    words: List[str],
    top_n: int = 5,
    diversity: float = 0.8,
) -> List[Tuple[str, float]]:
    """Calculate Maximal Marginal Relevance (MMR)
    between candidate keywords and the document.
    MMR considers the similarity of keywords/keyphrases with the
    document, along with the similarity of already selected
    keywords and keyphrases. This results in a selection of keywords
    that maximize their within diversity with respect to the document.
    Arguments:
        doc_embedding: The document embeddings
        word_embeddings: The embeddings of the selected candidate keywords/phrases
        words: The selected candidate keywords/keyphrases
        top_n: The number of keywords/keyphrases to return
        diversity: How diverse the select keywords/keyphrases are.
                   Values between 0 and 1 with 0 being not diverse at all
                   and 1 being most diverse.
    Returns:
         List[Tuple[str, float]]: The selected keywords/keyphrases with their distances
    """

    # Extract similarity within words, and between words and the document
    word_doc_similarity = cosine_similarity(word_embeddings, doc_embedding)
    word_similarity = cosine_similarity(word_embeddings)

    # Initialize candidates and already choose best keyword/keyphrase
    keywords_idx = [np.argmax(word_doc_similarity)]
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    for _ in range(min(top_n - 1, len(words) - 1)):
        # Extract similarities within candidates and
        # between candidates and selected keywords/phrases
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(
            word_similarity[candidates_idx][:, keywords_idx], axis=1
        )

        # Calculate MMR
        mmr = (
            1 - diversity
        ) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]

        # Update keywords & candidates
        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    # Extract and sort keywords in descending similarity
    keywords = [
        (words[idx], round(float(word_doc_similarity.reshape(1, -1)[0][idx]), 4))
        for idx in keywords_idx
    ]
    keywords = sorted(keywords, key=itemgetter(1), reverse=True)
    return keywords


def max_sum_distance(
    doc_embedding: np.ndarray,
    word_embeddings: np.ndarray,
    words: List[str],
    top_n: int,
    nr_candidates: int,
) -> List[Tuple[str, float]]:
    """Calculate Max Sum Distance for extraction of keywords
    We take the 2 x top_n most similar words/phrases to the document.
    Then, we take all top_n combinations from the 2 x top_n words and
    extract the combination that are the least similar to each other
    by cosine similarity.
    This is O(n^2) and therefore not advised if you use a large `top_n`
    Arguments:
        doc_embedding: The document embeddings
        word_embeddings: The embeddings of the selected candidate keywords/phrases
        words: The selected candidate keywords/keyphrases
        top_n: The number of keywords/keyphrases to return
        nr_candidates: The number of candidates to consider
    Returns:
         List[Tuple[str, float]]: The selected keywords/keyphrases with their distances
    """
    if nr_candidates < top_n:
        raise Exception("Make sure that the number of candidates exceeds the number "
                        "of keywords to return.")
    elif top_n > len(words):
        return []

    # Calculate distances and extract keywords
    distances = cosine_similarity(doc_embedding, word_embeddings)
    distances_words = cosine_similarity(word_embeddings, word_embeddings)

    # Get 2*top_n words as candidates based on cosine similarity
    words_idx = list(distances.argsort()[0][-nr_candidates:])
    words_vals = [words[index] for index in words_idx]
    candidates = distances_words[np.ix_(words_idx, words_idx)]

    # Calculate the combination of words that are the least similar to each other
    min_sim = 100_000
    candidate = None
    for combination in itertools.combinations(range(len(words_idx)), top_n):
        sim = sum(
            [candidates[i][j] for i in combination for j in combination if i != j]
        )
        if sim < min_sim:
            candidate = combination
            min_sim = sim

    return [(words_vals[idx], round(float(distances[0][words_idx[idx]]), 4))
                for idx in candidate]


@Language.factory(component_name)
class KeywordExtractor:

    def __init__(self, name, nlp):
        """Simply set extension method"""

        Doc.set_extension('extract_keywords', method=self.extract_keywords)

    def __call__(self, doc):
        """Override `vector` attributes (for debugging purpose)"""

        doc.user_hooks['vector']       = self.doc_tensor
        doc.user_span_hooks['vector']  = self.span_tensor
        doc.user_token_hooks['vector'] = self.token_tensor

        return doc

    def extract_keywords(
        self,
        doc: Doc,
        keyphrase_ngram_range: Tuple[int, int] = (1, 1),
        top_n: int = 5,
        min_df: int = 1,
        use_maxsum: bool = False,
        use_mmr: bool = False,
        diversity: float = 0.5,
        nr_candidates: int = 20,
    ) -> Tuple[List[Tuple[str, float]], Dict[str, List[List[Token]]]]:

        # Extract ngram embeddings
        vocab = self._count_vocab(doc, keyphrase_ngram_range)
        candidates, candidate_embeddings = [], []

        # Ngram is considered a candidate only if it passes
        #   minimum word frequency. The embedding of the ngram
        #   is the average contextualized embeddings that have
        #   the same normalized form.
        for ng_normal, ng_spans in vocab.items():
            if len(ng_spans) >= min_df:
                candidates.append(ng_normal)
                candidate_embeddings.append(np.array([self.token_tensor(t)
                                                for s in ng_spans for t in s]).mean(axis=0))

        # Reshape to have B x H
        doc_embedding        = self.doc_tensor(doc).reshape(1, -1)
        candidate_embeddings = np.array(candidate_embeddings)

        # Empty candidate
        if len(candidates) < 1:
            return [], vocab

        # Extract keywords with specified metrics
        try:
            # Maximal Marginal Relevance (MMR)
            if use_mmr:
                keywords = mmr(
                    doc_embedding,
                    candidate_embeddings,
                    candidates,
                    top_n,
                    diversity,
                )

            # Max Sum Distance
            elif use_maxsum:
                keywords = max_sum_distance(
                    doc_embedding,
                    candidate_embeddings,
                    candidates,
                    top_n,
                    nr_candidates,
                )

            # Cosine-based keyword extraction
            else:
                distances = cosine_similarity(doc_embedding, candidate_embeddings)
                keywords = [(candidates[idx], round(float(distances[0][idx]), 4))
                                for idx in distances.argsort()[0][-top_n:]][::-1]

            return keywords, vocab

        except Exception as e:
            warnings.warn(f'Unexpected error occurred when calculating similarities; ' +
                          f'no keyword extracted: <{e}>')
            return [], vocab

    def doc_tensor(self, doc: Doc) -> np.ndarray:
        """Use the average pooling to combine all span 
            embeddings. Although the transformer output has 
            a pooled output, it may not be suitable for calculating
            semantic similarities between Doc and Tokens/Spans since
            the pooling strategies (depending on model) may involve 
            other non-linear transformations, which results in that 
            Doc and Token embeddings resides in different sub-space."""

        # Reuse span embedding
        return self.span_tensor(doc[:])

    def span_tensor(self, span: Span) -> np.ndarray:
        """Get the span embedding as the average of all
            sub-word embeddings within the span."""

        aligns = span.doc._.trf_data.align
        hidden = span.doc._.trf_data.tensors[0]

        # Get alignment information for Span; flatten array for indexing.
        #   Use `hidden.shape[-1]` to get embedding dimension (e.g., 768 for BERT)
        sub_ix = aligns[span.start:span.end].data.flatten()
        tensor = hidden.reshape(-1, hidden.shape[-1])[sub_ix]

        # Try if using `cupy` backend
        try:
            return tensor.mean(axis=0).get()
        except AttributeError:
            return tensor.mean(axis=0)

    def token_tensor(self, token: Token) -> np.ndarray:
        """Get the token embedding as the average of all sub-word embeddings"""

        aligns = token.doc._.trf_data.align
        hidden = token.doc._.trf_data.tensors[0]

        # Get alignment information for Token; flatten array for indexing.
        #   Use `hidden.shape[-1]` to get embedding dimension (e.g., 768 for BERT)
        sub_ix = aligns[token.i].data.flatten()
        tensor = hidden.reshape(-1, hidden.shape[-1])[sub_ix]

        # Try if using `cupy` backend
        try:
            return tensor.mean(axis=0).get()
        except AttributeError:
            return tensor.mean(axis=0)

    def _count_vocab(self, doc, keyphrase_ngram_range):
        """Create vocabulary to store """

        vocab = defaultdict(list)
        analyze = functools.partial(self._word_ngrams, ngram_range=keyphrase_ngram_range)

        # Map spans to keys with same (normalized) text form
        for ng in analyze(doc):
            ng_normal = self._normalize(ng)
            vocab[ng_normal].append(ng)

        return vocab

    def _word_ngrams(
        self,
        doc: Doc,
        ngram_range: Tuple[int],
        *,
        filter_stops: bool = True,
        filter_punct: bool = True,
        filter_email: bool = True,
        filter_url:   bool = True,
        filter_nums:  bool = False,
    ) -> List[List[Token]]:
        """Turn tokens into a sequence of n-grams (List[Token]) after filtering"""

        # Sanity check argument `ngram_range`
        assert len(ngram_range) == 2, f'`ngram_range` should only contain min and max ngram, got {ngram_range} instead.'

        min_n, max_n = ngram_range
        if min_n > max_n:
            raise ValueError(f'Invalid value for ngram_range={ngram_range} '
                                'lower boundary larger than the upper boundary.')

        # Generate ngrams from filtered tokens only
        def _rm_token(t: Token) -> bool:
            """Combine into a predicate given the filter rule set"""
            #   TODO: SHOULD HAVE SOME SEPARATE METHOD(S) FOR PREDICATE GENERATION

            return (t.is_space
                    or (t.like_email and filter_email)
                    or (t.like_url   and filter_url)
                    or (t.like_num   and filter_nums)
                    or (t.is_stop    and filter_stops)
                    or (t.is_punct   and filter_punct))

        tokens = [t for t in doc if not _rm_token(t)]

        # Extract ngrams with sliding window
        if max_n != 1:
            original_tokens = tokens
            if min_n == 1:
                # No need to do any slicing for uni-grams,
                #   just iterate through the original tokens
                tokens = [[t] for t in tokens]
                min_n += 1
            else:
                tokens = []

            n_original_tokens = len(original_tokens)

            for n in range(min_n, min(max_n, n_original_tokens) + 1):
                for i in range(n_original_tokens - n + 1):
                    tokens.append(original_tokens[i:(i + n)])

            return tokens

        # Base uni-grams
        return [[t] for t in tokens]

    def _normalize(self, ts: List[Token]) -> str:
        """Map list of tokens to some normalized form"""
        #   TODO: SHOULD ALLOW CUSTOMIZABLE NORMALIZATION RULES

        return ' '.join(t.text for t in ts).lower()
