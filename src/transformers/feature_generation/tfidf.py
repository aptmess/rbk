from string import punctuation

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfTransformer,
    TfidfVectorizer,
)

nltk.download('stopwords')


class CountTfIdfVectorizer:
    def __init__(
        self,
        col: str,
        tfidf_transformer=TfidfTransformer(
            norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False
        ),
        count_vectorizer=CountVectorizer(
            input='content',
            encoding='utf-8',
            decode_error='strict',
            strip_accents=None,
            lowercase=True,
            preprocessor=None,
            tokenizer=None,
            stop_words=None,
            token_pattern=r'(?u)\b\w\w+\b',
            ngram_range=(1, 1),
            analyzer='word',
            max_df=1.0,
            min_df=1,
            max_features=None,
            vocabulary=None,
            binary=False,
            dtype=np.int64,
        ),
    ):
        self.tfidf_transformer = tfidf_transformer
        self.count_vectorizer = count_vectorizer
        self.col = col

        self.tfidf_matrix = None
        self.word2tfidf = None

    def fit(self, X):
        unique_values = X[~(X[self.col].isna())]
        unique_values = [
            x for xs in unique_values[self.col].tolist() for x in xs
        ]
        counter = self.count_vectorizer.fit_transform(unique_values)
        self.tfidf_matrix = self.tfidf_transformer.fit_transform(counter)
        self.word2tfidf = dict(
            zip(
                self.count_vectorizer.get_feature_names_out(),
                self.tfidf_transformer.idf_,
            )
        )
        return self

    def transform(self, X):
        counter = []
        for elem in X[self.col]:
            if elem is None:
                counter.append(None)
            else:
                counter_tfidf_sentence = 0
                for list_elem in elem:
                    for word in list_elem.split():
                        if word.lower() in self.word2tfidf:
                            counter_tfidf_sentence += self.word2tfidf[
                                word.lower()
                            ]
                counter.append(counter_tfidf_sentence)
        return counter


class TfidfVectorTransformer:
    def __init__(
        self,
        col: str,
        tfidf_vectorizer=TfidfVectorizer(
            input='content',
            encoding='utf-8',
            decode_error='strict',
            strip_accents=None,
            lowercase=True,
            preprocessor=None,
            tokenizer=None,
            analyzer='word',
            stop_words=None,
            token_pattern=r'(?u)\b\w\w+\b',
            ngram_range=(1, 1),
            max_df=1.0,
            min_df=1,
            max_features=None,
            vocabulary=None,
            binary=False,
            dtype=np.float64,
            norm='l2',
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=False,
        ),
        use_stop_words: bool = True,
        use_filtered_punct: bool = True,
    ):
        self.col = col
        self.tfidf_vectorizer = tfidf_vectorizer
        self.use_stop_words = use_stop_words
        self.stopwords = stopwords.words('russian')
        self.use_filtered_punct = use_filtered_punct

    def fit(self, X, split=False):
        text = X[self.col].dropna()
        if split:
            text = text.apply(lambda x: x.split())
        texts: pd.Series = text.apply(
            lambda x: x
            if self.use_stop_words is False
            else [word for word in x if word not in self.stopwords]
        )
        texts: pd.Series = texts.apply(
            lambda x: ' '.join(x)
            if self.use_filtered_punct is False
            else ' '.join([word for word in x if word not in punctuation])
        )
        self.tfidf_vectorizer.fit_transform(texts)
        return self

    def transform(self, X):
        df = self.tfidf_vectorizer.transform(
            X[self.col].fillna('').apply(lambda x: ' '.join(x))
        )
        train_ebmeddings = pd.DataFrame.sparse.from_spmatrix(
            data=df,
            columns=[
                f'{self.col.lower()}_{x}'
                for x in self.tfidf_vectorizer.get_feature_names_out()
            ],
            index=X.index,
        )
        return train_ebmeddings
