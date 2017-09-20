import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None,
                 lowercase=True, preprocessor=None, tokenizer=None,
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), analyzer='word',
                 max_df=1.0, min_df=1, max_features=None,
                 vocabulary=None, binary=False, dtype=np.int64):
        #vectorizer params
        self.input = input
        self.encoding = encoding
        self.decode_error = decode_error
        self.strip_accents = strip_accents
        self.lowercase = lowercase
        self.preprocessor = preprocessor
        self.stop_words = stop_words
        self.token_pattern = token_pattern
        self.ngram_range = ngram_range
        self.analyzer = analyzer
        self.max_df = max_df
        self.min_df = min_df
        self.max_features = max_features
        self.vocabulary = vocabulary
        self.binary = binary
        self.dtype = dtype
        self.last_used_vectorizer_ = None  # type: CountVectorizer
        self.last_used_vectorizer_feature_names_ = None
        self.last_used_tfidf_ = None  # type: TfidfVectorizer
        self.last_used_tfidf_feature_names_ = None
        self.last_used_lda_ = None  # type: LatentDirichletAllocation
        self.last_used_nmf_ = None  # type: NMF

    def fit(self, X, y=None, **fit_params):
        self.fit_transform(X=X, y=y, fit_params=fit_params)
        return self

    @staticmethod
    def _display_topics(model, feature_names, no_top_words):
        for topic_idx, topic in enumerate(model.components_):
            print("Topic %d:" % topic_idx)
            print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

    def fit_transform(self, X, y=None, **fit_params):
        self.last_used_vectorizer_ = CountVectorizer(
            input=self.input, encoding=self.encoding, decode_error=self.decode_error, strip_accents=self.strip_accents,
            lowercase=self.lowercase, preprocessor=self.preprocessor, stop_words=self.stop_words,
            token_pattern=self.token_pattern, ngram_range=self.ngram_range, analyzer=self.analyzer, max_df=self.max_df,
            min_df=self.min_df, max_features=self.max_features, vocabulary=self.vocabulary, binary=self.binary,
            dtype=self.dtype
        )
        if y is not None:
            no_topics = len(set(y)) * 1.5
        else:
            no_topics = 300
        self.last_used_tfidf_ = TfidfVectorizer(stop_words=self.stop_words)
        self.last_used_nmf_ = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd')
        self.last_used_lda_ = LatentDirichletAllocation(n_topics=no_topics, max_iter=5, learning_method='online', learning_offset=50.0,random_state=0)
        #fitting and transforming
        X_vect = self.last_used_vectorizer_.fit_transform(raw_documents=X, y=y)
        self.last_used_vectorizer_feature_names_ = self.last_used_vectorizer_.get_feature_names()
        X_tfidf = self.last_used_tfidf_.fit_transform(raw_documents=X, y=y)
        self.last_used_tfidf_feature_names_ = self.last_used_tfidf_.get_feature_names()
        X_lda = self.last_used_lda_.fit_transform(X=X, y=y)
        X_nmf = self.last_used_nmf_.fit_transform(X=X, y=y)
        
        return X_vect

    def transform(self, X):
        if self.last_used_vectorizer_ is None:
            return X
        # TODO potential for fuzzy-match correction here
        return self._do_transform(X=X)


