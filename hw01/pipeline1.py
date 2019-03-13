import numpy as np
from nltk import SnowballStemmer, WordPunctTokenizer
from nltk.stem import StemmerI
from nltk.tokenize.api import StringTokenizer
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from hw01.datasets import read_pos, read_neg


# class MyPipeline:
#     def __init__(self, text: str):
#         self.text = text
#         self.tokens = None
#
#     def tokenize(self, tokenizer: StringTokenizer):
#         self.tokens = tokenizer.tokenize(self.text)
#
#     def stem(self, stemmer: StemmerI):
#         self.tokens = [stemmer.stem(token) for token in self.tokens]
#
#     def filter_tokens(self, f):
#         self.tokens = list(filter(f, self.tokens))


class SklearnTokenizer(BaseEstimator):
    def __init__(self):
        self.tokenizer = WordPunctTokenizer()

    def fit(self, *args):
        return self

    def transform(self, docs, *args):
        return [filter(lambda w: len(w) > 1, self.tokenizer.tokenize(doc)) for doc in docs]


class SklearnLemmatizer(BaseEstimator):
    def __init__(self):
        self.stemmer = SnowballStemmer("russian")

    def fit(self, *args):
        return self

    def transform(self, docs):
        return [[self.stemmer.stem(word) for word in doc] for doc in docs]


# p = MyPipeline(state_union.raw())
# p.tokenize(SpaceTokenizer())
# # p.filter_tokens(lambda s: ''.)
# p.stem(SnowballStemmer("english"))


if __name__ == '__main__':
    pipeline = Pipeline([
        ('tokenizer', SklearnTokenizer()),
        ('stemmer', SklearnLemmatizer()),
        ('counts', CountVectorizer(tokenizer=lambda x:x, preprocessor=lambda x:x, lowercase=False)),
        ('tfidf', TfidfTransformer()),
    ])

    pos_sents, neg_sents = read_pos(), read_neg()
    all_sents = [(sent, 'pos') for sent in pos_sents] + [(sent, 'neg') for sent in neg_sents]

    train_set, test_set = tuple(train_test_split(all_sents))

    X, y = zip(*train_set)

    tfidf = pipeline.fit_transform(list(X), list(y))
    idf = pipeline.named_steps.tfidf.idf_
    counts = pipeline.named_steps.counts

    print(tfidf)

    tfidf_idx_max = list(np.unravel_index(np.argmax(tfidf, axis=None), tfidf.shape))
    tfidf_val = np.argmax(tfidf, axis=None)
    tfidf_word = counts.get_feature_names()[tfidf_idx_max[1]]

    idf_max = max(idf)
    idf_word = counts.get_feature_names()[np.argmax(idf)]

    tf_idx_max = list(np.unravel_index(np.argmax(tfidf / idf), tfidf.shape))
    tf_val = np.argmax(tfidf / idf)
    tf_word = counts.get_feature_names()[tf_idx_max[1]]

    for text, val, word in (('tfidf', tfidf_val, tfidf_word), ('tf', tf_val, tf_word), ('idf', idf_max, idf_word)):
        print('%s: \'%s\' (%f)' % (text, word, val))