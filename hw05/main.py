import logging
import os
import string
import sys
import random
import itertools
from collections import defaultdict

import numpy as np
import multiprocessing as mp
from nltk import pos_tag
from nltk.corpus import stopwords

from joblib import Parallel, delayed, Memory
from sklearn import metrics
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
import pickle

from commons.helpers import FeatureNamePipeline
from extra.taggers import PerceptronTagger


class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                s = line.split("\t")
                yield (s[1], s[0].split(" "))


class SklearnSentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                s = line.split("\t")
                yield (s[1].split(" ")[0].strip(), s[0])


class OneHotNormalizer(BaseEstimator):
    """
    Apply it to all the one hot encoded features separately, so that it sums up to one feature
    """

    def fit(self, *args):
        return self

    def transform(self, X, *args):
        d = X.shape[1]
        return X / d


class PosVectorizer(BaseEstimator):
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
        self.pos_list = None
        self.cv = CountVectorizer(tokenizer=tokenizer)
        self.memory = Memory('./cache', verbose=0)
        # self.worker = self.memory.cache(pos_tag_worker)
        self.worker = self.memory.cache(pos_tag, verbose=0)

    def transform(self, *args):
        return self.fit_transform(*args)

    def fit(self, docs, _=None):
        self.fit_transform(docs)
        return self

    def fit_transform(self, docs, _=None):
        tokenizer = self.tokenizer
        all_pos = set()

        N_docs = len(list(docs))

        dicts = [defaultdict(lambda: 0) for _ in docs]

        logging.info("POS: tagging")
        # tags = [[token.pos_ for token in self.nlp(doc)] for doc in docs]
        # tags = nltk.pos_tag_sents([tokenizer(doc) for doc in docs])
        # pool = mp.Pool()

        tags = Parallel(n_jobs=-1)(delayed(self.worker)(tokenizer(doc)) for doc in docs)

        # tags = pool.map(pos_tag_worker, docs, 250)  # [tagger.tag(doc, tokenize=False) for doc in docs]
        # tags = [TextBlob(doc, pos_tagger=tagger, tokenizer=False).pos_tags for doc in docs]

        logging.info("POS: counting")
        for counts, tag_data in zip(dicts, tags):
            for word, tag in tag_data:
                all_pos.add(tag)
                counts[tag] += 1

        if self.pos_list is None:
            pos_list = self.pos_list = list(all_pos)
        else:
            pos_list = self.pos_list

        unk_pos = set()

        X = np.zeros(shape=(N_docs, len(pos_list)))
        for counts, i in zip(dicts, range(N_docs)):
            for pos, count in counts.items():
                try:
                    idx = pos_list.index(pos)
                    X[i, idx] = count
                except ValueError:
                    unk_pos.add(pos)
                    continue

        logging.info("POS: done")
        logging.info('POS: unknown POS: %a', unk_pos)
        return X

    def get_feature_names(self):
        return self.pos_list


memory = Memory('./cache')
stops = set(stopwords.words('english'))


# @memory.cache(verbose=0)
def tokenize(s):
    # return s.split()
    return list(filter(lambda t: t not in stops and t not in string.punctuation, s.split()))


def get_datasets(N=None):
    corpus = SklearnSentences('/home/anth/prog/nlp/datasets/so_125k/data/')

    train_set = []
    test_set = []

    for label, sent in itertools.islice(corpus, N):
        (train_set if random.random() < 0.7 else test_set).append((label, sent))

    train_y, train_x = zip(*train_set)
    test_y, test_x = zip(*test_set)
    return train_x, train_y, test_x, test_y


def get_pipeline(train_x, train_y, load=True):
    # if load and os.path.isfile(PIPELINE_PICKLE):
    #     return pickle.load(open(PIPELINE_PICKLE, 'rb'))
    # else:
        bow_pipeline = FeatureNamePipeline([
            ('count_vectorizer', CountVectorizer(stop_words=stops)),
            ('tfidf', TfidfTransformer()),
            # ('norm', OneHotNormalizer())
        ])

        pos_pipeline = FeatureNamePipeline([
            ('pos', PosVectorizer(tokenizer=tokenize)),
            # ('norm', OneHotNormalizer())
        ])

        all_features = FeatureUnion([
            ('bow', bow_pipeline),
            ('pos', pos_pipeline),
        ])

        pipeline = Pipeline([
            ('all_features', all_features),
            ('classifier', LogisticRegression(solver='sag', n_jobs=-1, max_iter=1000))
        ], './cache')

        logging.info("Fitting the pipeline")
        pipeline.fit(train_x, train_y)
        # pickle.dump(pipeline, open(PIPELINE_PICKLE, 'wb'))
        return pipeline


def get_test_results(pipeline: Pipeline, test_x, test_y, load=True):
    # if load and os.path.isfile(TEST_PICKLE):
    #     return pickle.load(open(TEST_PICKLE, 'rb'))

    logging.info("Predicting test data")
    predicted = pipeline.predict(test_x)
    # pickle.dump((test_x, test_y, predicted), open(TEST_PICKLE, 'wb'))
    return test_x, test_y, predicted


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    random.seed(1337)

    N = int(sys.argv[1]) if len(sys.argv) > 1 else None

    DIR = os.path.dirname(__file__)
    PIPELINE_PICKLE = os.path.join(DIR, 'pipeline.pickle')
    TEST_PICKLE = os.path.join(DIR, 'test_results.pickle')

    train_x, train_y, test_x, test_y = get_datasets(N)

    pipeline = get_pipeline(train_x, train_y)

    test_x, test_y, predicted = get_test_results(pipeline, test_x, test_y)
    print(metrics.classification_report(test_y, predicted))

    print("Top features for each class: ")
    clf = pipeline.named_steps.classifier
    fnames = pipeline.named_steps.all_features.get_feature_names()
    for i, cls in enumerate(clf.classes_):
        top_features = np.argsort(clf.coef_[i])[::-1]
        print("%10s: %a" % (cls, [(fnames[idx], clf.coef_[i, idx]) for idx in top_features[:10]]))
