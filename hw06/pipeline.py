import logging
import os
import pickle
import random
import sys

import numpy as np
from gensim.models import Word2Vec
from joblib import Memory, Parallel, delayed
from sklearn import metrics
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion

from hw05.main import get_datasets, get_test_results, OneHotNormalizer, tokenize


def w2v_worker(model, tokenizer, doc):
    vector_size = model.wv.vector_size
    return sum([model.wv[w] if w in model.wv else np.zeros((vector_size,)) for w in tokenizer(doc)])


class Word2VecVectorizer(BaseEstimator):
    def __init__(self, tokenizer, model: Word2Vec):
        self.model = model
        self.tokenizer = tokenizer
        self.vector_size = self.model.wv.vector_size
        # self.memory = Memory('./cache', verbose=0)
        # self.worker = self.memory.cache(w2v_worker)
        # self.pool = mp.Pool()

    def transform(self, *args):
        return self.fit_transform(*args)

    def fit(self, docs, _=None):
        self.fit_transform(docs)
        return self

    def fit_transform(self, docs, *args):
        N_docs = len(list(docs))
        N_feats = self.vector_size

        logging.info("WORD2VEC : fit start")
        # X = np.ndarray(shape=(N_docs, N_feats))
        # X = Parallel(n_jobs=-1)(delayed(self.worker)(self.model, self.tokenizer, doc) for doc in docs)
        X = np.array([
            np.sum([self.model.wv[w] if w in self.model.wv else np.zeros((N_feats,)) for w in self.tokenizer(doc)], 0) for doc in docs
        ])
        # X = pool.map(w2v_worker, map(lambda doc: (self.model, self.tokenizer, doc), docs), 250)
        # X = pool.map(self._doc_vector, docs, 250)
        logging.info("WORD2VEC : fit end")
        return X


def get_pipeline(train_x, train_y, load=True):
    model = Word2Vec.load('/home/anth/prog/nlp/s6-ttb-hws/models/word2vec_2')

    if load and os.path.isfile(PIPELINE_PICKLE):
        return pickle.load(open(PIPELINE_PICKLE, 'rb'))
    else:
        bow_pipeline = Pipeline([
            ('count_vectorizer', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('norm', OneHotNormalizer())
        ])

        w2v_pipeline = Pipeline([
            ('word2vec', Word2VecVectorizer(tokenizer=tokenize, model=model)),
            ('norm', OneHotNormalizer())
        ])

        all_features = FeatureUnion([
            ('bow', bow_pipeline),
            ('word2vec', w2v_pipeline),
        ])

        pipeline = Pipeline([
            ('all_features', all_features),
            ('classifier', LogisticRegression(solver='sag', n_jobs=-1, max_iter=1000))
        ], './cache')

        logging.info("Fitting the pipeline")
        pipeline.fit(train_x, train_y)
        pickle.dump(pipeline, open(PIPELINE_PICKLE, 'wb'))
        return pipeline


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    random.seed(1337)

    N = int(sys.argv[1]) if len(sys.argv) > 1 else None

    DIR = os.path.dirname(__file__)
    PIPELINE_PICKLE = os.path.join(DIR, 'pipeline.pickle')
    TEST_PICKLE = os.path.join(DIR, 'test_results.pickle')

    # pool = mp.Pool(processes=1)

    train_x, train_y, test_x, test_y = get_datasets(N)

    pipeline = get_pipeline(train_x, train_y)

    test_x, test_y, predicted = get_test_results(pipeline, test_x, test_y)
    print(metrics.classification_report(test_y, predicted))
