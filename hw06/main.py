import os

import gensim, logging
from gensim.models import Word2Vec


class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split("\t")[0].split(" ")


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    corpus = MySentences('/home/anth/prog/nlp/datasets/so_125k/data/')

    # model = gensim.models.Word2Vec(corpus, size=200, iter=50)
    # model.save('/home/anth/prog/nlp/s6-ttb-hws/models/word2vec_2')

    model = Word2Vec.load('/home/anth/prog/nlp/s6-ttb-hws/models/word2vec_1')

    for word in ('android', 'java', 'program'):
        print("Most similar to %s:" % (word))
        print(model.wv.most_similar((word,)))


    print(model.wv['samsung'])
