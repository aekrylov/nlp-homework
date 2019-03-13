import random

from nltk import FreqDist, defaultdict, LaplaceProbDist, pprint, DictionaryProbDist, ProbDistI
from nltk.classify import ClassifierI
from nltk.sentiment import SentimentAnalyzer
from nltk.tokenize import WordPunctTokenizer

from hw04.datasets import *


class NaiveBayesClassifier(ClassifierI):

    def __init__(self, label_probdist: ProbDistI, feature_probdists: dict) -> None:
        self._labels = feature_probdists.keys()
        self._label_probdist = label_probdist
        self._probdists = feature_probdists

    def classify(self, featureset):
        """
        Classifies under bag of words assumption
        :param featureset: bag of words
        :return: most suitable label
        """
        max_logprob = -10000
        argmax = None

        for label in self._labels:
            probs = [self._probdists[label].logprob(w) for w in featureset]
            p = sum(probs) + self._label_probdist.logprob(label)
            if p > max_logprob:
                max_logprob = p
                argmax = label

        return argmax

    def labels(self):
        return self._labels

    @classmethod
    def train(cls, labeled_docs):
        # corresponds to P(c_i)
        label_freqdist = FreqDist()

        # corresponds to P(w_i|c_j)
        feature_freqdist = defaultdict(FreqDist)

        vocab = set()

        for doc, label in labeled_docs:
            label_freqdist[label] += 1
            for w in doc:
                feature_freqdist[label][w] += 1
                vocab.add(w)

        label_probdist = DictionaryProbDist(label_freqdist, normalize=True)
        feature_probdists = {k: LaplaceProbDist(v, len(vocab)) for k, v in feature_freqdist.items()}
        return cls(label_probdist, feature_probdists)


def divide_data(data, train_percent: float):
    tagged_data = [(item, 'train') if random.random() < train_percent else (item, 'test') for item in data]
    return list(map(lambda x: x[0], filter(lambda x: x[1] == 'train', tagged_data))), \
           list(map(lambda x: x[0], filter(lambda x: x[1] == 'test', tagged_data)))


if __name__ == '__main__':
    analyzer = SentimentAnalyzer()

    tokenizer = WordPunctTokenizer()

    print('Reading corpora')
    pos_sents = tokenizer.tokenize_sents(read_pos())
    neg_sents = tokenizer.tokenize_sents(read_neg())

    all_sents = [(sent, 'pos') for sent in pos_sents] + [(sent, 'neg') for sent in neg_sents]
    train_sents, test_sents = divide_data(all_sents, 0.7)

    classifier = analyzer.train(NaiveBayesClassifier.train, train_sents)
    pprint(analyzer.evaluate(test_sents))

    while True:
        s = input('Enter sentence: ')
        print(classifier.classify(tokenizer.tokenize(s)))
