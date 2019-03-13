import pprint
from collections import defaultdict
from nltk.book import text1, text5


def distr(tokens: list, N=2):
    # count
    counts = defaultdict(lambda: 0)

    for i in range(len(tokens)):
        for n in range(N):
            gram = tuple(map(lambda t: t.lower(), tokens[i-n:i+1]))
            counts[gram] += 1

    counts[tuple()] = len(tokens)

    # normalize
    d_norm = {k: v / counts[k[:-1]] for k, v in counts.items()}
    return d_norm


if __name__ == '__main__':
    pp = pprint.PrettyPrinter(indent=4)

    d1 = distr(text1.tokens)
    d2 = distr(text5.tokens)
    pp.pprint(sorted(filter(lambda t: len(t[0]) > 1, d1.items()), key=lambda x: x[1], reverse=True)[:20])
    pp.pprint(sorted(filter(lambda t: len(t[0]) > 1, d2.items()), key=lambda x: x[1], reverse=True)[:20])

    print(d1[('hi', 'guys')])
    print(d2[('hi', 'guys')])
