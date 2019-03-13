import numpy as np


def get_med(w1, w2):
    w1 = '#' + w1
    w2 = '#' + w2
    d = np.ndarray((len(w1), len(w2)))

    for i in range(len(w1)):
        d[i, 0] = i
    for j in range(len(w2)):
        d[0, j] = j

    for i in range(1, len(w1)):
        for j in range(1, len(w2)):
            c_i = c_d = 1
            c_s = 2 if w1[i] != w2[j] else 0
            d[i,j] = min([d[i-1,j]+c_d, d[i,j-1]+c_i, d[i-1,j-1]+c_s])

    return d[len(w1)-1, len(w2)-1], d


def print_backtrace(w1, w2, d):
    i, j = d.shape
    w1 = '#' + w1
    w2 = '#' + w2
    i -= 1
    j -= 1

    s1 = ''
    s2 = ''

    while i > 0 or j > 0:
        c_i = c_d = 1
        c_s = 2 if w1[i] != w2[j] else 0

        if d[i,j] == d[i-1,j-1] + c_s:
            # substitution
            s1 = w1[i] + s1
            s2 = w2[j] + s2
            i -= 1
            j -= 1
        elif d[i,j] == d[i,j-1] + c_i:
            # insertion
            s1 = '*' + s1
            s2 = w2[j] + s2
            j -= 1
        elif d[i,j] == d[i-1,j] + c_d:
            # deletion
            s1 = w1[i] + s1
            s2 = '*' + s2
            i -= 1

    print(s1)
    print(s2)


if __name__ == '__main__':
    w1 = 'разработка'
    w2 = 'ребята'
    med, d = get_med(w1, w2)

    print(med)

    print_backtrace(w1, w2, d)