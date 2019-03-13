import csv


def read_pos() -> list:
    with open('../data/positive1.csv', 'rU') as f:
        reader = csv.reader(f, delimiter=';')
        return list(map(lambda row: row[3], reader))


def read_neg() -> list:
    with open('../data/negative1.csv', 'rU') as f:
        reader = csv.reader(f, delimiter=';')
        return list(map(lambda row: row[3], reader))
