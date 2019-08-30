import re
from collections import Counter


def openfile(filename):
    fh = open(filename, "r+")
    str = fh.read()
    fh.close()
    return str


def removegarbage(str):
    # Replace one or more non-word (non-alphanumeric) chars with a space
    str = re.sub(r'\W+', ' ', str)
    str = str.lower()
    return str


def getwordbins(words):
    cnt = Counter()
    for word in words:
        cnt[word] += 1
    return cnt


def main(filename, topwords):
    txt = openfile(filename)
    txt = removegarbage(txt)
    words = txt.split(' ')
    bins = getwordbins(words)
    for key, value in bins.most_common(topwords):

        yield key, value

main("/Users/anushakonakalla/Desktop/test.txt", 500)

with open("/Users/anushakonakalla/Desktop/try.txt", "a") as f:
    for word, freq in main("/Users/anushakonakalla/Desktop/test.txt", 500):
        f.write('%s\t%d\n' % (word, freq))