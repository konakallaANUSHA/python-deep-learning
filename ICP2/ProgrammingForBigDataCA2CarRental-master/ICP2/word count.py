file = open("/Users/anushakonakalla/Desktop/test.txt", 'r')
file2 = open("/Users/anushakonakalla/Desktop/try.txt", 'w')

list = []

wordfreq = {}

for word in file.read().lower().split():
        if word not in wordfreq:
            wordfreq[word] = 1
        else:
            wordfreq[word] += 1

for k, v in wordfreq.items():
       output=  print(k,v)




def main(filename):
    wordfreq = {}

    for word in file.read().lower().split():
        if word not in wordfreq:
            wordfreq[word] = 1
        else:
            wordfreq[word] += 1

    for k, v in wordfreq.items():
        yield k,v


main("/Users/anushakonakalla/Desktop/test.txt")

with open("/Users/anushakonakalla/Desktop/try.txt", "w") as f:
    for word, freq in main("/Users/anushakonakalla/Desktop/test.txt"):
        f.write('%s\t%d\n' % (word, freq))