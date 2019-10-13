from bs4 import BeautifulSoup
import requests
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.util import trigrams
from nltk import ne_chunk
from nltk.tokenize import RegexpTokenizer

wikiDoc = requests.get("https://en.wikipedia.org/wiki/Google");
parsedDoc = BeautifulSoup(wikiDoc.content, "html.parser")
page = parsedDoc.get_text("\n")

with open('/Users/anushakonakalla/Desktop/test.txt', 'w') as f:
   for line in page:
      f.write(str(line))

file_content = open("/Users/anushakonakalla/Desktop/test.txt").read()
print('wrd_tokens')
tokenizer = RegexpTokenizer(r'\w+')
wrd_tokens=tokenizer.tokenize(page)
print(wrd_tokens)
with open('/Users/anushakonakalla/Desktop/token.txt', 'w') as f:
   f.write(str(wrd_tokens))

triData = list(trigrams(wrd_tokens))
print('Tridata')
print(triData)
with open('/Users/anushakonakalla/Desktop/trigram.txt', 'w') as f:
   f.write(str(triData))


PofS = nltk.pos_tag(wrd_tokens)
print('Parts of Speech')
print(PofS)

print('NAMED ENTITY')
namedEntityRecgData = ne_chunk(PofS)
print(namedEntityRecgData);

print('Stemming')
ps = PorterStemmer()
for word in wrd_tokens:
    print("Stem for {} is {}".format(word, ps.stem(word)))

print('LEMMATIZATION')
lem = WordNetLemmatizer()
for word in wrd_tokens:
    print("Lemma for {} is {}".format(word, lem.lemmatize(word, pos="a")))