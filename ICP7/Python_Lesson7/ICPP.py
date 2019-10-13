from bs4 import BeautifulSoup
import requests
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.util import trigrams
from nltk.tokenize import RegexpTokenizer
from nltk import ne_chunk

wikiDoc = requests.get("https://en.wikipedia.org/wiki/Google");
parsedDoc = BeautifulSoup(wikiDoc.content, "html.parser")
page = parsedDoc.get_text("\n")
print(page)

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

PS = nltk.pos_tag(wrd_tokens)
print('PS')
print(PS)

namedEntity = ne_chunk(PS)
print('NamedEntity')
print(namedEntity);

pStemmer = PorterStemmer();
stemmData = [pStemmer.stem(tagged_word[0]) for tagged_word in wrd_tokens]
print('Stemmed Data')
print(stemmData)

lemmetizer = WordNetLemmatizer()
lemmetizeData = [lemmetizer.lemmatize(tagged_word[0]) for tagged_word in PS]
print('Lemmetized Data')
print(lemmetizeData)


