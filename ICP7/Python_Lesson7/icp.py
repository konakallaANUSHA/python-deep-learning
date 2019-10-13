from bs4 import BeautifulSoup
import requests
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.util import trigrams
from nltk import ne_chunk
wikiDoc = requests.get("https://en.wikipedia.org/wiki/Google");
parsedDoc = BeautifulSoup(wikiDoc.content, "html.parser")
page = parsedDoc.get_text("\n")

print(page)

stokens = nltk.word_tokenize(page)

triData = list((trigrams(stokens)))
print('Tridata')
print(triData)


stokens = nltk.pos_tag(stokens)
namedEntityRecgData = ne_chunk(stokens)
print(namedEntityRecgData);

pStemmer = PorterStemmer();
stemmData = [pStemmer.stem(tagged_word[0]) for tagged_word in stokens]
print('Stemmed Data')
print(stemmData)

lemmetizer = WordNetLemmatizer()
lemmetizeData = [lemmetizer.lemmatize(tagged_word[0]) for tagged_word in stokens]
print('Lemmetized Data')
print(lemmetizeData)