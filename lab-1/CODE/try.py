from bs4 import BeautifulSoup
import requests
import nltk
import functools
import operator
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.util import trigrams
from nltk import ne_chunk
from nltk.tokenize import RegexpTokenizer


file_content = open("/Users/anushakonakalla/Desktop/f1.txt").read()

phrases = list(nltk.sent_tokenize(file_content))
print(phrases)

tokenizer = RegexpTokenizer(r'\w+')
print('TOKENS')
stokens = tokenizer.tokenize(file_content)
print(stokens)




triData = list(trigrams(stokens))
print('Tridata')
print(triData)


PS = nltk.pos_tag(stokens)
print('PS')
print(PS)
lemmetizer = WordNetLemmatizer()
lemmetizeData = [lemmetizer.lemmatize(tagged_word[0]) for tagged_word in PS]
print('Lemmetized Data')
print(lemmetizeData)


print('Top Trigrams')
tp = nltk.Counter(nltk.ngrams(stokens, 3)).most_common(10)
print(tp)


list=[]
for i in range(0, 10):
    str = " ".join(tp[i][0])
    print(str)
    list.append(str)
print(list)
for i in list:
 print(i)
 with open('/Users/anushakonakalla/Desktop/f1.txt', 'r') as file:
     matchedLine = ''
     array = []
     for line in file:
         if i in line:
             matchedLine += line
     print(matchedLine)






