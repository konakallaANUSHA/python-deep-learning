import nltk
nltk.download("brown")
from nltk.corpus import brown
print(brown.categories())
print(brown.words(categories ='news'))
from nltk.stem import WordNetLemmatizer

lem = WordNetLemmatizer()
print("better :", lem.lemmatize("better", pos ="a"))



from nltk.stem import 	WordNetLemmatizer
WordNet_lemmatizer = WordNetLemmatizer()
text = "studies studying cries cry"
tokenization = nltk.word_tokenize(text)
for w in tokenization:
	print("Lemma for {} is {}".format(w, WordNet_lemmatizer.lemmatize(w)))