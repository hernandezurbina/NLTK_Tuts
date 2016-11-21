from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

exampleText = "Hello Mr. Smith, how are you doing today? The weather is great and Python is awesome. The sky is pink blue and you should not eat cardboard"

print(sent_tokenize(exampleText))
print()
print(word_tokenize(exampleText))
print()

stopWords = set(stopwords.words("english"))
words = word_tokenize(exampleText)

filteredSentence = [w for w in words if not w in stopWords]

print(filteredSentence)

ps = PorterStemmer()

print()
exampleWords = ["Python", "pythoner", "pythoning", "pythoned", "pythonly", "pythonista"]
for w in exampleWords:
	print(ps.stem(w))

newText = "It is very important to be pythonly while you are pythoning with Python. All pythoners have pythoned poorly at least once."

print()
words = word_tokenize(newText)
for w in words:
	print(ps.stem(w))


