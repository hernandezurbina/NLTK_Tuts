from nltk.stem import WordNetLemmatizer

lemmatizer =  WordNetLemmatizer()

print(lemmatizer.lemmatize("dogs"))
print(lemmatizer.lemmatize("cacti"))
print(lemmatizer.lemmatize("geese"))
print(lemmatizer.lemmatize("rocks"))
print(lemmatizer.lemmatize("python"))
print(lemmatizer.lemmatize("formulae"))
print(lemmatizer.lemmatize("better", pos="a"))
print(lemmatizer.lemmatize("best", pos="a"))
print(lemmatizer.lemmatize("runs"))
print(lemmatizer.lemmatize("runs", pos="v"))

