import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

documents = [(list(movie_reviews.words(fileid)), category)
			for category in movie_reviews.categories()
			for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

allWords = []
for w in movie_reviews.words():
	allWords.append(w.lower())

allWords = nltk.FreqDist(allWords)
wordFeatures = list(allWords.keys())[:3000]

def findFeatures(document):
	words = set(document)
	features = {}
	for w in wordFeatures:
		features[w] = (w in words)

	return features

featureSets = [(findFeatures(rev), category) for (rev, category) in documents]

trainingSet = featureSets[:1900] 
testingSet = featureSets[1900:]

classifier = nltk.NaiveBayesClassifier.train(trainingSet)
print("NB (original) acc: ", nltk.classify.accuracy(classifier, testingSet) * 100)
classifier.show_most_informative_features(15)

# saveClassifier = open("NB.pickle", "wb")
# pickle.dump(classifier, saveClassifier)
# saveClassifier.close()
print()
mnbClassifier = SklearnClassifier(MultinomialNB())
mnbClassifier.train(trainingSet)
print("NB (multinomial) acc: ", nltk.classify.accuracy(mnbClassifier, testingSet) * 100)

# print()
# gaussClassifier = SklearnClassifier(GaussianNB())
# gaussClassifier.train(trainingSet)
# print("NB (Gaussian) acc: ", nltk.classify.accuracy(gaussClassifier, testingSet) * 100)

print()
bernClassifier = SklearnClassifier(BernoulliNB())
bernClassifier.train(trainingSet)
print("NB (Bernoulli) acc: ", nltk.classify.accuracy(bernClassifier, testingSet) * 100)

# LogisticRegression, SGDClassifier
# SVC, LinearSVC, NuSVC

print()
logRegClassifier = SklearnClassifier(LogisticRegression())
logRegClassifier.train(trainingSet)
print("Logistic Regression acc: ", nltk.classify.accuracy(logRegClassifier, testingSet) * 100)

print()
sgdClassifier = SklearnClassifier(SGDClassifier())
sgdClassifier.train(trainingSet)
print("SGD acc: ", nltk.classify.accuracy(sgdClassifier, testingSet) * 100)

print()
svcClassifier = SklearnClassifier(SVC())
svcClassifier.train(trainingSet)
print("SVC acc: ", nltk.classify.accuracy(svcClassifier, testingSet) * 100)

print()
linSvcClassifier = SklearnClassifier(LinearSVC())
linSvcClassifier.train(trainingSet)
print("Linear SVC acc: ", nltk.classify.accuracy(linSvcClassifier, testingSet) * 100)

print()
nuSvcClassifier = SklearnClassifier(NuSVC())
nuSvcClassifier.train(trainingSet)
print("Nu SVC acc: ", nltk.classify.accuracy(nuSvcClassifier, testingSet) * 100)

print()
