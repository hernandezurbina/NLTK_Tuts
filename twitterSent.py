import nltk
import random
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode

from nltk.tokenize import word_tokenize

class VoteClassifier(ClassifierI):
	def __init__(self, *classifiers):
		self._classifiers = classifiers

	def classify(self, features):
		votes = []
		for c in self._classifiers:
			v = c.classify(features)
			votes.append(v)
		return mode(votes)

	def confidence(self, features):
		votes = []
		for c in self._classifiers:
			v = c.classify(features)
			votes.append(v)
		choiceVotes = votes.count(mode(votes))
		conf = choiceVotes / len(votes)
		return conf

shortPos = open("positive.txt", "r", encoding="latin-1").read()
shortNeg = open("negative.txt", "r", encoding="latin-1").read()

documents = []
allWords = []

allowedWordTypes = ["J"] #J is adject

for p in shortPos.split('\n'):
	documents.append((p, "pos"))
	words = word_tokenize(p)
	pos = nltk.pos_tag(words)
	for w in pos:
		if w[1][0] in allowedWordTypes:
			allWords.append(w[0].lower())

for p in shortNeg.split('\n'):
	documents.append((p, "neg"))
	words = word_tokenize(p)
	pos = nltk.pos_tag(words)
	for w in pos:
		if w[1][0] in allowedWordTypes:
			allWords.append(w[0].lower())

with open("docs.pickle", "wb") as f:
	pickle.dump(documents, f)

allWords = nltk.FreqDist(allWords)
wordFeatures = list(allWords.keys())[:5000]

with open("wordFeatures.pickle", "wb") as f:
	pickle.dump(wordFeatures, f)

def findFeatures(document):
	words = word_tokenize(document)
	features = {}
	for w in wordFeatures:
		features[w] = (w in words)

	return features

featureSets = [(findFeatures(rev), category) for (rev, category) in documents]

with open("featureSets.pickle", "wb") as f:
	pickle.dump(featureSets, f)

random.shuffle(featureSets)

trainingSet = featureSets[:10000] 
testingSet = featureSets[10000:]

classifier = nltk.NaiveBayesClassifier.train(trainingSet)
print("NB (original) acc: ", nltk.classify.accuracy(classifier, testingSet) * 100)
with open("NB.pickle", "wb") as f:
	pickle.dump(classifier, f)

# classifier.show_most_informative_features(15)

# saveClassifier = open("NB.pickle", "wb")
# pickle.dump(classifier, saveClassifier)
# saveClassifier.close()
print()
mnbClassifier = SklearnClassifier(MultinomialNB())
mnbClassifier.train(trainingSet)
print("NB (multinomial) acc: ", nltk.classify.accuracy(mnbClassifier, testingSet) * 100)
with open("MNB.pickle", "wb") as f:
	pickle.dump(mnbClassifier, f)


# print()
# gaussClassifier = SklearnClassifier(GaussianNB())
# gaussClassifier.train(trainingSet)
# print("NB (Gaussian) acc: ", nltk.classify.accuracy(gaussClassifier, testingSet) * 100)

print()
bernClassifier = SklearnClassifier(BernoulliNB())
bernClassifier.train(trainingSet)
print("NB (Bernoulli) acc: ", nltk.classify.accuracy(bernClassifier, testingSet) * 100)
with open("Bern.pickle", "wb") as f:
	pickle.dump(bernClassifier, f)


# LogisticRegression, SGDClassifier
# SVC, LinearSVC, NuSVC

print()
logRegClassifier = SklearnClassifier(LogisticRegression())
logRegClassifier.train(trainingSet)
print("Logistic Regression acc: ", nltk.classify.accuracy(logRegClassifier, testingSet) * 100)
with open("LogReg.pickle", "wb") as f:
	pickle.dump(logRegClassifier, f)


print()
sgdClassifier = SklearnClassifier(SGDClassifier())
sgdClassifier.train(trainingSet)
print("SGD acc: ", nltk.classify.accuracy(sgdClassifier, testingSet) * 100)
with open("SDG.pickle", "wb") as f:
	pickle.dump(sgdClassifier, f)


# print()
# svcClassifier = SklearnClassifier(SVC())
# svcClassifier.train(trainingSet)
# print("SVC acc: ", nltk.classify.accuracy(svcClassifier, testingSet) * 100)

print()
linSvcClassifier = SklearnClassifier(LinearSVC())
linSvcClassifier.train(trainingSet)
print("Linear SVC acc: ", nltk.classify.accuracy(linSvcClassifier, testingSet) * 100)
with open("LinSVC.pickle", "wb") as f:
	pickle.dump(linSvcClassifier, f)


print()
nuSvcClassifier = SklearnClassifier(NuSVC())
nuSvcClassifier.train(trainingSet)
print("Nu SVC acc: ", nltk.classify.accuracy(nuSvcClassifier, testingSet) * 100)
with open("NuSVC.pickle", "wb") as f:
	pickle.dump(nuSvcClassifier, f)


votedClassifier = VoteClassifier(classifier, mnbClassifier, bernClassifier,
	logRegClassifier, sgdClassifier, linSvcClassifier, nuSvcClassifier)

print("Voted classifier acc: ", nltk.classify.accuracy(votedClassifier, testingSet) * 100)
print()
