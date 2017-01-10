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

with open("docs.pickle", "rb") as f:
	documents = pickle.load(f)

with open("wordFeatures.pickle", "rb") as f:
	wordFeatures = pickle.load(f)

def findFeatures(document):
	words = word_tokenize(document)
	features = {}
	for w in wordFeatures:
		features[w] = (w in words)

	return features

with open("featureSets.pickle", "rb") as f:
	featureSets = pickle.load(f)

random.shuffle(featureSets)
print(len(featureSets))

testingSet = featureSets[10000:]
trainingSet = featureSets[:10000]

with open("NB.pickle", "rb") as f:
	classifier = pickle.load(f)

with open("MNB.pickle", "rb") as f:
	mnbClassifier = pickle.load(f)

with open("Bern.pickle", "rb") as f:
	bernClassifier = pickle.load(f)

with open("logReg.pickle", "rb") as f:
	logRegClassifier = pickle.load(f)

with open("SDG.pickle", "rb") as f:
	sgdClassifier = pickle.load(f)

with open("linSVC.pickle", "rb") as f:
	linSvcClassifier = pickle.load(f)

with open("NuSVC.pickle", "rb") as f:
	nuSvcClassifier = pickle.load(f)

votedClassifier = VoteClassifier(classifier, mnbClassifier, bernClassifier,
	logRegClassifier, sgdClassifier, linSvcClassifier, nuSvcClassifier)

def sentiment(text):
	feats = findFeatures(text)
	return votedClassifier.classify(feats), votedClassifier.confidence(feats)




