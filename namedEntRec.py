import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

trainText = state_union.raw("2005-GWBush.txt")
sampleText = state_union.raw("2006-GWBush.txt")

customSentTokenizer = PunktSentenceTokenizer(trainText)

tokenized = customSentTokenizer.tokenize(sampleText)

def processContent():
	try:
		j = 0
		for i in tokenized:
			words = nltk.word_tokenize(i)
			tagged = nltk.pos_tag(words)

			namedEnt = nltk.ne_chunk(tagged, binary=True)
			if j < 10:
				namedEnt.draw()
				j += 1
	except Exception as e:
		print(str(e))


processContent()

