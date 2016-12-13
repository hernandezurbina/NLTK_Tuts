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
			#chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""
			chunkGram = r"""Chunk: {<PRP>+<MD>*<VB.?>*<RB.?>*<NN>*}"""
			chunkParser = nltk.RegexpParser(chunkGram)
			chunked = chunkParser.parse(tagged)
			print(chunked)

			for subtree in chunked.subtrees(filter=lambda t: t.label() == 'Chunk'):
				print(subtree)

			print("\n")

			if j < 10:
				chunked.draw()
			j += 1
	except Exception as e:
		print(str(e))


processContent()

