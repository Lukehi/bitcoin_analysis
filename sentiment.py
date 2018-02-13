from nltk.tokenize import sent_tokenize, word_tokenize


directory = '/Users/lukehindson/PycharmProjects/Bitcoin/'

short_pos = open(directory + 'Data/positive.txt','r').read()
short_neg = open(directory + 'Data/negative.txt','r').read()

documents = []

for r in short_pos.split('\n'):
    documents.append( (r, "pos") )

for r in short_neg.split('\n'):
    documents.append( (r, "neg") )


all_words = []

short_pos_words = word_tokenize(short_pos)
short_neg_words = word_tokenize(short_neg)

for w in short_pos_words:
    all_words.append(w.lower())

for w in short_neg_words:
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:5000]


def find_features(document):
	words = word_tokenize(document)
	features = {}
	for w in word_features:
		features[w] = (w in words)

	return features


featuresets = [(find_features(rev), category) for (rev, category) in documents]
random.shuffle(featuresets)