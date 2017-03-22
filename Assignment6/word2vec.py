# word2vec 


from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import logging
from gensim.models.word2vec import Text8Corpus

logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO)
data = Text8Corpus("/Users/admin/Downloads/text8")
model = Word2Vec (data,	size=50, window=8, min_count=5, workers=8)
model.most_similar('president') # the most similar word to "president"
model.most_similar('election') # the most similar to "election"

