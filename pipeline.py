import gensim
from gensim.models import KeyedVectors

model_filename = 'data/GoogleNews-vectors-negative300.bin'

wv = KeyedVectors.load_word2vec_format(model_filename, binary=True, limit=1000000)

wv.save_word2vec_format('data/vectors.csv')