from gensim.models import KeyedVectors
class WordVecModel:
    def __init__(self, file_path):
        self.model = KeyedVectors.load_word2vec_format(file_path, binary=False)

    def get_vector(self, word):
        return self.model[word] if word in self.model.key_to_index else None

    def has_word(self, word):
        return word in self.model.key_to_index
