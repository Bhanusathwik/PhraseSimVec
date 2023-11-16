from scipy.spatial.distance import cosine
from .utils import phrase_vector

class SimilarityCalculator:
    def __init__(self, model):
        self.model = model

    def calculate_similarity(self, phrase1, phrase2):
        vec1 = phrase_vector(phrase1, self.model)
        vec2 = phrase_vector(phrase2, self.model)
        return cosine(vec1, vec2)  # or use euclidean(vec1, vec2)
