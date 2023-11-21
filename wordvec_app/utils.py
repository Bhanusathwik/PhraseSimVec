import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')

def phrase_vector(phrase, model):
    words = word_tokenize(phrase.lower())
    valid_words = [word for word in words if model.has_word(word)]
    if not valid_words:
        return np.zeros(model.model.vector_size)
    word_vectors = np.array([model.get_vector(word) for word in valid_words if model.get_vector(word) is not None])
    sum_vector = word_vectors.sum(axis=0)
    norm_vector = sum_vector / np.linalg.norm(sum_vector) if np.linalg.norm(sum_vector) != 0 else sum_vector
    return norm_vector


