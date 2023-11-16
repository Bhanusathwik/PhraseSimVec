import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
import nltk

# Ensure nltk resources are downloaded (needed for tokenization)
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


def load_phrases(file_path):
    phrases_df = pd.read_csv(file_path)
    return phrases_df['phrase'].tolist()

def find_closest_match(input_vec, phrase_vectors, phrases):
    min_distance = float('inf')
    closest_match = None
    closest_index = -1
    for i, vec in enumerate(phrase_vectors):
        distance = np.linalg.norm(input_vec - vec)  # Euclidean distance
        if distance < min_distance:
            min_distance = distance
            closest_match = phrases[i]
            closest_index = i
    return closest_match, closest_index
