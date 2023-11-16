import pandas as pd
from wordvec_app.model import WordVecModel
from wordvec_app.utils import phrase_vector
from scipy.spatial.distance import cosine
import numpy as np
from tqdm import tqdm

print("1")

model = WordVecModel('data/vectors.csv')

print("2")


phrases_df = pd.read_csv('data/phrases.csv', encoding='ISO-8859-1')
phrases = phrases_df['Phrases'].tolist()

print("3")


phrase_vectors = [phrase_vector(phrase, model) for phrase in tqdm(phrases)]

# Function to calculate similarities
def calculate_similarities(phrase_vectors):
    n = len(phrase_vectors)
    distances = np.zeros((n, n))
    for i, vec1 in tqdm(enumerate(phrase_vectors)):
        for j, vec2 in enumerate(phrase_vectors):
            distances[i, j] = 1 - cosine(vec1, vec2) if np.any(vec1) and np.any(vec2) else 0
    return distances

similarity_matrix = calculate_similarities(phrase_vectors)


similarity_df = pd.DataFrame(similarity_matrix)
similarity_df.to_csv('data/phrase_similarities.csv', index=False)
