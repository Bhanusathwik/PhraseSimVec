import pandas as pd
from wordvec_app.model import WordVecModel
from wordvec_app.utils import phrase_vector,load_phrases
from scipy.spatial.distance import cosine
import numpy as np
from tqdm import tqdm

def find_closest_match(user_input):
    model = WordVecModel('data/vectors.csv')

    user_vector = phrase_vector(user_input, model)
    phrases_df = pd.read_csv('data/phrases.csv', encoding='ISO-8859-1')
    phrases = phrases_df['Phrases'].tolist()

    phrase_vectors = [phrase_vector(phrase, model) for phrase in tqdm(phrases)]

    similarities = 1 - np.array([cosine(user_vector, vec) if np.any(vec) else 0 for vec in tqdm(phrase_vectors)])
    

    closest_match_index = np.argmax(similarities)
    
    closest_match = phrases[closest_match_index]
    similarity_score = similarities[closest_match_index]
    
    return closest_match, similarity_score


user_input_phrase = input("Give the input phrase:")

closest_match, similarity_score = find_closest_match(user_input_phrase)

print(f"User Input: {user_input_phrase}")
print(f"Closest Match: {closest_match}")
print(f"Similarity Score: {similarity_score}")
