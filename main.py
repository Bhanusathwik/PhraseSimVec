import logging
from wordvec_app.model import WordVecModel
from wordvec_app.similarity import SimilarityCalculator
from wordvec_app.utils import phrase_vector, load_phrases, find_closest_match

def main():
    logging.basicConfig(level=logging.INFO)

    try:
        model = WordVecModel('data/vectors.csv')
        calculator = SimilarityCalculator(model)

        phrases = load_phrases('data/phrases.csv')

        phrase_vectors = [calculator.model.get_vector(phrase) for phrase in phrases]

        while True:
            input_phrase = input("Enter a phrase (or type 'exit' to quit): ")
            if input_phrase.lower() == 'exit':
                break
            
            input_vec = phrase_vector(input_phrase, model)

            # Find and display the closest match
            closest_match, distance = find_closest_match(input_vec, phrase_vectors, phrases)
            print(f"Closest match: {closest_match}, Distance: {distance}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
