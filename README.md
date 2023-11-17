# PhraseSimVec
A Python application for computing semantic similarities between phrases using Word2Vec embeddings.
1) Create a folder data
2) Download the GoogleNews-vectors-negative300.bin file from the google drive link 'https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM' and put the file in the data folder.
3) Put phrases.csv in the data folder, then use the command python pipline.py to run the pipeline file which loads the word embeddings and saves then in vectors.csv file in the data folder.
4) Then run the command python process_phrases.py for generating phrase similarities as required for the task. The csv file has already been generated, please refer to it in the submissions folder.
5) Run main.py file (it takes time to load the vectors.csv so please be patient) to find the closest match to phrases given a custom input.
