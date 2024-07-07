import gensim.downloader
from gensim.models import KeyedVectors
import json

# Load English dictionary words
with open('../data/words_dictionary.json', 'r') as file:
    words_dict = json.load(file)

words_list = list(words_dict.keys())


# Load word2vec model
word2vec = gensim.downloader.load('glove-wiki-gigaword-300')

# Filter the Word2Vec model for english words
filtered_vectors = {word: word2vec[word] for word in words_list if word in word2vec}

# Create a new KeyedVectors instance with the filtered vectors
filtered_word2vec = KeyedVectors(vector_size=word2vec.vector_size)
filtered_word2vec.add_vectors(list(filtered_vectors.keys()), list(filtered_vectors.values()))

# Save the filtered model for future use
filtered_word2vec.save("../models/word2vec.model")