from gensim.models import KeyedVectors
from PyDictionary import PyDictionary

model = KeyedVectors.load("../models/word2vec.model", mmap='r')

def find_opposite_word(word):
    all_sims = model.most_similar(word, topn=len(filtered_word2vec.key_to_index))
    most_dissimilar_word = all_sims[-1][0]
    return most_dissimilar_word
