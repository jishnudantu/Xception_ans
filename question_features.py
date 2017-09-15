import json
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
def get_ques_vector(question):
    metadata = json.load(open('resources/data_prepro.json', 'r'))
    metadata['ix_to_word'] = {str(word):int(i) for i,word in metadata['ix_to_word'].items()}
    question_vector = []
    seq_length = 26
    word_index = metadata['ix_to_word']
    for word in word_tokenize(question.lower()):
        if word in word_index:
            question_vector.append(word_index[word])
        else:
            question_vector.append(0)
    question_vector = np.array(pad_sequences([question_vector], maxlen=seq_length))[0]
    question_vector = question_vector.reshape((1,seq_length))
    return question_vector
