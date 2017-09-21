from flask import Flask
from flask import request
from flask import render_template
from image_features import *
from load_model import *
import json
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
metadata = json.load(open('resources/data_prepro.json', 'r'))
metadata['ix_to_word'] = {str(word):int(i) for i,word in metadata['ix_to_word'].items()}

def get_ques_vector(question):
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


app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template("website.html")


@app.route('/submit', methods=['POST','GET'])
def my_form_post():
    if request.method=='POST':
        image_url = request.form['image_url']
        question = request.form['question']
 
        img_features = image_features(image_url)
        ques_features = get_ques_vector(question)
        model = loadmodel()
        preds = model.predict([img_features, ques_features])[0]
        top_preds = preds.argsort()[-5:][::-1]
        top_pred = [(metadata['ix_to_ans'][str(_)].title(), round(preds[_]*100.0,2)) for _ in top_preds][0]

    return render_template("website.html", answer=[[question,image_url,top_pred]])


if __name__ == '__main__':
    app.run(debug=True)
