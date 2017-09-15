from flask import Flask
from flask import request
from flask import render_template
from image_features import *
from question_features import *
from load_model import *

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
        preds = model.predict([features, question_vector])[0]
        top_preds = pred.argsort()[-5:][::-1]
        top_pred = [(metadata['ix_to_ans'][str(_)].title(), round(pred[_]*100.0,2)) for _ in top_preds][0]

    return render_template("website.html", answer=[top_pred])


if __name__ == '__main__':
    app.run(debug=True)
