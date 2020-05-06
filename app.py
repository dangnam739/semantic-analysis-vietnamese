import os
import time
import tensorflow as tf
from model import *
from flask import Flask, render_template, url_for, request
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
from load_data import word2idx, words_list, word_vectors


# Enable Eager Execution
tf.enable_eager_execution()
tf.executing_eagerly()

# ------------------------------------- Build model -------------------------------------------
# Các hyperparameters
LSTM_UNITS = 128
N_LAYERS = 2
NUM_CLASSES = 2
MAX_SEQ_LENGTH = 200

#Build model
model = SentimentAnalysisModel(word_vectors, LSTM_UNITS, N_LAYERS, NUM_CLASSES)

#Đưa trọng số vào model
model.load_weights(tf.train.latest_checkpoint('model'))

# ------------------------------------- Connect to database ------------------------------------
os.environ["DATABASE_URL"] = "postgresql://postgres:123456@127.0.0.1:5432/sav"

engine = create_engine(os.getenv("DATABASE_URL"))
db = scoped_session(sessionmaker(bind=engine))

# ------------------------------------- Build Web app ------------------------------------------

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/sav')
def sav():
    return render_template('sav.html')

@app.route('/result', methods=['GET','POST'])
def result():

    if request.method == 'POST':
        sentence = request.form['inputSentence']
        start = time.time()
        my_prediction, prob_pos = predict(sentence, model, words_list, MAX_SEQ_LENGTH, word2idx)
        end = time.time()
        time2run = end - start

        #Insert to table
        db.execute("INSERT INTO labeled_paragraph (content, label) VALUES (:content, :label)",
                   {"content": sentence, "label": int(my_prediction)})
        db.commit()

    else:
        return f"<h1>Please enter your paragraph vietnamese in text box !</h1>"

    return render_template('result.html', prediction = my_prediction, sentence = sentence,
                           prob_pos = prob_pos, time2run = time2run)

if __name__ == '__main__':
    app.run(debug=True)