import time
import numpy as np
import tensorflow as tf
from functions import *
from flask import Flask, render_template, url_for, request

# Enable Eager Execution
# tf.enable_eager_execution()
# tf.executing_eagerly()

# ------------------------------------- Build model -------------------------------------------

# Tạo word embeddings
words_list = np.load('data/words_list.npy')
words_list = words_list.tolist()
word_vectors = np.load('data/word_vectors.npy')
word_vectors = np.float32(word_vectors)

# Ma trận index của từ ánh xạ sang v
word2idx = {w: i for i, w in enumerate(words_list)}

# Các hyperparameters
LSTM_UNITS = 128
N_LAYERS = 2
NUM_CLASSES = 2
MAX_SEQ_LENGTH = 200

#Build model
model = SentimentAnalysisModel(word_vectors, LSTM_UNITS, N_LAYERS, NUM_CLASSES)

#Đưa trọng số vào model
model.load_weights(tf.train.latest_checkpoint('model'))

# ------------------------------------- Build Web app -------------------------------------------

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

    else:
        return f"<h1>Please enter your paragraph vietnamese in text box !</h1>"

    return render_template('result.html', prediction = my_prediction, sentence = sentence,
                           prob_pos = prob_pos, time2run = time2run)

if __name__ == '__main__':
    app.run(debug=True)