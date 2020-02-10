import re
import numpy as np
import tensorflow as tf
from pyvi import ViTokenizer, ViPosTagger
from flask import Flask, render_template, url_for, request

# Enable Eager Execution
tf.enable_eager_execution()
tf.executing_eagerly()


# ------------------------------------- Build model -------------------------------------------

# Tạo word embeddings
words_list = np.load('data/words_list.npy')
words_list = words_list.tolist()
word_vectors = np.load('data/word_vectors.npy')
word_vectors = np.float32(word_vectors)

# Ma trận index của từ ánh xạ sang v
word2idx = {w: i for i, w in enumerate(words_list)}


#Hàm clean_sent: chuẩn hóa lại câu văn.
strip_special_chars = re.compile("[^\w0-9]+")

def clean_sentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())


def get_sentence_indices(sentence, max_seq_length, _words_list):
    """
    Hàm này dùng để lấy index cho từng từ
    trong câu (không có dấu câu, có thể in hoa)
    Parameters
    ----------
    sentence là câu cần xử lý
    max_seq_length là giới hạn số từ tối đa trong câu
    _words_list là bản sao local của words_list, được truyền vào hàm
    """
    indices = np.zeros((max_seq_length), dtype='int32')

    # Tách câu thành từng tiếng
    words = [word.lower() for word in sentence.split()]

    # Lấy chỉ số của UNK
    unk_idx = word2idx['UNK']

    for idx, word in enumerate(words):
        if idx < max_seq_length:
            if (word in _words_list):
                word_idx = word2idx[word]
            else:
                word_idx = word2idx['UNK']

            indices[idx] = word_idx
        else:
            break

    return indices


class SentimentAnalysisModel(tf.keras.Model):
    """
    Mô hình phân tích cảm xúc của câu

    Properties
    ----------
    word2vec: numpy.array
        word vectors
    lstm_layers: list
        list of lstm layers, lstm cuối cùng sẽ chỉ trả về output của lstm cuối cùng
    dropout_layers: list
        list of dropout layers
    dense_layer: Keras Dense Layer
        lớp dense layer cuối cùng nhận input từ lstm,
        đưa ra output bằng số lượng class thông qua hàm softmax
    """

    def __init__(self, word2vec, lstm_units, n_layers, num_classes, dropout_rate=0.25):
        """
        Khởi tạo mô hình

        Paramters
        ---------
        word2vec: numpy.array
            word vectors
        lstm_units: int
            số đơn vị lstm
        n_layers: int
            số layer lstm xếp chồng lên nhau
        num_classes: int
            số class đầu ra
        dropout_rate: float
            tỉ lệ dropout giữa các lớp
        """
        super().__init__(name='sentiment_analysis')

        # Khởi tạo các đặc tính của model
        self.word2vec = word2vec

        self.lstm_layers = []  # List chứa các tầng LSTM
        self.dropout_layers = []  # List chứa các tầng dropout

        # Khởi tạo các layer
        for i in range(n_layers):
            new_lstm = tf.keras.layers.LSTM(units=lstm_units, return_sequences=True)
            self.lstm_layers.append(new_lstm)
            new_dropout = tf.keras.layers.Dropout(rate=dropout_rate)
            self.dropout_layers.append(new_dropout)

        # Tầng cuối cùng
        new_lstm = tf.keras.layers.LSTM(units=lstm_units, return_sequences=False)
        self.lstm_layers.append(new_lstm)

        self.dense_layer = tf.keras.layers.Dense(num_classes, activation="softmax")


    def call(self, inputs):
        # Thực hiện các bước biến đổi khi truyền thuận input qua mạng

        inputs = tf.cast(inputs, tf.int32)
        # Input hiện là indices, cần chuyển sang dạng vector
        # sử dụng:
        # tf.nn.embeddings_lookup(embeddings, indices)

        x = tf.nn.embedding_lookup(self.word2vec, inputs)

        # Truyền thuận inputs lần lượt qua các tầng
        # ở mỗi tầng, truyền input qua các layer: lstm > dropout
        n_layers = len(self.dropout_layers)

        for i in range(n_layers):
            x = self.lstm_layers[i](x)
            x = self.dropout_layers[i](x)

        x = self.lstm_layers[-1](x)

        x = self.dense_layer(x)

        # Gán giá trị tầng cuối cùng vào out và trả về
        out = x

        return out


# Các hyperparameters
LSTM_UNITS = 128
N_LAYERS = 2
NUM_CLASSES = 2
MAX_SEQ_LENGTH = 200

#Build model
model = SentimentAnalysisModel(word_vectors, LSTM_UNITS, N_LAYERS, NUM_CLASSES)

#Đưa trọng số vào model
model.load_weights(tf.train.latest_checkpoint('model'))


def predict(sentence, model, _word_list=words_list, _max_seq_length=MAX_SEQ_LENGTH):
    """
    Dự đoán cảm xúc của một câu

    Parameters
    ----------
    sentence: str
        câu cần dự đoán
    model: model keras
        model keras đã được train/ load trọng số vừa train
    _word_list: numpy.array
        danh sách các từ đã biết
    _max_seq_length: int
        giới hạn số từ tối đa trong mỗi câu

    Returns
    -------
    int
        0 nếu là negative, 1 nếu là positive
    """

    # Tokenize/Tách từ trong câu
    tokenized_sent = ViTokenizer.tokenize(sentence)

    # Đưa câu đã tokenize về dạng input_data thích hợp để truyền vào model
    tokenized_sent = clean_sentences(tokenized_sent)
    input_data = get_sentence_indices(tokenized_sent, _max_seq_length, _word_list)

    input_data = input_data.reshape(-1, _max_seq_length)

    # Truyền input_data qua model để nhận về xác suất các nhãn
    # Chọn nhãn có xác suất cao nhất và return
    pred = model(input_data)
    predictions = tf.argmax(pred, 1).numpy().astype(np.int32)

    return predictions



# ------------------------------------ Build Web app -------------------------------------------

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/sva')
def sva():
    return render_template('sva.html')

@app.route('/result', methods=['GET','POST'])
def result():
    if request.method == 'POST':
        sentence = request.form['inputSentence']
        my_prediction = predict(sentence, model)

    return render_template('result.html', prediction = my_prediction)

if __name__ == '__main__':
    app.run(debug=True)