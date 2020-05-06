import re
import numpy as np
from tqdm import tqdm

strip_special_chars = re.compile("[^\w0-9 ]+")

def clean_sentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())


def get_sentence_indices(sentence, max_seq_length, _words_list, word2idx):
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


def text2ids(df, max_length, _word_list):
    """
    Biến đổi các text trong dataframe thành ma trận index

    Parameters
    ----------
    df: DataFrame
        dataframe chứa các text cần biến đổi
    max_length: int
        độ dài tối đa của một text
    _word_list: numpy.array
        array chứa các từ trong word vectors

    Returns
    -------
    numpy.array
        len(df) x max_length contains indices of text
    """
    ids = np.zeros((len(df), max_length), dtype='int32')
    for idx, text in enumerate(tqdm(df['text'])):
        ids[idx, :] = get_sentence_indices(clean_sentences(text), max_length, _word_list)
    return ids

