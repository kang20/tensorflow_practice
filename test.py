
import json
import tensorflow as tf
import numpy as np
import urllib

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Flatten,Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint


def solution_model():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/sarcasm.json'
    urllib.request.urlretrieve(url, 'sarcasm.json')
    with open('sarcasm.json') as f:
        datas = json.load(f)

    # DO NOT CHANGE THIS CODE OR THE TESTS MAY NOT WORK
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type='post'
    padding_type='post'
    oov_tok = "<OOV>"
    training_size = 20000

    sentences = []
    labels = []
    # YOUR CODE HERE
    for data in datas:
        sentences.append(data['headline'])
        labels.append(data['is_sarcastic'])
    training_size = 20000
    train_sentences = sentences[:training_size]
    train_labels = labels[:training_size]
    validation_sentences = sentences[training_size:]
    validation_labels = labels[training_size:]
    vocab_size = 1000
    oov_tok = "<OOV>"
    tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
    tokenizer.fit_on_texts(train_sentences)
    train_sequences = tokenizer.texts_to_sequences(train_sentences)
    validation_sequences = tokenizer.texts_to_sequences(validation_sentences)
    train_labels = np.array(train_labels)
    validation_labels = np.array(validation_labels)
    x = Embedding(vocab_size, embedding_dim, input_length=max_length)

    embedding_dim = 16

    # 한 문장의 최대 단어 숫자
    max_length = 120

    # 잘라낼 문장의 위치
    trunc_type='post'

    # 채워줄 문장의 위치
    padding_type='post'
    train_padded = pad_sequences(train_sequences, maxlen=max_length, truncating=trunc_type, padding=padding_type)
    validation_padded = pad_sequences(validation_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_length),

        Bidirectional(LSTM(64, return_sequences=True)),
        Bidirectional(LSTM(64)),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid'),
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

    checkpoint_path = 'my_checkpoint.ckpt'
    checkpoint = ModelCheckpoint(checkpoint_path,
                                save_weights_only=True,
                                save_best_only=True,
                                monitor='val_loss',
                                verbose=1)
    history = model.fit(train_padded, train_labels,
                    validation_data=(validation_padded, validation_labels),
                    callbacks=[checkpoint],
                        batch_size = 8,
                    epochs=12)
    model.load_weights(checkpoint_path)
    model.evaluate(validation_padded, validation_labels)
    return model


# Note that you'll need to save your model as a .h5 like this
# This .h5 will be uploaded to the testing infrastructure
# and a score will be returned to you
if __name__ == '__main__':
    model = solution_model()
    model.save("TF4-sarcasm.h5")
