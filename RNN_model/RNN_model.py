import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, model_from_json
from keras.layers import LSTM, Dense, Embedding, Dropout
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import random, pickle, os
from collections import OrderedDict

seed = 100
random.seed(seed)

WORD_INDEX = {}
EMBEDDING_DIM = 30
MAX_SEQUENCE_LENGTH = 30
BATCH_SIZE = 32

def load_data(file_name, testMode=False):
    print('loading data...')
    df = pd.read_csv(file_name, encoding='utf-8')
    texts = df['tweet'].values
    labels = df['handle'].values
    texts = np.array([text.lower() for text in texts])
    if testMode:
        return texts
    else:
        return texts, encode_label(labels)


def encode_label(labels):
    y = np.zeros(len(labels))
    for i, l in np.ndenumerate(labels):
        if l == 'realDonaldTrump':
            y[i] = 1
        else:
            y[i] = 0
    return y


def vectorize_text(texts, tokenizer=None):
    """
    Turn texts into 2D 
    :param texts: 
    :param tokenizer: 
    :return: 
    """
    global WORD_INDEX
    if tokenizer is None:
        tokenizer = Tokenizer()
        # OR restrict to
        #tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
        tokenizer.fit_on_texts(texts)

    sequences = tokenizer.texts_to_sequences(texts)
    WORD_INDEX = tokenizer.word_index
    vec_data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='pre', truncating='pre')
    with open('tokenizer.pickle', 'wb') as pk:
        pickle.dump(tokenizer, pk, protocol=pickle.HIGHEST_PROTOCOL)
    return vec_data, tokenizer


# TODO: load pretrained langugage model, w2v or Glove
def embedding_matrix(WORD_INDEX):
    print('Building embedding layer...')
    word_embedding = np.zeros((len(WORD_INDEX) + 1, EMBEDDING_DIM))
    for word, idx in WORD_INDEX.items():
        word_embedding[idx] = np.random.uniform(-0.25, 0.25, EMBEDDING_DIM)
    embedding_layer = Embedding(len(WORD_INDEX) + 1,
                                EMBEDDING_DIM,
                                weights=[word_embedding],
                                input_length=MAX_SEQUENCE_LENGTH)
    return embedding_layer


def create_model():
    print('Building model...')
    model = Sequential()
    embedding_layer = embedding_matrix(WORD_INDEX)
    model.add(embedding_layer)
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model



def train_save_model(input_file, out_model_file, out_weight_file):
    texts, labels = load_data(input_file)
    x_train, x_val, y_train, y_val = train_test_split(texts, labels, stratify=labels, test_size=0.2,
                                                       random_state=10)
    x_train, tokenzier = vectorize_text(x_train)
    x_val, _ = vectorize_text(x_val, tokenizer=tokenzier)

    print('Training...')
    model = create_model()
    model.fit(x_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=20,
              validation_data=(x_val, y_val))
    score, acc = model.evaluate(x_val, y_val,
                                batch_size=BATCH_SIZE)
    print('Test score:', score)
    print('Test accuracy:', acc)

    print('Saving model...')
    model_json = model.to_json()
    with open(out_model_file, 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(out_weight_file)
    print('Model saved.')


def predict(saved_model, saved_weights, test_file):
    json_file = open(saved_model, 'r')
    loaded_model = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model)
    model.load_weights(saved_weights)
    with open('tokenizer.pickle', 'rb') as pk:
        tokenizer = pickle.load(pk)
    print('Loaded model from disk.')
    x_test = load_data(test_file, testMode=True)
    x_test, _ = vectorize_text(x_test, tokenizer)

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    # model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    y_pred = model.predict(x_test, batch_size=BATCH_SIZE)
    return y_pred


if __name__ == '__main__':
    input_file = '../data/train.csv'
    out_model_file = './rnnClassifier.json'
    out_weight_file = './rnnWeight.json'
    # Only train once
    if not os.path.isfile(out_model_file):
        train_save_model(input_file, out_model_file, out_weight_file)
    # Predict test file
    test_file = '../data/test.csv'
    predictions_DT = predict(out_model_file, out_weight_file, test_file).ravel()
    predictions_HC = np.ones(predictions_DT.shape) - predictions_DT
    # write result
    data = OrderedDict()
    data['id'] = [i for i in range(0, len(predictions_DT))]
    data['realDonaldTrump'] = predictions_DT
    data['HillaryClinton'] = predictions_HC
    df = pd.DataFrame(data)
    df.to_csv('../result/rnn_result.csv', index=False)
