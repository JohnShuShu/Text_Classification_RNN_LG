import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

""""
author: Wen Cui
Date: Nov 11, 2017
"""

def load_tweets_label(file_name, trainMode=True):
    df = pd.read_csv(file_name, encoding='utf-8')
    texts = df['tweet'].values
    labels = df['handle'].values
    texts = np.array([text.lower() for text in texts])
    labels = np.array(labels)
    if trainMode:
        return texts, encode_label(labels)
    else:
        return texts


#TODO: smart way for labelling
def encode_label(labels):
    DT = [1, 0]
    HC = [0, 1]
    return [DT if l == 'realDonaldTrump' else HC for l in labels]



def batch_iter(data, batch_size, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches = int((len(data)-1)/batch_size) + 1

    # Shuffle the data everytime
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
    else:
        shuffled_data = data
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield shuffled_data[start_index:end_index]




if __name__ == '__main__':
    input_file = '../data/train.csv'
    test_file = '../data/test.csv'
    SPLIT = 0.2
    BATCH_SIZE = 256
    NUM_EPOCHS = 10

    x, y = load_tweets_label(input_file)

    # Building vocabulary
    # TODO: how to handle url, this is not adding url into x
    max_doc_length = max([len(sent.split(' ')) for sent in x])
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_doc_length)
    vocab_size = len(vocab_processor.vocabulary_)
    x = np.array(list(vocab_processor.fit_transform(x)))
    vocab_size = len(vocab_processor.vocabulary_)
    print('Vocabulary size: ', vocab_size)

    # Split for train and evaluation
    # TODO: cross-validation
    x_train, x_val, y_train, y_val = train_test_split(x, y, stratify=y, test_size=SPLIT, random_state=10)
    print('Train/val split:{0}/{1}'.format(len(x_train), len(y_train)))

    batches = batch_iter(list(zip(x_train, y_train)), BATCH_SIZE, NUM_EPOCHS)
    for batch in batches:
        x_batch, y_batch = zip(*batch)
        print('x_batch ', x_batch)
        print('y_batch ', y_batch)
