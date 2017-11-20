import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import rnn, data_helper
import pandas as pd
import matplotlib.pyplot as plt

""""
author: Wen Cui
Date: Nov 11, 2017
"""

def train_graph(sess, textRNN, x_train, y_train, batch_size, dropout_prob, trainMode=True):
    """
    Train model on training set
    """
    loss_list, acc_list = [], []
    loss, acc, ct = 0, 0, 0

    batches = data_helper.batch_iter(list(zip(x_train, y_train)), batch_size)
    for batch in batches:
        x_batch, y_batch = zip(*batch)
        feed = {
            textRNN.x: x_batch,
            textRNN.y: y_batch,
            textRNN.dropout_keep_prob: dropout_prob
        }
        if trainMode:
            sess.run([textRNN.train_step], feed_dict=feed)
        curr_loss, curr_acc = sess.run([textRNN.loss, textRNN.accuracy], feed_dict=feed)


        # Contain loss/acc per batch
        loss_list.append(curr_loss)
        acc_list.append(curr_acc)
        # Contain loss/acc per epoch
        loss, acc, ct = loss + curr_loss, acc + curr_acc, ct + 1

    return loss/float(ct), acc/float(ct), loss_list, acc_list


if __name__ == '__main__':
    # Data loading parameters
    np.random.seed(10)
    input_file = '../data/train.csv'
    test_file = '../data/test.csv'
    pretrained_model = '../data/glove.twitter.27B.100d.txt'
    SPLIT = 0.2

    # Model parameters
    PRETRAIN = False
    BIDIRECTIONAL = True
    NUM_CLASSES = 2
    BATCH_SIZE = 128
    DROPOUT_KEEP_PROB = 0.8  # 0.5
    NUM_EPOCHS = 3
    # For optimizer
    LEARNING_RATE = 0.001
    DECAY_STEPS = 1200
    DECAY_RATE = 0.9


    # Get embedding_dim and embedding_index
    if PRETRAIN:
        pretrained_vocb, pretrained_embedding = rnn.loadGlove(pretrained_model)
        EMBEDDING_DIM = len(pretrained_embedding[0])
    else:
        EMBEDDING_DIM = 200
        pretrained_embedding = None


    print('loading data...')
    x, y = data_helper.load_tweets_label(input_file)

    # Building vocabulary
    max_doc_length = max([len(sent.split(' ')) for sent in x])
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_doc_length)
    if PRETRAIN:
        pretrain = vocab_processor.fit(pretrained_vocb)
        x = np.array(list(vocab_processor.transform(x)))
        vocab_size = len(pretrained_embedding)
    else:
        x = np.array(list(vocab_processor.fit_transform(x)))
        vocab_size = len(vocab_processor.vocabulary_)
    print('Vocabulary size: ', vocab_size)
    print('Embedding size: ', EMBEDDING_DIM)
    print('Bidirectinoal Model: ', BIDIRECTIONAL)


    # Split for train and evaluation
    x_train, x_val, y_train, y_val = train_test_split(x, y, stratify=y, test_size=SPLIT, random_state=10)


    # Training
    print('Start training...\n')
    val_loss, val_acc = 10, 1

    rnn.reset_graph()
    with tf.Session() as sess:
        textRNN = rnn.TextRNN(NUM_CLASSES, max_doc_length, LEARNING_RATE, DECAY_STEPS, DECAY_RATE, EMBEDDING_DIM, vocab_size, PRETRAIN, pretrained_embedding, BIDIRECTIONAL)
        sess.run(tf.global_variables_initializer())
        current_epoch = 1
        while current_epoch <= NUM_EPOCHS:
            curr_train_loss, curr_train_acc, train_loss_list, train_acc_list = train_graph(sess, textRNN, x_train, y_train, BATCH_SIZE, DROPOUT_KEEP_PROB)
            _,_,val_loss_list, val_acc_list = train_graph(sess, textRNN, x_val, y_val, BATCH_SIZE, DROPOUT_KEEP_PROB, trainMode=False)
            # TODO: early stop
            # print("Epoch %d\tTrain Loss: %.3f\tTrain Accuracy: %.3f\tValidation Loss: %.3f\tValidation Accuracy: %.3f"
            #       % (current_epoch, curr_train_loss, curr_train_acc, curr_val_loss, curr_val_acc))
            print("Epoch %d\tTrain Loss: %.3f\tTrain Accuracy: %.3f"
                  % (current_epoch, curr_train_loss, curr_train_acc))

            ax = plt.subplot(2, 2, 1)
            ax.set_xlabel('batch numbers')
            ax.set_ylabel('training loss')
            ax.plot(train_loss_list, label='epoch%s' %(current_epoch))
            ax.legend(loc='upper right', markerscale=0.2)
            # ax.set_title('self-learn embedding 200 dimension')


            ax = plt.subplot(2, 2, 2)
            ax.set_xlabel('batch numbers')
            ax.set_ylabel('training accuracy')
            ax.plot(train_acc_list, label='epoch%s' % current_epoch)
            ax.legend(loc='upper right', markerscale=0.2)

            ax = plt.subplot(2, 2, 3)
            ax.set_xlabel('batch numbers')
            ax.set_ylabel('validation loss')
            ax.plot(val_loss_list, label='epoch%s' % current_epoch)
            ax.legend(loc='upper right', markerscale=0.2)

            ax = plt.subplot(2, 2, 4)
            ax.set_xlabel('batch numbers')
            ax.set_ylabel('validation accuracy')
            ax.plot(val_acc_list, label='epoch%s' % current_epoch)
            ax.legend(loc='upper right',  markerscale=0.2)

            current_epoch += 1

        print('End of training.')

        plt.show()
        # Run final model on test dataset
        # x_test_text = data_helper.load_tweets_label(test_file, trainMode=False)
        # x_test = np.array(list(vocab_processor.transform(x_test_text)))
        # batches = data_helper.batch_iter(x_test, BATCH_SIZE, shuffle=False)
        # predictions = []
        # for batch in batches:
        #     feed = {
        #         textRNN.x: batch,
        #         textRNN.dropout_keep_prob: DROPOUT_KEEP_PROB
        #     }
        #     prediction = sess.run([textRNN.predictions], feed_dict=feed)[0].tolist()
        #     predictions += prediction
        # # Save to file
        # df = pd.DataFrame(data=predictions, columns=['realDonaldTrump', 'HillaryClinton'])
        # df.index.name = 'id'
        # df.to_csv("output.csv")



