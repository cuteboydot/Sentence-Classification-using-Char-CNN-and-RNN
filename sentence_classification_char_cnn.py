from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import rnn
import time
import datetime
import os

np.random.seed(0)
print("PROGRAM START !!!")


'''''''''''''''''''''''''''''''''''''''''''''
READING & PARSING DATA
'''''''''''''''''''''''''''''''''''''''''''''
csvfile = "./csv/sentence_full.tsv"
list_data = []
list_len = []
total = ""

with open(csvfile, 'r', encoding="utf8") as tsv:
    for line in tsv:
        sep = line.split("\t")

        sentence_class = int(sep[0].replace("\ufeff", ""))
        sentence_english = sep[1].lower()

        total += sentence_english
        list_data.append([sentence_english, sentence_class])

dic = list(set(total))  # id -> char
dic.insert(0, "P")  # P:PAD symbol(0)
rdic = {w: i for i, w in enumerate(dic)}

print("RDIC")
print(rdic)

VOCAB_SIZE = len(dic)
print("VOCABULARY SIZE : %d" % VOCAB_SIZE)

CLASS_SIZE = len(set([c[1] for c in list_data]))
print("CLASS SIZE : %d" % CLASS_SIZE)

list_data.sort(key=lambda s: len(s[0]))
MAX_LEN = len(list_data[-1][0]) + 1
print("SENTENCE MAX LEN : %d" % MAX_LEN)

list_data = [[[rdic[char] for char in data[0]], data[1]] for data in list_data]

random.shuffle(list_data, random.random)

list_data_test = list_data[:200]
list_data = list_data[200:]
print("TOTAL TRAIN DATASET : %d" % (len(list_data)))
print("TOTAL TEST DATASET : %d" % (len(list_data_test)))


'''''''''''''''''''''''''''''''''''''''''''''
GENERATING BATCH
'''''''''''''''''''''''''''''''''''''''''''''
def generate_batch(size):
    assert size <= len(list_data)

    data_x = np.zeros((size, MAX_LEN), dtype=np.int)
    data_y = np.zeros((size, CLASS_SIZE), dtype=np.int)

    index = np.random.choice(range(len(list_data)), size, replace=False)
    for a in range(len(index)):
        idx = index[a]

        x = list_data[idx][0]
        x = x[:MAX_LEN - 1] + [0] * max(MAX_LEN - len(x), 1)
        y = list_data[idx][1]
        y = np.eye(CLASS_SIZE)[y]

        data_x[a] = x
        data_y[a] = y

    return data_x, data_y


def generate_batch_test(size):
    assert size <= len(list_data_test)

    data_x = np.zeros((size, MAX_LEN), dtype=np.int)
    data_y = np.zeros((size, CLASS_SIZE), dtype=np.int)

    index = np.random.choice(range(len(list_data_test)), size, replace=False)
    for a in range(len(index)):
        idx = index[a]

        x = list_data_test[idx][0]
        x = x[:MAX_LEN - 1] + [0] * max(MAX_LEN - len(x), 1)
        y = list_data_test[idx][1]
        y = np.eye(CLASS_SIZE)[y]

        data_x[a] = x
        data_y[a] = y

    return data_x, data_y


'''''''''''''''''''''''''''''''''''''''''''''
MAKE TENSORFLOW GRAPH
'''''''''''''''''''''''''''''''''''''''''''''
with tf.Graph().as_default():

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    sess = tf.Session(config=config)
    with sess.as_default():

        NUM_EPOCH = 351
        BATCH_SIZE = 30
        EMBED_DIM = 100
        CNN_NUM_FILTER = 20
        CNN_SEQ_FEAT_DIM = 6

        '''''''''''''''''''''''''''''''''''''''''''''
        DEFINE VARIABLE AND OPs
        '''''''''''''''''''''''''''''''''''''''''''''
        with tf.device("/cpu:0"):
            # Input parameter
            X = tf.placeholder(tf.int32, [None, MAX_LEN], name="X")
            Y = tf.placeholder(tf.int32, [None, CLASS_SIZE], name="Y")
            dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

            # Char embeddings
            with tf.name_scope("char_embedding"):
                embeddings = tf.get_variable("embeddings", [VOCAB_SIZE, EMBED_DIM])
                embed_X = tf.nn.embedding_lookup(embeddings, X)
                embed_X = tf.expand_dims(embed_X, -1)

            # CNN
            filter_sizes = [3, 6, 9]
            pooled_outputs = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, EMBED_DIM, 1, CNN_NUM_FILTER]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[CNN_NUM_FILTER]), name="b")
                    conv = tf.nn.conv2d(
                        embed_X,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")

                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    print("h.shape = %s" % (h.get_shape()))

                    # Max-pooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, MAX_LEN - filter_size + 2 - CNN_SEQ_FEAT_DIM, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="pool")
                    pooled_outputs.append(pooled)
                    print("pooled.shape = %s" % (pooled.get_shape()))

            # Combine all the pooled features
            with tf.name_scope("reshape_flat"):
                num_filters_total = CNN_NUM_FILTER * CNN_SEQ_FEAT_DIM * len(filter_sizes)
                h_pool = tf.concat(pooled_outputs, axis=3)
                h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
                print("h_pool_flat.shape = %s" % (h_pool_flat.get_shape()))

            # Add dropout
            with tf.name_scope("dropout"):
                h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)
                print("h_drop.shape = %s" % (h_drop.get_shape()))

            with tf.name_scope("output_dense_layer"):
                W = tf.Variable(tf.truncated_normal([num_filters_total, CLASS_SIZE], stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[CLASS_SIZE]), name="b")
                scores = tf.nn.xw_plus_b(h_drop, W, b, name="scores")
                predictions = tf.argmax(scores, 1, name="predictions")

            # Calculate mean cross-entropy loss
            with tf.name_scope("l2_loss"):
                #losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=Y)
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=Y) \
                         + 0.01 * tf.nn.l2_loss(W) + 0.01 * tf.nn.l2_loss(b)
                loss = tf.reduce_mean(losses, name="loss")

            # Calculate Accuracy
            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(predictions, tf.argmax(Y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

            # Train optimizer
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-4)
            grads_and_vars = optimizer.compute_gradients(loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        '''''''''''''''''''''''''''''''''''''''''''''
        CHECK POINT & SUMMARY
        '''''''''''''''''''''''''''''''''''''''''''''
        # Output directory for models and summaries
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "summary", timestamp))
        print("LOGDIR = %s" % out_dir)

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", loss)
        acc_summary = tf.summary.scalar("accuracy", accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary])
        #train_summary_op = tf.summary.merge_all()
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Test summaries
        test_summary_op = tf.summary.merge([loss_summary, acc_summary])
        #test_summary_op = tf.summary.merge_all()
        test_summary_dir = os.path.join(out_dir, "summaries", "test")
        test_summary_writer = tf.summary.FileWriter(test_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)


        '''''''''''''''''''''''''''''''''''''''''''''
        TRAIN
        '''''''''''''''''''''''''''''''''''''''''''''
        sess.run(tf.global_variables_initializer())

        def train(train_x, train_y, writer=False):
            feed_dict = {
                X: train_x,
                Y: train_y,
                dropout_keep_prob: 0.7
            }
            _, g_step, summaries, train_loss, train_acc = sess.run(
                [train_op, global_step, train_summary_op, loss, accuracy],
                feed_dict)

            if writer:
                train_summary_writer.add_summary(summaries, g_step)

            return train_loss, train_acc, g_step

        def test(test_x, test_y, writer=False):
            feed_dict = {
                X: test_x,
                Y: test_y,
                dropout_keep_prob: 1.0
            }
            _, g_step, summaries, test_loss, test_acc = sess.run(
                [train_op, global_step, test_summary_op, loss, accuracy],
                feed_dict)

            if writer:
                test_summary_writer.add_summary(summaries, g_step)

            return test_loss, test_acc, g_step

        def predict(pred_x):
            feed_dict = {
                X: pred_x,
                dropout_keep_prob: 1.0
            }
            pred_y = sess.run([predictions], feed_dict)
            return pred_y

        epoch = -1
        epoch_tmp = -1
        step = 0
        while(True):
            batch_x, batch_y = generate_batch(BATCH_SIZE)
            _1, _2, g_step = train(batch_x, batch_y, False)

            step += 1
            epoch_tmp = int((BATCH_SIZE * step) / len(list_data))

            if (epoch_tmp != epoch) and (epoch_tmp % 10 == 0):
                epoch = epoch_tmp

                batch_x, batch_y = generate_batch(len(list_data))
                train_loss, train_acc, _3 = train(batch_x, batch_y, train_summary_writer)
                time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print("%s:[TRAIN] epoch[%04d], glob-step[%06d], loss=%.4f, acc=%.3f" % (time_str, epoch, g_step, train_loss, train_acc))

                batch_x, batch_y = generate_batch_test(len(list_data_test))
                test_loss, test_acc, _3 = test(batch_x, batch_y, test_summary_writer)
                time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print("%s: [TEST] epoch[%04d], glob-step[%06d], loss=%.4f, acc=%.3f" % (time_str, epoch, g_step, test_loss, test_acc))

                current_step = tf.train.global_step(sess, global_step)
                saver.save(sess, checkpoint_prefix, global_step=current_step)

            if epoch >= NUM_EPOCH:
                print("TRAIN COMPLETED !!!")
                break

        '''''''''''''''''''''''''''''''''''''''''''''
        PREDICTION TEST
        '''''''''''''''''''''''''''''''''''''''''''''
        batch_x, batch_y = generate_batch(BATCH_SIZE)
        label_y = np.argmax(batch_y, axis=1)
        pred_y = predict(batch_x)
        pred_y = pred_y[0]
        for a in range(len(label_y)):
            ox = "X"
            if label_y[a] == pred_y[a]:
                ox = "O"
            print("TEST[%03d] label_y=%02d, pred_y=%02d => %s" % (a, label_y[a], pred_y[a], ox))

print("PROGRAM END!!")

