import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

from dataset import Dataset, Datasets

import os
import sys
import pickle


# load data
height = pickle.load(open('height.pkl', 'rb'))


# network parameters
input_dim = 1 # height data input
encoder_hidden_dim = 16
decoder_hidden_dim = 16
latent_dim = 2
lam = 0 # lambda


# define weight shape
weights = {
    "encoder_h1": tf.Variable(tf.zeros([input_dim, encoder_hidden_dim])),
    "encoder_mu": tf.Variable(tf.zeros([encoder_hidden_dim, latent_dim])),
    "encoder_logvar": tf.Variable(tf.zeros([encoder_hidden_dim, latent_dim])),
    "decoder_h1": tf.Variable(tf.zeros([latent_dim, decoder_hidden_dim])),
    "decoder_reconstruction": tf.Variable(tf.zeros([decoder_hidden_dim, input_dim]))
}


# define bias shape
biases = {
    "encoder_h1": tf.Variable(tf.zeros([encoder_hidden_dim])),
    "encoder_mu": tf.Variable(tf.zeros([latent_dim])),
    "encoder_logvar": tf.Variable(tf.zeros([latent_dim])),
    "decoder_h1": tf.Variable(tf.zeros([decoder_hidden_dim])),
    "decoder_reconstruction": tf.Variable(tf.zeros([input_dim]))
}


# encoder
## define input shape
x = tf.placeholder(tf.float32, [None, input_dim])

## encoder_h1
l2_loss = tf.constant(0.0) # l2 norm
l2_loss += tf.nn.l2_loss(weights["encoder_h1"])
hidden_encoder = tf.nn.relu(tf.matmul(x, weights["encoder_h1"]) + biases["encoder_h1"])

## encoder_mu
l2_loss += tf.nn.l2_loss(weights["encoder_mu"])
mu_encoder = tf.matmul(hidden_encoder, weights["encoder_mu"]) + biases["encoder_mu"]

## encoder_logvar
l2_loss += tf.nn.l2_loss(weights["encoder_logvar"])
logvar_encoder = tf.matmul(hidden_encoder, weights["encoder_logvar"]) + biases["encoder_logvar"]


# get latent representation over test set
batch_size = 50
train_iter = int(height.train.num_examples / batch_size)
test_iter = int(height.test.num_examples / batch_size)
train = np.zeros((train_iter * batch_size, latent_dim*2+1)) # [mu, logvar, class_label]
test = np.zeros((test_iter * batch_size, latent_dim*2+1)) # [mu, logvar, class_label]

with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state('out_models/')
    if ckpt:
        last_model = ckpt.model_checkpoint_path
        saver = tf.train.Saver()
        saver.restore(sess, last_model)
    else:
        print("No trained model was found.")
        sys.exit(0)

    for i in range(train_iter):
        batch = height.train.next_batch(batch_size)
        feed_dict = {x: batch[0]}
        mu, logvar = sess.run([mu_encoder, logvar_encoder], feed_dict=feed_dict)

        _train = np.concatenate((mu, logvar, batch[1]), axis=1)

        train[batch_size*i:batch_size*(i+1)] = _train

    for i in range(test_iter):
        batch = height.test.next_batch(batch_size)
        feed_dict = {x: batch[0]}
        mu, logvar = sess.run([mu_encoder, logvar_encoder], feed_dict=feed_dict)

        _test = np.concatenate((mu, logvar, batch[1]), axis=1)

        test[batch_size*i:batch_size*(i+1)] = _test


# classification
estimator = LinearSVC(C=100.0)
estimator.fit(train[:,:-1], train[:,-1].astype(np.int32))
predictions = estimator.predict(test[:,:-1])

# evaluation
print("# Evaluation")
print(" - Male: 0")
print(" - Female: 1")
print(classification_report(test[:,-1], predictions))

print('Done.')
