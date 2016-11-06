import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from dataset import Dataset, Datasets

import os
import pickle


# load data
height = pickle.load(open('height.pkl', 'rb'))

# parameters
n_epoch = 20
batch_size = 50
display_step = 1

# network parameters
input_dim = 1 # height data input
encoder_hidden_dim = 64
decoder_hidden_dim = 64
latent_dim = 16
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


# sampling
## sample epsilon
epsilon = tf.random_normal(tf.shape(logvar_encoder), name='epsilon')

## sample latent variable
std_encoder = tf.exp(0.5 * logvar_encoder)
z = mu_encoder + tf.mul(std_encoder, epsilon)


# decoder
## decoder_h1
l2_loss += tf.nn.l2_loss(weights["decoder_h1"])
hidden_decoder = tf.nn.relu(tf.matmul(z, weights["decoder_h1"]) + biases["decoder_h1"])

## decoder_reconstruction
l2_loss += tf.nn.l2_loss(weights["decoder_reconstruction"])
x_hat = tf.matmul(hidden_decoder, weights["decoder_reconstruction"]) + biases["decoder_reconstruction"]

# calculate loss
kl_divergence = -0.5 * tf.reduce_sum(1 + logvar_encoder - tf.pow(mu_encoder, 2) - tf.exp(logvar_encoder), reduction_indices=1)
bce = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(x_hat, x), reduction_indices=1)
loss = tf.reduce_mean(kl_divergence + bce)
regularized_loss = loss + lam * l2_loss

# optimization
train_step = tf.train.AdamOptimizer(0.01).minimize(regularized_loss)

# logging
loss_summary = tf.scalar_summary("lower_bound", loss)
summary_op = tf.merge_all_summaries()
saver = tf.train.Saver()


# training
n_epoch = 1000
batch_size = 50
display_step = 1

with tf.Session() as sess:
    summary_writer = tf.train.SummaryWriter('experiment', graph=sess.graph)

    if os.path.isfile("out_models/model.ckpt"):
        saver.restore(sess, "out_models/model.ckpt")
    else:
        sess.run(tf.initialize_all_variables())

    for epoch in range(1, n_epoch+1):
        batch = height.train.next_batch(batch_size)
        feed_dict = {x: batch[0]}
        _, current_loss, summary_str = sess.run([train_step, loss, summary_op], feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, epoch)

        print("# epoch{0:2d}: loss={1:.2f}".format(epoch, current_loss))

    save_path = saver.save(sess, "out_models/model.ckpt")
