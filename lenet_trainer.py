import tensorflow as tf
import pickle
import numpy as np
import math
import random

from pdb import set_trace as st
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

class LeNetTrainer():
    def preprocess_train(self, raw_X, raw_y):
        print('Preprocessing X,y train...')
        X = np.expand_dims(np.array(raw_X), axis=3)
        y = np.array(raw_y)
        return X,y 

    def preprocess_test(self, raw_X_test):
        X_test = np.expand_dims(np.array(raw_X_test), axis=3)
        return X_test

    def next_batch(self, num, X, y):
        '''
        input: numpy arrays
        '''
        idx = np.random.choice(X.shape[0], num, replace=False)
        X_shuffle = X[idx,:,:,:]
        y_shuffle = y[idx,:]
        
        return X_shuffle, y_shuffle

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev = 0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool(self, x, width_ksize):
        return tf.nn.max_pool(x, ksize=[1, 1, width_ksize, 1], strides=[1, 1, width_ksize, 1], padding='SAME')

    def deepnn(self, x,im_h,im_w,im_c,conv_width_ksize,maxpool_width_ksize):
        x_image = tf.reshape(x, [-1, im_h, im_w, im_c])
        
        # First convolutional layer - maps one grayscale image to 32 feature maps.
        W_conv1 = self.weight_variable([1, conv_width_ksize, im_c, 32])
        b_conv1 = self.bias_variable([32])
        h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)
        # Pooling layer - downsamples by 2X.
        h_pool1 = self.max_pool(h_conv1,maxpool_width_ksize)
        
        # Second convolutional layer -- maps 32 feature maps to 64.
        W_conv2 = self.weight_variable([1, conv_width_ksize, 32, 64])
        b_conv2 = self.bias_variable([64])
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
        # Second pooling layer.
        h_pool2 = self.max_pool(h_conv2,maxpool_width_ksize)
        
        # the image size after 2 conv and 2 pool layers
        im_h_fc1 = int(h_pool2.get_shape()[1])
        im_w_fc1 = int(h_pool2.get_shape()[2])

        # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
        # is down to 7x7x64 feature maps -- maps this to 1024 features.
        W_fc1 = self.weight_variable([im_w_fc1*im_h_fc1*64, 1024])
        b_fc1 = self.bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, im_w_fc1*im_h_fc1*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        
        # Dropout - controls the complexity of the model, prevents co-adaptation of
        # features.
        keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # Map the 1024 features to 5 classes, one for each digit
        W_fc2 = self.weight_variable([1024, 5])
        b_fc2 = self.bias_variable([5])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        weights = {}
        weights['W_conv1'] = W_conv1
        weights['W_conv2'] = W_conv2
        weights['W_fc1'] = W_fc1
        weights['W_fc2'] = W_fc2

        return weights, y_conv, keep_prob

    def one_layer_deepnn(self, x,im_h,im_w,im_c, conv_width_ksize, conv1_num_filter, fc1_num_feat, maxpool_width_ksize):
        x_image = tf.reshape(x, [-1, im_h, im_w, im_c])
        
        # First convolutional layer - maps one grayscale image to 32 feature maps.
        W_conv1 = self.weight_variable([1, conv_width_ksize, im_c, conv1_num_filter])
        b_conv1 = self.bias_variable([conv1_num_filter])
        h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)
        # Pooling layer - downsamples by 2X.
        h_pool1 = self.max_pool(h_conv1,maxpool_width_ksize)
        
        # the image size after 2 conv and 2 pool layers
        im_h_fc1 = int(h_pool1.get_shape()[1])
        im_w_fc1 = int(h_pool1.get_shape()[2])

        # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
        # is down to 7x7x64 feature maps -- maps this to 1024 features.
        W_fc1 = self.weight_variable([im_w_fc1*im_h_fc1*conv1_num_filter, fc1_num_feat])
        b_fc1 = self.bias_variable([fc1_num_feat])

        h_pool1_flat = tf.reshape(h_pool1, [-1, im_w_fc1*im_h_fc1*conv1_num_filter])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)
        
        # Dropout - controls the complexity of the model, prevents co-adaptation of
        # features.
        keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # Map the 1024 features to 10 classes, one for each digit
        W_fc2 = self.weight_variable([fc1_num_feat, 5])
        b_fc2 = self.bias_variable([5])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        weights = {}
        weights['W_conv1'] = W_conv1
        weights['W_fc1'] = W_fc1
        weights['W_fc2'] = W_fc2
        
        return weights, y_conv, keep_prob

    def deep_train(self, X_train, y_train, X_test, y_test, im_h, im_w, im_c, conv_width_ksize, conv1_num_filter, fc1_num_feat, maxpool_width_ksize, restore_sess, reg_beta, keep_prob_rate, max_iterations):
        x = tf.placeholder(tf.float32, [None,im_h,im_w,im_c], name = 'x')
        y_ = tf.placeholder(tf.float32, [None, 5], name = 'y_')
        # Build the graph for the deep net
        # weights, y_conv, keep_prob = self.deepnn(x,im_h,im_w,im_c,conv_width_ksize,maxpool_width_ksize)
        weights, y_conv, keep_prob = self.one_layer_deepnn(x,im_h,im_w,im_c,conv_width_ksize, conv1_num_filter, fc1_num_feat, maxpool_width_ksize)
        pred_prob = tf.nn.softmax(logits=y_conv, name = "pred_prob")

        # Loss function with L2 Regularization with beta=0.01
        regularizers = tf.zeros([], tf.float32)
        for key in weights:
            regularizers = tf.nn.l2_loss(weights[key])

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_conv) + reg_beta*regularizers)
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cost)
        correct_prediction = tf.cast(tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1)), tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)

        tf.add_to_collection('x', x)
        tf.add_to_collection('y_', y_)
        tf.add_to_collection('pred_prob', pred_prob)
        tf.add_to_collection('keep_prob', keep_prob)

        saver = tf.train.Saver()
        sess = tf.Session()
        if restore_sess:
            saver = tf.train.import_meta_graph("model.meta")
            saver.restore(sess, tf.train.latest_checkpoint('./'))
            print("Model restored.")
            max_iters = 2
        else:       
            sess.run(tf.global_variables_initializer())
            # sess.run(tf.initialize_all_variables())
            max_iters = max_iterations

        print("Start training...")
        for i in range(max_iters):
            batch_X, batch_y = self.next_batch(50, X_train, y_train)
            if i % 10 == 0:
                train_accuracy = accuracy.eval(session=sess,feed_dict={x: batch_X, y_: batch_y, keep_prob: 1.0})
                print('step {}, training accuracy {}'.format(i, train_accuracy))

                if X_test is not None and y_test is not None:
                    test_accuracy = accuracy.eval(session=sess, feed_dict={x: X_test, y_: y_test, keep_prob: 1.0})
                    print('         test accuracy: {}'.format(test_accuracy))

            # print('step {}'.format(i))
            train_step.run(session=sess,feed_dict={x: batch_X, y_: batch_y, keep_prob: keep_prob_rate})

            if (i+1) % 100 == 0:
                save_path = saver.save(sess, "./model")
        
        # if X_test is not None:
        #     prediction = pred_prob.eval(session=sess, feed_dict={x: X_test, keep_prob: 1.0})
        #     return prediction

    def train(self, raw_X, raw_y, raw_X_test, raw_y_test, conv_width_ksize, conv1_num_filter, fc1_num_feat, maxpool_width_ksize, restore_sess, reg_beta, keep_prob_rate, max_iterations):
        X, y = self.preprocess_train(raw_X, raw_y)
        im_h, im_w, im_c = X.shape[1], X.shape[2], 1

        if  raw_X_test is not None and raw_y_test is not None:
            X_test = self.preprocess_test(raw_X_test)
            y_test = np.array(raw_y_test)
            self.deep_train(X, y, X_test, y_test, im_h, im_w, im_c, conv_width_ksize, conv1_num_filter, fc1_num_feat, maxpool_width_ksize, restore_sess, reg_beta, keep_prob_rate, max_iterations)
        else:
            self.deep_train(X, y, None, None, im_h, im_w, im_c, conv_width_ksize, conv1_num_filter, fc1_num_feat, maxpool_width_ksize, restore_sess, reg_beta, keep_prob_rate, max_iterations)

    def predict(self, raw_X_test):
        # Create a clean graph and import the MetaGraphDef nodes.
        new_graph = tf.Graph()
        with tf.Session(graph=new_graph) as sess:
            # Load meta graph and restore weights
            saver = tf.train.import_meta_graph("model.meta")
            saver.restore(sess, tf.train.latest_checkpoint('./'))
            graph = tf.get_default_graph()

            x = graph.get_tensor_by_name("x:0")
            pred_prob = graph.get_tensor_by_name("pred_prob:0")
            keep_prob = graph.get_tensor_by_name('keep_prob:0')

            X_test = self.preprocess_test(raw_X_test)
            prediction = pred_prob.eval(session=sess, feed_dict={x: X_test, keep_prob: 1.0})

        return prediction