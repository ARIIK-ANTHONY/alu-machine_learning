#!/usr/bin/env python3
""" Train Mini-Batch using TensorFlow 1.x """

import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """ trains a loaded neural network model using mini-batch gradient descent

    Args:
        X_train (np.array): matrix of shape (m, 784) containing the training data
        Y_train (np.array): one-hot matrix of shape (m, 10) containing training labels
        X_valid (np.array): matrix of shape (m, 784) containing validation data
        Y_valid (np.array): one-hot matrix of shape (m, 10) containing validation labels
        batch_size (int): number of data points in a batch
        epochs (int): number of times training passes through the whole dataset
        load_path (str): path from which to load the model
        save_path (str): path to where the model should be saved
        
    Returns:
        str: the path where the model was saved
    """
    with tf.Session() as sess:
        # Load the model
        saver = tf.train.import_meta_graph(load_path + '.meta')
        saver.restore(sess, load_path)
        
        # Get model collections
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        train_op = tf.get_collection('train_op')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        
        # Calculate number of mini-batches
        m = X_train.shape[0]
        batches = m // batch_size
        if m % batch_size != 0:
            batches += 1
        
        # Training loop
        for epoch in range(epochs + 1):
            # Calculate training metrics
            cost_t, acc_t = sess.run([loss, accuracy],
                                     feed_dict={x: X_train, y: Y_train})
            cost_v, acc_v = sess.run([loss, accuracy],
                                     feed_dict={x: X_valid, y: Y_valid})
            
            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(cost_t))
            print("\tTraining Accuracy: {}".format(acc_t))
            print("\tValidation Cost: {}".format(cost_v))
            print("\tValidation Accuracy: {}".format(acc_v))
            
            # Skip training after the last epoch
            if epoch == epochs:
                break
            
            # Shuffle training data
            X_shuffled, Y_shuffled = shuffle_data(X_train, Y_train)
            
            # Mini-batch training
            for step in range(batches):
                start = step * batch_size
                end = start + batch_size
                X_mini = X_shuffled[start:end]
                Y_mini = Y_shuffled[start:end]
                
                # Run training step
                sess.run(train_op, feed_dict={x: X_mini, y: Y_mini})
                
                # Print progress every 100 steps
                if (step + 1) % 100 == 0:
                    cost_step, acc_step = sess.run([loss, accuracy],
                                                   feed_dict={x: X_mini, y: Y_mini})
                    print("\tStep {}:".format(step + 1))
                    print("\t\tCost: {}".format(cost_step))
                    print("\t\tAccuracy: {}".format(acc_step))
            
            print()  # Empty line between epochs
        
        # Save and return the model path
        save_path_result = saver.save(sess, save_path)
        return save_path_result
