# Load pickled data
import sys
import csv
import pickle
import cv2

# TODO: Fill this in based on where you saved the training and testing data

training_file = "dataset/train.p"
validation_file= "dataset/valid.p"
testing_file = "dataset/test.p"
signnames_file = "dataset/signnames.csv"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
with open(signnames_file, mode='r') as f:
    reader = csv.reader(f)
    next(reader)  # skip the header
    signnames = {rows[0]: rows[1] for rows in reader}

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

## Replace each question mark with the appropriate value.
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(set(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

import random
import numpy as np
import matplotlib.pyplot as plt

plt.figure(1)
index = random.randint(0, n_train)
plt.imshow(X_train[index].squeeze(), cmap='gray')
plt.title(signnames[str(0)])
plt.savefig('images/one')

plt.figure(figsize=(15,3))
for i in range(0,5):
    plt.subplot(1, 5, i+1)
    plt.imshow(X_train[i])
    plt.title(signnames[str(i)])
plt.figure(figsize=(15,3))
for i in range(0,5):
    plt.subplot(1, 5, i+1)
    plt.imshow(X_test[i])
    plt.title(signnames[str(i)])
plt.savefig('images/multiple_signs')

from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def normalize_grayscale(img):
    a = 0.
    b = 1.0
    xmax = 255
    xmin = 0
    return (a + (img - xmin) * (b-a) / (xmax - xmin))

def preprecess_images(images):
    shape = images[0].shape
    img_list = rgb2gray(images)
    img_list = normalize_grayscale(img_list)
    return np.reshape(img_list, [len(images), shape[0], shape[1], 1]) 

X_train = preprecess_images(X_train)
X_valid = preprecess_images(X_valid)
X_test = preprecess_images(X_test)

print(X_train.shape)
print(X_test.shape)

from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(
            X_train,
            y_train,
            test_size=0.05,
            random_state=832289)

import tensorflow as tf
EPOCHS = 5
BATCH_SIZE=128

from tensorflow.contrib.layers import flatten
def LeNet(x):
    mu = 0
    sigma = 0.1

    # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)

    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)

    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
    fc0   = flatten(conv2)

    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b

    # SOLUTION: Activation.
    fc1    = tf.nn.relu(fc1)

    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b

    # SOLUTION: Activation.
    fc2    = tf.nn.relu(fc2)

    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, 43), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)

rate = 0.001

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './lenet')
    print("Model saved")

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))

import matplotlib.image as mping
test_images = []
for i in range(1, 6):
    img = mping.imread("/notebooks/test/test_image_" + str(i) + ".jpg")
    test_images.append(img)

for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(test_images[i])
plt.show()
X_test1 = preprecess_images(np.array(test_images))
print(X_test1.shape)

pred_prob = tf.nn.softmax(logits)
pred_prob5 = tf.nn.top_k(pred_prob, 5)
pred_class = tf.argmax(logits, 1)

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    r_pred_class = sess.run(pred_class, feed_dict={x: X_test1})
    r_pred_prob5 = sess.run(pred_prob5, feed_dict={x: X_test1})

for i in range(5):
    print("test_image_" + str(i+1) + "belongs to class " + str(r_pred_class[i]) + " - " + signnames[str(r_pred_class[i])])

for i in range(5):
    plt.imshow(X_test1[i].squeeze(), cmap="gray")
    plt.show()
    print("top 5 class probablities:")
    for j in range(5):
        print("Class #" + str(r_pred_prob5[1][i][j]) + " - " + signnames[str(r_pred_prob5[1][i][j])] + " : " + str(round(r_pred_prob5[0][i][j],3)))
