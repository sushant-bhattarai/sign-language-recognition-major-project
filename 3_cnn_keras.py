import numpy as np
import pickle
import cv2, os
import tensorflow as tf
from glob import glob
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from matplotlib import pyplot as plt

K.image_data_format()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def get_image_size():
    img = cv2.imread('gestures/1/100.jpg', 0)
    return img.shape


def get_num_of_classes():
    return len(glob('gestures/*'))


image_x, image_y = get_image_size()


def cnn_model():
    num_of_classes = get_num_of_classes()
    model = Sequential()
    model.add(Conv2D(16, (2, 2), input_shape=(image_x, image_y, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='same'))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_of_classes, activation='softmax'))
    # sgd = optimizers.SGD(lr=1e-2)
    sgd = tf.keras.optimizers.SGD(learning_rate=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    filepath = "cnn_model_keras.h5"
    checkpoint1 = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint1]
    # from keras.utils import plot_model
    from tensorflow.keras.utils import plot_model
    plot_model(model, to_file='model.png', show_shapes=True)
    return model, callbacks_list


def train():
    with open("train_images", "rb") as f:
        train_images = np.array(pickle.load(f))
    with open("train_labels", "rb") as f:
        train_labels = np.array(pickle.load(f), dtype=np.int32)

    with open("val_images", "rb") as f:
        val_images = np.array(pickle.load(f))
    with open("val_labels", "rb") as f:
        val_labels = np.array(pickle.load(f), dtype=np.int32)

    train_images = np.reshape(train_images, (train_images.shape[0], image_x, image_y, 1))
    val_images = np.reshape(val_images, (val_images.shape[0], image_x, image_y, 1))
    train_labels = np_utils.to_categorical(train_labels)
    val_labels = np_utils.to_categorical(val_labels)

    print(val_labels.shape)

    model, callbacks_list = cnn_model()
    model.summary()
    epochs = 20
    hist = model.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=epochs,
                     batch_size=500,
                     callbacks=callbacks_list)

    # plotting epoch vs loss graph
    loss_train = hist.history['loss']
    loss_val = hist.history['val_loss']
    epoch = range(1, epochs + 1)
    plt.plot(epoch, loss_train, 'g', label='Training loss')
    plt.plot(epoch, loss_val, 'b', label='Validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig('epoch_vs_loss_graph.png')
    plt.show()

    # plotting epoch vs accuracy graph
    loss_train = hist.history['accuracy']
    loss_val = hist.history['val_accuracy']
    epoch = range(1, epochs + 1)
    plt.plot(epoch, loss_train, 'g', label='Training accuracy')
    plt.plot(epoch, loss_val, 'b', label='Validation accuracy')
    plt.title('Training and Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig('epoch_vs_accuracy_graph.png')
    plt.show()

    scores = model.evaluate(val_images, val_labels, verbose=0)
    print("CNN Error: %.2f%%" % (100 - scores[1] * 100))
    model.save('cnn_model_keras.h5')


train()
K.clear_session()
