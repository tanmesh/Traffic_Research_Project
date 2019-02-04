import time

from tensorflow.python.keras import Sequential, optimizers
from tensorflow.python.layers.core import Dense
import matplotlib.pyplot as plt
from load_dataset import get_training_data, test_model
import numpy as np


def evaluate_model(history):
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(history.history['loss'], color='b', label="Training loss")
    ax[0].plot(history.history['val_loss'], color='r', label="validation loss", axes=ax[0])
    legend = ax[0].legend(loc='best', shadow=True)

    ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
    ax[1].plot(history.history['val_acc'], color='r', label="Validation accuracy")
    legend = ax[1].legend(loc='best', shadow=True)
    plt.show()


def build_model():
    model = Sequential()
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # print(model.summary())
    return model


def using_back_prop(img_set_size):
    x_train, x_test, y_train, y_test, classes = get_training_data(img_set_size, partition_ratio=0.2, img_size=64)

    # print(type(x_train))
    # print(type(x_test))
    # print(type(y_train))
    # print(type(y_test))

    # img = image.img_to_array(x_train[0])
    # plt.imshow(img / 255.)
    # plt.show()
    #
    # img = image.img_to_array(x_test[0])
    # plt.imshow(img / 255.)
    # plt.show()
    #
    # print(y_train[0])
    # print(y_test[0])

    # m_train = x_train.shape[0]
    # num_px = x_train.shape[1]
    # m_test = x_test.shape[0]
    #
    # print("Number of training examples: " + str(m_train))
    # print("Number of testing examples: " + str(m_test))
    # print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
    # print("train_x_orig shape: " + str(x_train.shape))
    # print("train_y shape: " + str(y_train))
    # print("test_x_orig shape: " + str(x_test.shape))
    # print("test_y shape: " + str(y_test))

    x_train = x_train/255.
    x_test = x_test / 255.

    # print("train_x's shape: " + str(x_train.shape))
    # print("test_x's shape: " + str(x_test.shape))

    # n_x = num_px * num_px * 3
    # n_h = 7
    # n_y = 1

    model = build_model()

    model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=2e-5), metrics=['accuracy'])
    hist = model.fit(x_train, y_train, batch_size=32, epochs=2, validation_data=(x_test, y_test), validation_split=0)
    # print('Training time: %s' % (t - time.time()))
    (loss, accuracy) = model.evaluate(x_test, y_test, batch_size=32, verbose=1)
    print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))

    evaluate_model(hist)

    # negative scenario
    test_image_path = '/Users/tanmesh/dev/traffic/vehicles_test/4593_not_vehicle.jpg'
    test_model(model, test_image_path)

    # positive scenario
    test_image_path = '/Users/tanmesh/dev/traffic/vehicles_train/02033_vehicle.jpg'
    test_model(model, test_image_path)


# fix random seed for reproducibility
np.random.seed(7)
using_back_prop(img_set_size=50)
