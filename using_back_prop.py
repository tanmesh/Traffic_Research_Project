import time

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.keras import Sequential, optimizers
from tensorflow.python.layers.core import Dense
import pandas as pd
from load_dataset import get_training_data, test_back_prop_model


def evaluate_model(history):
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(history.history['loss'], color='b', label="Training loss")
    ax[0].plot(history.history['val_loss'], color='r', label="validation loss", axes=ax[0])
    legend = ax[0].legend(loc='best', shadow=True)

    ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
    ax[1].plot(history.history['val_acc'], color='r', label="Validation accuracy")
    legend = ax[1].legend(loc='best', shadow=True)
    plt.show()


def build_model(h, w, n_c):
    model = Sequential()
    model.add(Dense(128, input_dim=h * w * n_c, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    return model


def using_back_prop(img_set_size):
    img_size = 64
    x_train, x_test, y_train, y_test, classes = get_training_data(img_set_size, partition_ratio=0.2, img_size=img_size)

    n_train = x_train.shape[0]
    n_test = x_test.shape[0]
    height = x_train.shape[1]
    width = x_train.shape[2]
    n_channels = x_train.shape[3]
    print(n_train, n_test, height, width, n_channels)

    x_train = x_train.reshape((n_train, height * width * n_channels))
    x_test = x_test.reshape((n_test, height * width * n_channels))
    print("train_x's shape: " + str(x_train.shape))
    print("test_x's shape: " + str(x_test.shape))

    # x_train = x_train_flatten/255.
    # x_test = x_test_flatten / 255.

    model = build_model(height, width, n_channels)

    model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=2e-5), metrics=['accuracy'])
    t = time.time()
    hist = model.fit(x_train, y_train, batch_size=32, epochs=2, validation_data=(x_test, y_test))
    print('Training time: %s' % (t - time.time()))
    (loss, accuracy) = model.evaluate(x_test, y_test, batch_size=32, verbose=1)
    print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))

    evaluate_model(hist)

    if img_set_size == 50:
        df.loc[0, "50 data-size"] = accuracy*100
    elif img_set_size == 2000:
        df.loc[0, "2000 data-size"] = accuracy*100
    elif img_set_size == 4000:
        df.loc[0, "4000 data-size"] = accuracy*100
    elif img_set_size == 6000:
        df.loc[0, "6000 data-size"] = accuracy*100
    else:
        df.loc[0, "8000 data-size"] = accuracy*100

    # negative scenario
    test_image_path = '/Users/tanmesh/dev/traffic/vehicles_test/4593_not_vehicle.jpg'
    test_back_prop_model(model, test_image_path, img_size)

    # positive scenario
    test_image_path = '/Users/tanmesh/dev/traffic/vehicles_train/02033_vehicle.jpg'
    test_back_prop_model(model, test_image_path, img_size)

    print(df)
    df.to_csv("accuracy_table_using_BP.csv")


# fix random seed for reproducibility
np.random.seed(7)
df = pd.DataFrame(columns=["50 data-size", "2000 data-size", "4000 data-size", "6000 data-size", "8000 data-size"])
using_back_prop(img_set_size=50)
using_back_prop(img_set_size=2000)
using_back_prop(img_set_size=4000)
using_back_prop(img_set_size=6000)
using_back_prop(img_set_size=8000)

