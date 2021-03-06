import time

import matplotlib.pyplot as plt
import pandas as pd
from keras import Sequential
from keras import optimizers
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, Flatten

from load_dataset import get_training_data, test_model


def get_training_model():
    # use keras model with pre-trained weights
    resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # creating model for vehicle classifier
    model = Sequential()
    model.add(resnet)
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    print(model.summary())
    return model


def evaluate_model(history):
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(history.history['loss'], color='b', label="Training loss")
    ax[0].plot(history.history['val_loss'], color='r', label="validation loss", axes=ax[0])
    legend = ax[0].legend(loc='best', shadow=True)

    ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
    ax[1].plot(history.history['val_acc'], color='r', label="Validation accuracy")
    legend = ax[1].legend(loc='best', shadow=True)
    plt.show()


def classify_vehicle(img_set_size):
    # creating training and validation set
    x_train, x_test, y_train, y_test, classes = get_training_data(img_set_size, partition_ratio=0.2, img_size=224)

    # create training model using keras pre-training image classification model resnet50
    model = get_training_model()

    # we will use RMS prop with learning rate .0001 and binary_crossentropy for binary classification
    model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=2e-5), metrics=['accuracy'])

    # train the model
    t = time.time()
    hist = model.fit(x_train, y_train, batch_size=32, epochs=2, validation_data=(x_test, y_test))
    print('Training time: %s' % (t - time.time()))
    (loss, accuracy) = model.evaluate(x_test, y_test, batch_size=32, verbose=1)
    print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))
    if img_set_size == 50:
        df.loc[0, "50 data-size"] = accuracy*100
    elif img_set_size == 2000:
        df.loc[1, "2000 data-size"] = accuracy*100
    elif img_set_size == 4000:
        df.loc[2, "4000 data-size"] = accuracy*100
    elif img_set_size == 6000:
        df.loc[3, "6000 data-size"] = accuracy*100
    else:
        df.loc[4, "8000 data-size"] = accuracy*100

    # plot accuracy and loss for model
    evaluate_model(hist)

    # save model
    model_name = "cnn_model_of_size_" + str(img_set_size) + ".h5"
    model.save(model_name)

    # negative scenario
    test_image_path = '/Users/tanmesh/dev/traffic/vehicles_test/4593_not_vehicle.jpg'
    test_model(model, test_image_path)

    # positive scenario
    test_image_path = '/Users/tanmesh/dev/traffic/vehicles_train/02033_vehicle.jpg'
    test_model(model, test_image_path)


df = pd.DataFrame(columns=["50 data-size", "2000 data-size", "4000 data-size", "6000 data-size", "8000 data-size"])
classify_vehicle(img_set_size=100//2)
classify_vehicle(img_set_size=2000//2)
classify_vehicle(img_set_size=4000//2)
classify_vehicle(img_set_size=6000//2)
classify_vehicle(img_set_size=8000//2)
print(df)
df.to_csv("accuracy_table.csv")
