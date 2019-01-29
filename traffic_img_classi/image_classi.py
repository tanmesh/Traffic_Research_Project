import os
import time

import matplotlib.pyplot as plt
import numpy as np
from keras import Sequential
from keras import optimizers
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.layers import Dense, Flatten
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def get_training_data():
    # Preparing the training data
    data_dir_list = ['vehicles_test/', 'vehicles_train/']
    data_dir_path = '/Users/tanmesh/dev/traffic/traffic_img_classi'
    img_data_list = []
    labels = []
    for dataset in data_dir_list:
        img_list = os.listdir(data_dir_path + '/' + dataset)

        # reading only 1000 entries
        img_list = img_list[0:500]
        print('Loaded the images of dataset- ' + '{}\n'.format(dataset))
        for img in img_list:
            if 'not_vehicle' in img:
                labels.append(0)
            else:
                labels.append(1)
            img_path = data_dir_path + '/' + dataset + '/' + img
            if (img_path != '/Users/tanmesh/dev/traffic/traffic_img_classi/vehicles_test//.DS_Store') and (
                    img_path != '/Users/tanmesh/dev/traffic/traffic_img_classi/vehicles_train//.DS_Store'):
                img = image.load_img(img_path, target_size=(224, 224))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                img_data_list.append(x)

    img_data = np.array(img_data_list)
    print(img_data.shape)
    img_data = np.rollaxis(img_data, 1, 0)
    print(img_data.shape)
    img_data = img_data[0]
    print(img_data.shape)

    # shuffle the positive and negative data
    x, y = shuffle(img_data, labels, random_state=2)
    return x, y


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


def test_model(model, image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(img)
    plt.imshow(img / 255.)
    plt.show()
    x = preprocess_input(np.expand_dims(img.copy(), axis=0))
    print(model.predict(x))


def evaluate_model(history):
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(history.history['loss'], color='b', label="Training loss")
    ax[0].plot(history.history['val_loss'], color='r', label="validation loss", axes=ax[0])
    legend = ax[0].legend(loc='best', shadow=True)

    ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
    ax[1].plot(history.history['val_acc'], color='r', label="Validation accuracy")
    legend = ax[1].legend(loc='best', shadow=True)
    plt.show()


def classify_vehicle():
    # loading data from disk
    x, y = get_training_data()

    # creating training and validation set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

    # create training model using keras pre-training image classification model resnet50
    model = get_training_model()

    # we will use RMS prop with learning rate .0001 and binary_crossentropy for binary classification
    model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=2e-5), metrics=['accuracy'])

    # train the model
    t = time.time()
    hist = model.fit(x_train, y_train, batch_size=32, epochs=2, verbose=1, validation_data=(x_test, y_test))
    print('Training time: %s' % (t - time.time()))
    (loss, accuracy) = model.evaluate(x_test, y_test, batch_size=32, verbose=1)
    print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))

    # plot accuracy and loss for model
    evaluate_model(hist)

    # save model
    model.save('my_model.h5')

    # negative scenario
    test_image_path = '/Users/tanmesh/dev/traffic/traffic_img_classi/vehicles_test/4593_not_vehicle.jpg'
    test_model(model, test_image_path)

    # positive scenario
    test_image_path = '/Users/tanmesh/dev/traffic/traffic_img_classi/vehicles_train/02033_vehicle.jpg'
    test_model(model, test_image_path)


classify_vehicle()
