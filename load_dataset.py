import os
from sklearn.utils import shuffle

import matplotlib.pyplot as plt
import numpy as np
from keras.applications.imagenet_utils import preprocess_input
from keras_preprocessing import image
from sklearn.model_selection import train_test_split


def get_training_data(img_set_size, partition_ratio, img_size):
    # Preparing the training data
    data_dir_list = ['vehicles_test/', 'vehicles_train/']
    data_dir_path = '/Users/tanmesh/dev/traffic'
    img_data_list = []
    labels = []
    for dataset in data_dir_list:
        img_list = os.listdir(data_dir_path + '/' + dataset)

        img_list = img_list[0:img_set_size]
        print('Loaded the images of dataset-' + '{}\n'.format(dataset))
        for img in img_list:
            if 'not_vehicle' in img:
                labels.append(0)
            else:
                labels.append(1)
            img_path = data_dir_path + '/' + dataset + '/' + img

            img = image.load_img(img_path, target_size=(img_size, img_size))
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

    print(type(labels))
    # shuffle the positive and negative data
    x, y = shuffle(img_data, labels)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=partition_ratio, random_state=2)

    classes = np.array({"not_vehicle", "vehicle"})

    return x_train, x_test, y_train, y_test, classes


def test_model(model, image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(img)
    plt.imshow(img / 255.)
    plt.show()
    x = preprocess_input(np.expand_dims(img.copy(), axis=0))
    print(model.predict(x))


# get_training_data(img_set_size=50, partition_ratio=0.2, img_size=64)
