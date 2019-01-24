import os

import imageio as imageio
import numpy as np
from keras import Sequential
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from keras.preprocessing import image
from matplotlib.pyplot import imshow
from sklearn.model_selection import train_test_split

from process_data import split_data, prepare_data

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def build_model(img_width, img_height):
    model = Sequential()

    print("Running the first layer...")
    model.add(Conv2D(32, (3, 3), input_shape=(img_width, img_height, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    print("Running the second layer...")
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    print("Running the third layer...")
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    print("Running the last layer...")
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    # try:
    #     model.add(Dropout(0.5))
    # except Exception as e:
    #     print("There is error........."+str(e))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    print("Compiling the model...")
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model


def do_prediction(model, img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    my_image = imageio.imread(img_path)
    imshow(my_image)
    print(model.predict(x))


def img_classi():
    print("Splitting data into train and test...")
    total_img_data = split_data()
    img_width = 150
    img_height = 150

    print("Preparing the train data...")
    x, y = prepare_data(total_img_data, img_width, img_height)

    print("Splitting the train data into training and validation set...")
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=1)
    # x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=1)

    n_train = len(x_train)
    n_val = len(x_val)
    print("ntrain is %d" % n_train)
    print("nval is %d" % n_val)
    batch_size = 120
    epoch = 2

    print("Building the model..")
    model = build_model(img_width, img_height)
    print(model.__str__())
    print("Model build.")

    # print('Data augmentation...')
    # train_data_gen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    # val_data_gen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    #
    # print('Preparing generators for training and validation sets...')
    # train_generator = train_data_gen.flow(np.array(x_train), y_train, batch_size=batch_size)
    # validation_generator = val_data_gen.flow(np.array(x_val), y_val, batch_size=batch_size)

    x_train = np.array(x_train)
    x_val = np.array(x_val)

    print("Normalize data...")
    x_train = x_train / 255.0
    x_val = x_val / 255.0

    print("Reshape data...")
    x_train = x_train.reshape(-1, 150, 150, 3)
    x_val = x_val.reshape(-1, 150, 150, 3)

    print('Fitting the model...')
    model.fit(x_train, y_train, batch_size, epoch)
    model.save('model_keras.csv')  ## serialising and saving the data

    score = model.evaluate(x_val, y_val)
    for items in score:
        print(items)
    # print("Validation accuracy is %d" % score)

    # print('Saving the model...')
    # model.save_weights('model_weights.csv')
    # model.save('model_keras.csv')
    # print("Model saved...")
    #
    # print('Generating test data...')
    # test_data_gen = ImageDataGenerator(rescale=1. / 255)
    # test_generator = test_data_gen.flow(np.array(x_test), batch_size=batch_size)
    #
    # print("Predicting...")
    # pred = model.predict_generator(test_generator, verbose=1, steps=len(test_generator))
    # # print("Prediction is " + str(pred))
    # prediction = pd.DataFrame(pred, columns=['predictions']).to_csv('prediction.csv')

    print("Predicting for input image...")

    do_prediction(model)


img_classi()
do_prediction("")
