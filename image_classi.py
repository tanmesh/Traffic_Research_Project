import numpy as np
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.models import Sequential
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

from process_data import split_data, prepare_data


def img_classi():
    print("Splitting data into train and test...")
    total_img_data = split_data()
    img_width = 150
    img_height = 150

    print("Preparing the train data...")
    x, y = prepare_data(total_img_data, img_width, img_height)
    print("Splitting the train data into training and validation set...")
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=1)
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=1)
    n_train = len(x_train)
    n_val = len(x_val)

    batch_size = 16

    print("Building the model..")
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

    print("Model build.")

    print('Data augmentation...')
    train_data_gen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    val_data_gen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

    print('Preparing generators for training and validation sets...')
    train_generator = train_data_gen.flow(np.array(x_train), y_train, batch_size=batch_size)
    validation_generator = val_data_gen.flow(np.array(x_val), y_val, batch_size=batch_size)

    print('Fitting the model...')
    model.fit_generator(train_generator, steps_per_epoch=n_train // batch_size, epochs=30,
                        validation_data=validation_generator, validation_steps=n_val // batch_size)

    print('Saving the model...')
    model.save_weights('model_wieghts.h5')
    model.save('model_keras.h5')
    print("Model saved...")

    print('Generating test data...')
    test_data_gen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_data_gen.flow(np.array(x_test), batch_size=batch_size)

    print("Predicting...")
    pred = model.predict_generator(test_generator, verbose=1, steps=len(test_generator))
    print("Prediction is " + str(pred))


img_classi()
