import os
import cv2


def prepare_data(list_of_images_path, img_width, img_height):
    global tmp
    x = []
    y = []
    for image_path in list_of_images_path:
        try:
            read_image = cv2.imread(image_path)
            tmp = cv2.resize(read_image, (img_width, img_height), interpolation=cv2.INTER_CUBIC)
        except Exception as e:
            print(e)
        x.append(tmp)

    for i in list_of_images_path:
        if 'not_vehicle' in i:
            y.append(0)
        else:
            y.append(1)

    return x, y


def split_data():
    train_dir = "/Users/tanmesh/dev/traffic/dataset/vehicles_train/"
    test_dir = "/Users/tanmesh/dev/traffic/dataset/vehicles_test/"
    img1 = [train_dir + i for i in os.listdir(train_dir)]
    img2 = [test_dir + i for i in os.listdir(test_dir)]

    # print(len(img1))
    # print(len(img2))
<<<<<<< HEAD
    total_img_data = img1 + img2
=======
    total_img_data = img1[:2000] + img2[6000:8000]
>>>>>>> 1a71d4b17eab4f840491374b6dcf87c6f03701d4
    # print(len(total_img_data))
    return total_img_data
