import cv2
import os
from operator import itemgetter
from numpy import array
import csv
import time
import tensorflow as tf
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D


start_time = time.time()

p_test = '/home/ruben/Desktop/test2018'

IMG_HEIGHT = 224
IMG_WIDTH = 224

def create_model(no_classes):
    densenet = DenseNet121(include_top=False, weights=None, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    model = tf.keras.Sequential(densenet)
    model.add(GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dropout(0.5))
    if no_classes == 2:
        model.add(Dense(units=no_classes, activation="softmax"))
    else:
        model.add(Dense(units=no_classes, activation="softmax"))
    return model

def import_dataset(path_dataset, mode, dataset_unsorted, dataset):
    dataset = []
    dataset_unsorted = []
    print("Start importing " + mode + " images...")
    for filename in os.listdir(path_dataset):
        if filename.endswith(".jpg"):
            complete_path = os.path.join(path_dataset, filename)
            image = cv2.imread(complete_path, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # from BGR to RGB
            dim = (IMG_HEIGHT, IMG_WIDTH)  # image dimensions
            image = cv2.resize(image, dsize=dim, interpolation=cv2.INTER_AREA)
            image_filename = [filename, image]
            dataset_unsorted.append(image_filename)
        else:
            continue

    # orders the list of images by alphabetic order of its name
    dataset_unsorted = sorted(dataset_unsorted, key=itemgetter(0))

    names = []
    # creates a list with only the image and not its filename
    for x in dataset_unsorted:
        dataset.append(x[1])
        names.append(x[0])
    return names, array(dataset)  # Converts the lists into numpy.ndarray

flat = create_model(7)
flat_path = '/home/ruben/Desktop/Small/Flat/densenet/cp.ckpt'
flat.load_weights(flat_path)

dataset_unsorted = []
dataset = []
target = []

names, x_test = import_dataset(p_test, 'validation', dataset_unsorted, dataset)
y_test = flat.predict_classes(x_test)

path = '/home/ruben/PycharmProjects/SkinLesions6/Flat/isic2018_flat.csv'
with open(path, mode='w') as csv_file:
    fieldnames = ['image', 'MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(0, len(names)):
        if y_test[i] == 0:
            writer.writerow(
                {'image': names[i][0:-4], 'MEL': float(94/100), 'NV': float(1/100), 'BCC': float(1/100), 'AKIEC': float(1/100),
                 'BKL': float(1/100), 'DF': float(1/100), 'VASC': float(1/100)})
        elif y_test[i] == 1:
            writer.writerow(
                {'image': names[i][0:-4], 'MEL': float(1/100), 'NV': float(94/100), 'BCC': float(1/100), 'AKIEC': float(1/100),
                 'BKL': float(1/100), 'DF': float(1/100), 'VASC': float(1/100)})
        elif y_test[i] == 2:
            writer.writerow(
                {'image': names[i][0:-4], 'MEL': float(1/100), 'NV': float(1/100), 'BCC': float(94/100), 'AKIEC': float(1/100),
                 'BKL': float(1/100), 'DF': float(1/100), 'VASC': float(1/100)})
        elif y_test[i] == 3:
            writer.writerow(
                {'image': names[i][0:-4], 'MEL': float(1/100), 'NV': float(1/100), 'BCC': float(1/100), 'AKIEC': float(94/100),
                 'BKL': float(1/100), 'DF': float(1/100), 'VASC': float(1/100)})
        elif y_test[i] == 4:
            writer.writerow(
                {'image': names[i][0:-4], 'MEL': float(1/100), 'NV': float(1/100), 'BCC': float(1/100), 'AKIEC': float(1/100),
                 'BKL': float(94/100), 'DF': float(1/100), 'VASC': float(1/100)})
        elif y_test[i] == 5:
            writer.writerow(
                {'image': names[i][0:-4], 'MEL': float(1/100), 'NV': float(1/100), 'BCC': float(1/100), 'AKIEC': float(1/100),
                 'BKL': float(1/100), 'DF': float(94/100), 'VASC': float(1/100)})
        elif y_test[i] == 6:
            writer.writerow(
                {'image': names[i][0:-4], 'MEL': float(1/100), 'NV': float(1/100), 'BCC': float(1/100), 'AKIEC': float(1/100),
                 'BKL': float(1/100), 'DF': float(1/100), 'VASC': float(94/100)})
        else:
            print("ERROR!")
            exit(0)


print("--- %s minutes ---" % ((time.time() - start_time)/60))