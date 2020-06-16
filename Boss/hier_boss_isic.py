from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten, Input
import csv
from numpy import array
from operator import itemgetter
import cv2
import os
import time
import tensorflow as tf
import numpy as np


IMG_HEIGHT = 224
IMG_WIDTH = 224

def index_of_max(array, size):
    maxi = 0
    for i in range(1, size):
        if array[i] > array[maxi]:
            maxi = i
    return maxi

def import_dataset(path_dataset, mode):
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

def create_labels(path):
    target = []


    counter = {'MEL': 0, 'NV': 0, 'BCC': 0, 'AKIEC': 0, 'BKL': 0, 'DF': 0, 'VASC': 0}
    i = 0
    with open(path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if i == 0:
                i += 1
                continue
            if row[1] == '1.0':  # MEL
                counter['MEL'] += 1
                target.append(0)

            elif row[2] == '1.0':  # NV
                counter['NV'] += 1
                target.append(1)

            elif row[3] == '1.0':  # BCC
                counter['BCC'] += 1
                target.append(2)

            elif row[4] == '1.0':  # AKIEC
                counter['AKIEC'] += 1
                target.append(3)

            elif row[5] == '1.0':  # BKL
                counter['BKL'] += 1
                target.append(4)

            elif row[6] == '1.0':  # DF
                counter['DF'] += 1
                target.append(5)

            elif row[7] == '1.0':  # VASC
                counter['VASC'] += 1
                target.append(6)

            else:
                continue
    print(counter)
    file.close()
    return target


def create_model():

    inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)) # inputs.shape, inputs.dtype
    densenet = DenseNet121(include_top=False, weights='imagenet')
    x = densenet(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)

    a = Dense(2, activation='softmax', name="model_a")(x)
    b = Dense(2, activation='softmax', name="model_b")(x)
    c = Dense(2, activation='softmax', name="model_c")(x)
    d = Dense(3, activation='softmax', name="model_d")(x)
    e = Dense(2, activation='softmax', name="model_e")(x)

    model = tf.keras.Model(inputs=inputs, outputs=[a, b, c, d, e], name='global_loss')
    model.summary()
    return model

def split_in_a(y_pred, x, names):
    x_b = []
    x_c = []
    names_b = []
    names_c = []
    y = y_pred[0] # y_a

    if len(x) != len(y):
        print("oops", len(x), len(y), model)
        exit(0)

    for i in range(len(y)):
        if y[i][0] > y[i][1]:
            x_b.append(x[i])
            names_b.append(names[i])
        else:
            x_c.append(x[i])
            names_c.append(names[i])

    return array(x_b), array(x_c), names_b, names_c

def split_in_c(y_pred, x, names):
    x_d = []
    x_e = []
    names_d = []
    names_e = []
    y = y_pred[2]  # y_c

    if len(x) != len(y):
        print("oops", len(x), len(y), model)
        exit(0)

    for i in range(len(y)):
        if y[i][0] > y[i][1]:
            x_e.append(x[i])
            names_e.append(names[i])
        else:
            x_d.append(x[i])
            names_d.append(names[i])

    return array(x_d), array(x_e), names_d, names_e

def get_max_index(l):
    index = 0
    max = l[0]
    for i in range(1, len(l)):
        if l[i] > max:
            index = i
            max = l[i]
    return index

def update_conf_matrix(model, pred, true, conf, offset, total):
    pred = pred[model]
    d = {0: 0, 1: 0, 2: 0}
    if len(pred) != len(true):
        print("meh!")
        exit(0)
    for i in range(len(pred)):
        total += 1
        prediction = get_max_index(pred[i])
        d[prediction] += 1
        conf[true[i]][prediction+offset] += 1
    print("model: ", model, " - ", d)
    print("####################")
    return conf, total


if __name__ == '__main__':

    start_time = time.time()
    p_val = '/home/ruben/Desktop/test2018'

    model = create_model()

    names, x_val = import_dataset(p_val, 'val')

    checkpoint_path = "/home/ruben/Desktop/Boss/small_dataset_weights/cp.ckpt"
    model.load_weights(checkpoint_path)

    y_pred = model.predict(x_val)
    x_b, x_c, names_b, names_c = split_in_a(y_pred, x_val, names)

    y_pred = model.predict(x_c)
    x_d, x_e, names_d, names_e = split_in_c(y_pred, x_c, names_c)

    y_pred_b = model.predict(x_b)[1]
    y_pred_d = model.predict(x_d)[3]
    y_pred_e = model.predict(x_e)[4]

    name_pred = []

    if len(x_b) >= 1:
        for i in range(0, len(x_b)):
            index = index_of_max(y_pred_b[i, :], 2)
            name_pred.append((names_b[i], index))

    if len(x_d) >= 1:
        for i in range(0, len(x_d)):
            index = index_of_max(y_pred_d[i, :], 3) + 4
            name_pred.append((names_d[i], index))

    if len(x_e) >= 1:
        for i in range(0, len(x_e)):
            index = index_of_max(y_pred_e[i, :], 2) + 2
            name_pred.append((names_e[i], index))

    name_pred = sorted(name_pred, key=itemgetter(0))
    print(len(name_pred))

    path = '/home/ruben/PycharmProjects/SkinLesions6/Boss/isic2018_boss.csv'
    with open(path, mode='w') as csv_file:
        fieldnames = ['image', 'MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(0, len(name_pred)):
            if name_pred[i][1] == 0:
                writer.writerow(
                    {'image': name_pred[i][0][0:-4], 'MEL': float(94 / 100), 'NV': float(1 / 100),
                     'BCC': float(1 / 100), 'AKIEC': float(1 / 100),
                     'BKL': float(1 / 100), 'DF': float(1 / 100), 'VASC': float(1 / 100)})
            elif name_pred[i][1] == 1:
                writer.writerow(
                    {'image': name_pred[i][0][0:-4], 'MEL': float(1 / 100), 'NV': float(94 / 100),
                     'BCC': float(1 / 100), 'AKIEC': float(1 / 100),
                     'BKL': float(1 / 100), 'DF': float(1 / 100), 'VASC': float(1 / 100)})
            elif name_pred[i][1] == 2:
                writer.writerow(
                    {'image': name_pred[i][0][0:-4], 'MEL': float(1 / 100), 'NV': float(1 / 100),
                     'BCC': float(94 / 100), 'AKIEC': float(1 / 100),
                     'BKL': float(1 / 100), 'DF': float(1 / 100), 'VASC': float(1 / 100)})
            elif name_pred[i][1] == 3:
                writer.writerow(
                    {'image': name_pred[i][0][0:-4], 'MEL': float(1 / 100), 'NV': float(1 / 100), 'BCC': float(1 / 100),
                     'AKIEC': float(94 / 100),
                     'BKL': float(1 / 100), 'DF': float(1 / 100), 'VASC': float(1 / 100)})
            elif name_pred[i][1] == 4:
                writer.writerow(
                    {'image': name_pred[i][0][0:-4], 'MEL': float(1 / 100), 'NV': float(1 / 100), 'BCC': float(1 / 100),
                     'AKIEC': float(1 / 100),
                     'BKL': float(94 / 100), 'DF': float(1 / 100), 'VASC': float(1 / 100)})
            elif name_pred[i][1] == 5:
                writer.writerow(
                    {'image': name_pred[i][0][0:-4], 'MEL': float(1 / 100), 'NV': float(1 / 100), 'BCC': float(1 / 100),
                     'AKIEC': float(1 / 100),
                     'BKL': float(1 / 100), 'DF': float(94 / 100), 'VASC': float(1 / 100)})
            elif name_pred[i][1] == 6:
                writer.writerow(
                    {'image': name_pred[i][0][0:-4], 'MEL': float(1 / 100), 'NV': float(1 / 100), 'BCC': float(1 / 100),
                     'AKIEC': float(1 / 100),
                     'BKL': float(1 / 100), 'DF': float(1 / 100), 'VASC': float(94 / 100)})
            else:
                print("ERROR!")
                exit(0)

    print("--- %s seconds ---" % (time.time() - start_time))