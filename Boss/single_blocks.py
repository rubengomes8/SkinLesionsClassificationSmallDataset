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

    # creates a list with only the image and not its filename
    for x in dataset_unsorted:
        dataset.append(x[1])
    return array(dataset)  # Converts the lists into numpy.ndarray

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

def split_in_a(y_pred, x, y_true):
    x_b = []
    x_c = []
    y_true_b = []
    y_true_c = []
    y = y_pred[0] # y_a
    conf_a = [[0,0],[0,0]]
    print(y_true)
    if len(x) != len(y):
        print("oops", len(x), len(y), model)
        exit(0)

    for i in range(len(y)):
        if y[i][0] > y[i][1]:
            x_b.append(x[i])
            y_true_b.append(y_true[i])
            if y_true[i] == 0 or y_true[i] == 1:
                conf_a[0][0] += 1
            else:
                conf_a[1][0] += 1

        else:
            x_c.append(x[i])
            y_true_c.append(y_true[i])
            if y_true[i] == 0 or y_true[i] == 1:
                conf_a[0][1] += 1
            else:
                conf_a[1][1] += 1
    print("conf_a")
    print(conf_a)
    return array(x_b), array(x_c), y_true_b, y_true_c

def split_in_c(y_pred, x, y_true):
    x_d = []
    x_e = []
    y_true_d = []
    y_true_e = []
    y = y_pred[2]  # y_c
    conf_c = [[0,0],[0,0]]

    if len(x) != len(y):
        print("oops", len(x), len(y), model)
        exit(0)

    for i in range(len(y)):
        if y[i][0] > y[i][1]:
            x_e.append(x[i])
            y_true_e.append(y_true[i])
            if y_true[i] == 2 or y_true[i] == 3:
                conf_c[0][0] += 1
            elif y_true[i] == 4 or y_true[i] == 5 or y_true[i] == 6:
                conf_c[1][0] += 1

        else:
            x_d.append(x[i])
            y_true_d.append(y_true[i])
            if y_true[i] == 4 or y_true[i] == 5 or y_true[i] == 6:
                conf_c[1][1] += 1
            elif y_true[i] == 2 or y_true[i] == 3:
                conf_c[0][1] += 1

    print("conf_c")
    print(conf_c)

    return array(x_d), array(x_e), y_true_d, y_true_e

def conf_b(y_pred, y_true):

    y = y_pred[1]  # y_b
    conf = [[0,0],[0,0]]

    for i in range(len(y)):
        if y[i][0] > y[i][1]:
            if y_true[i] == 0:
                conf[0][0] += 1
            elif y_true[i] == 1:
                conf[1][0] += 1

        else:
            if y_true[i] == 0:
                conf[0][1] += 1
            elif y_true[i] == 1:
                conf[1][1] += 1

    print("conf_b")
    print(conf)

def conf_e(y_pred, y_true):

    y = y_pred[4]
    conf = [[0,0],[0,0]]

    for i in range(len(y)):
        if y[i][0] > y[i][1]:
            if y_true[i] == 2:
                conf[0][0] += 1
            elif y_true[i] == 1:
                conf[1][0] += 1

        else:
            if y_true[i] == 3:
                conf[0][1] += 1
            elif y_true[i] == 1:
                conf[1][1] += 1

    print("conf_e")
    print(conf)

def conf_d(y_pred, y_true):

    y = y_pred[3]
    print(len(y))
    print(len(y_true))
    conf = [[0,0,0],[0,0,0],[0,0,0]]

    for i in range(len(y)):

        if y[i][0] > y[i][1]:
            if y[i][0] > y[i][2]:
                index = 0
            else:
                index = 2
        else:
            if y[i][1] > y[i][2]:
                index = 1
            else:
                index = 2

        if y_true[i] == 4 or y_true[i] == 5 or y_true[i] == 6:
            true = y_true[i] - 4
            conf[true][index] += 1


    print("conf_d")
    print(conf)


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
    p_val = '/home/ruben/Desktop/smallval2018'
    t_val = '/home/ruben/Desktop/smallval2018/labels.csv'

    model = create_model()

    x_val = import_dataset(p_val, 'val')
    y_val = create_labels(t_val)

    checkpoint_path = "/home/ruben/Desktop/Boss/small_dataset_weights/cp.ckpt"
    model.load_weights(checkpoint_path)

    y_pred = model.predict(x_val)
    x_b, x_c, y_true_b, y_true_c = split_in_a(y_pred, x_val, y_val)

    y_pred = model.predict(x_c)
    x_d, x_e, y_true_d, y_true_e = split_in_c(y_pred, x_c, y_true_c)

    y_pred_b = model.predict(x_b)
    y_pred_d = model.predict(x_d)
    y_pred_e = model.predict(x_e)




    conf_matrix = np.zeros(shape=(7, 7), dtype=int)
    conf_matrix, total = update_conf_matrix(1, y_pred_b, y_true_b, conf_matrix, 0, 0)
    conf_matrix, total = update_conf_matrix(3, y_pred_d, y_true_d, conf_matrix, 4, total)
    conf_matrix, total = update_conf_matrix(4, y_pred_e, y_true_e, conf_matrix, 2, total)

    print(conf_matrix)
    print(total)

    conf_b(y_pred_b, y_true_b)
    conf_e(y_pred_e, y_true_e)
    conf_d(y_pred_d, y_true_d)

    print("--- %s seconds ---" % (time.time() - start_time))