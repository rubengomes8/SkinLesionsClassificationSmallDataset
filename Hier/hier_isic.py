import cv2
import os
from operator import itemgetter
from numpy import array
import csv
import time
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import shutil
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.resnet import ResNet101
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import pandas as pd


start_time = time.time()

p_test = '/home/ruben/Desktop/test2018'


IMG_HEIGHT = 224
IMG_WIDTH = 224

def index_of_max(array, size):
    maxi = 0
    for i in range(1, size):
        if array[i] > array[maxi]:
            maxi = i
    return maxi

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

def create_model(no_classes):
    densenet = DenseNet121(include_top=False, weights=None, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    model = tf.keras.Sequential(densenet)
    model.add(GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(Dense(units=no_classes, activation="softmax"))
    return model
######################## CREATE MODELS AND LOAD WEIGHTS ######################

model1 = create_model(2) # MEL vs NMEL
model2 = create_model(2) # NV vs MELA
model3 = create_model(2) # BEN vs MAL
model4 = create_model(3) # BKL vs DF vs VASC
model5 = create_model(2) # AKIEC vs BCC

ckpt1_path = '/home/ruben/Desktop/Small/Hier/a/cp.ckpt'
ckpt2_path = '/home/ruben/Desktop/Small/Hier/b/cp.ckpt'
ckpt3_path = '/home/ruben/Desktop/Small/Hier/c/cp.ckpt'
ckpt4_path = '/home/ruben/Desktop/Small/Hier/d/cp.ckpt'
ckpt5_path = '/home/ruben/Desktop/Small/Hier/e/cp.ckpt'

model1.load_weights(ckpt1_path)
model2.load_weights(ckpt2_path)
model3.load_weights(ckpt3_path)
model4.load_weights(ckpt4_path)
model5.load_weights(ckpt5_path)

dataset_unsorted = []
dataset = []
target = []

names, x_test = import_dataset(p_test, 'test', dataset_unsorted, dataset)
y_pred = model1.predict_classes(x_test)

x_mel = []
x_nmel = []
x_ben = []
x_mal = []

y_nmel = []
y_mel = []
y_mal = []
y_ben = []

name_pred = []
names_nmel = []
names_mel = []

for i in range(0, len(y_pred)):
    if y_pred[i] == 1: # NMEL
        x_nmel.append(x_test[i])
        names_nmel.append(names[i])

    elif y_pred[i] == 0: # MEL
        x_mel.append(x_test[i])
        names_mel.append(names[i])

x_mel = array(x_mel)
x_nmel = array(x_nmel)

y_mel = model2.predict(x_mel)
y_nmel = model3.predict_classes(x_nmel)

names_mal = []
names_ben = []

for i in range(0, len(y_nmel)):
    if y_nmel[i] == 1: #
        x_mal.append(x_nmel[i])
        names_mal.append(names_nmel[i])

    elif y_nmel[i] == 0: # BEN
        x_ben.append(x_nmel[i])
        names_ben.append(names_nmel[i])

x_ben = array(x_ben)
x_mal = array(x_mal)

y_ben = model4.predict(x_ben)
y_mal = model5.predict(x_mal)

name_pred = sorted(name_pred, key=itemgetter(0))
print(len(name_pred))

if len(x_mel) >= 1:
    for i in range(0, len(x_mel)):
        index = index_of_max(y_mel[i, :], 2)
        name_pred.append((names_mel[i], index))

if len(x_ben) >= 1:
    for i in range(0, len(x_ben)):
        index = index_of_max(y_ben[i, :], 3) + 4
        name_pred.append((names_ben[i], index))

if len(x_mal) >= 1:
    for i in range(0, len(x_mal)):
        index = index_of_max(y_mal[i, :], 2) + 2
        name_pred.append((names_mal[i], index))

name_pred = sorted(name_pred, key=itemgetter(0))
print(len(name_pred))

path = '/home/ruben/PycharmProjects/SkinLesions6/Hier/isic2018_hier.csv'
with open(path, mode='w') as csv_file:
    fieldnames = ['image', 'MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(0, len(name_pred)):
        if name_pred[i][1] == 0:
            writer.writerow(
                {'image': name_pred[i][0][0:-4], 'MEL': float(94/100), 'NV': float(1/100), 'BCC': float(1/100), 'AKIEC': float(1/100),
                 'BKL': float(1/100), 'DF': float(1/100), 'VASC': float(1/100)})
        elif name_pred[i][1] == 1:
            writer.writerow(
                {'image': name_pred[i][0][0:-4], 'MEL': float(1/100), 'NV': float(94/100), 'BCC': float(1/100), 'AKIEC': float(1/100),
                 'BKL': float(1/100), 'DF': float(1/100), 'VASC': float(1/100)})
        elif name_pred[i][1] == 2:
            writer.writerow(
                {'image': name_pred[i][0][0:-4], 'MEL': float(1/100), 'NV': float(1/100), 'BCC': float(94/100), 'AKIEC': float(1/100),
                 'BKL': float(1/100), 'DF': float(1/100), 'VASC': float(1/100)})
        elif name_pred[i][1] == 3:
            writer.writerow(
                {'image': name_pred[i][0][0:-4], 'MEL': float(1/100), 'NV': float(1/100), 'BCC': float(1/100), 'AKIEC': float(94/100),
                 'BKL': float(1/100), 'DF': float(1/100), 'VASC': float(1/100)})
        elif name_pred[i][1] == 4:
            writer.writerow(
                {'image': name_pred[i][0][0:-4], 'MEL': float(1/100), 'NV': float(1/100), 'BCC': float(1/100), 'AKIEC': float(1/100),
                 'BKL': float(94/100), 'DF': float(1/100), 'VASC': float(1/100)})
        elif name_pred[i][1] == 5:
            writer.writerow(
                {'image': name_pred[i][0][0:-4], 'MEL': float(1/100), 'NV': float(1/100), 'BCC': float(1/100), 'AKIEC': float(1/100),
                 'BKL': float(1/100), 'DF': float(94/100), 'VASC': float(1/100)})
        elif name_pred[i][1] == 6:
            writer.writerow(
                {'image': name_pred[i][0][0:-4], 'MEL': float(1/100), 'NV': float(1/100), 'BCC': float(1/100), 'AKIEC': float(1/100),
                 'BKL': float(1/100), 'DF': float(1/100), 'VASC': float(94/100)})
        else:
            print("ERROR!")
            exit(0)