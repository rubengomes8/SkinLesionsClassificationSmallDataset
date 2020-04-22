import cv2
import os
from operator import itemgetter
from numpy import array
import csv
import time
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import pandas as pd


start_time = time.time()

p_train = '/home/ruben/Desktop/smalltrain2018'
t_train = '/home/ruben/Desktop/smalltrain2018/labels.csv'

p_val = '/home/ruben/Desktop/smallval2018'
t_val = '/home/ruben/Desktop/smallval2018/labels.csv'

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

def assign_labels(path_groundtruth):
    target = []
    counter = {'MEL': 0, 'NV': 0, 'BCC': 0, 'AKIEC': 0, 'BKL': 0, 'DF': 0, 'VASC': 0}
    i=0
    with open(path_groundtruth, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if i == 0:
                i += 1
                continue
            if row[1] == '1.0': # MEL
                counter['MEL'] += 1
                target.append(0)
            elif row[2] == '1.0': # NV
                counter['NV'] += 1
                target.append(1)
            elif row[3] == '1.0': # BCC
                counter['BCC'] += 1
                target.append(2)
            elif row[4] == '1.0': # AKIEC
                counter['AKIEC'] += 1
                target.append(3)
            elif row[5] == '1.0': # BKL
                counter['BKL'] += 1
                target.append(4)
            elif row[6] == '1.0': # DF
                counter['DF'] += 1
                target.append(5)
            elif row[7] == '1.0':   # VASC
                counter['VASC'] += 1
                target.append(6) # BCC
            else:
                continue
    print(counter)
    file.close()
    return target


def plot_val_train_error(fit):
    plt.plot(fit.history['accuracy'])
    plt.plot(fit.history['val_accuracy'])
    plt.grid(True)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    plt.plot(fit.history['loss'])
    plt.plot(fit.history['val_loss'])
    plt.grid(True)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

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

ckpt1_path = "/home/ruben/Desktop/Small/Hier/a/cp.ckpt"
ckpt2_path = "/home/ruben/Desktop/Small/Hier/b/cp.ckpt"
ckpt3_path = "/home/ruben/Desktop/Small/Hier/c/cp.ckpt"
ckpt4_path = "/home/ruben/Desktop/Small/Hier/d/cp.ckpt"
ckpt5_path = "/home/ruben/Desktop/Small/Hier/e/cp.ckpt"

model1.load_weights(ckpt1_path)
model2.load_weights(ckpt2_path)
model3.load_weights(ckpt3_path)
model4.load_weights(ckpt4_path)
model5.load_weights(ckpt5_path)

#############################  IMPORT LABELS  ################################
indexes_1_nmel = []
indexes_1_mel = []
indexes_2_mal = []
indexes_2_ben = []

y_val = assign_labels(t_val)

y_nmel = []
y_mel = []
y_mal = []
y_ben = []

############################# IMPORT DATASETS ################################
x_val = import_dataset(p_val, 'validation')

# Evaluation
y_pred = model1.predict_classes(x_val)

x_mel = []
x_nmel = []

div1 = []
div2 = []
div3 = []
div4 = []
div5 = []

err_bef1 = []
err_bef2 = []
err_bef3 = []
err_bef4 = []
err_bef5 = []

for i in range(0, len(y_pred)):
    if y_pred[i] == 1: # NMEL
        indexes_1_nmel.append(i)
        x_nmel.append(x_val[i])
        y_nmel.append(y_val[i])
        if y_val[i] == 0 or y_val[i] == 1:
            div1.append('0') # False
        else:
            div1.append('1') # True

    elif y_pred[i] == 0: # MEL
        indexes_1_mel.append(i)
        x_mel.append(x_val[i])
        y_mel.append(y_val[i])
        if y_val[i] == 0 or y_val[i] == 1:
            div1.append('1') # True
        else:
            div1.append('0') # False

    err_bef1.append('1')

x_mel = array(x_mel)
x_nmel = array(x_nmel)

y_mel_pred = model2.predict_classes(x_mel)
y_nmel_pred = model3.predict_classes(x_nmel)

x_ben = []
x_mal = []

for i in range(0, len(y_nmel_pred)):
    if y_nmel[i] == 0 or y_nmel[i] == 1:
        err_bef3.append('0')
    else:
        err_bef3.append('1')

    if y_nmel_pred[i] == 1: # MAL
        indexes_2_mal.append(i)
        x_mal.append(x_nmel[i])
        y_mal.append(y_nmel[i])

        if y_nmel[i] == 2 or y_nmel[i] == 3:
            div3.append('1') # True
        else:
            div3.append('0') # False

    elif y_nmel_pred[i] == 0: # BEN
        indexes_2_ben.append(i)
        x_ben.append(x_nmel[i])
        y_ben.append(y_nmel[i])
        if y_nmel[i] == 2 or y_nmel[i] == 3:
            div3.append('0') # False
        else:
            div3.append('1') # True


x_ben = array(x_ben)
x_mal = array(x_mal)

y_ben_pred = model4.predict_classes(x_ben)
y_mal_pred = model5.predict_classes(x_mal)

zippedList = list(zip(y_mel_pred, y_mel))
df_mel = pd.DataFrame(zippedList, columns = ['MEL_PRED', 'MEL_TRUTH'])

zippedList = list(zip(y_ben_pred, y_ben))
df_ben = pd.DataFrame(zippedList, columns = ['BEN_PRED', 'BEN_TRUTH'])

zippedList = list(zip(y_mal_pred, y_mal))
df_mal = pd.DataFrame(zippedList, columns = ['MAL_PRED', 'MAL_TRUTH'])

for i in range(0, len(y_mel_pred)):
    if y_mel[i] == 0:
        if y_mel_pred[i] == 0:
            div2.append('1') # True
        else:
            div2.append('0') # False
        err_bef2.append('1')
    elif y_mel[i] == 1:
        if y_mel_pred[i] == 1:
            div2.append('1')  # True
        else:
            div2.append('0')  # False
        err_bef2.append('1')
    else:
        div2.append('0')
        err_bef2.append('0')

for i in range(0, len(y_ben_pred)):
    if y_ben[i] == 4:

        if y_ben_pred[i] + 4 == 4:
            div4.append('1')  # True
        else:
            div4.append('0')  # False
        err_bef4.append('1')

    elif y_ben[i] == 5:

        if y_ben_pred[i] + 4 == 5:
            div4.append('1')  # True
        else:
            div4.append('0')  # False
        err_bef4.append('1')

    elif y_ben[i] == 6:

        if y_ben_pred[i] + 4 == 6:
            div4.append('1')  # True
        else:
            div4.append('0')  # False
        err_bef4.append('1')
    else:
        div4.append('0')
        err_bef4.append('0')

for i in range(0, len(y_mal_pred)):
    if y_mal[i] == 2: #BCC

        if y_mal_pred[i] + 2 == 2:
            div5.append('1')  # True
        else:
            div5.append('0')  # False
        err_bef5.append('1')

    elif y_mal[i] == 3: # AKIEC

        if y_mal_pred[i] + 2 == 3:
            div5.append('1')  # True
        else:
            div5.append('0')  # False
        err_bef5.append('1')
    else:
        div5.append('0')
        err_bef5.append('0')


df_mel.to_csv('/home/ruben/Desktop/teste_mel3.csv', index=False)
df_ben.to_csv('/home/ruben/Desktop/teste_ben3.csv', index=False)
df_mal.to_csv('/home/ruben/Desktop/teste_mal3.csv', index=False)


y_pred1 = model1.predict(x_val)
y_mel_pred2 = model2.predict(x_mel)
y_nmel_pred3 = model3.predict(x_nmel)
y_ben_pred4 = model4.predict(x_ben)
y_mal_pred5 = model5.predict(x_mal)

import numpy as np

y_pred1 = np.array(y_pred1)
df_1 = pd.DataFrame({'mel': y_pred1[:,0], 'nmel': y_pred1[:,1], 'div': div1, 'not_err_bef': err_bef1})

y_mel_pred2 = np.array(y_mel_pred2)
df_2 = pd.DataFrame({'0': y_mel_pred2[:,0], '1': y_mel_pred2[:,1] , 'div': div2, 'not_err_bef': err_bef2})

y_nmel_pred3 = np.array(y_nmel_pred3)
df_3 = pd.DataFrame({'ben': y_nmel_pred3[:,0], 'mal': y_nmel_pred3[:,1] , 'div': div3, 'not_err_bef': err_bef3})

y_ben_pred4 = np.array(y_ben_pred4)
df_4 = pd.DataFrame({'4': y_ben_pred4[:,0], '5': y_ben_pred4[:,1] , '6': y_ben_pred4[:,2], 'div': div4, 'not_err_bef': err_bef4})

y_mal_pred5 = np.array(y_mal_pred5)
df_5 = pd.DataFrame({'2': y_mal_pred5[:,0], '3': y_mal_pred5[:,1] , 'div': div5, 'not_err_bef': err_bef5})


df_1.to_csv('/home/ruben/Desktop/pred1_3.csv', index=False)
df_2.to_csv('/home/ruben/Desktop/pred2_3.csv', index=False)
df_3.to_csv('/home/ruben/Desktop/pred3_3.csv', index=False)
df_4.to_csv('/home/ruben/Desktop/pred4_3.csv', index=False)
df_5.to_csv('/home/ruben/Desktop/pred5_3.csv', index=False)



print("--- %s minutes ---" % ((time.time() - start_time)/60))