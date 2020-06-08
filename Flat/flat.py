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
    i = 0
    with open(path_groundtruth, 'r') as file:
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
                target.append(6)  # BCC
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


def create_model(modelo, tl=True):
    if modelo == 'vgg':
        from tensorflow.keras.applications.vgg19 import VGG19
        from tensorflow.keras.layers import Dense

        if tl == False:
            vgg = VGG19(include_top=False, weights=None, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
        else:
            vgg = VGG19(include_top=False, weights='imagenet', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

        model = tf.keras.Sequential(vgg)
        model.add(tf.keras.layers.Flatten())
        model.add(Dense(units=4096, activation="relu"))
        model.add(Dense(units=4096, activation="relu"))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(Dense(units=7, activation="softmax"))

    elif modelo == 'resnet50':
        from tensorflow.keras.applications.resnet import ResNet50
        from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

        if tl == False:
            resnet = ResNet50(include_top=False, weights=None, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
        else:
            resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

        model = tf.keras.Sequential(resnet)
        model.add(GlobalAveragePooling2D())
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(Dense(units=7, activation="softmax"))

    elif modelo == 'resnet101':

        from tensorflow.keras.applications.resnet import ResNet101
        from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

        if tl == False:
            resnet = ResNet101(include_top=False, weights=None, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
        else:
            resnet = ResNet101(include_top=False, weights='imagenet', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

        model = tf.keras.Sequential(resnet)
        model.add(GlobalAveragePooling2D())
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(Dense(units=7, activation="softmax"))

    elif modelo == 'densenet':

        from tensorflow.keras.applications.densenet import DenseNet121
        from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

        if tl == False:
            densenet = DenseNet121(include_top=False, weights=None, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
        else:
            densenet = DenseNet121(include_top=False, weights='imagenet', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

        model = tf.keras.Sequential(densenet)
        model.add(GlobalAveragePooling2D())
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(Dense(units=7, activation="softmax"))

    else:
        print("Esse modelo não existe!")
        exit(0)

    return model

x_train = import_dataset(p_train, 'training')
y_train = assign_labels(t_train)

x_val = import_dataset(p_val, 'validation')
y_val = assign_labels(t_val)

exit(0)
print("Images imported.")

no_epochs = 20
lr = 1e-5
no_classes = 7
batch_size = 10

# NÃO ESQUECER DE DEFINIR !!!
w_0 = 10608. / 1212
w_1 = 10608. / 7149
w_2 = 10608. / 519
w_3 = 10608. / 360
w_4 = 10608. / 1101
w_5 = 10608. / 114
w_6 = 10608. / 153

#class_weight = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1}
class_weight = {0: w_0, 1: w_1, 2: w_2, 3: w_3, 4: w_4, 5: w_5, 6: w_6}

y_train_cat = keras.utils.to_categorical(y_train, no_classes)
y_val_cat = keras.utils.to_categorical(y_val, no_classes)

################################ MODEL ################################
from tensorflow.keras.callbacks import EarlyStopping
import datetime

# ADJUST LEARNING RATE IF VALIDATION DOES NOT IMPROVE
def scheduler(epoch):
    global no_epochs
    global lr
    if epoch < int(0.5 * no_epochs):
        return lr
    elif epoch < int(0.75 * no_epochs):
        return lr / 10
    else:
        return lr / 100

scheduler_cb = tf.keras.callbacks.LearningRateScheduler(scheduler)


checkpoint_path = {'vgg': "/home/ruben/Desktop/Small/Flat/vgg/cp.ckpt",
                   'resnet': "/home/ruben/Desktop/Small/Flat/resnet/cp.ckpt",
                   'densenet': "/home/ruben/Desktop/Small/Flat/densenet/cp.ckpt"}

checkpoint_dir = os.path.dirname(checkpoint_path['densenet'])

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path['densenet'],
                                                 save_weights_only=True,
                                                 verbose=1, save_best_only=True)

model = create_model("densenet", tl=True)  # vgg, resnet50, resnet101, densenet121
model.summary()
#######################################################################

adam = tf.keras.optimizers.Adam(lr=1e-5)
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=adam, metrics=['accuracy'])

fit = model.fit(x_train, y_train_cat, batch_size=batch_size, class_weight=class_weight,
                callbacks=[scheduler_cb, cp_callback], epochs=no_epochs, shuffle=True,
                validation_data=(x_val, y_val_cat))
model.load_weights(checkpoint_path['densenet'])

########################### Evaluate in test set ############################
y_pred = model.predict_classes(x_val)

conf_matrix = confusion_matrix(y_val, y_pred)
print(conf_matrix)

score = model.evaluate(x_val, y_val_cat, verbose=1)
print(f'Val loss: {score[0]} / Val accuracy: {score[1]}')

plot_val_train_error(fit)