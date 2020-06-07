from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten, Input
import csv
from numpy import array
from operator import itemgetter
import cv2
import os
import time
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import numpy as np
import tensorflow as tf


start_time = time.time()

no_epochs = 25
lr = 1e-5

IMG_HEIGHT = 224
IMG_WIDTH = 224

p_train = '/home/ruben/Desktop/smalltrain2018'
t_train = '/home/ruben/Desktop/smalltrain2018/labels.csv'

p_val = '/home/ruben/Desktop/smallval2018'
t_val = '/home/ruben/Desktop/smallval2018/labels.csv'


def create_labels(path):
    target = []
    target_a = []
    target_b = []
    target_c = []
    target_d = []
    target_e = []

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
                target_a.append(0)
                target_b.append(0)
                target_c.append(-1)
                target_d.append(-1)
                target_e.append(-1)

            elif row[2] == '1.0':  # NV
                counter['NV'] += 1
                target.append(1)
                target_a.append(0)
                target_b.append(1)
                target_c.append(-1)
                target_d.append(-1)
                target_e.append(-1)

            elif row[3] == '1.0':  # BCC
                counter['BCC'] += 1
                target.append(2)
                target_a.append(1)
                target_b.append(-1)
                target_c.append(0)
                target_d.append(-1)
                target_e.append(0)

            elif row[4] == '1.0':  # AKIEC
                counter['AKIEC'] += 1
                target.append(3)
                target_a.append(1)
                target_b.append(-1)
                target_c.append(0)
                target_d.append(-1)
                target_e.append(1)

            elif row[5] == '1.0':  # BKL
                counter['BKL'] += 1
                target.append(4)
                target_a.append(1)
                target_b.append(-1)
                target_c.append(1)
                target_d.append(0)
                target_e.append(-1)

            elif row[6] == '1.0':  # DF
                counter['DF'] += 1
                target.append(5)
                target_a.append(1)
                target_b.append(-1)
                target_c.append(1)
                target_d.append(1)
                target_e.append(-1)

            elif row[7] == '1.0':  # VASC
                counter['VASC'] += 1
                target.append(6)  # BCC
                target_a.append(1)
                target_b.append(-1)
                target_c.append(1)
                target_d.append(2)
                target_e.append(-1)

            else:
                continue
    print(counter)
    file.close()
    return target, target_a, target_b, target_c, target_d, target_e, counter


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


def to_categorical_custom(array, num_classes=2):
    categorical = []
    for i in range(len(array)):
        if array[i] == -1:
            categorical.append(np.zeros(num_classes, dtype=np.float))
        elif array[i] == 0:
            a = np.zeros(num_classes, dtype=np.float)
            a[0] = float(1)
            categorical.append(a)
        elif array[i] == 1:
            a = np.zeros(num_classes, dtype=np.float)
            a[1] = float(1)
            categorical.append(a)
        elif array[i] == 2:
            a = np.zeros(num_classes, dtype=np.float)
            a[2] = float(1)
            categorical.append(a)
    return categorical

def count_labels(array):
    d = {-1: 0, 0: 0, 1: 0, 2: 0}
    for i in range(len(array)):
        d[array[i]] += 1

    print(d)

if __name__ == "__main__":

    print(tf.executing_eagerly())
    x_train = import_dataset(p_train, 'training')
    x_val = import_dataset(p_val, 'val')

    y_train, y_a, y_b, y_c, y_d, y_e, counter_train = create_labels(t_train)
    y_val, y_val_a, y_val_b, y_val_c, y_val_d, y_val_e, counter_val = create_labels(t_val)

    print(counter_train)
    print(counter_val)
    exit(0)
    count_labels(y_a)
    count_labels(y_b)
    count_labels(y_c)
    count_labels(y_d)
    count_labels(y_e)


    model = create_model()

    losses = {
        "model_a": "categorical_crossentropy",
        "model_b": "categorical_crossentropy",
        "model_c": "categorical_crossentropy",
        "model_d": "categorical_crossentropy",
        "model_e": "categorical_crossentropy"
    }

    adam = Adam(lr=1e-5)
    model.compile(optimizer=adam, loss=losses,
                  metrics=["accuracy"])

    y_a_cat = to_categorical_custom(y_a, num_classes=2) # fazer customizado,
    y_b_cat = to_categorical_custom(y_b, num_classes=2)
    y_c_cat = to_categorical_custom(y_c, num_classes=2)
    y_d_cat = to_categorical_custom(y_d, num_classes=3)
    y_e_cat = to_categorical_custom(y_e, num_classes=2)
    y_val_a_cat = to_categorical_custom(y_val_a, num_classes=2)
    y_val_b_cat = to_categorical_custom(y_val_b, num_classes=2)
    y_val_c_cat = to_categorical_custom(y_val_c, num_classes=2)
    y_val_d_cat = to_categorical_custom(y_val_d, num_classes=3)
    y_val_e_cat = to_categorical_custom(y_val_e, num_classes=2)


    y_val_cat = to_categorical(y_val)
    tf.compat.v1.enable_eager_execution()
    print(tf.executing_eagerly())

    checkpoint_path = "/home/ruben/Desktop/Boss/small_dataset_weights/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1, save_best_only=True)

    def scheduler(epoch):
        global no_epochs
        global lr
        if epoch < int(0.5 * no_epochs):
            return lr
        elif epoch < int(0.75 * no_epochs):
            print("Redução do lr /10")
            return lr / 10.
        else:
            print("Redução do lr /100")
            return lr / 100.

    scheduler_cb = tf.keras.callbacks.LearningRateScheduler(scheduler)

    class_weights_a = {0: 24012. / 18633, 1: 24012. / 5379}
    class_weights_b = {0: 18633. / 2697, 1: 18633. / 15936}
    class_weights_c = {0: 5379. / 2106, 1: 5379. / 3273}
    class_weights_d = {0: 3273. / 2625, 1: 3273. / 291, 2: 3273. / 357}
    class_weights_e = {0: 2106. / 1272, 1: 2106. / 834}
    class_weights = [class_weights_a, class_weights_b, class_weights_c, class_weights_d, class_weights_e]

    fit = model.fit(x_train, [y_a_cat, y_b_cat, y_c_cat, y_d_cat, y_e_cat], batch_size=10, epochs=10, callbacks=[cp_callback, scheduler_cb],
                    class_weight=class_weights, validation_data=(x_val, [y_val_a_cat, y_val_b_cat, y_val_c_cat, y_val_d_cat, y_val_e_cat])) # class_weights?


    print("--- %s seconds ---" % (time.time() - start_time))