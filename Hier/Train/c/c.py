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

p_train = '/home/ruben/Desktop/HierSmall/c'
t_train = '/home/ruben/Desktop/HierSmall/c/labels.csv'

p_val = '/home/ruben/Desktop/HierSmall/val/c'
t_val = '/home/ruben/Desktop/HierSmall/val/c/labels.csv'

IMG_HEIGHT = 224
IMG_WIDTH = 224


def import_dataset(path_dataset, mode, val=False):
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
    counter = {'BEN': 0, 'MAL': 0}
    i = 0
    with open(path_groundtruth, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if i == 0:
                i += 1
                continue

            if row[1] == '1.0': # BEN
                counter['BEN'] += 1
                target.append(0)
            elif row[2] == '1.0': # MAL
                counter['MAL'] += 1
                target.append(1)
    print(counter)
    file.close()
    return target, counter


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


def create_model(modelo, tl=False):
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
        model.add(Dense(units=2, activation="softmax"))

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
        model.add(Dense(units=2, activation="softmax"))

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
        model.add(Dense(units=2, activation="softmax"))

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
        model.add(Dense(units=2, activation="softmax"))

    else:
        print("Esse modelo não existe!")
        exit(0)

    return model


x_train = import_dataset(p_train, 'training', False)
print("x_train: ", len(x_train))
y_train, counter = assign_labels(t_train)
print("y_train: ", len(y_train))
print(counter)

# EM PRINCIPIO ESTA OK!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

n_0_float = float(counter['BEN'])
n_1_float = float(counter['MAL'])
n_total = n_0_float + n_1_float
w_0 = n_total / n_0_float
w_1 = n_total / n_1_float

print("n_0: ", n_0_float)
print("n_1: ", n_1_float)
print("n_total: ", n_total)

dict_labels_val = {}
y_val, counter = assign_labels(t_val)
print("y_val: ", len(y_val))
x_val = import_dataset(p_val, 'validation', True)
print("x_val: ", len(x_val))

print("Images imported.")

no_epochs = 20
lr = 1e-5
no_classes = 2
batch_size = 10

class_weight = {0: w_0, 1: w_1}
y_train_cat = keras.utils.to_categorical(y_train, no_classes)
y_val_cat = keras.utils.to_categorical(y_val, no_classes)

################################ MODEL ################################

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

checkpoint_path = "/home/ruben/Desktop/Small/Hier/c/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1, save_best_only=True)

model = create_model("densenet", tl=True)  # vgg, resnet50, resnet101, densenet121
model.summary()
#######################################################################

adam = tf.keras.optimizers.Adam(lr=1e-5)
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=adam, metrics=['accuracy'])
# não meti earlystop_cb
fit = model.fit(x_train, y_train_cat, batch_size=batch_size, class_weight=class_weight,
                callbacks=[scheduler_cb, cp_callback], epochs=no_epochs, shuffle=True,
                validation_data=(x_val, y_val_cat))
model.load_weights(checkpoint_path)

########################### Evaluate in test set ############################
y_pred = model.predict_classes(x_val)

conf_matrix = confusion_matrix(y_val, y_pred)
print(conf_matrix)

score = model.evaluate(x_val, y_val_cat, verbose=1)
print(f'Val loss: {score[0]} / Val accuracy: {score[1]}')

plot_val_train_error(fit)

print("--- %s seconds ---" % (time.time() - start_time))