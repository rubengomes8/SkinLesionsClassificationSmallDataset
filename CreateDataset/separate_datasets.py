import cv2
import os
from operator import itemgetter
from numpy import array
import csv
import time
from random import randint

start_time = time.time()

print("--- %s seconds ---" % (time.time() - start_time))

# Dataset de treino original
p_origin = "/home/ruben/Desktop/train/"
t_origin = "/home/ruben/Desktop/truth/labels.csv"

# Dataset de treino - porção 70% do treino -> 35
p_train = "/home/ruben/Desktop/smalltrain2018/"
t_train = "/home/ruben/Desktop/smalltrain2018/labels.csv"

# Dataset de validação - porção 17% do treino -> 9
p_val = "/home/ruben/Desktop/smallval2018/"
t_val = "/home/ruben/Desktop/smallval2018/labels.csv"

# Dataset de teste - porção 13% do treino -> 7
p_test = "/home/ruben/Desktop/smalltest2018/"
t_test = "/home/ruben/Desktop/smalltest2018/labels.csv"

def import_dataset(path_dataset, mode, dataset_unsorted, dataset):
    dataset = []
    dataset_unsorted = []
    print("Start importing " + mode + " images...")
    for filename in os.listdir(path_dataset):
        if filename.endswith(".jpg"):
            complete_path = os.path.join(path_dataset, filename)
            image = cv2.imread(complete_path, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # from BGR to RGB
            dim = (224, 224)  # image dimensions
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

labels_dict = {}
val_csv_list = []
train_csv_list = []
test_csv_list = []
i = 0

with open(t_origin, 'r') as filee:
    reader = csv.reader(filee)
    for row in reader:
        if i == 0:
            i+= 1
        if row[1] == '1.0':  # MEL
            labels_dict[row[0]] = 1
        elif row[2] == '1.0':  # NV
            labels_dict[row[0]] = 2
        elif row[3] == '1.0':  # BCC
            labels_dict[row[0]] = 3
        elif row[4] == '1.0':  # AKIEC
            labels_dict[row[0]] = 4
        elif row[5] == '1.0':  # BKL
            labels_dict[row[0]] = 5
        elif row[6] == '1.0':  # DF
            labels_dict[row[0]] = 6
        elif row[7] == '1.0':  # VASC
            labels_dict[row[0]] = 7
filee.close()

file_dir = p_origin # input directory path
for root, _, files in os.walk(file_dir):
    #print(root)
    pass

for file in files:
    ext = file[-4:]
    if ext == ".jpg":
        index = file[:len(file) - 4]  # tira o .jpg
        image = cv2.imread(root + file)
        value = randint(1, 100)
        if 1 <= value <= 9: # 17% -> 9
            # Image goes to validation set
            cv2.imwrite(p_val+'%s' %str(file), image)
            if labels_dict[index] == 1:  # MEL
                val_csv_list.append((str(index), '1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0'))
            elif labels_dict[index] == 2:  # NV
                val_csv_list.append((str(index), '0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0'))
            elif labels_dict[index] == 3:  # BCC
                val_csv_list.append((str(index), '0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0'))
            elif labels_dict[index] == 4:  # AKIEC
                val_csv_list.append((str(index), '0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0'))
            elif labels_dict[index] == 5:  # BKL
                val_csv_list.append((str(index), '0.0', '0.0', '0.0', '0.0', '1.0', '0.0', '0.0'))
            elif labels_dict[index] == 6:  # DF
                val_csv_list.append((str(index), '0.0', '0.0', '0.0', '0.0', '0.0', '1.0', '0.0'))
            elif labels_dict[index] == 7:  # VASC
                val_csv_list.append((str(index), '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '1.0'))
        elif 18 <= value <= 24: # 13% -> 7
            # Image goes to test set
            cv2.imwrite(p_val+'%s' %str(file), image)
            if labels_dict[index] == 1:  # MEL
                test_csv_list.append((str(index), '1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0'))
            elif labels_dict[index] == 2:  # NV
                test_csv_list.append((str(index), '0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0'))
            elif labels_dict[index] == 3:  # BCC
                test_csv_list.append((str(index), '0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0'))
            elif labels_dict[index] == 4:  # AKIEC
                test_csv_list.append((str(index), '0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0'))
            elif labels_dict[index] == 5:  # BKL
                test_csv_list.append((str(index), '0.0', '0.0', '0.0', '0.0', '1.0', '0.0', '0.0'))
            elif labels_dict[index] == 6:  # DF
                test_csv_list.append((str(index), '0.0', '0.0', '0.0', '0.0', '0.0', '1.0', '0.0'))
            elif labels_dict[index] == 7:  # VASC
                test_csv_list.append((str(index), '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '1.0'))
        elif 31 <= value <= 65:
            # Image goes to training set
            cv2.imwrite(p_train+'%s' %str(file), image)
            if labels_dict[index] == 1:  # MEL
                train_csv_list.append((str(index), '1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0'))
            elif labels_dict[index] == 2:  # NV
                train_csv_list.append((str(index), '0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0'))
            elif labels_dict[index] == 3:  # BCC
                train_csv_list.append((str(index), '0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0'))
            elif labels_dict[index] == 4:  # AKIEC
                train_csv_list.append((str(index), '0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0'))
            elif labels_dict[index] == 5:  # BKL
                train_csv_list.append((str(index), '0.0', '0.0', '0.0', '0.0', '1.0', '0.0', '0.0'))
            elif labels_dict[index] == 6:  # DF
                train_csv_list.append((str(index), '0.0', '0.0', '0.0', '0.0', '0.0', '1.0', '0.0'))
            elif labels_dict[index] == 7:  # VASC
                train_csv_list.append((str(index), '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '1.0'))

train_csv_list = sorted(train_csv_list, key=itemgetter(0))
val_csv_list = sorted(val_csv_list, key=itemgetter(0))
test_csv_list = sorted(test_csv_list, key=itemgetter(0))

with open (t_train, mode = 'w') as csv_file:
    fieldnames = ['image', 'MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    for _tuple in train_csv_list:
        writer.writerow({'image': _tuple[0], 'MEL': _tuple[1], 'NV': _tuple[2], 'BCC': _tuple[3], 'AKIEC': _tuple[4], 'BKL': _tuple[5], 'DF': _tuple[6], 'VASC': _tuple[7]})

with open (t_val, mode = 'w') as csv_file:
    fieldnames = ['image', 'MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    for _tuple in val_csv_list:
        writer.writerow({'image': _tuple[0], 'MEL': _tuple[1], 'NV': _tuple[2], 'BCC': _tuple[3], 'AKIEC': _tuple[4], 'BKL': _tuple[5], 'DF': _tuple[6], 'VASC': _tuple[7]})

with open (t_test, mode = 'w') as csv_file:
    fieldnames = ['image', 'MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    for _tuple in test_csv_list:
        writer.writerow({'image': _tuple[0], 'MEL': _tuple[1], 'NV': _tuple[2], 'BCC': _tuple[3], 'AKIEC': _tuple[4], 'BKL': _tuple[5], 'DF': _tuple[6], 'VASC': _tuple[7]})
