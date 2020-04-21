import numpy as np
import pickle
import time
import cv2
import os
from operator import itemgetter
import csv

start_time = time.time()

class DataAugmentation:

    def __init__(self, path, image_name):
        self.path = path
        self.name = image_name
        self.image = cv2.imread(path + image_name)

    def rotate(self, image, angle=90, scale=1.0):
        '''
        Rotate the image
        :param image: image to be processed, cv2.imread(path)
        :param angle: Rotation angle in degrees. Positive values mean counter-clockwise rotation (the coordinate origin is assumed to be the top-left corner).
        :param scale: Isotropic scale factor.
        '''
        w = image.shape[1]
        h = image.shape[0]
        # rotate matrix
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        # rotate
        image = cv2.warpAffine(image, M, (w, h))
        return image

    def flip(self, image, vflip=False, hflip=False):
        '''
        Flip the image
        :param image: image to be processed
        :param vflip: whether to flip the image vertically
        :param hflip: whether to flip the image horizontally
        '''

        if hflip or vflip:
            if hflip and vflip:
                c = -1
            else:
                c = 0 if vflip else 1
            image = cv2.flip(image, flipCode=c)
        return image

    def image_augment(self, path, class_lesion, list_total):
        '''
        Create the new image with image augmentation
        :param path: the path to store the new image
        '''
        name_int = self.name[:len(self.name)-4] #tira o ".jpg"
        dim = (400, 400)
        if class_lesion == 1: # Melanoma -> 2 artificial
            img = self.image.copy()
            img_square = cv2.resize(img, dsize=dim, interpolation=cv2.INTER_AREA)
            img_rot = self.rotate(img_square)
            img_flip = self.flip(img, vflip=True, hflip=False)
            list_total.append((str(name_int), '1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0'))
            cv2.imwrite(path+'%s' %str(name_int)+'_rot.jpg', img_rot)
            list_total.append((str(name_int)+'_rot', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0'))
            cv2.imwrite(path + '%s' % str(name_int) + '_vf.jpg', img_flip)
            list_total.append((str(name_int)+'_vf', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0'))

        elif class_lesion == 2: # Nevus -> 2 artificial
            img = self.image.copy()
            img_square = cv2.resize(img, dsize=dim, interpolation=cv2.INTER_AREA)
            img_rot = self.rotate(img_square)
            img_flip = self.flip(img, vflip=True, hflip=False)
            list_total.append((str(name_int), '0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0'))
            cv2.imwrite(path+'%s' %str(name_int)+'_rot.jpg', img_rot)
            list_total.append((str(name_int)+'_rot', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0'))
            cv2.imwrite(path + '%s' % str(name_int) + '_vf.jpg', img_flip)
            list_total.append((str(name_int)+'_vf', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0'))

        elif class_lesion == 3: # BCC -> 2 artificial
            img = self.image.copy()
            img_square = cv2.resize(img, dsize=dim, interpolation=cv2.INTER_AREA)
            img_rot = self.rotate(img_square)
            img_flip = self.flip(img, vflip=True, hflip=False)
            list_total.append((str(name_int), '0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0'))
            cv2.imwrite(path+'%s' %str(name_int)+'_rot.jpg', img_rot)
            list_total.append((str(name_int)+'_rot', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0'))
            cv2.imwrite(path + '%s' % str(name_int) + '_vf.jpg', img_flip)
            list_total.append((str(name_int)+'_vf', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0'))

        elif class_lesion == 4: # AKIEC -> 2 artificial
            img = self.image.copy()
            img_square = cv2.resize(img, dsize=dim, interpolation=cv2.INTER_AREA)
            img_rot = self.rotate(img_square)
            img_flip = self.flip(img, vflip=True, hflip=False)
            list_total.append((str(name_int), '0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0'))
            cv2.imwrite(path + '%s' % str(name_int) + '_rot.jpg', img_rot)
            list_total.append((str(name_int) + '_rot', '0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0'))
            cv2.imwrite(path + '%s' % str(name_int) + '_vf.jpg', img_flip)
            list_total.append((str(name_int) + '_vf', '0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0'))

        elif class_lesion == 5: # BKL -> 2 artificial
            img = self.image.copy()
            img_square = cv2.resize(img, dsize=dim, interpolation=cv2.INTER_AREA)
            img_rot = self.rotate(img_square)
            img_flip = self.flip(img, vflip=True, hflip=False)
            list_total.append((str(name_int), '0.0', '0.0', '0.0', '0.0', '1.0', '0.0', '0.0'))
            cv2.imwrite(path + '%s' % str(name_int) + '_rot.jpg', img_flip)
            list_total.append((str(name_int)+'_rot', '0.0', '0.0', '0.0', '0.0', '1.0', '0.0', '0.0'))
            cv2.imwrite(path + '%s' % str(name_int) + '_vf.jpg', img_flip)
            list_total.append((str(name_int)+'_vf', '0.0', '0.0', '0.0', '0.0', '1.0', '0.0', '0.0'))

        elif class_lesion == 6: # DF -> 2 artificial
            img = self.image.copy()
            img_square = cv2.resize(img, dsize=dim, interpolation=cv2.INTER_AREA)
            img_flip = self.flip(img, vflip=True, hflip=False)
            img_rot = self.rotate(img_square)
            list_total.append((str(name_int), '0.0', '0.0', '0.0', '0.0', '0.0', '1.0', '0.0'))
            cv2.imwrite(path + '%s' % str(name_int) + '_vf.jpg', img_flip)
            list_total.append((str(name_int)+'_vf', '0.0', '0.0', '0.0', '0.0', '0.0', '1.0', '0.0'))
            cv2.imwrite(path+'%s' %str(name_int)+'_rot90.jpg', img_rot)
            list_total.append((str(name_int)+'_rot90', '0.0', '0.0', '0.0', '0.0', '0.0', '1.0', '0.0'))

        elif class_lesion == 7: # VASC -> 2 artificial
            img = self.image.copy()
            img_square = cv2.resize(img, dsize=dim, interpolation=cv2.INTER_AREA)
            img_flip = self.flip(img, vflip=True, hflip=False)
            img_rot = self.rotate(img_square)
            list_total.append((str(name_int), '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '1.0'))
            cv2.imwrite(path + '%s' % str(name_int) + '_vf.jpg', img_flip)
            list_total.append((str(name_int) + '_vf', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '1.0'))
            cv2.imwrite(path + '%s' % str(name_int) + '_rot90.jpg', img_rot)
            list_total.append((str(name_int) + '_rot', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '1.0'))

        else:
            print("Invalid class...")
            exit()
        return list_total

path_truth = '/home/ruben/Desktop/smalltrain2018/labels.csv' # path of labels of 80% for training
train_dir = '/home/ruben/Desktop/smalltrain2018/' # path of 80% for training
output_dir = '/home/ruben/Desktop/smalltrain2018/'
path_labels = '/home/ruben/Desktop/smalltrain2018/labels1.csv'

list_total = []
labels_dict = {}
i=0

with open(path_truth, 'r') as filee:
    reader = csv.reader(filee)
    for row in reader:
        if i == 0:
            i += 1
        if row[1] == '1.0': # MEL
            labels_dict[row[0]] = 1
        elif row[2] == '1.0': # NV
            labels_dict[row[0]] = 2
        elif row[3] == '1.0': # BCC
            labels_dict[row[0]] = 3
        elif row[4] == '1.0': # AKIEC
            labels_dict[row[0]] = 4
        elif row[5] == '1.0': # BKL
            labels_dict[row[0]] = 5
        elif row[6] == '1.0': # DF
            labels_dict[row[0]] = 6
        elif row[7] == '1.0': # VASC
            labels_dict[row[0]] = 7
filee.close()

print(labels_dict)

file_dir = "/home/acsdc/Desktop/train2018/"
for root, _, files in os.walk(file_dir):
    #print(root)
    pass

for file in files:
    ext = file[-4:]
    if ext == ".jpg":
        index = file[:len(file)-4] # tira o '.jpg'
        raw_image = DataAugmentation(root, file)
        list_total = raw_image.image_augment(output_dir, labels_dict[index], list_total)

list_total_sorted = sorted(list_total, key=itemgetter(0))

print(len(list_total))
print(len(labels_dict))


with open(path_labels, mode='w') as csv_file:
    fieldnames = ['image', 'MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    for _tuple in list_total_sorted:
        writer.writerow({'image': _tuple[0], 'MEL': _tuple[1], 'NV': _tuple[2], 'BCC': _tuple[3], 'AKIEC': _tuple[4], 'BKL': _tuple[5], 'DF': _tuple[6], 'VASC': _tuple[7]})

print("--- %s seconds ---" % (time.time() - start_time))