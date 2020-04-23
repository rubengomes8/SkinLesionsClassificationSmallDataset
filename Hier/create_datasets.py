from operator import itemgetter
import csv
import time
import  os
import shutil

start_time = time.time()

def create_list_labels(path_in, dict_labels, boolean):
    list_labels = []
    dict_out = {}
    path_in = path_in + '/labels.csv'
    with open(path_in, 'r') as file:
        i = 0
        reader = csv.reader(file)
        for row in reader:
            if i == 0:
                i += 1
                continue

            if boolean[1] and row[1] == '1.0':  # MEL
                list_labels.append((row[0], dict_labels[1]))
                dict_out[row[0]] = dict_labels[1]
            elif boolean[2] and row[2] == '1.0':  # NV
                list_labels.append((row[0], dict_labels[2]))
                dict_out[row[0]] = dict_labels[2]
            elif boolean[3] and row[3] == '1.0':  # BCC
                list_labels.append((row[0], dict_labels[3]))
                dict_out[row[0]] = dict_labels[3]
            elif boolean[4] and row[4] == '1.0':  # AKIEC
                list_labels.append((row[0], dict_labels[4]))
                dict_out[row[0]] = dict_labels[4]
            elif boolean[5] and row[5] == '1.0':  # BKL
                list_labels.append((row[0], dict_labels[5]))
                dict_out[row[0]] = dict_labels[5]
            elif boolean[6] and row[6] == '1.0':  # DF
                list_labels.append((row[0], dict_labels[6]))
                dict_out[row[0]] = dict_labels[6]
            elif boolean[7] and row[7] == '1.0':  # VASC
                list_labels.append((row[0], dict_labels[7]))
                dict_out[row[0]] = dict_labels[7]
            else:
                continue
    file.close()
    print(list_labels)
    list_sorted = sorted(list_labels, key=itemgetter(0))
    print(dict_out)
    print("\n")
    return list_sorted, dict_out

def write_labels_file_a(path_out, list_sorted):
    with open(path_out, mode='w') as csv_file:
        fieldnames = ['image', 'MEL', 'NMEL']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for element in list_sorted:
            if element[1] == '1':
                writer.writerow({'image': element[0], 'MEL': '0.0', 'NMEL': '1.0'})
            elif element[1] == '0':
                writer.writerow({'image': element[0], 'MEL': '1.0', 'NMEL': '0.0'})

    csv_file.close()

def write_labels_file_b(path_out, list_sorted):
    with open(path_out, mode='w') as csv_file:
        fieldnames = ['image', 'MEL', 'NV']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for element in list_sorted:
            if element[1] == '1':  # MEL
                writer.writerow({'image': element[0], 'MEL': '1.0', 'NV': '0.0'})
            elif element[1] == '0':  # NV
                writer.writerow({'image': element[0], 'MEL': '0.0', 'NV': '1.0'})

    csv_file.close()

def write_labels_file_c(path_out, list_sorted):
    with open(path_out, mode='w') as csv_file:
        fieldnames = ['image', 'BEN', 'MAL']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for element in list_sorted:
            if element[1] == '0':  # BEN
                writer.writerow({'image': element[0], 'BEN': '1.0', 'MAL': '0.0'})
            elif element[1] == '1':  # MAL
                writer.writerow({'image': element[0], 'BEN': '0.0', 'MAL': '1.0'})
            else:
                continue

    csv_file.close()

def write_labels_file_d(path_out, list_sorted):
    with open(path_out, mode='w') as csv_file:
        fieldnames = ['image', 'BKL', 'DF', 'VASC']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for element in list_sorted:

            if element[1] == '3':  # BKL
                writer.writerow({'image': element[0], 'BKL': '1.0', 'DF': '0.0', 'VASC': '0.0'})
            elif element[1] == '4':  # DF
                writer.writerow({'image': element[0], 'BKL': '0.0', 'DF': '1.0', 'VASC': '0.0'})
            elif element[1] == '5':  # VASC
                writer.writerow({'image': element[0], 'BKL': '0.0', 'DF': '0.0', 'VASC': '1.0'})
            else:
                continue

    csv_file.close()

def write_labels_file_e(path_out, list_sorted):
    with open(path_out, mode='w') as csv_file:
        fieldnames = ['image', 'BCC', 'AKIEC']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for element in list_sorted:

            if element[1] == '1':  # BCC
                writer.writerow({'image': element[0], 'BCC': '1.0', 'AKIEC': '0.0'})
            elif element[1] == '2':  # AKIEC
                writer.writerow({'image': element[0], 'BCC': '0.0', 'AKIEC': '1.0'})
            else:
                continue

    csv_file.close()

def copy_images(dict_labels, file_dir, path_out):
    for root, _, files in os.walk(file_dir):
        pass

    for file in files:
        # print(file)
        ext = file[-4:]
        if ext == '.jpg':
            index = file[:len(file) - 4]
            # print(index)
            if index in dict_labels:
                print("OLA")
                shutil.copy(file_dir + file, path_out)

boolean_a =[None, True, True, True, True, True, True, True]
boolean_b =[None, True, True, False, False, False, False, False]
boolean_c =[None, False, False, True, True, True, True, True]
boolean_d =[None, False, False, False, False, True, True, True]
boolean_e =[None, False, False, True, True, False, False, False]
dict_labels_a = {1: '0', 2: '0', 3: '1', 4: '1', 5: '1', 6: '1', 7: '1'}
dict_labels_b = {1: '1', 2: '0'}
dict_labels_c = {3: '1', 4: '1', 5: '0', 6: '0', 7: '0'}
dict_labels_d = {5: '3', 6: '4', 7: '5'}
dict_labels_e = {3: '1', 4: '2'}

path_origin = '/home/ruben/Desktop/smalltrain2018'
path_val = '/home/ruben/Desktop/smallval2018'
# CNN A
list_a, dict_a = create_list_labels(path_origin, dict_labels_a, boolean_a)
write_labels_file_a('/home/ruben/Desktop/HierSmall/a/labels.csv', list_a)

list_a, dict_a = create_list_labels(path_val, dict_labels_a, boolean_a)
write_labels_file_a('/home/ruben/Desktop/HierSmall/val/a/labels.csv', list_a)

# CNN B
'''
path_b = '/home/ruben/Desktop/HierSmall/b'
list_b, dict_b = create_list_labels(path_origin, dict_labels_b, boolean_b)
write_labels_file_b(path_b+'/labels.csv', list_b)
copy_images(dict_b, path_origin+'/', path_b)

path_b = '/home/ruben/Desktop/HierSmall/val/b'
list_b, dict_b = create_list_labels(path_val, dict_labels_b, boolean_b)
write_labels_file_b(path_b+'/labels.csv', list_b)
copy_images(dict_b, path_val+'/', path_b)

# CNN C
path_c = '/home/ruben/Desktop/HierSmall/c'
list_c, dict_c = create_list_labels(path_origin, dict_labels_c, boolean_c)
write_labels_file_c(path_c+'/labels.csv', list_c)
copy_images(dict_c, path_origin+'/', path_c)

path_c = '/home/ruben/Desktop/HierSmall/val/c'
list_c, dict_c = create_list_labels(path_val, dict_labels_c, boolean_c)
write_labels_file_c(path_c+'/labels.csv', list_c)
copy_images(dict_c, path_val+'/', path_c)

# CNN D
path_d = '/home/ruben/Desktop/HierSmall/d'
list_d, dict_d = create_list_labels(path_origin, dict_labels_d, boolean_d)
write_labels_file_d(path_d+'/labels.csv', list_d)
copy_images(dict_d, path_origin+'/', path_d)

path_d = '/home/ruben/Desktop/HierSmall/val/d'
list_d, dict_d = create_list_labels(path_val, dict_labels_d, boolean_d)
write_labels_file_d(path_d+'/labels.csv', list_d)
copy_images(dict_d, path_val+'/', path_d)

# CNN E
path_e = '/home/ruben/Desktop/HierSmall/e'
list_e, dict_e = create_list_labels(path_origin, dict_labels_e, boolean_e)
write_labels_file_e(path_e+'/labels.csv', list_e)
copy_images(dict_e, path_origin+'/', path_e)

path_e = '/home/ruben/Desktop/HierSmall/val/e'
list_e, dict_e = create_list_labels(path_val, dict_labels_e, boolean_e)
write_labels_file_e(path_e+'/labels.csv', list_e)
copy_images(dict_e, path_val+'/', path_e)
'''
print("--- %s seconds ---" % (time.time() - start_time))