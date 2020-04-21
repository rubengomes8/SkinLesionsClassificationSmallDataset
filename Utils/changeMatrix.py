no_classes = 7
path = '/home/ruben/PycharmProjects/SkinLesions2/20epochs_noTL/VGG/vgg.txt'

with open(path, "r") as f:
    data = f.readlines()
    print(data)
    print("#######################")
    for i in range(0, no_classes):
        data[i] = data[i][:-1].split(" ")
        print(data[i])
    print("#######################")


data = data[0:7]
s = '['
r = 0
for row in data:
    col = 0
    s = s + '['
    for c in row:
        if col < 6:
            s = s + c + ','
        elif col == 6:
            if r < 6:
                s = s + c + '],\n'
            elif r == 6:
                s = s + c + ']'
        col += 1
    r += 1

print(s+']')