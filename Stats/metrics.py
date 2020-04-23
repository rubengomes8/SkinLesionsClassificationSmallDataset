no_classes = 7
# {'MEL': 94, 'NV': 621, 'BCC': 40, 'AKIEC': 24, 'BKL': 94, 'DF': 7, 'VASC': 12} total = 892
p = {0: 94/892, 1: 621/892, 2: 40/892, 3: 24/892, 4: 94/892, 5: 7/892, 6: 12/892} # val
# p = {0: 1607/892, 1: 404/892} # block 1
# p = {0: 178/1502, 1: 1324/1502} # block 2
# p = {0: 234/364, 1: 131/365} # block 3
# p = {0: 181/216, 1: 13/216, 2: 22/216} # block 4
# p = {0: 79/120, 1: 41/120} # block 5
def read_matrix(path):
    global no_classes
    with open(path, "r") as f:
        data = f.readlines()
        for i in range(0,no_classes):
            data[i] = data[i][:-1].split(" ")
            print(data[i])
    print(data)
    return data

def calculate_acc(lesion, matrix):
    global no_classes
    tp = int(matrix[lesion][lesion])
    fp = 0
    fn = 0
    tn = 0

    # fp -> percorrer coluna i
    for i in range(0, no_classes):
        if i != lesion:
            fp += int(matrix[i][lesion])

    # fn -> percorrer linha i
    for i in range(0, no_classes):
        if i != lesion:
            fn += int(matrix[lesion][i])

    # tn -> percorrer matriz e se linha & coluna != lesion somar
    for i in range(0, no_classes):
        for j in range(0, no_classes):
            if i != lesion and j != lesion:
                tn += int(matrix[i][j])

    print("Class " + str(lesion))
    print("True Positive: " + str(tp))
    print("True Negative: " + str(tn))
    print("False Positive: " + str(fp))
    print("False Negative: " + str(fn))

    acc = (tp + tn)/(tp + tn + fp + fn)
    se = (tp)/(tp + fn)
    sp = (tn)/(tn + fp)
    prec = (tp)/(tp+fp)
    return acc, se, sp, tp/(tp+fp), prec


matrix = read_matrix("/home/ruben/PycharmProjects/SkinLesions6/Flat/DN/results.txt")
print(matrix)
acc_total = 0.0
se_total = 0.0
bacc_total = 0.0
sp_total = 0.0
prec_total = 0.0
weighted_bacc = 0.0


for i in range(0, no_classes):
    print("")
    acc, se, sp, bacc_i, prec = calculate_acc(i, matrix)
    print("Sensibility for lesion " + str(i) + ": " + str(se))
    print("Specificity for lesion " + str(i) + ": " + str(sp))
    print("Precision for lesion " + str(i) + ": " + str(prec))
    print("Accuracy for lesion " + str(i) + ": " + str(acc))
    print("BACC_" + str(i) + ": " + str(bacc_i))
    acc_total += acc
    se_total += se
    bacc_total += bacc_i
    sp_total += sp
    prec_total += prec
    weighted_bacc += p[i]*se


prec_avg = prec_total/float(no_classes)
avg_acc = acc_total/float(no_classes)
avg_se = se_total/float(no_classes)
bacc = bacc_total/float(no_classes)
avg_sp = sp_total/float(no_classes)

print("")
print("Balanced sensitivity: " + str(avg_se))
print("Balanced specificity: " + str(avg_sp))
print("Weighted BACC: " + str(weighted_bacc))