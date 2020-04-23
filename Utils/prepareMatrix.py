matrix = '[[ 52  10   3  11  17   0   1]\n[ 72 420  17  13  62  19  18]\n[  1   1  28   4   2   2   2]\n[  1   0   2  15   6   0   0]\n[ 22   6   7  11  48   0   0]\n[  0   0   2   0   0   5   0]\n[  0   0   1   0   0   1  10]]'
for i in range(0, len(matrix)):
    try:
        if matrix[i] == ']':
            if i != len(matrix)-1:
                matrix = matrix[0:i] + matrix[i+1:]
            else:
                matrix = matrix[0:i]
        elif matrix[i] == '[':
            if i != 0:
                matrix = matrix[0:i] + matrix[i + 1:]
            else:
                matrix = matrix[1:]
        if matrix[i] == ' ':
            if matrix[i+1] == ' ':
                if matrix[i+2] == ' ':
                    matrix = matrix[0:i] + matrix[i+2:]
                else:
                    matrix = matrix[0:i] + matrix[i+1:]
            else:
                matrix = matrix[0:i] + matrix[i:]
    except:
        print(matrix)
        exit(0)

print(matrix)
