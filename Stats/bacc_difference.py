import matplotlib.pyplot as plt
import numpy as np

y = [0, 892]
bacc = [0.643, 0.658, 0.660, 0.660, 0.671, 0.674, 0.677, 0.676, 0.684, 0.694, 0.694, 0.691, 0.685, 0.690, 0.682, 0.681, 0.679, 0.673, 0.669, 0.659, 0.645]
sen = [0.7731, 0.7871, 0.7868, 0.7807, 0.7887, 0.8007, 0.8047, 0.8036, 0.8120, 0.8194, 0.8194, 0.8192, 0.8200, 0.8299, 0.8284, 0.8274, 0.8252, 0.8171, 0.8147, 0.8155, 0.8099]
spe = [0.9667, 0.9675, 0.9675, 0.9676, 0.9681, 0.9687, 0.9685, 0.9684, 0.9689, 0.9690, 0.9690, 0.9693, 0.9692, 0.9697, 0.9689, 0.9688, 0.9687, 0.9684, 0.9679, 0.9671, 0.9662]

wbacc=[0.8255, 0.8309, 0.8319, 0.8334, 0.8359, 0.8384, 0.8384, 0.8389, 0.8429, 0.8434, 0.8444, 0.8444, 0.8439, 0.8458, 0.8419, 0.8414, 0.8409, 0.8389, 0.8364, 0.8319, 0.8260]

x = [w for w in range(0, 105, 5)]
print(x)

plt.plot(x, y, marker='o')
plt.ylabel('No. Flat Classifier Predictions', fontsize=16)
plt.xlabel('η (%)', fontsize=16)
plt.xticks(np.arange(0, 105, 10), fontsize=16)
plt.yticks(fontsize=16)
plt.grid()
plt.show()

plt.plot(x, bacc, marker='o', label='Mixed')
hier = [0.643]*21
plt.plot(x, hier, 'r--', marker='', label='Hier')
flat = [0.645]*21
plt.plot(x, flat, 'g:', label='Flat')
#plt.ylabel('BACC.', fontsize=16)
plt.xlabel('η (%)', fontsize=16)

legend = plt.legend(loc='upper left', fontsize=16)
plt.xticks(np.arange(0, 105, 10), fontsize=16)
plt.grid()
plt.yticks(fontsize=16)
plt.show()


plt.plot(x, wbacc, marker='o', label='Mixed')
hier = [0.8255]*21
plt.plot(x, hier, 'r--', marker='', label='Hier')
flat = [0.8260]*21
plt.plot(x, flat, 'g:', label='Flat')
#plt.ylabel(fontsize=16)
plt.xlabel('η (%)', fontsize=16)
plt.grid()
legend = plt.legend(loc='upper left', fontsize=16)
plt.xticks(np.arange(0, 105, 10), fontsize=16)
plt.yticks(fontsize=16)
plt.show()