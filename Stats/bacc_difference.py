import matplotlib.pyplot as plt
import numpy as np

y = [0, 143, 201, 247, 293, 333, 373, 422, 482, 549, 892]
bacc = [0.6443, 0.6675, 0.6725, 0.7003, 0.7190, 0.7218, 0.7219, 0.7209, 0.7281, 0.7227, 0.7341]
wbacc = [0.7433, 0.7556, 0.7612, 0.7668, 0.7720, 0.7668, 0.7612, 0.7601, 0.7545, 0.7489, 0.7478]

x = [w for w in range(0, 105, 10)]
print(x)

plt.plot(x, y, marker='o')
plt.ylabel('No. Flat Classifier Predictions', fontsize=16)
plt.xlabel('η (%)', fontsize=16)
plt.xticks(np.arange(0, 105, 10), fontsize=16)
plt.yticks(fontsize=16)
plt.grid()
plt.show()

plt.plot(x, bacc, marker='o', label='Mixed')
hier = [0.6443]*11
plt.plot(x, hier, 'r--', marker='', label='Hier')
flat = [0.7341]*11
plt.plot(x, flat, 'g:', label='Flat')
#plt.ylabel('BACC.', fontsize=16)
plt.xlabel('η (%)', fontsize=16)

legend = plt.legend(loc='upper left', fontsize=16)
plt.xticks(np.arange(0, 105, 10), fontsize=16)
plt.grid()
plt.yticks(fontsize=16)
plt.show()


plt.plot(x, wbacc, marker='o', label='Mixed')
hier = [0.7433]*11
plt.plot(x, hier, 'r--', marker='', label='Hier')
flat = [0.7478]*11
plt.plot(x, flat, 'g:', label='Flat')
#plt.ylabel(fontsize=16)
plt.xlabel('η (%)', fontsize=16)
plt.grid()
legend = plt.legend(loc='upper left', fontsize=16)
plt.xticks(np.arange(0, 105, 10), fontsize=16)
plt.yticks(fontsize=16)
plt.show()