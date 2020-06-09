
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


multiclass = np.array([[57,14,5,5,13,0,0],
[60,490,13,8,35,9,6],
[0,1,29,3,2,2,3],
[0,0,1,19,2,2,0],
[14,9,2,11,56,2,0],
[0,0,0,1,0,5,1],
[0,1,0,0,0,0,11]])

class_names = ['MEL', 'NV', 'BKL', 'DF', 'VASC', 'BCC', 'AKIEC']

fig, ax = plot_confusion_matrix(conf_mat=multiclass,
                                colorbar=True,
                                show_absolute=False,
                                show_normed=True,
                                class_names=class_names)
plt.show()