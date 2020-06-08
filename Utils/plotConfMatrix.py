
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


multiclass = np.array([[51,7,6,6,14,7,3],
[137,422,8,5,26,6,17],
[1,1,22,5,3,0,8],
[0,0,12,11,1,0,0],
[16,3,9,4,52,3,7],
[0,0,1,0,2,1,3],
[0,0,2,0,0,0,10]])

class_names = ['MEL', 'NV', 'BKL', 'DF', 'VASC', 'BCC', 'AKIEC']

fig, ax = plot_confusion_matrix(conf_mat=multiclass,
                                colorbar=True,
                                show_absolute=False,
                                show_normed=True,
                                class_names=class_names)
plt.show()