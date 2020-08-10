import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(fig,ax,txt,colorbar, y_true,
                          y_pred):
    """
    This function prints and plots the confusion matrix.
    """
    cmap = plt.cm.Blues
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    colorbar.update_normal(im)
    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt[i+2*j].set_text(format(cm[i, j], 'd'))
            txt[i + 2 * j].set_color("white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

# https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
# https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html