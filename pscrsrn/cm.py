import matplotlib.pyplot as plt
import numpy as np

def plot(cm, names, cmap='viridis'):
    n_classes = cm.shape[0]
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap=cmap, interpolation='nearest')
    cmap_min = im.cmap(0)
    cmap_max = im.cmap(256)
    thresh = (cm.max() + cm.min()) / 2
    for i in range(n_classes):
        for j in range(n_classes):
            color = cmap_max if cm[i,j] < thresh else cmap_min
            ax.text(j, i, format(cm[i,j], '.2%'), ha='center', va='center', color=color)
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks = np.arange(n_classes),
        yticks = np.arange(n_classes),
        xticklabels = names,
        yticklabels = names,
        xlabel = 'Predicted label',
        ylabel = 'True label')
    ax.set_ylim((n_classes - 0.5, -0.5))
    return fig
