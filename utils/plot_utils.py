"""
File contains functions to plot data.
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches


def plot_batch(batch):
    img, mask = batch
    batch_size = img.shape[0]
    fig, axes = plt.subplots(batch_size, 2, figsize=(7, 1.5 * batch_size))
    axes[0][0].set_title('Image')
    axes[0][1].set_title('Label')
    cmap = ListedColormap([
        'white', 'red', 'green', 'blue', 'cyan', 'yellow', 'magenta'
    ])

    for i, (plt_img, plt_mask) in enumerate(zip(img, mask)):
        plt_img = plt_img.permute(1, 2, 0)
        plt_mask = plt_mask.squeeze()
        axes[i][0].imshow(plt_img)
        axes[i][0].axis('off')

        presented_labels = np.unique(plt_mask)
        local_cmap = ListedColormap(cmap.colors[:len(presented_labels)])
        axes[i][1].imshow(plt_img)
        axes[i][1].imshow(plt_mask, alpha=0.3)
        axes[i][1].axis('off')

        # names = [idx2label[v] for v in presented_labels]
        # patches = [mpatches.Patch(color=color, label=name) for color, name in zip(local_cmap.colors, names)]
        # axes[i][2].legend(handles=patches, loc='right', fontsize='small')
        # axes[i][2].axis('off')

    plt.tight_layout()
    plt.show()
