import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

from utils.data_utils import get_config

config = get_config()


def plot_batch(batch):
    img, mask = batch
    fig, axes = plt.subplots(config['BATCH_SIZE'], 3, figsize=(7, 1.5 * config['BATCH_SIZE']))
    axes[0][0].set_title('Image')
    axes[0][1].set_title('Label')
    cmap = ListedColormap(['white', 'red'])

    for i, (plt_img, plt_mask) in enumerate(zip(img, mask)):
        plt_img = plt_img.permute(1, 2, 0)
        plt_mask = plt_mask.squeeze()
        axes[i][0].imshow(plt_img)
        axes[i][0].axis('off')

        axes[i][1].imshow(plt_img)
        local_cmap = ListedColormap([cmap.colors[idx] for idx in np.unique(plt_mask)])
        axes[i][1].imshow(plt_mask, alpha=0.3, cmap=local_cmap)
        axes[i][1].axis('off')

        class_names = {idx: label for idx, label in IDX2LABEL_TRAIN.items()
                       if label != '_background_'}
        legend_patches = [mpatches.Patch(color=cmap.colors[idx], label=name)
                          for idx, name in class_names.items() if idx in np.unique(plt_mask)]
        axes[i][2].legend(handles=legend_patches, loc='right', fontsize='small')
        axes[i][2].axis('off')

    plt.tight_layout()
    plt.show()