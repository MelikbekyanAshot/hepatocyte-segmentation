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
    fig, axes = plt.subplots(batch_size, 3, figsize=(7, 1.5 * batch_size))
    axes[0][0].set_title('Image')
    axes[0][1].set_title('Label')
    cmap = ListedColormap(['white', 'red', 'green', 'blue', 'cyan', 'yellow', 'magenta'])

    for i, (plt_img, plt_mask) in enumerate(zip(img, mask)):
        plt_img = plt_img.permute(1, 2, 0)
        plt_mask = plt_mask.squeeze()
        axes[i][0].imshow(plt_img)
        axes[i][0].axis('off')

        axes[i][1].imshow(plt_img)
        local_cmap = ListedColormap([cmap.colors[idx] for idx in np.unique(plt_mask)])
        axes[i][1].imshow(plt_mask, alpha=0.3, cmap=local_cmap)
        axes[i][1].axis('off')

        idx2label = {
            1: 'balloon_dystrophy',
            2: 'hepatocyte_inclusion',
            3: 'hepatocyte_non_nuclei',
            4: 'hepatocyte_relatively_normal',
            5: 'hepatocyte_steatosis',
            6: 'mesenchymal_cells'}
        class_names = {idx: label for idx, label in idx2label.items()}
        legend_patches = [mpatches.Patch(color=cmap.colors[idx], label=name)
                          for idx, name in class_names.items() if idx in np.unique(plt_mask)]
        axes[i][2].legend(handles=legend_patches, loc='right', fontsize='small')
        axes[i][2].axis('off')

    plt.tight_layout()
    plt.show()
