import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d import Axes3D


def draw_heatmaps_3d(heatmaps, region, region_mask=None):
    height, width = heatmaps.shape[1:]
    xs, ys = np.meshgrid(np.arange(width), np.arange(height))
    if region_mask is not None:
        region_mask = np.logical_not(region_mask)
    for heatmap in heatmaps:
        heatmap = heatmap.copy()
        if region_mask is not None:
            heatmap[region_mask] = 0
        fig = plt.figure(figsize=(30, 10))
        ax = fig.add_subplot(121, projection='3d')
        ax.plot_surface(xs, ys, heatmap, cmap='jet')
        y1, y2 = ax.get_ylim()
        ax.set_ylim(y2, y1)
        ax = fig.add_subplot(122)
        ax.imshow(heatmap, cmap='jet')
        p = Polygon(region, fill=False)
        ax.add_patch(p)
        plt.show()
        plt.close(fig)
