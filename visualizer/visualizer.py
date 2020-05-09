import matplotlib as mpl
import numpy as np
import torch
from detectron2.utils.visualizer import Visualizer as dt_visualizer
from matplotlib import pyplot as plt
from matplotlib.patches import Arrow, Polygon

from ..detector.base import ObjectType
from .color import ColorManager


class Visualizer(object):

    def __init__(self, box_3d_color_sync=False):
        self.color_manager = ColorManager()
        self.box_3d_color_sync = box_3d_color_sync

    def plt_imshow(self, image, figsize=(16, 9), dpi=120, axis='off',
                   show=True):
        fig = plt.figure(figsize=figsize, dpi=dpi)
        plt.axis(axis)
        plt.imshow(image)
        if show:
            plt.show()
            plt.close(fig)

    def draw_scene(self, frame, image_roi=None, mois=None):
        image_rgb = frame.image.numpy()[:, :, ::-1]
        visualizer = dt_visualizer(image_rgb, None)
        instances = frame.instances.to('cpu')
        labels, colors, masks = [], [], []
        for obj_i in range(len(instances)):
            obj_type = ObjectType(instances.pred_classes[obj_i].item())
            obj_id = obj_i
            if instances.has('track_ids'):
                obj_id = instances.track_ids[obj_i].item()
            obj_type = obj_type.name
            score = instances.scores[obj_i] * 100
            label = '%s-%s %.0f%%' % (obj_type, obj_id, score)
            labels.append(label)
            x0, y0, x1, y1 = instances.pred_boxes.tensor[obj_i].type(torch.int)
            roi = image_rgb[y0:y1, x0:x1]
            color = self.color_manager.get_color((obj_type, obj_id), roi)
            colors.append(color)
            mask = [np.array([0, 0])]
            if instances.has('pred_masks'):
                mask = instances.pred_masks[obj_i].numpy()
            if instances.has('contours'):
                contour = instances.contours[obj_i]
                if contour is not None:
                    contour = contour[contour[:, 0] >= 0]
                    mask = [contour.cpu().numpy()]
            masks.append(mask)
        visualizer.overlay_instances(
            masks=masks, boxes=instances.pred_boxes, labels=labels,
            assigned_colors=colors)
        if instances.has('boxes_3d'):
            for obj_i in range(len(instances)):
                angle = instances.angles[obj_i]
                if angle < 0:
                    continue
                box_3d = instances.boxes_3d[obj_i]
                color = colors[obj_i] if self.box_3d_color_sync else None
                self._draw_box_3d(visualizer, box_3d, color)
        if instances.has('boxes_3d_gt'):
            for obj_i in range(len(instances)):
                box_3d = instances.boxes_3d_gt[obj_i]
                if box_3d[0, 0, 0] < 0:
                    continue
                color = colors[obj_i] if self.box_3d_color_sync else None
                self._draw_box_3d(
                    visualizer, box_3d, color, linestyle='dotted')
        self._draw_roi_mois(visualizer, image_roi, mois)
        visual_image = visualizer.get_output().get_image()
        return visual_image

    def _draw_box_3d(self, visualizer, box_3d, color=None, alpha=0.5,
                     linestyle='dashdot'):
        linewidth = max(
            visualizer._default_font_size / 4, 1) * visualizer.output.scale
        if color is None:
            colors = ['red'] * 4 + ['green'] * 4 + ['blue'] * 4
        else:
            colors = [color] * 16
        bottom = box_3d[:, :, 1].mean(axis=1).argmax()
        xs_list, ys_list = [], []
        for layer_i in [bottom, 1 - bottom]:
            for point_i in range(4):
                xs_list.append([box_3d[layer_i, point_i, 0],
                                box_3d[layer_i, (point_i + 1) % 4, 0]])
                ys_list.append([box_3d[layer_i, point_i, 1],
                                box_3d[layer_i, (point_i + 1) % 4, 1]])
        for point_i in range(4):
            xs_list.append(
                [box_3d[0, point_i, 0], box_3d[1, 3 - point_i, 0]])
            ys_list.append(
                [box_3d[0, point_i, 1], box_3d[1, 3 - point_i, 1]])
        ax = visualizer.output.ax
        for xs, ys, color in zip(xs_list, ys_list, colors):
            ax.add_line(mpl.lines.Line2D(
                xs, ys, linewidth=linewidth,
                color=color, alpha=alpha, linestyle=linestyle))

    def _draw_roi_mois(self, visualizer, roi, mois):
        ax = visualizer.output.ax
        linewidth = max(
            visualizer._default_font_size / 4, 1) * visualizer.output.scale
        if roi is not None:
            roi = Polygon(roi, fill=False, color='red')
            ax.add_patch(roi)
        if mois is not None:
            for label, moi in sorted(mois.items(), key=lambda x: int(x[0])):
                color = self.color_manager.get_color(('moi', label))
                line = Polygon(
                    moi, closed=False, fill=False, edgecolor=color,
                    label=label, linewidth=linewidth)
                ax.add_patch(line)
                dx, dy = moi[-1][0] - moi[-2][0], moi[-1][1] - moi[-2][1]
                arrow = Arrow(*moi[-2], dx, dy, edgecolor=color,
                              facecolor=color, width=5 * linewidth)
                ax.add_patch(arrow)
            ax.legend(loc=0)
