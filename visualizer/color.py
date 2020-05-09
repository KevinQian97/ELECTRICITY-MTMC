import numpy as np

PALETTE_HEX = [
    "#000000", "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
    "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87",
    "#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80",
    "#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100",
    "#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F",
    "#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09",
    "#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66",
    "#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C",
    "#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81",
    "#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00",
    "#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700",
    "#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329",
    "#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C",
    "#83AB58", "#001C1E", "#D1F7CE", "#004B28", "#C8D0F6", "#A3A489", "#806C66", "#222800",
    "#BF5650", "#E83000", "#66796D", "#DA007C", "#FF1A59", "#8ADBB4", "#1E0200", "#5B4E51",
    "#C895C5", "#320033", "#FF6832", "#66E1D3", "#CFCDAC", "#D0AC94",
    "#7ED379", "#012C58"]


def _parse_hex_color(s):
    r = int(s[1:3], 16) / 256
    g = int(s[3:5], 16) / 256
    b = int(s[5:7], 16) / 256
    return (r, g, b)


COLORS = [*map(_parse_hex_color, PALETTE_HEX)]
PALETTE_RGB = np.asarray(COLORS)
COLOR_DIFF_WEIGHT = np.asarray((3, 4, 2))


class ColorManager(object):

    def __init__(self):
        self.color_index_scores = np.zeros(len(COLORS))
        self.color_map = {}

    def get_color(self, object_id, roi=None):
        if object_id not in self.color_map:
            if roi is None:
                color_id = self.color_index_scores.argmax()
                self.color_map[object_id] = COLORS[color_id]
                self.color_index_scores[color_id] -= 1
            else:
                mean_color = roi.mean(axis=(0, 1)) / 256
                color_scores = (np.square(PALETTE_RGB - mean_color) *
                                COLOR_DIFF_WEIGHT).sum(axis=1)
                color_id = (color_scores + self.color_index_scores).argmax()
                self.color_index_scores[color_id] -= 1
                self.color_map[object_id] = COLORS[color_id]
        return self.color_map[object_id]
