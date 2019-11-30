from __future__ import print_function
import colorsys
import numpy as np
from keras.models import Model
from collections import namedtuple
import numpy as np

Label = namedtuple('Label', [
    'name',
    'id',
    'color'
])

labels = [Label('background', 0, (0, 0, 0)),
          Label('aeroplane', 1, (128, 0, 0)),
          Label('bicycle', 2, (0, 128, 0)),
          Label('bird', 3, (128, 128, 0)),
          Label('boat', 4, (0, 0, 128)),
          Label('bottle', 5, (128, 0, 128)),
          Label('bus', 6, (0, 128, 128)),
          Label('car', 7, (128, 128, 128)),
          Label('cat', 8, (64, 0, 0)),
          Label('chair', 9, (192, 0, 0)),
          Label('cow', 10, (64, 128, 0)),
          Label('diningtable', 11, (192, 128, 0)),
          Label('dog', 12, (64, 0, 128)),
          Label('horse', 13, (192, 0, 128)),
          Label('motorbike', 14, (64, 128, 128)),
          Label('person', 15, (192, 128, 128)),
          Label('pottedplant', 16, (0, 64, 0)),
          Label('sheep', 17, (128, 64, 0)),
          Label('sofa', 18, (0, 192, 0)),
          Label('train', 19, (128, 192, 0)),
          Label('tvmonitor', 20, (0, 64, 128)),
          Label('void', 21, (128, 64, 12))]

voc_id2label = {label.id: label for label in labels}


def class_image_to_image(class_id_image, class_id_to_rgb_map):
    """Map the class image to a rgb-color image."""
    colored_image = np.zeros(
        (class_id_image.shape[0], class_id_image.shape[1], 3), np.uint8)
    for row in range(class_id_image.shape[0]):
        for col in range(class_id_image.shape[1]):
            try:
                colored_image[row, col, :] = class_id_to_rgb_map[
                    int(class_id_image[row, col])].color
            except KeyError as key_error:
                print("Warning: could not resolve classid %s" % key_error)
    return colored_image


def color_class_image(class_image, model_name):
    """Color classed depending on the model used."""
    colored_image = class_image_to_image(class_image, voc_id2label)
    return colored_image

