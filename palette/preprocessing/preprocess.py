# This file is for converting png image to jpeg

from PIL import Image
import os
import sysconfig

SAVE_PATH = sysconfig.get_config_var("SAVE_PATH")
DATASET_PATH = sysconfig.get_config_var("DATASET_PATH")


def to_jpeg_image(file_name):
    """
    This function is used for converting png image to jpeg
    :param file_name: name of the file
    :return: a PIL image (JPEG)
    """
    im = Image.open(os.path.join(DATASET_PATH, file_name))
    rgb_im = im.convert('RGB')
    rgb_im.save(os.path.join(SAVE_PATH + file_name))
    return rgb_im
