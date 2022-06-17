import argparse
import numpy as np
from matplotlib import pyplot as plt

from src.config import load_config

# Load config to use in helper
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='../conf.yml', help='path to the config.yaml file')
args = parser.parse_args()
config = load_config(args.config)


def view_npy_image(npy_file, cmap='gray'):
    img_array = np.load(npy_file)
    plt.imshow(img_array, cmap=cmap)
