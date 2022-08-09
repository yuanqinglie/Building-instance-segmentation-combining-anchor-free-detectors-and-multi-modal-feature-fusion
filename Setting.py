
import os
import sys
import random
import itertools
import colorsys

import numpy as np
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
import IPython.display

import rasterio
import gdal
from rasterio.plot import show
import cv2.imshow
os.chdir('/Builings instance segmentation')

! pip install rasterio-1.2.8-cp37-cp37m-manylinux1_x86_64.whl
! pip install h5py-2.10.0-cp37-cp37m-manylinux1_x86_64.whl

#%tensorflow_version 2.0
import tensorflow as tf
tf.__version__
