import cv2
import os
from os import listdir
import sys
import argparse
from posixpath import join
import shutil

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import face_recognition as fr
import imquality.brisque as brisque

import imutils
import time
from imutils.object_detection import non_max_suppression
#from google.colab.patches import cv2_imshow
from tqdm import tqdm

from datetime import datetime
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import PIL
from PIL import Image, ImageStat

prefix = 'C:/Users/leebr/Documents/GitHub'
prefix = '/home/hume-users/leebri2n/Documents/'

# modify customized_path
#proj_path = os.path.join(os.path.join(prefix, 'hume-eshaep'), 'eshaep_gans')
#data_path = os.path.join(prefix, 'data')
proj_path = os.path.join(os.path.join(prefix, 'hume-eshaep'), 'eshaep_gans')
data_path = os.path.join(prefix, 'data')
data_path = os.path.join(prefix, 'testdata')

print('Path to project files: {}'.format(proj_path))
print('Path to data files: {}'.format(data_path))

def display_graphics(input_path, output_path=os.getcwd()):
    num_img = len(os.listdir(input_path))
    num_rowcol = int(np.sqrt(num_img))

    plt.figure(figsize=(50,50))
    for i in range(num_img):
        plt.subplot(num_rowcol+1, num_rowcol+1, i+1)

        cur_name = os.listdir(input_path)[i]
        cur_img = os.path.join(input_path, cur_name)

        img = Image.open(cur_img)
        plt.imshow(img)
        plt.axis('off')

    plt.show()

input_path = os.path.join(os.path.join('output', 'accepted'), 'coffee')
img_paths = os.path.join(data_path, input_path)

display_graphics(img_paths)
