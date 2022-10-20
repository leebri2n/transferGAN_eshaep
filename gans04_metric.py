import os
import sys
import cv2
import shutil
import argparse
from os import listdir
from posixpath import join

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time
from tqdm import tqdm

from datetime import datetime
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class MetricDisplay():
    def __init__(self):
        self.start = 0

    def read_json(self, path):
        with open(path, 'r') as json_file:
            fid50k = json.load(json_file)
            print(type(fid50k))

        return fid50k

m = MetricDisplay()
print(m.read_json('./metric-fid50k_full.json'))
