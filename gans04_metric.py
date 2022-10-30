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
import matplotlib.font_manager
import matplotlib.colors as mcolors

import time
from tqdm import tqdm

from datetime import datetime
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class MetricDisplay():
    def __init__(self, in_dir, out_dir):
        self.metriclog = []
        self.in_dir = in_dir
        self.out_dir = out_dir

        plt.figure(figsize=(10,8))
        plt.yscale('log')
        plt.title('Transfer Learning Performance', fontname="Times")
        plt.xlabel("Training Length (kimg)")
        plt.ylabel("Performance (FID)")


    def read_json(self, in_file):
        score_data = []
        with open(in_file, 'r') as fid_file:
            for snap in fid_file:
                fidkimg = json.loads(snap)
                #print(fidkimg, type(fidkimg))
                #print(fidkimg['results']['fid50k_full'])
                score_data.append(fidkimg['results']['fid50k_full'])

        return score_data

    def visualize_fid_line(self, in_file, col, lab):
        score_data = self.read_json(os.path.join(self.in_dir, in_file))
        print(score_data)

        plt.plot(np.linspace(1, len(score_data), len(score_data), endpoint=True), score_data, \
            color=col, label=lab)
        self.visualize()

    def visualize(self):
        plt.legend()


in_dir = os.path.join(os.getcwd(), 'fidscoring')
out_dir = os.path.join(os.getcwd(), os.path.join('itea_mat', 'matplt-fig'))
print(matplotlib.get_cachedir())
#067-metric-fid50k_full-stock.json
print(in_dir)
print(out_dir)

vis = MetricDisplay(in_dir, out_dir)
vis.visualize_fid_line('062-metric-fid50k_full-stock.json', 'orange', 'Beach sunset')
vis.visualize_fid_line('087-metric-fid50k_full-jet.json', 'coral', 'Fighter jet')
vis.visualize_fid_line('088-metric-fid50k_full-latte.json', 'sienna', 'Latte art')
vis.visualize_fid_line('089-metric-fid50k_full-betta.json', 'aqua', 'Betta fish')
vis.visualize_fid_line('099-metric-fid50k_full-trains.json', 'tomato', 'Train')
vis.visualize_fid_line('100-metric-fid50k_full-corgi.json', 'green', 'Corgi')
plt.show()
