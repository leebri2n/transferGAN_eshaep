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
    def __init__(self, in_dir, out_dir, type):
        self.metriclog = []
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.type = type
        self.tnrfont = {'fontname':'Times New Roman'}


        plt.figure(figsize=(9,6))
        plt.yscale('log')
        plt.grid()
        plt.xlabel("Training Length (kimg)", fontsize=12, verticalalignment='top', **self.tnrfont)
        plt.ylabel("Performance (FID)", fontsize=12, verticalalignment='bottom', **self.tnrfont)


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
        #print(score_data)

        plt.plot(np.linspace(1, len(score_data), len(score_data), endpoint=True), score_data, \
            color=col, label=lab)

        self.visualize(self.type)

    def visualize(self, type):
        #print(type)
        legend = plt.legend()
        if type == 'dataset':
            plt.title('Training Performance Across Target Datasets', \
                fontsize=20, verticalalignment='bottom', **self.tnrfont)
            legend.set_title('Target Datasets')
        elif type == 'network':
            plt.title('Training Performance Across Starting Network', \
                fontsize=20, verticalalignment='bottom', **self.tnrfont)
            legend.set_title('Pretrained Network')
        else:
            raise Exception("Invalid experiment type.")


in_dir = os.path.join(os.getcwd(), 'fidscoring')
out_dir = os.path.join(os.getcwd(), os.path.join('itea_mat', 'matplt-fig'))
print(matplotlib.get_cachedir())
#067-metric-fid50k_full-stock.json
print(in_dir)
print(out_dir)

vis_dataset = MetricDisplay(in_dir, out_dir, 'dataset')
vis_dataset.visualize_fid_line('087-metric-fid50k_full-jet.json', 'green', 'Fighter jet')
vis_dataset.visualize_fid_line('099-metric-fid50k_full-trains.json', 'lightgreen', 'Train')
vis_dataset.visualize_fid_line('088-metric-fid50k_full-latte.json', 'tomato', 'Latte art')
vis_dataset.visualize_fid_line('089-metric-fid50k_full-betta.json', 'blue', 'Betta fish')
vis_dataset.visualize_fid_line('100-metric-fid50k_full-corgi.json', 'darkblue', 'Corgi')
vis_dataset.visualize_fid_line('062-metric-fid50k_full-stock.json', 'violet', 'Beach sunset')

vis_network = MetricDisplay(in_dir, out_dir, 'network')
vis_network.visualize_fid_line('088-metric-fid50k_full-latte.json', 'tomato', 'Animal Faces')
vis_network.visualize_fid_line('093-metric-fid50k_full-latteff.json', 'blue', 'Flickr Faces')
vis_network.visualize_fid_line('095-metric-fid50k_full-lattemet.json', 'green', 'Metfaces')

plt.show()
