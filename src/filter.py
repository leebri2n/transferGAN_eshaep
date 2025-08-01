# ~~~~~~~~~~~~~~~~~~~~ Packages ~~~~~~~~~~~~~~~~~~~~
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


class Pipeline():
    """A class containing all necessary functions used for data filtration"""

    def __init__(self, proj_path='', data_path ='', input_folder='input', output_folder='output', \
               size=300, blur_thresh=30, text_thresh=0.9999, text_area=0.1) -> None:
        #Simple paths
        self.proj_path = proj_path
        self.data_path = data_path
        self.input_path = os.path.join(data_path, input_folder)
        self.output_path = os.path.join(data_path, output_folder)

        #Default conditions
        self.valid_ext = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.gif',\
                   '.eps', '.raw', '.cr2', '.nef', '.orf', '.sr2','.webp']
        self.size = size
        self.blur_thresh = blur_thresh
        self.text_thresh = text_thresh
        self.text_area = text_area
        self.blurry_input = []
        self.text_input = []

        #Storage
        self.allfiles_input = [] #directory paths
        self.imgfiles_input = [] #directory paths
        self.acc_img = [] #directory paths
        self.rej_img = [] #directory paths
        self.display_img = [] #PIL Image objects

        #Logging
        self.input_widths = [] #int
        self.input_heights = [] #int
        self.input_formats = set() #str
        self.img_dict = dict()

        self.start = time.time()
        self.end = time.time()

    def filter(self, input_path, output_path, size, v=False):
        """
          The main filtering function including a filter walk and standardization
          of accepted image data.

          Parameters:
            input_path: Directory path to input images.
            output_path: Directory path to output images.
            size: Desired pixel size of (square) output images.

          Return:
            The dictionary of input data metadata.
        """
        self.start = time.time()

        self.walk(self.input_path, self.allfiles_input) #Possibly unnecessary
        print("NUMBER OF POTENTIAL INPUTS: ", str(len(self.allfiles_input)))

        accept_path = os.path.join(output_path, 'accepted')
        reject_path = os.path.join(output_path, 'rejected')

        try:
            shutil.rmtree(accept_path)
            shutil.rmtree(reject_path)
        except:
            print("Director(ies) not found! Creating new output folders...")

        os.makedirs(accept_path, exist_ok=True)
        os.makedirs(reject_path, exist_ok=True)

        # ~~~~~~~~~~~~~ Rejection criteria ~~~~~~~~~~~~~
        self.reject_image('0', input_path, output_path)
        self.reject_image('1', accept_path, output_path)
        self.reject_image('2', accept_path, output_path)
        self.reject_image('3', accept_path, output_path)
        self.reject_image('4', accept_path, output_path)
        self.reject_image('5', accept_path, output_path)

        self.walk(accept_path, self.acc_img)
        img_num = 0
        print("Standardizing images...")
        pbar = tqdm(self.acc_img)
        for img in self.acc_img:
            # ~~~~~~~~~~~~~~~~ Start Loop ~~~~~~~~~~~~~~~~
            cur_name = os.path.basename(os.path.normpath(img))
            new_tag = cur_name.split('-')[0]
            tag_path = os.path.join(accept_path, new_tag)

            os.makedirs(tag_path, exist_ok=True)
            file_num = len(os.listdir(tag_path))

            out_name = new_tag+'-'+str(file_num).zfill(6)+'.jpg'
            if v: print(cur_name, "ACCEPTED as", out_name)

            #assemble dict ~~~
            self.add_entry(img, out_name, self.img_dict)

            #Image handling ~~~
            img_standard = self.img_standardize(img, self.size)
            try:
                img_standard.save(os.path.join(tag_path, out_name))
            except:
                if v: print("Converting", cur_name, "to .jpg format...")
                img_standard = img_standard.convert("RGB")
                # Save-able as jpg now!
                img_standard.save(os.path.join(tag_path, out_name))

            self.display_img.append(img_standard)
            os.remove(os.path.join(accept_path, cur_name))

            #Update valid img number
            if v: print("Standardizing image", str(img_num), " of ", str(len(self.acc_img)), cur_name)
            img_num += 1
            pbar.update()
            # ~~~~~~~~~~~~~~~~ End Loop ~~~~~~~~~~~~~~~~
        pbar.close()
        self.end = time.time()

        #save to json to /output ~~~
        self.writejson(self.output_path)

        #save log file
        self.writelog(self.imgfiles_input, self.input_widths, self.input_heights,
                 self.input_formats, len(self.acc_img))

        # Peek at cropped images
        self.display_accepted(self.display_img)

        # Returns dictionary
        return self.img_dict

    #~~~~~~~~~~~~ Helper functions ~~~~~~~~~~~~~~~~~
    def reject_image(self, criteria, input_path, output_path, v=False):
        """
          Function that handles rejection of images for a single filter.
          Specified criterion engages the associated specific filter.

        Parameters:
          criteria: A string of the filter number being requested.
          input_path: Directory path to input images.
          output_path: Directory path to output images.
          v: Verbose print execution.

        Returns:
          Boolean of whether or not an image passes a filter. True if it
          passes, false, if it does not.
        """

        reject_path = os.path.join(output_path, "rejected")
        accept_path = os.path.join(output_path, "accepted")

        input_list = []
        self.walk(input_path, input_list)

        if criteria == '0': # Is image?
            print("Identifying non-image files...")
            pbar = tqdm(input_list)
            for cur_img in input_list:
                cur_tag = os.path.basename(os.path.dirname(cur_img))
                cur_name = os.path.basename(os.path.normpath(cur_img))
                cur_ext = os.path.splitext(cur_img)[1]

                self.input_formats.add(cur_ext)

                if cur_ext not in self.valid_ext:
                    if v: print(cur_name, " REJECTED.", " Warning: Not an image.")
                    shutil.copy(cur_img, os.path.join(reject_path, 'rjnotim_'+cur_tag+'-'+cur_name))
                else:
                    shutil.copy(cur_img, os.path.join(accept_path, cur_tag+'-'+cur_name))
                pbar.update()
            pbar.close()

        if criteria == '1': # High resolution?
            print("Checking for low resolution...")
            pbar = tqdm(input_list)

            for cur_img in input_list:
                cur_name = os.path.basename(os.path.normpath(cur_img))
                img = Image.open(cur_img)
                width, height = img.size

                self.imgfiles_input.append(cur_img)
                self.input_widths.append(width)
                self.input_heights.append(height)

                if (width <= 0.8*self.size) or (height <= int(0.8*self.size)):
                    if v: print(cur_name, " REJECTED.", " Warning: Resolution too low.")
                    shutil.move(cur_img, os.path.join(reject_path, 'rjres_'+cur_name))
                else:
                    shutil.move(cur_img, os.path.join(accept_path, cur_name))

                pbar.update()
            pbar.close()

        if criteria == '2': # Not grayscale?
            print("Checking for grayscale...")
            pbar = tqdm(input_list)
            for cur_img in input_list:
                cur_name = os.path.basename(os.path.normpath(cur_img))
                img = cv2.imread(cur_img)

                if len(img.shape) < 3:
                    if v: print(cur_name, "REJECTED.", "Warning: Grayscale image.")
                    shutil.move(cur_img, os.path.join(reject_path, 'rjgray_'+cur_name))
                else:
                    shutil.move(cur_img, os.path.join(accept_path, cur_name))

                pbar.update()
            pbar.close()

        if criteria == '3': # Blurry?
            print("Blur threshold", self.blur_thresh)
            self.blur_detection(input_path, output_path, v=v, thresh=self.blur_thresh, split=True)

        if criteria == '4': # Text ?
            print("Allowed text area", self.text_area)
            self.text_detection(input_path, output_path, v=v, confidence=self.text_thresh, allowed_area = self.text_area)

        if criteria == '5': #Faces?
            self.face_detection(input_path, output_path, v)

        # Passable images for the various criteria
        return os.listdir(accept_path)


    def walk(self, input_path, file_list):
        """
            Searches for and collects directory pathways to files.

            Parameters:
                input_path: Directory path to input images.
                file_list: A list to store full-length directory pathways
                to files.

            Returns:
                file_list: A list of full-length pathways to searched files.
        """
        input_list = os.listdir(input_path)
        skip_list = ['classifications', 'displayfolder', 'zipdatasets']
        for item in input_list:
            if item in skip_list:
                continue

            if os.path.isdir(os.path.join(input_path,item)):
                self.walk(os.path.join(input_path, item), file_list)
            else:
                file_list.append(os.path.join(input_path, item))

        return file_list

    def display_accepted(self, display_img):
        """
          Displays images that passed all filter criteria.

          Parameters:
            displayimg: List of Pillow Image objects to plot.
        """
        plt.figure(figsize=(50,50))
        assert(len(display_img) == len(self.acc_img))

        for i in range(len(display_img)):
          plt.subplot(len(display_img), len(display_img), i+1)
          plt.axis('off')

        plt.show()

    def img_standardize(self, cur_img, size):
        """
          Crops and resizes an image to the desired size.

          Parameters:
            cur_img: The directory path to the image being standardized.
            size: The desired size to which to resize the input image.

          Returns:
            img: Pillow Image object of the standardized image.
        """
        img = Image.open(cur_img)
        width, height = img.size

        if (width > height):
            left_start = (width-height)/2
            img = img.crop((left_start, 0, left_start+height,height))
        elif (height > width):
            top_start = (height-width)/2
            img = img.crop((0, top_start, width, top_start+width))

        img = img.resize((size,size))
        return img

    def blur_detection(self, input_path, output_path, thresh=30, split=True, v=True):
        """
          Helper function that classifies images in an input pathway as blurry
          or sharp and relocates them accordingly.

          Parameters:
            input_path: Directory path to input images.
            output_path: Directory path to output images.
            thresh: Threshold at which to classify as "blurry." See "LPC scores."
            split: Boolean for whether or not to copy blurry images into a
              "blurry" subdirectory. True for subdirectory split, false, for
              no changes.
            verbose: Verbose execution.

          Returns:
            A list of the blurry image filenames.
        """
        reject_path = os.path.join(output_path, "rejected")
        accept_path = os.path.join(output_path, "accepted")
        blurry_path = reject_path
        sharp_path = accept_path

        input_list = []
        self.walk(input_path, input_list)
        input_list.sort()

        print("Identifying blurry images...")
        lpcs = []
        pbar = tqdm(input_list)
        # ~~~~~~~~~~~~~~~ Start loop ~~~~~~~~~~~~~~~
        for img_path in input_list:
            cur_name = os.path.basename(os.path.normpath(img_path))

            if v:
                print("Processing %s" % img_path)

            img_cv2 = cv2.imread(img_path)
            img_gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
            lpc = cv2.Laplacian(img_gray, cv2.CV_64F).var()
            lpcs.append(lpc)

            if v: print(lpc)
            if split:
                if lpc >= thresh:
                    cv2.imwrite(os.path.join(sharp_path, cur_name), img_cv2)
                else:
                    if v: print(cur_name, " REJECTED.", " Warning: Blurry image.")
                    cv2.imwrite(os.path.join(blurry_path, 'rjblur_'+cur_name), img_cv2)
                    os.remove(os.path.join(sharp_path, cur_name))

        # ~~~~~~~~~~~~~~~ End loop ~~~~~~~~~~~~~~~
            pbar.update()
        pbar.close()

        return os.listdir(accept_path)

    def text_detection(self, input_path, output_path, confidence, allowed_area=0.25, resize=640, v=False):
        """
          Helper function that detects supposed text that appears in images from
          an input pathway and relocates them accordingly.

          Parameters:
            input_path: Directory path to input images.
            output_path: Directory path to output images.
            confidence: Confidence level required to deem a region as a
                text classification.
            allowed_area: The pixel area ratio of text regions to the full image.
            resize: Specific pixel size to which to resize in the text
                classificiation operation.

          Returns:
            The number of distinct text objects recognized.
        """
        # Testing purposes
        class_path = os.path.join(self.input_path, 'classifications')
        text_path = os.path.join(class_path, 'textdetection')
        #shutil.rmtree(text_path)
        os.makedirs(text_path, exist_ok=True)
        os.makedirs(os.path.join(os.path.join(text_path,\
         'acceptable')), exist_ok=True)
        os.makedirs(os.path.join(os.path.join(text_path,\
         'rejectable')), exist_ok=True)
        reject_path = os.path.join(output_path, "rejected")
        accept_path = os.path.join(output_path, "accepted")

        input_list = []
        self.walk(input_path, input_list)
        input_list.sort()
        model_path = os.path.join(os.path.join(self.proj_path, 'tools'), \
        'frozen_east_text_detection.pb')

        print("Identifying text in images...")
        pbar = tqdm(input_list)
        for cur_img in input_list:
            cur_name = os.path.basename(os.path.normpath(cur_img))
            wid_new = resize
            hei_new = resize #Adjustable

            try:
                img = cv2.imread(cur_img)
                img_orig = img.copy()
            except:
                continue

            (hei_orig, wid_orig) = img.shape[:2]
            wid_ratio = wid_orig / float(wid_new)
            hei_ratio = hei_orig / float (hei_new)

            img = cv2.resize(img, (wid_new, hei_new))
            #Grab dimensions?

            layerNames = ["feature_fusion/Conv_7/Sigmoid",
                "feature_fusion/concat_3"]

            net = cv2.dnn.readNet(model_path)
            blob = cv2.dnn.blobFromImage(img, 1.0, (wid_new, hei_new), \
                (123.68, 116.78, 103.94), swapRB=True, crop=False)
            net.setInput(blob)
            (scores, geometry) = net.forward(layerNames)

            (numRows, numCols) = scores.shape[2:4]
            rects = []
            rects_areas = []
            confidences = []

            for y in range(0, numRows):
                # extract the scores (probabilities), followed by the geometrical data
                scoresData = scores[0, 0, y]
                xData0 = geometry[0, 0, y]
                xData1 = geometry[0, 1, y]
                xData2 = geometry[0, 2, y]
                xData3 = geometry[0, 3, y]
                anglesData = geometry[0, 4, y]

                # loop over the number of columns
                for x in range(0, numCols):
                    # if our score does not have sufficient probability, ignore it
                    if scoresData[x] < confidence:
                        continue
                    # compute the offset factor as our resulting feature maps will 4x smaller
                    (offsetX, offsetY) = (x * 4.0, y * 4.0)
                    # extract the rotation angle for the prediction and then compute sin, cos
                    angle = anglesData[x]
                    cos = np.cos(angle)
                    sin = np.sin(angle)
                    # use the geometry volume to derive the width and height of pred box
                    h = xData0[x] + xData2[x]
                    w = xData1[x] + xData3[x]
                    # compute both the starting and ending (x, y)-coordinates for pred box
                    endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                    endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                    startX = int(endX - w)
                    startY = int(endY - h)
                    # add the bounding box coordinates and probability score to resp. lists
                    rects.append((startX, startY, endX, endY))
                    confidences.append(scoresData[x])

            boxes = non_max_suppression(np.array(rects), probs=confidences)
            for (startX, startY, endX, endY) in boxes:
                #area check
                area = np.abs(endX - startX) * np.abs(endY-startY)
                rects_areas.append(area)
                #print(endX-startX, endY-startY)

                # scale the bounding box coordinates based on the respective ratios
                startX = int(startX * wid_ratio)
                startY = int(startY * hei_ratio)
                endX = int(endX * wid_ratio)
                endY = int(endY * hei_ratio)
                # draw the bounding box on the image
                cv2.rectangle(img_orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

                #area check?
            text_area = np.sum(rects_areas)
            total_area = wid_orig * hei_orig
            #print("TEXT TO IMAGE RATIO:", text_area/total_area)

            tti_ratio = text_area / total_area
            if tti_ratio >= allowed_area:
                print("TEXT ENCOUNTERED")
                if v: print(cur_name, " REJECTED.", " Warning: Significant text detected.")
                shutil.move(cur_img, os.path.join(reject_path, 'rjtext_'+cur_name))
                cv2.imwrite(os.path.join(os.path.join(text_path, 'rejectable'), \
                    str(tti_ratio)[2:]+'-'+cur_name), img_orig)
            else:
                shutil.move(cur_img, os.path.join(accept_path, cur_name))
                cv2.imwrite(os.path.join(os.path.join(text_path, 'acceptable'), \
                    str(tti_ratio)[2:]+'-'+cur_name), img_orig)

            pbar.update()
        pbar.close()
        return os.listdir(accept_path)

    def face_detection(self, input_path, output_path, v=False):
        """
          Helper function that detects human faces in images from an input
          pathway and relocates them accordingly.

          Parameters:
            input_path: Directory path to input images.
            output_path: Directory path to output images.

          Returns:
            The directory path of accepted images.
        """
        reject_path = os.path.join(output_path, "rejected")
        accept_path = os.path.join(output_path, "accepted") #temporary

        input_list = []
        self.walk(input_path, input_list)
        input_list.sort()

        print("Identifying faces in images...")
        pbar = tqdm(input_list)
        for cur_img in input_list:
            cur_name = os.path.basename(os.path.normpath(cur_img))
            cur_img_fr = fr.load_image_file(cur_img)
            face_landmarks = fr.face_locations(cur_img_fr, model='hog')

            if len(face_landmarks) != 0:
                if v: print(cur_name," REJECTED.", " Warning: Faces detected.")
                shutil.move(cur_img, os.path.join(reject_path, 'rjface_'+cur_name))
            else:
                shutil.move(cur_img, os.path.join(accept_path, cur_name))

            pbar.update()
        pbar.close()

        return os.listdir(accept_path)

    def writejson(self, output_path):
        try:
            metadata = open(os.path.join(output_path, 'input_metadata.json'), 'w')
            json.dump(self.img_dict, metadata, indent=1)
            metadata.close()
            print('input_metadata.json written successfully.')
        except:
            print('Unable to write input_metadata.json.')

    def writelog(self, imgfiles_inputlist, widths, heights, formats, acc):
        """
          Writes a text file containing statistics about the input dataset.

          Params:
            imgfiles_inputlist: list of valid input image paths
            widths: list of image widths
            heights: list of image heights
            formats: set of image formats that appear
            acc: number of accepted images
        """
        print(acc)
        datetime_exc = datetime.now() #Date and time of finished execution
        num = len(imgfiles_inputlist)
        minw = np.min(widths)
        maxw = np.max(widths)
        minh = np.min(heights)
        maxh = np.max(heights)
        avw = np.average(widths)
        avh = np.average(heights)
        rej = num - acc
        percentage = (acc / (num))

        #Write stuff to file
        log_path = os.path.join(self.proj_path, 'logs')
        with open(os.path.join(log_path, \
            (datetime_exc.strftime("%m%d%Y_%H%M%S")))+'log.txt', 'w') as f:
            #START TEXT
            f.write(join("Latest successful execution finished at:", \
                datetime_exc.strftime("%m/%d/%Y, %H:%M:%S")))
            f.write('\n')
            f.write('\n')

            f.write("Number of valid images: %s" % str(num))
            f.write('\n')
            f.write('\n')

            f.write('Minimum input width: %s' % str(minw))
            f.write('\n')
            f.write('Maximum input width: %s' % str(maxw))
            f.write('\n')
            f.write('\n')

            f.write('Minimum input height: %s' % str(minh))
            f.write('\n')
            f.write('Maximum input height: %s' % str(maxh))
            f.write('\n')
            f.write('\n')

            f.write('Average input width: %s' % str(avw))
            f.write('\n')
            f.write('Average input height: %s' % str(avh))
            f.write('\n')
            f.write('\n')

            f.write(join('Percentage of accepted images:', \
                join(str(percentage*100), '%')))
            f.write('\n')
            f.write('\n')

            f.write('Total execution time: %s' \
                % str((self.end-self.start)/60) + ' minutes')
            f.write('\n')
            f.write('\n')

            # END TEXT
            print('log.txt successfully written.')

    def add_entry(self, img, img_outname, img_dict):
        """
          Adds metadata entry to the metadictionary to be written.

          Params:
            img_dict: The dictionary to which to store metadata in.
            img: The directory path to the image file.
            img_outname: The accepted image's output file name.
        """
        res = Image.open(img).size
        ext = os.path.splitext(img)[1]
        par = os.path.abspath(os.path.join(img, os.pardir))
        self.img_dict[img_outname]=dict()
        self.img_dict[img_outname]['original_resolution'] = res
        self.img_dict[img_outname]['original_format'] = ext
        self.img_dict[img_outname]['original_path'] = par

# ~~~~~~~~~~~~~~~~~~~~ Pathing ~~~~~~~~~~~~~~~~~~~~
prefix = 'C:/Users/leebr/Documents/GitHub'
prefix = '/home/hume-users/leebri2n/Documents/'

# modify customized_path
#proj_path = os.path.join(os.path.join(prefix, 'hume-eshaep'), 'eshaep_gans')
#data_path = os.path.join(prefix, 'data')
proj_path = os.path.join(os.path.join(prefix, 'eshaep_gans'))
data_path = os.path.join(prefix, 'testdata')
data_path = os.path.join(prefix, 'data')

print('Path to project files: {}'.format(proj_path))
print('Path to data files: {}'.format(data_path))

input_path = os.path.join(data_path, 'input') #entire input
output_path = os.path.join(data_path, 'output')

#~~~~~~~~~~~~~~~~~~ Filtering Job ~~~~~~~~~~~~~~~~~~~
start_t = time.time()
print("TIME OF EXECUTION", datetime.now())

pipeline = Pipeline(proj_path=proj_path, input_folder=input_path, output_folder=output_path, \
    size=512, blur_thresh=45, text_thresh=0.99, text_area=0.005)

pipeline.filter(input_path = pipeline.input_path, output_path = pipeline.output_path, size=pipeline.size)

end_t = time.time()
print("FILTERING EXECUTION TIME: ", str((end_t-start_t)/60), "MINUTES")

pipeline.filter(input_path = pipeline.input_path, output_path = pipeline.output_path, size=pipeline.size)
