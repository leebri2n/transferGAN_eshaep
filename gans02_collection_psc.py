import cv2
import os
from os import listdir
import sys
import argparse
from posixpath import join
import shutil
import json
import numpy as np
import matplotlib.pyplot as plt
import time
import instaloader
from datetime import datetime
from itertools import dropwhile, takewhile
import csv
import os
from tqdm import tqdm

#from instascrape import *

#Append the directory to your python path
prefix = 'C:/Users/leebr/Documents/GitHub/'
prefix = '/home/hume-users/leebri2n/Documents/'

# modify customized_path
customized_path = 'hume-rsc/eshaep_gans/'
proj_path = prefix + customized_path

destination = os.path.join(os.path.join(prefix, 'testdata'), 'input')
destination = os.path.join(os.path.join(prefix, 'data'), 'input')

print('Path to data: {}'.format(destination))

class InstagramScraper():
    """
        A class containing custom functions that use Instaloader's functions
        to scrape photo and image data from Instagram.
    """
    
    def __init__(self, login_user='', login_pass='', dest_path='', \
        date_start=(2022, 5, 1), date_end=(2022, 6, 1)):

        self.L = instaloader.Instaloader(dirname_pattern=dest_path,
            download_pictures=True,
            download_videos=False, #HARDCODED
            download_video_thumbnails=False,
            save_metadata=False,
            compress_json=False)

        self.post_errors = 0

        self.user = login_user
        self.passw = login_pass
        print(self.user)
        print(self.passw)
        try:
            self.L.login(self.user, self.passw)
            print("Login successful.")
        except:
            print("Login unsuccessful.")

    #HASHTAG
    def download_hashtag_posts(self, hashtags=[], supercategory='misc', max_count=25):
        """
          Scrapes post media from public posts that appear under a search
          of the given hashtag, ordered by recency.

          Parameters:
            hashtags: A list of hashtags from which to scrape posts from.
            supercategory: A string to make the superfolder under which a
                hashtag's posts will be downloaded to.
            max_count: The number of posts to scrape.

        """
        if len(hashtags) == 0:
          print("Specify at least one hashtag into the hashtags list. Supercategory optional.")
          return

        supcat_path = os.path.join(destination, supercategory)
        #os.makedirs(supcat_path, exist_ok=True)

        for tag in hashtags:
          iter = 0
          req = 0
          limit = max_count
          self.L.dirname_pattern = os.path.join(supcat_path, tag)
          print("Scraping for ", tag)

          #self.L.download_hashtag(tag, max_count=1000,profile_pic=False, posts=False)
          pbar = tqdm(total=max_count)
          for post in instaloader.Hashtag.from_name(self.L.context, tag).get_posts_resumable():
              try:
                  print("Saving post ", str(iter), " of ", str(limit))

                  file_exists = self.L.download_post(post, target='#'+tag)
                  if not file_exists:
                      print("File already exists!")
                  else:
                      iter += 1
                      pbar.update()
                  req += 1
              except : #High-quality image error
                  self.post_errors += 1
                  print("Error encountered: ", sys.exc_info()[0])
                  continue

              if iter == limit: #max_count reached
                  break
              if req % 10 == 0: #10 requests made: Activate sleep
                  print("Sleeping to prevent lockout... (45sec)")
                  time.sleep(45)

        pbar.close()

        #Reset directory
        self.L.dirname_pattern = os.path.join(destination, '')
        print("Scraping job completed. Resetting directory...")


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~ SCRAPING JOB ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
scraper = InstagramScraper(login_user='gram.scrape2', login_pass='insta$8scrape88', dest_path=destination)
print("Saving media to: ", scraper.L.dirname_pattern)


start = time.time()
# ~~~~~~~~~~~~~~~~~~ ENTER SCRAPING SUBJECTS ~~~~~~~~~~~~~~~~~
scraper.download_hashtag_posts(hashtags=sample_tags, supercategory='animals', max_count=1000)
# ~~~~~~~~~~~~~~~~ END SCRAPING ~~~~~~~~~~~~~~~~~~~~~~~~~
end = time.time()

print("TOTAL EXECUTION TIME: ", str((end-start)/60), "MINUTES")
