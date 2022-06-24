import instaloader
from datetime import datetime
from itertools import dropwhile, takewhile
import csv
import os

destination = r'C:\Users\leebr\Documents\GitHub\data\gans'

class InstagramScraper():
    """
        Class credit to: @HKN MZ on medium.com
    """
    def __init__(self, login_user='', login_pass='', dest_path='', \
        date_start=(2022, 5, 1), date_end=(2022, 6, 1)) -> None:

        self.L = instaloader.Instaloader(dirname_pattern=destination,
            download_videos=False, #HARDCODED
            download_video_thumbnails=True,
            save_metadata=False,
            compress_json=False)

        self.L.login(login_user, login_pass)
        self.date_start = datetime(date_start[0], date_start[1], date_start[2])
        self.date_end = datetime(date_end[0], date_end[1], date_end[2])

    #HASHTAG
    def download_hashtag_posts(self, hashtag, max_count):
        iter = 0
        limit = 30
        self.L.dirname_pattern=os.path.join(destination, hashtag) #@leebri2n
        for post in instaloader.Hashtag.from_name(self.L.context, hashtag).get_posts():
            try:
                self.L.download_post(post, target='#'+hashtag)
                print("Saving image ", str(iter), " of ", str(limit))
                if iter == limit:
                    break
                iter += 1
            except: #@leebri2n
                print("ERROR ENCOUNTERED!")
                continue

        self.L.dirname_pattern = os.path.join(destination, '')

    #USERS
    def download_users_posts_with_periods(self,username, date_start=(2022, 5, 1), date_end=(2022, 6, 1)):
        SINCE = datetime(date_start[0], date_start[1], date_start[2])
        UNTIL = datetime(date_end[0], date_end[1], date_end[2])

        self.L.dirname_pattern=os.path.join(destination, '@'+username)
        posts = instaloader.Profile.from_username(self.L.context, username).get_posts()
        for post in takewhile(lambda p: p.date > SINCE, dropwhile(lambda p: p.date > UNTIL, posts)):
            try:
                self.L.download_post(post, username)
            except:
                print("ERROR ENCOUNTERED!")
                continue

        self.L.dirname_pattern = os.path.join(destination, '')

cls = InstagramScraper(login_user='gramy.scrape', login_pass='insta$8scrape', dest_path=destination)
#cls.L.download_hashtag('burgers', max_count=30)
#cls.download_hashtag_posts("capitalshockey", 30)
cls.download_users_posts_with_periods('nopoopjune')
