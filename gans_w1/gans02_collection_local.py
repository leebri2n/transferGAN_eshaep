import instaloader
from datetime import datetime
from itertools import dropwhile, takewhile
import csv
import os

sys_path = os.getcwd()
destination = os.path.join(sys_path, 'input')

class InstagramScraper():
    """
        Class credit to:
    """
    def __init__(self, login_user='', login_pass='', dest_path='', \
        date_start=(2022, 5, 1), date_end=(2022, 6, 1)) -> None:

        self.L = instaloader.Instaloader(dirname_pattern=destination,
            download_videos=False,
            download_video_thumbnails=True,
            save_metadata=False,
            compress_json=False)

        self.L.login(login_user, login_pass)
        self.date_start = datetime(date_start[0], date_start[1], date_start[2])
        self.date_end = datetime(date_end[0], date_end[1], date_end[2])

    #HASHTAG
    def download_hashtag_posts(self, hashtag, max_count):
        self.L.dirname_pattern=os.path.join(destination, hashtag) #leebri2n
        iter = 0
        for post in enumerate(instaloader.Hashtag.from_name(self.L.context, hashtag).get_posts()):
            try:
                self.L.download_post(post, target='#'+hashtag)
                iter += 1
            except: #leebri2n
                print("ERROR ENCOUNTERED!")
                continue
            if iter == max_count:
              break

        self.L.dirname_pattern = os.path.join(destination, '')

    #USERS
    def download_users_profile_picture(self,username):
        self.L.download_profile(username, profile_pic_only=True)

    def download_users_posts_with_periods(self,username):
        posts = instaloader.Profile.from_username(self.L.context, username).get_posts()
        SINCE = datetime(2021, 8, 28)
        UNTIL = datetime(2021, 9, 30)

        for post in takewhile(lambda p: p.date > SINCE, dropwhile(lambda p: p.date > UNTIL, posts)):
            self.L.download_post(post, username)

    def get_users_followers(self,user_name):
        '''Note: login required to get a profile's followers.'''
        self.L.login(input("input your username: gramy.scrape"), \
            input("input your username: insta$8scrape") )
        profile = instaloader.Profile.from_username(self.L.context, user_name)
        file = open("follower_names.txt","a+")
        for followee in profile.get_followers():
            username = followee.username
            file.write(username + "\n")
            print(username)

    def get_users_followings(self,user_name):
        '''Note: login required to get a profile's followings.'''
        self.L.login(input("input your username: gramy.scrape"), input("input your username: insta$8scrape") )
        profile = instaloader.Profile.from_username(self.L.context, user_name)
        file = open("following_names.txt","a+")
        for followee in profile.get_followees():
            username = followee.username
            file.write(username + "\n")
            print(username)

    #POSTS
    def get_post_comments(self,username):
        posts = instaloader.Profile.from_username(self.L.context, username).get_posts()
        for post in posts:
            for comment in post.get_comments():
                print("comment.id  : "+str(comment.id))
                print("comment.owner.username  : "+comment.owner.username)
                print("comment.text  : "+comment.text)
                print("comment.created_at_utc  : "+str(comment.created_at_utc))
                print("************************************************")

    def get_post_info_csv(self,username):
        with open(username+'.csv', 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            posts = instaloader.Profile.from_username(self.L.context, username).get_posts()
            for post in posts:
                print("post date: " + str(post.date))
                print("post profile: "+post.profile)
                print("post caption: "+post.caption)
                print("post location: "+str(post.location))

                posturl = "https://www.instagram.com/p/"+post.shortcode
                print("post url: "+posturl)
                writer.writerow(["post",post.mediaid, post.profile, post.caption, post.date, post.location, posturl,  post.typename, post.mediacount, post.caption_hashtags, post.caption_mentions, post.tagged_users, post.likes, post.comments,  post.title,  post.url ])

                for comment in post.get_comments():
                    writer.writerow(["comment",comment.id, comment.owner.username,comment.text,comment.created_at_utc])
                    print("comment username: "+comment.owner.username)
                    print("comment text: "+comment.text)
                    print("comment date : "+str(comment.created_at_utc))
                print("\n\n")

cls = InstagramScraper(login_user='gramy.scrape', login_pass='insta$8scrape', dest_path=destination)
#cls.L.download_hashtag("capitals", max_count=30)

cls.download_hashtag_posts("capitalshockey", 30)
