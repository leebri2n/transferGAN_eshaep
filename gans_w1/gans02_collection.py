import instagram_scraper

args = {"login_user": "gramy.scrape",
        "login_pass": "insta$8scrape",
        'media_types': ['image', 'story-image'],
        'tag': True}

insta_scraper = instagram_scraper.InstagramScraper(**args)
insta_scraper.authenticate_with_login()

print(insta_scraper.tag)
#insta_scraper.usernames.append('dprlive')
insta_scraper.usernames.append('attackontitaneditsb')

insta_scraper.query_hashtag_gen('attackontitaneditsb')
insta_scraper.scrape()
