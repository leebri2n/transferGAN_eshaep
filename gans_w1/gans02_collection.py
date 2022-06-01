import instagram_scraper

args = {"login_user": "gramy.scrape",
        "login_pass": "insta$8scrape",
        'media_types': ['image', 'story-image'],
        'tag': True}

insta_scraper = instagram_scraper.InstagramScraper(login_user = 'gramy.scrape',
                                                    login_pass = 'insta$8scrape',
                                                    media_types = ['image', 'story-image'],
                                                    tag='attackontitaneditsb')

insta_scraper.authenticate_with_login()

insta_scraper.usernames.append('dprlive')


insta_scraper.scrape()
