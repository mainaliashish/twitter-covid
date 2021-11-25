"""
 -- author : Anish Basnet
 -- email : anishbasnetworld@gmail.com
 -- date : Feb 20, 2020
"""
import os
import time
import re

from selenium import webdriver
from bs4 import BeautifulSoup
from config import Config


class Scrapping:
    def __init__(self, header=None, driver_path=None):
        self.header = header
        self.driver = None
        self.url = None
        self.load_driver(driver_path)
        self.num_pages = None
        self.page = None

    def set_url(self, url=None):
        assert type(url) == str
        if url is None:
            return None
        self.url = url

    def get_page(self, pause_time=None, activate_scroll_down=False, running_hrs=4, tweet_gap_in_sec=60):
        if pause_time is None:
            pause_time = 0
        self.driver.get(self.url)
        time.sleep(4)
        start_range = 1
        if activate_scroll_down:
            for j in range(1, running_hrs, 1):
                start = time.time()
                for i in range(start_range, 1000000000, 50):
                    self.driver.execute_script("window.scrollTo(0, {});".format(i))
                    end = time.time()
                    if (end - start) > tweet_gap_in_sec:
                        print("{} - gap paused - ".format(j))
                        time.sleep(3700)
                        start_range = i
                        break
        time.sleep(pause_time)
        page = self.driver.page_source
        self.page = page
        return page

    def sub_get_content(self):
        parser = BeautifulSoup(self.page, "html5lib")
        containers = parser.find_all('div', {"class": "css-1dbjc4n r-eqz5dr r-16y2uox r-1wbh5a2"})
        print('total approx tweets : ', len(containers))
        if len(containers) == 0:
            return None
        contents = {'datetime': [], 'content': []}
        for i in range(len(containers)):
            if containers[i]:
                time_container = containers[i].find('time')
                if time_container:
                    datetime = time_container.get('datetime', None)
                else:
                    datetime = ''
            else:
                datetime = ''
            filter_text = re.findall(r'[\u0900-\u097F]+', str(containers[i]), re.IGNORECASE)
            if filter_text:
                contents['content'].append(' '.join(filter_text))
                contents['datetime'].append(datetime)
        return contents

    def load_driver(self, driver_path=None):
        if driver_path is None:
            driver_path = Config.get_instance()['chrome_driver_path']
        if os.path.isfile(driver_path):
            self.driver = webdriver.Chrome(driver_path)

    def close_driver(self):
        self.driver.close()
