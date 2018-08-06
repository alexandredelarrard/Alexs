# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 13:26:20 2018

@author: JARD
"""

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import pandas as pd
import os
import multiprocessing
from datetime import datetime
from queue import Queue
from threading import Thread
import numpy as np
import glob

class Crawling(object):
    
    def __init__(self):
        self.cores = multiprocessing.cpu_count() - 1 ### allow a main thread
        
    def initialize_driver(self):
        """
        Initialize the web driver with Firefox driver as principal driver geckodriver
        parameters are here to not load images and keep the default css --> make page loading faster
        """
        firefox_profile = webdriver.FirefoxProfile()
        firefox_profile.set_preference('permissions.default.stylesheet', 2)
        firefox_profile.set_preference('permissions.default.image', 2)
        firefox_profile.set_preference('dom.ipc.plugins.enabled.libflashplayer.so', 'false')
        firefox_profile.set_preference('disk-cache-size', 8000)
        firefox_profile.set_preference("http.response.timeout", 20)
    
        driver = webdriver.Firefox(firefox_profile=firefox_profile, log_path= os.environ["DIR_PATH"] + "/webdriver/geckodriver.log")#firefox_options=options, 
        driver.delete_all_cookies()
        driver.set_page_load_timeout(200)     
        return driver
        
    
    def initialize_driver_chrome(self):
        """
        Initialize the web driver with chrome driver as principal driver chromedriver.exe, headless means no open web page. But seems slower than firefox driver  
        parameters are here to not load images and keep the default css --> make page loading faster
        """
        
        options = Options()
        prefs = {"profile.managed_default_content_settings.images":2,
                 "profile.default_content_setting_values.notifications":2,
                 "profile.managed_default_content_settings.stylesheets":2,
                 "profile.managed_default_content_settings.cookies":2,
                 "profile.managed_default_content_settings.javascript":1,
                 "profile.managed_default_content_settings.plugins":1,
                 "profile.managed_default_content_settings.popups":2,
                 "profile.managed_default_content_settings.geolocation":2,
                 "profile.managed_default_content_settings.media_stream":2,
                 }
        options.add_experimental_option("prefs",prefs)
        options.add_argument("--headless") # Runs Chrome in headless mode.
        options.add_argument('--no-sandbox') # Bypass OS security model
        options.add_argument('--disable-gpu')  # applicable to windows os only
        options.add_argument('start-maximized') # 
        options.add_argument('disable-infobars')
        options.add_argument("--disable-extensions")
        
        service_args =["--verbose", "--log-path={0}".format(os.environ["DIR_PATH"] + "/webdriver/chrome.log")]
        
        driver = webdriver.Chrome(executable_path= os.environ["DIR_PATH"] + "/webdriver/chromedriver.exe", 
                                  chrome_options=options, service_args=service_args)
        return driver
    
    
    def initialize_queue_drivers(self):
        self.driver_queue = Queue()
        for i in range(self.cores):
             self.driver_queue.put(self.initialize_driver())
        
    
    def close_queue_drivers(self):
        for i in range(self.driver_queue.qsize()):
            driver = self.driver_queue.get()
            driver.close()


    def start_threads_and_queues(self, function):
        for i in range(self.cores):
             t = Thread(target= self.queue_calls, args=(function, self.queues,))
             t.daemon = True
             t.start()
             
             
    def queue_calls(self, function, queues):
        
        queue_url = queues["urls"]
        queue_results = queues["results"]
        
        while True:
            driver = queues["drivers"].get()
            url = queue_url.get()
            self.handle_timeout(driver, url)
                
            information, driver = self.handle_information(function, driver, queues)
            
            queues["drivers"].put(driver)
            queue_url.task_done()
            
            if information.shape[0] > 0:
                queue_results.put(information) 
            
                if self.check_in_date(information, url):
                    ### empty the queue of all its urls
                    while queue_url.qsize()>0:
                         queue_url.get()
                         queue_url.task_done()
   
    
    def handle_information(self, function, driver, queues):
        
        try:
            information = function(driver, queues)
        except Exception:
            driver.close()
            driver = self.initialize_driver()
            information, driver = self.handle_information(function, driver, queues)
            
        return information, driver
                        
    def handle_timeout(self, driver, url):
        try:
            driver.get(url)
        except Exception:
            driver.close()
            driver = self.initialize_driver()
            driver = self.handle_timeout(driver, url)
        return driver
        
    
    def save_results(self, journal):
        
        if not os.path.isdir(os.environ["DIR_PATH"] + "/data"):
            os.mkdir(os.environ["DIR_PATH"] + "/data")
            
        if not os.path.isdir(os.environ["DIR_PATH"] + "/data/" + journal):
            os.makedirs(os.environ["DIR_PATH"] + "/data/" + journal)
            
        if not os.path.isdir("/".join([os.environ["DIR_PATH"], "data", journal, self.queues["carac"]["url_article"]])):
            os.makedirs("/".join([os.environ["DIR_PATH"], "data", journal, self.queues["carac"]["url_article"]]))
            
        if not os.path.isdir("/".join([os.environ["DIR_PATH"], "data", journal, self.queues["carac"]["url_article"], datetime.now().strftime("%Y-%m-%d")])):
            os.makedirs("/".join([os.environ["DIR_PATH"], "data", journal, self.queues["carac"]["url_article"], datetime.now().strftime("%Y-%m-%d")]))
            
        #### if reached the min date then empty the queue of urls and save all results 
        path_name = "/".join([os.environ["DIR_PATH"], "data", journal, self.queues["carac"]["url_article"], datetime.now().strftime("%Y-%m-%d"), "extraction_0.csv"]) 
        if os.path.isfile(path_name):
            len_files = len(glob.glob("/".join([os.environ["DIR_PATH"], "data", journal, self.queues["carac"]["url_article"], datetime.now().strftime("%Y-%m-%d"), "*.csv"])))
            path_name = "/".join([os.environ["DIR_PATH"], "data", journal, self.queues["carac"]["url_article"], datetime.now().strftime("%Y-%m-%d"), "extraction_{0}.csv".format(len_files)]) 
            
        articles = np.array([])
        i = 0
        while self.queues["results"].qsize()>0:
            article = self.queues["results"].get()
            if i ==0:
                articles = article
                i +=1
            else:
                articles = np.concatenate((articles,article), axis=0)
                 
        article_bdd = pd.DataFrame(articles)
        article_bdd.to_csv(path_name, index=False)
        print("{0} data extracted".format(article_bdd.shape))

      
    def check_in_date(self, information, url):
        ### 0 is always the date column by convention, 1 is always the url by convention
        if min(pd.to_datetime(information[:,0])) < self.end_date: 
            return True
        return False
    