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
from threading import Thread, get_ident
import numpy as np
import glob
import random
import time
from selenium.webdriver.firefox.firefox_profile import FirefoxProfile
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary


class Crawling(object):
    
    def __init__(self):
        self.cores = int(multiprocessing.cpu_count()*2) 
        self.agents =  pd.read_csv(os.environ["DIR_PATH"] + "/webdriver/agents.csv")["agents"].tolist()

    def initialize_driver(self):
        """
        Initialize the web driver with chrome driver as principal driver chromedriver.exe, headless means no open web page. But seems slower than firefox driver  
        parameters are here to not load images and keep the default css --> make page loading faster
        """
        
        options = Options()
        prefs = {"profile.managed_default_content_settings.images":2,
                 "profile.default_content_setting_values.notifications":2,
                 "profile.managed_default_content_settings.stylesheets":2,
                 "profile.managed_default_content_settings.cookies":2,
                 "profile.managed_default_content_settings.javascript":2,
                 "profile.managed_default_content_settings.plugins":2,
                 "profile.managed_default_content_settings.popups":2,
                 "profile.managed_default_content_settings.geolocation":2,
                 "profile.managed_default_content_settings.media_stream":2,
                 }
        
        options.add_experimental_option("prefs",prefs)
        options.add_argument("--incognito")
        options.add_argument('--no-sandbox') # Bypass OS security model
        options.add_argument('--disable-gpu')  # applicable to windows os only
        options.add_argument('start-maximized') # 
        options.add_argument('disable-infobars')
        options.add_argument("--disable-extensions")
        options.add_argument("user-agent={0}".format(self.agents[random.randint(0,len(self.agents) - 1)]))
        
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
            driver.quit()


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
            driver = self.handle_timeout(driver, url)
                
            information, driver = self.handle_information(function, driver, queues, url, 0)
            time.sleep(random.uniform(0.1,0.3))
            
            queues["drivers"].put(driver)
            queue_url.task_done()
            
            queue_results.put(information) 
                    
            ### if queue has more than X elements in queue then save elements to have a queue size not too big
#            if queue_results.qsize() > 500 or queue_url.qsize() == 0:
#                self.save_results()
#   
    
    def handle_information(self, function, driver, queues, url, compteur):
        try:
            information = function(driver, queues)
        except Exception as e:
            if compteur < 0: 
                driver.quit()
                driver = self.initialize_driver()
                driver = self.handle_timeout(driver, url)
                information, driver = self.handle_information(function, driver, queues, url, compteur +1)
            else: 
                print("thread : {0}, url : {1}, error : {2}".format(get_ident(), driver.current_url, e))
                return np.array([]), driver
            
        return information, driver
                        
    def handle_timeout(self, driver, url):
#        try:
            driver.get(url)
#            driver.execute_script("window.alert = function() {};")
#        except Exception:
#            driver.quit()
#            driver = self.initialize_driver()
#            time.sleep(5)
#            driver = self.handle_timeout(driver, url)
            return driver
        
    
    def save_results(self):
        
        if not os.path.isdir(os.environ["DIR_PATH"] + "/data"):
            os.mkdir(os.environ["DIR_PATH"] + "/data")
            
        if not os.path.isdir(os.environ["DIR_PATH"] + "/data/continuous_run"):
            os.makedirs(os.environ["DIR_PATH"] + "/data/continuous_run")
            
        if not os.path.isdir("/".join([os.environ["DIR_PATH"], "data", "continuous_run", "article"])):
            os.makedirs("/".join([os.environ["DIR_PATH"], "data", "continuous_run", "article"]))
            
        if not os.path.isdir("/".join([os.environ["DIR_PATH"], "data", "continuous_run", "article", datetime.now().strftime("%Y-%m-%d")])):
            os.makedirs("/".join([os.environ["DIR_PATH"], "data", "continuous_run", "article", datetime.now().strftime("%Y-%m-%d")]))
            
        #### if reached the min date then empty the queue of urls and save all results 
        path_name = "/".join([os.environ["DIR_PATH"], "data", "continuous_run", "article", datetime.now().strftime("%Y-%m-%d"), "extraction_0.csv"]) 
        if os.path.isfile(path_name):
            len_files = len(glob.glob("/".join([os.environ["DIR_PATH"], "data", "continuous_run", "article", datetime.now().strftime("%Y-%m-%d"), "*.csv"])))
            path_name = "/".join([os.environ["DIR_PATH"], "data", "continuous_run", "article", datetime.now().strftime("%Y-%m-%d"), "extraction_{0}.csv".format(len_files)]) 
        
        cols = ["journal", "url", "restricted", "titre", "auteur", "article", "categorie", "description_article"]
        i = 0
        while self.queues["results"].qsize()>0:
            article = self.queues["results"].get()
            if i ==0:
                articles = pd.DataFrame([article], columns = cols)
                i +=1
            else:
                articles = pd.concat([articles, pd.DataFrame([article], columns = cols)], axis=0)
                 
        articles.to_csv(path_name, index=False, sep='#')
        print("{0} data extracted".format(articles.shape))
