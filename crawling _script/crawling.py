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
        
    def initialize_driver_firefox(self):
        """
        Initialize the web driver with Firefox driver as principal driver geckodriver
        parameters are here to not load images and keep the default css --> make page loading faster
        """
        firefox_profile = webdriver.FirefoxProfile()
        firefox_profile.set_preference('permissions.default.stylesheet', 2)
        firefox_profile.set_preference('permissions.default.image', 2)
        firefox_profile.set_preference('dom.ipc.plugins.enabled.libflashplayer.so', 'false')
        firefox_profile.set_preference('disk-cache-size', 4096)
        firefox_profile.set_preference("http.response.timeout", 10)
    
        driver = webdriver.Firefox(firefox_profile=firefox_profile, log_path= os.environ["DIR_PATH"] + "/webdriver/geckodriver.log")#firefox_options=options, 
        driver.delete_all_cookies()
        driver.set_page_load_timeout(100)     
        return driver
        
    
    def initialize_driver(self):
        """
        Initialize the web driver with Firefox driver as principal driver geckodriver
        parameters are here to not load images and keep the default css --> make page loading faster
        """
        
        options = Options()
        options.add_argument("--headless") # Runs Chrome in headless mode.
        options.add_argument('--no-sandbox') # Bypass OS security model
        options.add_argument('--disable-gpu')  # applicable to windows os only
        options.add_argument('start-maximized') # 
        options.add_argument('disable-infobars')
        options.add_argument("--disable-extensions")
        
        service_args =["--verbose", "--log-path={0}".format(os.environ["DIR_PATH"] + "/webdriver/chrome.log")]
        
        driver = webdriver.Chrome(executable_path= os.environ["DIR_PATH"] + "/webdriver/chromedriver.exe", 
                                  chrome_options=options, service_args=service_args)
        driver.delete_all_cookies()
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
                
            information = self.handle_information(function, driver, queues)
            
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
        except OSError:
            driver.close()
            driver = self.initialize_driver()
            self.handle_information(function, driver, queues)
            
        return information
                        
    def handle_timeout(self, driver, url):
        try:
            driver.get(url)
        except Exception:
            driver.quit()
            driver = self.initialize_driver()
            self.handle_timeout(driver, url)
        return driver
        
    
    def save_results(self, journal):
        
        if not os.path.isdir(os.environ["DIR_PATH"] + "/data"):
            os.mkdir(os.environ["DIR_PATH"] + "/data")
            
        if not os.path.isdir(os.environ["DIR_PATH"] + "/data/" + journal):
            os.makedirs(os.environ["DIR_PATH"] + "/data/" + journal)
        
        #### if reached the min date then empty the queue of urls and save all results 
        path_name = os.environ["DIR_PATH"] + "/data/" + journal + "/extraction_0_{0}.csv".format(datetime.now().strftime("%Y-%m-%d"))
        if os.path.isfile(path_name):
            len_files = len(glob.glob(os.environ["DIR_PATH"] + "/data/" + journal+ "/*.csv"))
            path_name = os.environ["DIR_PATH"] + "/data/" + journal + "/extraction_{0}_{1}.csv".format(len_files, datetime.now().strftime("%Y-%m-%d"))
            
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
    
    
    def get_menu_liste(self, url, caracteristics):
        try:
            if len(caracteristics["not_in_liste"]) > 0:
                self.driver.get(url)
                liste_menu_href = self.get_lis_from_nav(caracteristics["nav_menu"])
                liste_menu_href = [x for x in liste_menu_href if x not in [url] + caracteristics["not_in_liste"]] 
                
                if len(liste_menu_href)>0:
                    return liste_menu_href
        except Exception:
            pass
        return [url]
     
    
    