# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 13:26:20 2018

@author: JARD
"""

from selenium import webdriver
import pandas as pd
import os
import multiprocessing
import time
from datetime import datetime
from queue import Queue
from threading import Thread
import numpy as np


class Crawling(object):
    
    def __init__(self):
        
        self.id_col_date = 0
        self.end_date = pd.to_datetime("2018-07-01", format = "%Y-%m-%d")
        self.cores = multiprocessing.cpu_count() - 1 ### allow a main thread
        self.driver = self.initialize_driver()
        self.initialize_queue_drivers()
        
    def initialize_driver(self):
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
    
    def initialize_queue_drivers(self):
        self.driver_queue = Queue()
        for i in range(self.cores):
             self.driver_queue.put(self.initialize_driver())
        print("Initialized {0} drivers for crawling".format(self.cores))
        
    
    def close_queue_drivers(self):
        ### close all drivers
        for i in  range(self.cores):
            driver = self.driver_queue.get()
            driver.close()
    
                 
    def get_lis_from_nav(self, class_id, id_ul):
        """
        click on input when there is a value to set.
        Useful for checking captcha
        """
        nav = self.driver.find_element_by_xpath("//nav[@{0}='{1}']".format(class_id, id_ul))
        liste_href = nav.find_element_by_tag_name("ul")
        liste = []
        for li in liste_href.find_elements_by_tag_name("a"):
            liste.append(li.get_attribute("href"))
        return liste   


    def start_threads_and_queues(self, function):
        self.queues = {"drivers" : self.driver_queue, "urls" :  Queue(), "results": Queue()}
        for i in range(self.cores):
             t = Thread(target= self.queue_calls, args=(function, self.queues, ))
             t.daemon = True
             t.start()
             
            
    def queue_calls(self, function, queues):
        
        queue_driver = queues["drivers"]
        queue_url = queues["urls"]
        queue_results = queues["results"]
        
        while True:
            driver = queue_driver.get()
            url = queue_url.get()
            driver.get(url)
                
            information = function(driver)
           
            queue_driver.put(driver)
            queue_results.put(information)
            queue_url.task_done()

            #### if reached the min date then empty the queue of urls and save all results 
            if self.check_in_date(information, url) or queue_url.empty():
                path_name = os.environ["DIR_PATH"] + "/data/{0}_{1}.csv".format(os.path.dirname(url).replace("https://www.", "").replace(".fr",""), datetime.now().strftime("%Y-%m-%d"))
                if not os.path.isfile(path_name):
                    
                    ### close all drivers 
                    while queue_driver.qsize()> 0:
                        driver = queue_driver.get()
                        driver.close()
                        queue_driver.task_done()   
                        
                    ### artificially empty the queue
                    while queue_url.qsize()> 0:
                        queue_url.get()
                        queue_url.task_done()   
                        
                    self.queues["urls"] = Queue()
                    
                    articles = np.array([])
                    i = 0
                    while queue_results.qsize()>0:
                        article = queue_results.get()
                        if i ==0:
                            articles = article
                            i +=1
                        else:
                            articles = np.concatenate((articles,article),axis=0)
                        
                    article_bdd = pd.DataFrame(articles)
                    article_bdd.to_csv(path_name, index=False)

                else:
                    time.sleep(3)
                

    def check_in_date(self, information, url):
        
        if min(pd.to_datetime(information[:,self.id_col_date])) < self.end_date:
            root = os.environ["DIR_PATH"] + "/data"
            if not os.path.isdir(root):
                os.mkdir(root)
            if not os.path.isdir(root + "/{0}".format(url.replace("https://www.", "").replace(".fr","").split("/")[0])):
                os.mkdir(root + "/{0}".format(url.replace("https://www.", "").replace(".fr","").split("/")[0]))
            return True
        return False
    