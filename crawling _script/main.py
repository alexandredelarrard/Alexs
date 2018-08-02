# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 11:30:04 2018

@author: JARD
"""


import os
import configparser
import platform
import pandas as pd
import multiprocessing
from queue import Queue
from crawling import Crawling
from lemonde import LemondeScrapping
from mediapart import MediapartScrapping
from lefigaro import LefigaroScrapping
from lesechos import LesechosScrapping

def environment_variables():
    configParser = configparser.RawConfigParser() 
    if platform.system() in  ["Darwin", "Linux"]: # Mac or Linux
        configFilePath = os.environ["PROFILE"] + '/config_alexs.txt' # to check if PROFILE in os environ for Mac
    else:
        configFilePath = os.environ["USERPROFILE"] + '/config_alexs.txt'
    configParser.read(configFilePath)
    os.environ["DIR_PATH"] = configParser.get("config-Alexs", "project_path")
    

class MainUrl(object):
    
    def __init__(self):
        
        crawl = Crawling()
        self.end_date = pd.to_datetime("2018-07-01", format = "%Y-%m-%d")
        self.cores = multiprocessing.cpu_count() - 1 
        self.driver = crawl.initialize_driver()
        
        self.driver_queue = Queue() 
        for i in range(self.cores):
             self.driver_queue.put(crawl.initialize_driver())
             
        self.queues = {"drivers" : self.driver_queue, "urls" :  Queue(), "results": Queue()}
        self.main_url()


    def main_url(self):
        
        ### crawl each competitor
        MediapartScrapping(self.end_date, self.queues, self.driver)
        LesechosScrapping(self.end_date, self.queues, self.driver)
        LefigaroScrapping(self.end_date, self.queues, self.driver)
        LemondeScrapping(self.end_date, self.queues, self.driver)
        

if __name__ == "__main__":
     environment_variables()
     MainUrl()
     