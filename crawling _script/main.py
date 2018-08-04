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
from url_crawling import URLCrawling
from article_crawling import ArticleCrawling

def environment_variables():
    configParser = configparser.RawConfigParser() 
    if platform.system() in  ["Darwin", "Linux"]: # Mac or Linux
        configFilePath = os.environ["PROFILE"] + '/config_alexs.txt' # to check if PROFILE in os environ for Mac
    else:
        configFilePath = os.environ["USERPROFILE"] + '/config_alexs.txt'
    configParser.read(configFilePath)
    os.environ["DIR_PATH"] = configParser.get("config-Alexs", "project_path")
    

class Main(object):
    
    def __init__(self, url_article, min_date, journals):
        
        crawl = Crawling()
        self.end_date = pd.to_datetime(min_date, format = "%Y-%m-%d")
        self.cores   =  multiprocessing.cpu_count() - 1
        self.queues = {"drivers": Queue(), "urls" :  Queue(), "results": Queue()}

        self.driver = crawl.initialize_driver()
        for i in range(self.cores):
             self.queues["drivers"].put(crawl.initialize_driver())
        
        self.main(journals, url_article)


    def main(self, journals, url_article):
        
        for journal in journals:
            self.specificities(journal)
            
            if url_article=="url":
                URLCrawling(self.end_date, self.queues, self.driver)
            elif url_article=="article":
                ArticleCrawling(self.end_date, self.queues, self.driver)
            else:
                print("Currently two actions handled: url extraction or article crawling")


    def specificities(self, journal):
        if journal == "lemonde":
            self.queues["carac"] = {"journal": journal, 
                                   "url": "https://www.lemonde.fr/", 
                                   "nav_menu": "id='navigation-generale'",
                                   "not_in_liste": ['https://www.lemonde.fr/campus/','https://www.lemonde.fr/m-le-mag/',
                                                    'https://www.lemonde.fr/pixels/','https://www.lemonde.fr/teaser/presentation.html#xtor=CS1-263[BOUTONS_LMFR]-[BARRE_DE_NAV]-5-[Home]',
                                                    'https://www.lemonde.fr/grands-formats/','https://www.lemonde.fr/les-decodeurs/',
                                                    'https://www.lemonde.fr/videos/','https://www.lemonde.fr/data/',
                                                    'https://www.lemonde.fr/guides-d-achat/'],
                                    "time_element":"time[@class='grid_1 alpha']",
                                    "href_element":"div[@class='grid_11 conteneur_fleuve omega']/div/h3/a",
                                    "article_element":"",
                                    "fill_queue":["div[@class='padt8']/div/ul","adroite","{0}.html"]}
            
        elif journal == "lefigaro":
            self.queues["carac"] = {"journal": journal, 
                                    "url": "http://articles.lefigaro.fr/", 
                                    "nav_menu": "",
                                    "not_in_liste": [],
                                    "time_element":"",
                                    "href_element":"div[@class='SiteMap']/a",
                                    "article_element":"div[@class='SiteMap']/a",
                                    "fill_queue":[]}
                
        elif journal == "lesechos":
            self.queues["carac"] = {"journal": journal, 
                                   "url": "https://www.lesechos.fr/recherche", 
                                   "nav_menu": "",
                                   "not_in_liste": [],
                                   "time_element":"article[@class='liste-article']/div/time",
                                   "href_element":"article[@class='liste-article']/h2/a",
                                   "article_element":"article[@class='liste-article']",
                                   "fill_queue":[]}
            
        elif journal == "mediapart":
            self.queues["carac"] = {"journal": journal, 
                                   "url": "https://www.mediapart.fr/", 
                                   "nav_menu": "class='main-menu'",
                                   "not_in_liste": ['/studio','//blogs.mediapart.fr/edition/le-club-mode-demploi',
                                                    "//blogs.mediapart.fr/",'https://www.mediapart.fr/studio'],
                                   "time_element":"div[@class='post-list universe-journal']/div/div/time",
                                   "href_element":"div[@class='post-list universe-journal']/div/h3/a",
                                   "article_element":"",
                                   "fill_queue":["ul[@class='pager']","pager-last","?page={0}"]}
            
        else:
            print("only following journals are currently handled : lemonde, lefigaro, mediapart, lesechos")
        

if __name__ == "__main__":
     environment_variables()
     Main("url", 
          "2018-07-01",
          [ "mediapart", "lemonde", "lesechos", "lefigaro"])