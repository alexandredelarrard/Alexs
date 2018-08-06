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
    
    def __init__(self, pick_url_article, min_date, journals):
        
        crawl = Crawling()
        self.end_date = pd.to_datetime(min_date, format = "%Y-%m-%d")
        self.cores   =  multiprocessing.cpu_count() - 1
        self.queues = {"drivers": Queue(), "urls" :  Queue(), "results": Queue()}

        self.driver = crawl.initialize_driver()
        for i in range(self.cores):
             self.queues["drivers"].put(crawl.initialize_driver())
        
        self.main(journals, pick_url_article)


    def main(self, journals, pick_url_article):
        
        for journal in journals:
            self.specificities(journal)
            
            for url_article in pick_url_article:
                self.queues["carac"]["url_article"] = url_article
                if url_article=="url":
                    URLCrawling(self.end_date, self.queues, self.driver)
                elif url_article=="article":
                    ArticleCrawling(self.end_date, self.queues, self.driver)
                else:
                    print("Currently two actions handled: url extraction or article crawling")


    def specificities(self, journal):
        
        
        if journal == "lemonde":
            self.queues["carac"] = {"url_crawl":{"url": "https://www.lemonde.fr/", 
                                                "in_liste": ['https://www.lemonde.fr/international/','https://www.lemonde.fr/politique/',
                                                             'https://www.lemonde.fr/societe/','https://www.lemonde.fr/economie/',
                                                             'https://www.lemonde.fr/culture/',
                                                             'https://www.lemonde.fr/idees/','https://www.lemonde.fr/planete/',
                                                             'https://www.lemonde.fr/sport/','https://www.lemonde.fr/sciences/', 
                                                             'https://www.lemonde.fr/campus/'], # missing pixels, m-mag
                                                "time_element":["time[@class='grid_1 alpha']"],
                                                "href_element":["div[@class='grid_11 conteneur_fleuve omega']/div/h3/a"],
                                                "article_element":["article"],
                                                "fill_queue":["div[@class='conteneur_pagination']/ul","adroite","{0}.html"]
                                                  },
                                    "article_crawl": {"article_crawl" :"",
                                                      "restricted" : ""}
                                    }
            
        elif journal == "lefigaro":
            self.queues["carac"] = {"url_crawl":{"url": "http://articles.lefigaro.fr/", 
                                                "in_liste": ["http://articles.lefigaro.fr/"],
                                                "time_element":[],
                                                "href_element":["div[@class='SiteMap']/a"],
                                                "article_element":["div[@class='SiteMap']/a"],
                                                "fill_queue":[]},
                                    "article_crawl": {"article_crawl" :"",
                                                      "restricted" : ""}
                                    }
                
        elif journal == "lesechos":
            self.queues["carac"] = {"url_crawl":{ "url": "https://www.lesechos.fr/recherche", 
                                                   "in_liste": ["https://www.lesechos.fr/recherche"],
                                                   "time_element":["article[@class='liste-article']/div/time"],
                                                   "href_element":["article[@class='liste-article']/h2/a"],
                                                   "article_element":["article[@class='liste-article']"],
                                                   "fill_queue":[]},
                                   "article_crawl": {"article_crawl" :"",
                                                      "restricted" : ""}
                                    }
            
        elif journal == "mediapart":
            self.queues["carac"] = {"url_crawl":{ "url": "https://www.mediapart.fr/", 
                                                   "in_liste": ["https://www.mediapart.fr/journal/international",
                                                                "https://www.mediapart.fr/journal/france",
                                                                "https://www.mediapart.fr/journal/economie",
                                                                "https://www.mediapart.fr/journal/culture-idees",
                                                                "https://www.mediapart.fr/journal/dossiers"], # missing le studio, le club
                                                   "time_element":["div[@data-type='article']/div/time", "li[@data-type='case']/div/time"],
                                                   "href_element":["div[@data-type='article']/h3/a","li[@data-type='case']/h3/a"],
                                                   "article_element":["div[@data-type='article']","li[@data-type='case']"],
                                                   "fill_queue":["ul[@class='pager']","pager-last","?page={0}"]},
                                    "article_crawl": {"article_crawl" :"",
                                                      "restricted" : ""}
                                    }
                                    
        elif journal == "latribune":
            self.queues["carac"] = {"url_crawl":{ "url": "https://www.latribune.fr", 
                                                  "in_liste":["https://www.latribune.fr/entreprises-finance-11/", "https://www.latribune.fr/economie-2/", 
                                                               "https://www.latribune.fr/technos-medias-28/", "https://www.latribune.fr/vos-finances-38/", 
                                                               "https://www.latribune.fr/opinions-65/", "https://www.latribune.fr/regions-70/"], # missing bourse = webfg/articles/marches-france/
                                                   "time_element":[],
                                                   "href_element":["article/section/h2/a"],
                                                   "article_element":["article"],
                                                   "fill_queue":[]},
                                    "article_crawl": {"article_crawl" :"",
                                                      "restricted" : ""}
                                    }
                                    
        elif journal == "leparisien":
            self.queues["carac"] = {"url_crawl":{ "url": "http://www.leparisien.fr/", 
                                                   "in_liste":["http://www.leparisien.fr/international/", "http://www.leparisien.fr/politique/",
                                                               "http://www.leparisien.fr/economie/", "http://www.leparisien.fr/high-tech/",
                                                               "http://www.leparisien.fr/societe/", "http://www.leparisien.fr/environnement/",
                                                               "http://www.leparisien.fr/societe/sante/", "http://www.leparisien.fr/faits-divers/",
                                                               "http://www.leparisien.fr/info-paris-ile-de-france-oise/", 
                                                               "http://www.leparisien.fr/sports/", "http://www.leparisien.fr/culture-loisirs/"],
                                                   "time_element":[],
                                                   "href_element":["article[@class='article article_small hover-expand-animation ']/h3/a",
                                                                   "article[@class='article article_large hover-expand-animation ']/h3/a",
                                                                   "article[@class='article article_medium hover-expand-animation ']/h3/a",
                                                                   "div[@class='article__list-infos']/a",
                                                                   "div[@class='article__list-infos']/h2/a"],
                                                   "article_element":["article[@class='article article_small hover-expand-animation ']",
                                                                       "article[@class='article article_large hover-expand-animation ']",
                                                                       "article[@class='article article_medium hover-expand-animation ']",
                                                                       "div[@class='container article__list  ']",
                                                                       "div[@class='container article__list  article__list_pagin']",
                                                                       "div[@class='container article__list  article__list_pagin article__list_pagin_small']"],
                                                   "fill_queue":[]},
                                    "article_crawl": {"article_crawl" :"",
                                                      "restricted" : ""}
                                    }
                                    
        elif journal == "lexpress":
            self.queues["carac"] = {"url_crawl":{ "url": "https://www.lexpress.fr", 
                                                   "in_liste":["https://www.lexpress.fr/actualite/politique/liste.html", 
                                                               "https://www.lexpress.fr/actualite/monde/liste.html", 
                                                               "https://www.lexpress.fr/actualite/societe/liste.html", 
                                                               "https://www.lexpress.fr/actualite/sport/liste.html", 
                                                               "https://www.lexpress.fr/culture/liste.html", 
                                                               "https://www.lexpress.fr/actualite/sciences/liste.html", 
                                                               "https://www.lexpress.fr/actualite/medias/liste.html", 
                                                               "https://www.lexpress.fr/education/liste.html", 
                                                               "https://lexpansion.lexpress.fr/high-tech/liste.html", 
                                                               "https://www.lexpress.fr/region/liste.html", ],
                                                   "time_element":[],
                                                   "href_element":["div[@class='groups']/div/a"],
                                                   "article_element":["div[@class='groups']/div"],
                                                   "fill_queue":[]},
                                    "article_crawl": {"article_crawl" :"",
                                                      "restricted" : ""}
                                    }
                                    
        else:
            print("only following journals are currently handled : lemonde, lefigaro, mediapart, lesechos")
        
        self.queues["carac"]["journal"] = journal

if __name__ == "__main__":
     environment_variables()
     Main(["url"], 
          "2018-07-01",
          ["mediapart", "latribune", "leparisien", "lexpress", "lefigaro", "lemonde", "lesechos"]) 