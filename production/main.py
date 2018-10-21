# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 11:30:04 2018

@author: JARD
"""

import os
import configparser
import platform
import time
import datetime
from queue import Queue
from url_crawling import UrlCrawling
from article_crawling import ArticleCrawling
import pandas as pd

def environment_variables():
    configParser = configparser.RawConfigParser() 
    if platform.system() in  ["Darwin", "Linux"]: 
        configFilePath = os.environ["PROFILE"] + '/config_alexs.txt'
    else:
        configFilePath = os.environ["USERPROFILE"] + '/config_alexs.txt'
    configParser.read(configFilePath)
    os.environ["DIR_PATH"] = configParser.get("config-Alexs", "project_path")
    

class Main(object):
    
    def __init__(self):
        
        self.queues = {"drivers": Queue(), "urls" :  Queue(), "results": Queue()}        
        self.specificities()
        self.main()


    def main(self):
        
        time_tot = time.time()
        
        t0 = time.time()
        self.liste_urls = UrlCrawling(self.queues).main_url_crawling()
        print("[{0}] Crawling in {1}s\n {2} articles to crawl".format(datetime.datetime.today().strftime("%Y-%m-%d"), time.time() - t0, self.liste_urls.shape[0]))
        
        self.liste_urls = pd.read_csv(r"C:\Users\User\Documents\Alexs\data\continuous_run\url\2018-10-20.csv")
        t0 = time.time()
        ArticleCrawling(self.queues, self.liste_urls["url"].tolist()).main_article_crawling()
        print("[{0}] Crawling in {1}s\n".format(datetime.datetime.today().strftime("%Y-%m-%d"), time.time() - t0))
                
        print("total crawling time {0}s".format(time.time() - time_tot))
        
        #### kill drivers
        while self.queues["drivers"].qsize()> 0:
            driver = self.queues["drivers"].get()
            driver.quit()

    def specificities(self):

        self.queues["lemonde"] = {"article" :["div[@id='articleBody']", "article[@class='article article_normal']"],
                                  "restricted" : ["div[@id='teaser_article']", "div[@class='Paywall']/div/div/div"],
                                  "title": ["h1[@itemprop='Headline']", "h1[@class='article__title']", "article[@class='main']/header/h1"],
                                  "author":"span[@itemprop='author']",
                                  "categorie": "nav[@id='nav_ariane']/ul/li",
                                  "description_article": "p[@itemprop='description']",
                                  "not_to_crawl" : ["/live/", "/portfolio/", "/visuel/", "/blog-mediateur/"]}
        
        self.queues["lefigaro"] = {"article" :["div[@class='fig-content__body']"],
                                  "restricted" : ["div[@class='fig-premium-paywall']", "div[@class='fig-premium-paywall__infos']"],
                                  "title": ["h1[@class='fig-main-title']", "div[@class='fig-article-headline']/h1"],
                                  "author":"a[@class='fig-content-metas__author']",
                                  "categorie": "li[@class='fig-breadcrumb__item fig-breadcrumb__item--current']",
                                  "description_article":"p[@class='fig-content__chapo']",
                                  "not_to_crawl" : []}
                
        self.queues["lesechos"] = {"article" :["div[@class='paywall']", "div[@class='main-content content-article']"],
                                  "title": ["h1[@class='title-article']"],
                                  "author":"div[@class='signature-article']",
                                  "categorie": "li[@class='opened']",
                                  "description_article":"p[@itemprop='articleBody']",
                                  "not_to_crawl" : []}
        
        self.queues["mediapart"] = {"article" :["div[@class='content-article content-restricted']", "div[@class='content-article']"],
                                  "restricted" : ["div[@class='content-article content-restricted']"],
                                  "title": ["h1[@class='title']"],
                                  "author": "div[@class='author']/a",
                                  "categorie": "ul[@class='taxonomy']",
                                  "description_article":"div[@class='introduction']",
                                  "not_to_crawl" : []}
        
        self.queues["latribune"] = {"article" :["div[@class='article-content-wrapper']"],
                                   "restricted" : [],
                                   "title": ["h1[@itemprop='Headline']"],
                                   "author":"span[@itemprop='name']",
                                   "categorie": "div[@class='article-title-wrapper']/ol/li[2]",
                                   "description_article":"section[@class='chapo']",
                                   "not_to_crawl" : []}
        
        self.queues["leparisien"] = {"article" :["div[@class='article-full__body-content']"],
                                   "restricted" : ["iframe[@allowtransparency='true']"],
                                   "title": ["h1[@class='article-full__title']"],
                                   "author":"span[@class='article-full__infos-author']",
                                   "categorie": "span[@class='breadcrumb__item']",
                                   "description_article":"h2[@class='article-full__header']",
                                   "not_to_crawl" : ["/video/"]}
        
        self.queues["lexpress"] = {"article" :["div[@class='article_container']"],
                                  "restricted" : [],
                                  "title": ["h1[@class='title_alpha']"],
                                  "author":"span[@itemprop='author']",
                                  "categorie": "span[@itemprop='name']",
                                  "description_article":"h2[@itemprop='description']",
                                  "not_to_crawl" : []}
        
        self.queues["humanite"] = {"article" :["div[@class='group-pool-cache field-group-div']"],
                                  "restricted" : ["div[@id='app']"],
                                  "title": ["div[@class='group-ft-header-node-article field-group-div']/div[1]"],
                                  "author": "div[@class='group-ft-header-node-article field-group-div']/div[2]/div[2]",
                                  "categorie": "div[@class='group-top-node-article field-group-div']",
                                  "description_article":"div[@class='field-item even']",
                                  "not_to_crawl" : []}
                                    
        self.queues["liberation"] = {"article" :[ "div[@class='article-body read-left-padding']", "div[@class='width-wrap']/article/div[2]"],
                                      "restricted" : ["div[@class='paywall-deny']/div[2]"],
                                      "title": ["h1[@class='article-headline']"],
                                      "author":"span[@class='author']",
                                      "categorie": "div[@class='article-subhead']",
                                      "description_article":"h2[@class='article-standfirst read-left-padding']",
                                      "not_to_crawl" : []}

if __name__ == "__main__":
     environment_variables()
     m = Main() 