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
import pymongo
from datetime import timedelta
import pandas as pd

from production.crawling.url_crawling import UrlCrawling
from production.crawling.article_crawling import ArticleCrawling
from production.nlp.clustering import ClusteringArticles
from production.nlp.classification import ClassificationSujet


def environment_variables():
    configParser = configparser.RawConfigParser() 
    if platform.system() in  ["Darwin", "Linux"]: 
        configFilePath = os.environ["PROFILE"] + '/config_alexs.txt'
    else:
        configFilePath = os.environ["USERPROFILE"] + '/config_alexs.txt'
    configParser.read(configFilePath)
    os.environ["DIR_PATH"] = configParser.get("config-Alexs", "project_path")
    return configParser


class Main(object):
    
    def __init__(self):
        
        self.analytics = False
        self.config = environment_variables()
        self.queues = {"drivers": Queue(), "urls" :  Queue(), "results": Queue()}        
        self.specificities()
        self.main()


    def main(self):
        
        time_tot = time.time()
        
        if not self.analytics:
# =============================================================================
#             #### url crawling
# =============================================================================
            t0 = time.time()
            liste_urls = UrlCrawling(self.queues).main_url_crawling()
            print("[{0}] URL Crawling in {1}s\n {2} articles to crawl".format(datetime.datetime.today().strftime("%Y-%m-%d"), time.time() - t0, liste_urls.shape[0]))
            
# =============================================================================
#             #### article crawling
# =============================================================================
            t0 = time.time()
            new_articles = ArticleCrawling(self.queues, liste_urls).main_article_crawling()
            print("[{0}] Article Crawling in {1}s\n".format(datetime.datetime.today().strftime("%Y-%m-%d"), time.time() - t0))
        
        else:
            new_articles = pd.read_csv(os.environ["DIR_PATH"] + "/data/continuous_run/article/extraction_%s.csv"%(datetime.datetime.today()-timedelta(days=0)).strftime("%Y-%m-%d"), sep = "#")
#            new_articles = pd.read_csv(os.environ["DIR_PATH"] + "/data/continuous_run/article/extraction_2018-12-05.csv", sep = "#")
        
# =============================================================================
#         ### clustering articles
# =============================================================================
        t0 = time.time()
        new_articles, new_clusters = ClusteringArticles(new_articles).main_article_clustering()
        print("[{0}] Clustering in {1}s\n".format(datetime.datetime.today().strftime("%Y-%m-%d"), time.time() - t0))
            
# =============================================================================
#         ### classification sujets des articles
# =============================================================================
        try:
            t0 = time.time()
            new_articles, new_clusters = ClassificationSujet(new_articles, new_clusters).main_classification_sujets()
            print("[{0}] Classification sujets in {1}s\n".format(datetime.datetime.today().strftime("%Y-%m-%d"), time.time() - t0))
        except Exception as e:
            print(e)
            pass
        
        if not self.analytics:
            ### enregistrer articles supplementaires
            path_name = "/".join([os.environ["DIR_PATH"], "data", "continuous_run", "article", "extraction_{0}.csv".format(datetime.datetime.now().strftime("%Y-%m-%d"))]) 
            new_articles.to_csv(path_name, index=False, sep='#')
                                
        new_clusters = pd.DataFrame.from_dict(new_clusters, orient='index') #.to_csv("/".join([os.environ["DIR_PATH"], "data", "continuous_run", "article", "cluster_{0}.csv".format(datetime.datetime.now().strftime("%Y-%m-%d"))]), index=False, sep='#')
       
# =============================================================================
#         #### mongodb
# =============================================================================
        connection = pymongo.MongoClient(self.config.get("config-Alexs", "mongodb"))
        db=connection[self.config.get("config-Alexs", "mongo_db_name")]
        
        collection_articles = db.get_collection("articles") 
        collection_articles.delete_many({})
        collection_articles.insert_many(new_articles.to_dict(orient='records'))
        
        collection_clusters = db.get_collection("clusters")
        collection_clusters.delete_many({})
        collection_clusters.insert_many(new_clusters.to_dict(orient='records'))
        
        print("total updating time {0}s".format(time.time() - time_tot))


    def getCollectionKeys(collection):
        """Get a set of keys from a collection"""
        
        keys_list = []
        collection_list = collection.find()
        for document in collection_list:
            for field in document.keys():
                keys_list.append(field)
        keys_set =  set(keys_list)
        return keys_set


    def specificities(self):

        self.queues["lemonde"] = {"article" :["div[@itemprop='articleBody']", "article[@class='article article_normal']", "div[@class='description']", "section[@class='article__content']", "section[@class='article__content article__content--restricted']", 
                                  "section[@class='article__content article__content--restricted-media']"],
                                  "restricted" : ["div[@id='teaser_article']", "div[@class='Paywall']/div/div/div"],
                                  "title": ["h1[@itemprop='Headline']", "h1[@class='article__title']", "article[@class='main']/header/h1"],
                                  "author":["span[@itemprop='author']", "span[@itemprop='Publisher']"],
                                  "categorie": ["nav[@id='nav_ariane']/ul/li", "ul[@class='breadcrumb']"],
                                  "description_article": ["p[@itemprop='description']", "p[@class='article__desc']"],
                                  "not_to_crawl" : ["/live/", "/video/", "/blog-mediateur/"]}
        
        self.queues["lefigaro"] = {"article" :["div[@class='fig-content__body']", "div[@itemprop='articleBody']"],
                                  "restricted" : ["div[@class='fig-premium-paywall']", "div[@class='fig-premium-paywall__infos']"],
                                  "title": ["h1[@class='fig-main-title']", "div[@class='fig-article-headline']/h1"],
                                  "author":["a[@class='fig-content-metas__author']", "span[@itemprop='autor']"],
                                  "categorie": ["li[@class='fig-breadcrumb__item fig-breadcrumb__item--current']", "span[@itemprop='name']"],
                                  "description_article":["p[@class='fig-content__chapo']", "p[@class='fig-chapo']"],
                                  "not_to_crawl" : ["/story/", "/photos/"]}
                
        self.queues["lesechos"] = {"article" :["div[@class='paywall']", "div[@class='main-content content-article']", "div[@itemprop='articleBody']"],
                                   "restricted": [],
                                  "title": ["h1[@class='title-article']","h1[@itemprop='Headline']"],
                                  "author":["div[@class='signature-article']"],
                                  "categorie": ["li[@class='opened']"],
                                  "description_article":["p[@itemprop='articleBody']"],
                                  "not_to_crawl" : []}
        
        self.queues["mediapart"] = {"article" :["div[@class='content-article content-restricted']", "div[@class='content-article']"],
                                  "restricted" : ["div[@class='content-article content-restricted']"],
                                  "title": ["h1[@class='title']"],
                                  "author": ["div[@class='author']/a"],
                                  "categorie": ["ul[@class='taxonomy']"],
                                  "description_article":["div[@class='introduction']"],
                                  "not_to_crawl" : []}
        
        self.queues["latribune"] = {"article" :["div[@class='article-content-wrapper']"],
                                   "restricted" : [],
                                   "title": ["h1[@itemprop='Headline']"],
                                   "author":["span[@itemprop='name']"],
                                   "categorie": ["div[@class='article-title-wrapper']/ol/li[2]"],
                                   "description_article":["section[@class='chapo']"],
                                   "not_to_crawl" : []}
        
        self.queues["leparisien"] = {"article" :["div[@class='article-full__body-content']"],
                                   "restricted" : ["iframe[@allowtransparency='true']"],
                                   "title": ["h1[@class='article-full__title']"],
                                   "author":["span[@class='article-full__infos-author']"],
                                   "categorie": ["span[@class='breadcrumb__item']"],
                                   "description_article":["h2[@class='article-full__header']"],
                                   "not_to_crawl" : ["/video/"]}
        
        self.queues["lexpress"] = {"article" :["div[@class='article_container']"],
                                  "restricted" : [],
                                  "title": ["h1[@class='title_alpha']"],
                                  "author":["span[@itemprop='author']"],
                                  "categorie": ["span[@itemprop='name']"],
                                  "description_article":["h2[@itemprop='description']"],
                                  "not_to_crawl" : []}
        
        self.queues["humanite"] = {"article" :["div[@class='group-pool-cache field-group-div']"],
                                  "restricted" : ["div[@id='app']"],
                                  "title": ["div[@class='group-ft-header-node-article field-group-div']/div[1]"],
                                  "author": ["div[@class='group-ft-header-node-article field-group-div']/div[2]/div[2]"],
                                  "categorie": ["div[@class='group-top-node-article field-group-div']"],
                                  "description_article":["div[@class='field-item even']"],
                                  "not_to_crawl" : []}
                                    
        self.queues["liberation"] = {"article" :[ "div[@class='article-body read-left-padding']", "div[@class='width-wrap']/article/div[2]", "div[@class='article-body']"],
                                      "restricted" : ["div[@class='paywall-deny']/div[2]"],
                                      "title": ["h1[@class='article-headline']"],
                                      "author":["span[@class='author']"],
                                      "categorie": ["div[@class='article-subhead']"],
                                      "description_article":["h2[@class='article-standfirst read-left-padding']","h2[@class='article-standfirst']"],
                                      "not_to_crawl" : ['/photographie/']}

if __name__ == "__main__":
     os.environ["DIR_PATH"] = r"C:\Users\User\Documents\Alexs"
     m = Main() 