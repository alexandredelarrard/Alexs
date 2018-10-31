# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 10:55:26 2018

@author: JARD
"""


import time
import numpy as np
import pandas as pd
import os
import glob
from datetime import datetime
from random import shuffle

try:
    from crawling import Crawling
except Exception:
    import sys
    sys.path.append(os.environ["DIR_PATH"] + "/script/crawling_script")
    from crawling import Crawling


class ArticleCrawling(Crawling):
    
    def __init__(self, queues):
        """
        """
        Crawling.__init__(self)
        self.journal = queues["carac"]["journal"]
        self.queues = queues
        self.main_article_crawling()
        
    
    def main_article_crawling(self):
         
        self.get_liste_urls()
        self.start_threads_and_queues(self.crawl_article)
        t0 = time.time()
        for url in self.liste_urls.to_dict(orient='records'):
            self.queues["urls"].put(url)
        print('*** Main thread waiting')
        self.queues["urls"].join()
        print('*** Done in {0}'.format(time.time() - t0))
             
        
    def get_liste_urls(self):
        
        data = pd.read_csv(os.environ["DIR_PATH"] + "/data/history/url/history/%s_history.csv"%self.journal)
        shape = data.shape[0]
        index = data["url"].apply(lambda x  : True if  "/video/" not in x else False)
        data= data[index]
        self.liste_urls = data[["date", "url"]]
#        self.liste_urls["short_url"] =  self.liste_urls['url'].apply(lambda x: x.replace("http://www.","https://www."))
        
        #### already done:
        files= glob.glob( "/".join([os.environ["DIR_PATH"], "data", self.journal, self.queues["carac"]["url_article"], "*", "extraction_*.csv"]) )
        if len(files)>0:
            for i, f in enumerate(files):
                try:
                    if i ==0:
                        total = pd.read_csv(f, sep= "#")
                    else:
                        total = pd.concat([total, pd.read_csv(f, sep= "#")], axis=0)
                except Exception:
                    print(f)
                    
            self.liste_urls = pd.merge(data[["date", "url"]],total[["url", "article"]], on ="url", how= "left")
            self.liste_urls = self.liste_urls.loc[pd.isnull(self.liste_urls["article"])][["date", "url"]]
        else:
            total= pd.DataFrame([])
        print("total number of articles to crawl is {0} / {1}, already crawled : {2} articles".format(self.liste_urls.shape[0], shape, total.shape[0]))
    
    
    def crawl_article(self, driver, queues, date):
        
        url = driver.current_url
        journal = self.journal
        queue = queues["carac"]["article_crawl"]
        
        if len([1 for x in queue["not_to_crawl"] if x in url]) == 0:

            # =============================================================================
            #         Is article restricted
            # =============================================================================
            restricted = 0
            for string in queue["restricted"]:
                 if len(driver.find_elements_by_xpath("//" + string)) >0:
                    restricted = 1
                
            # =============================================================================
            #             Article Title
            # =============================================================================
            title = ""
            for string in queue["title"]:
                if len(driver.find_elements_by_xpath("//" + string)) >0:
                        title += driver.find_element_by_xpath("//" + string).text
                        break
                    
            # =============================================================================
            #             Article Categorie
            # =============================================================================
            categorie = ""
            for string in queue["categorie"]:
                if len(driver.find_elements_by_xpath("//" + string)) >0:
                    categorie += driver.find_element_by_xpath("//" + string).text 

            
            # =============================================================================
            #             Article Categorie
            # =============================================================================
            description_article = ""
            for string in queue["description_article"]:
                if len(driver.find_elements_by_xpath("//" + string)) >0:
                    description_article += driver.find_element_by_xpath("//" + string).text 
            
            # =============================================================================
            #             Article author
            # =============================================================================
            author = ""
            for string in queue["author"]:
                if len(driver.find_elements_by_xpath("//" + string)) >0:
                    author += driver.find_element_by_xpath("//" + string).text 

            # =============================================================================
            #             Article content
            # =============================================================================
            article = ""
            for string in queue["article"]:
                if len(driver.find_elements_by_xpath("//" + string)) >0:
                    article += driver.find_element_by_xpath("//" + string).text
                    break

            information = [date, journal, driver.current_url, restricted, str(title), str(author), str(article), str(categorie), str(description_article)]
        else:
            information = [date, journal, driver.current_url, '', '', '', '', '', '']
            
        return information
    