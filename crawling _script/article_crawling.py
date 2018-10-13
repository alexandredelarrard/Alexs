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
        for url in self.liste_urls:
            self.queues["urls"].put(url)
        print('*** Main thread waiting')
        self.queues["urls"].join()
        print('*** Done in {0}'.format(time.time() - t0))
             
        
    def get_liste_urls(self):
        
        data = pd.read_csv(os.environ["DIR_PATH"] + "/data/clean_data/history/%s_history.csv"%self.journal)
        shape = data.shape[0]
        index = data["url"].apply(lambda x  : True if  "/www.dailymotion.com/video/" not in x else False)
        data= data[index]
        self.liste_urls =  data['url'].tolist()
        
        #### already done:
        files= glob.glob( "/".join([os.environ["DIR_PATH"], "data", self.journal, self.queues["carac"]["url_article"], "*", "extraction_*.csv"]) )
        if len(files)>0:
            for i, f in enumerate(files):
                try:
                    if i ==0:
                        total = pd.read_csv(f, error_bad_lines=False)
                    else:
                        total = pd.concat([total, pd.read_csv(f, error_bad_lines=False)], axis=0)
                except Exception:
                    print(f)
            already_crawled = total["0"]
            self.liste_urls = list(set(self.liste_urls) - set(already_crawled))
        else:
            self.liste_urls = data["url"].tolist()
            already_crawled = []
            
        shuffle(self.liste_urls)
        print("total number of articles to crawl is {0} / {1}, already crawled : {2} articles".format(len(self.liste_urls), shape, len(already_crawled)))
    
    
    def crawl_article(self, driver, queues):
        
        restricted = []
        try:
            if len(queues["carac"]["article_crawl"]["restricted"]) > 0:    
                for string in queues["carac"]["article_crawl"]["restricted"]:
                    element_restrict = driver.find_elements_by_xpath("//" + string)
                    restricted.append(1 if len(element_restrict) >0 else 0)
                restricted = max(restricted)    
        except Exception:
            restricted = 0
            pass 
        
        ### have to reopen a driver with other IP because article is not fully available
#        if restricted == 1:
#            raise Exception
        
        if len(queues["carac"]["article_crawl"]["head_article"])>0:
            head = ""
            for string in queues["carac"]["article_crawl"]["head_article"]:
                try:
                    head += driver.find_element_by_xpath("//" + string).text + "\n"
                except Exception:
                    pass
        else:
            head = ""
           
        for string in queues["carac"]["article_crawl"]["main"]:
            if len(driver.find_elements_by_xpath("//" + string))>0:
                main = driver.find_element_by_xpath("//" + string)
                count_paragraphs = len(main.find_elements_by_tag_name("p"))
                count_h1 = "\n".join([x.text for x in main.find_elements_by_tag_name("h1")])
                count_h2 = "\n".join([x.text for x in main.find_elements_by_tag_name("h2")])
                count_h3 = "\n".join([x.text for x in main.find_elements_by_tag_name("h3")])
                texte = main.text
                break
                
        information = np.column_stack([driver.current_url, restricted, count_paragraphs, count_h1, count_h2, count_h3, str(head), str(texte.replace("\"","'"))])
        return information