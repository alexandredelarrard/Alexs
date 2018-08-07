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

try:
    from crawling import Crawling
except Exception:
    import sys
    sys.path.append(os.environ["DIR_PATH"] + "/script/crawling_script")
    from crawling import Crawling


class ArticleCrawling(Crawling):
    
    def __init__(self, end_date, queues):
        """
        """
        Crawling.__init__(self)
        self.journal = queues["carac"]["journal"]
        self.end_date = end_date
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
        self.save_results(self.queues["carac"])
             
        
    def get_liste_urls(self):

        liste_files = glob.glob("/".join([os.environ["DIR_PATH"], "data", self.journal, "url", "*/*.csv"]))
            
        for i, f in enumerate(liste_files):
            if i == 0:
                data = pd.read_csv(f, encoding = "latin1")
            else:
                data= pd.concat([data, pd.read_csv(f, encoding = "latin1")], axis=0)
        self.liste_urls = set(data["1"].tolist())
        print("total number of articles to crawl is {0}".format(len(self.liste_urls)))
             
   
    def crawl_article(self, driver, queues):
        
        if len(queues["carac"]["article_crawl"]["restricted"]) > 0:    
            for string in self.queues["carac"]["article_crawl"]["restricted"]:
                try:
                    element_restrict = driver.find_elements_by_xpath("//" + string)
                    restricted = 1 if len(element_restrict) >0 else 0
                    text_restriction = element_restrict[0].text
                except Exception:
                    pass 
        else:
            restricted = 0
            text_restriction = ""
        
        ### have to reopen a driver with other IP because article is not fully available
        if restricted == 1:
            raise Exception
        
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
            try:
                main = driver.find_element_by_xpath("//" + string)
            except Exception:
                pass

        count_paragraphs = len(main.find_elements_by_tag_name("p"))
        count_h1 = "\n".join([x.text for x in main.find_elements_by_tag_name("h1")])
        count_h2 = "\n".join([x.text for x in main.find_elements_by_tag_name("h2")])
        count_h3 = "\n".join([x.text for x in main.find_elements_by_tag_name("h3")])
        
        texte = main.text
        information = np.transpose(np.array([[datetime.now(), driver.current_url, restricted, text_restriction, count_paragraphs, count_h1, count_h2, count_h3, head, texte]]))
        
        return information