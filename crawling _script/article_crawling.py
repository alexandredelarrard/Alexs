# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 10:55:26 2018

@author: JARD
"""


import time
import numpy as np
import pandas as pd
from datetime import datetime

try:
    from crawling import Crawling
except Exception:
    import sys
    import os
    sys.path.append(os.environ["DIR_PATH"] + "/script/crawling_script")
    from crawling import Crawling


class ArticleCrawling(Crawling):
    
    def __init__(self, liste_urls, queues):
        """
        """
    
        Crawling.__init__(self)
        self.liste_urls = liste_urls
        self.queues = queues
    
    def main_url_crawling(self):
         
        self.start_threads_and_queues(self.crawl_article)
        t0 = time.time()
        for url in self.liste_urls:
            self.queues["urls"].put(url)
        print('*** Main thread waiting')
        self.queues["urls"].join()
        print('*** Done in {0}'.format(time.time() - t0))
        self.save_results(self.queues["carac"])
             
   
    def crawl_article(self, driver):
        
        main = driver.find_element_by_xpath("//" + self.queues["carac"]["main"])
        restricted = 1 if len(main.find_elements_by_xpath("//" + self.queues["carac"]["restricted"])) >0 else 0
        
        count_paragraphs = len(main.find_elements_by_tag_name("p"))
        count_h1 = "\n".join([x.text for x in main.find_elements_by_tag_name("h1")])
        count_h2 = "\n".join([x.text for x in main.find_elements_by_tag_name("h2")])
        count_h3 = "\n".join([x.text for x in main.find_elements_by_tag_name("h3")])
        
        texte = main.text
        information = np.transpose(np.array([[restricted, count_paragraphs, count_h1, count_h2, count_h3, texte]]))
        
        return information
    