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
        
        def filtered_url(x, not_to_crawl):
            for a in not_to_crawl:
                if a in x:
                    return 0
            return 1
        
        liste_files = glob.glob("/".join([os.environ["DIR_PATH"], "data", self.journal, "url", "*/*.csv"]))
            
        for i, f in enumerate(liste_files):
            if i == 0:
                data = pd.read_csv(f, encoding = "latin1")
            else:
                data= pd.concat([data, pd.read_csv(f, encoding = "latin1")], axis=0)
        data["0-1"] = data["1"].apply(lambda x : filtered_url(x, self.queues["carac"]["article_crawl"]["not_to_crawl"]))
        self.liste_urls =  set(data.loc[data["0-1"] == 1, "1"].tolist())
        print("total number of articles to crawl is {0} / {1}".format(len(self.liste_urls), data.shape[0]))

    
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
           
        try:
            for string in queues["carac"]["article_crawl"]["main"]:
                if len(driver.find_elements_by_xpath("//" + string))>0:
                    main = driver.find_element_by_xpath("//" + string)
                    count_paragraphs = len(main.find_elements_by_tag_name("p"))
                    count_h1 = "\n".join([x.text for x in main.find_elements_by_tag_name("h1")])
                    count_h2 = "\n".join([x.text for x in main.find_elements_by_tag_name("h2")])
                    count_h3 = "\n".join([x.text for x in main.find_elements_by_tag_name("h3")])
                    texte = main.text
                    break
                
        except Exception:
            count_paragraphs = -1
            count_h1 = -1
            count_h2 = -1
            count_h3 = -1
            texte = ""
            pass
                
        information = np.column_stack([datetime.now(), driver.current_url, restricted, count_paragraphs, count_h1, count_h2, count_h3, head, texte])
        return information