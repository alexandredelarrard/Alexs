# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 16:26:58 2018

@author: JARD
"""

import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

try:
    from crawling import Crawling
except Exception:
    import sys
    import os
    sys.path.append(os.environ["DIR_PATH"] + "/script/crawling_script")
    from crawling import Crawling

class LefigaroScrapping(Crawling):
    
    def __init__(self, min_date, queues, driver):
    
        Crawling.__init__(self)
        self.url= "http://articles.lefigaro.fr/"
        self.end_date = pd.to_datetime(min_date, format = "%Y-%m-%d")
        self.id_col_date = 0
        self.driver = driver
        self.queues = queues
        self.main_lefigaro()


    def main_lefigaro(self):
        
        t0 = time.time()
        self.start_threads_and_queues(self.mediapart_article_information)
        self.get_max_pages()
        print('*** Main thread waiting')
        self.queues["urls"].join()
        print('*** Done in {0}'.format(time.time() - t0))
        self.save_results(self.url)
            
    def mediapart_article_information(self, driver):

        url = "http://www.lefigaro.fr/"
        # =============================================================================
        #  get all desired infromation of the list of articles  
        # =============================================================================
        ### liste_href_articles
        href = driver.find_elements_by_xpath("//div[@class='SiteMap']/a")
        liste_href = []
        liste_times =[]
        for h in href:
            link = h.get_attribute("href")
            integers = [x for x in link.replace(url, "").split("/") if x.isdigit()]
            time = "-".join(integers)
            liste_href.append(link)
            liste_times.append(time)
            
        articles = driver.find_elements_by_xpath("//div[@class='SiteMap']/a")
        liste_text = []
        for ar in articles:
            liste_text.append(ar.text)

        information = np.array(np.transpose([x for x in [liste_times, liste_href, liste_text] if x != []]))
        
        try:
            assert len(liste_times) == len(liste_href) == len(liste_text)
        except AssertionError:
            pass
        
        return information
            

    def get_max_pages(self):
        
         self.driver.get(self.url)
         delta = datetime.now() - self.end_date
         
         print("max pages to crawl for {0} : {1}".format(self.url, delta.days))
         #### fill the queue with all possible urls
         for i in range(delta.days + 1):
             new_date = self.end_date + timedelta(i)
             new_month = str(new_date.month) if len(str(new_date.month)) ==2 else "0" +  str(new_date.month)
             new_day = str(new_date.day) if len(str(new_date.day)) ==2 else "0" +  str(new_date.day)
             self.queues["urls"].put(self.url + "{0}/{1}".format(str(new_date.year) + new_month, new_day))

### test unitaire
if __name__ == "__main__":
    lefigaro = LefigaroScrapping("2018-07-01")
    