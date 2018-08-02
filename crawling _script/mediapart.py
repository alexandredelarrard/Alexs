# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 16:26:58 2018

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

class MediapartScrapping(Crawling):
    
    def __init__(self, min_date, queues, driver):
    
        Crawling.__init__(self)
        self.url= "https://www.mediapart.fr/"
        self.end_date = pd.to_datetime(min_date, format = "%Y-%m-%d")
        self.id_col_date = 0
        self.driver = driver
        self.queues = queues
        self.main_mediapart()

    def main_mediapart(self):
        
        self.driver.get(self.url)
        liste_menu_href = self.get_lis_from_nav("class","main-menu")
        liste_menu_href = [x for x in liste_menu_href if x not in [self.url, 
                                                                   '/studio'
                                                                   '//blogs.mediapart.fr/',
                                                                   '//blogs.mediapart.fr/edition/le-club-mode-demploi']]   
        for element in liste_menu_href:
            try:
                self.start_threads_and_queues(self.mediapart_article_information)
                t0 = time.time()
                self.get_max_pages(element)
                print('*** Main thread waiting')
                self.queues["urls"].join()
                print('*** Done in {0}'.format(time.time() - t0))
                self.save_results(element)
                
            except Exception as e:
                print(e)
                print(element)

            
    def mediapart_article_information(self, driver):

        # =============================================================================
        #  get all desired infromation of the list of articles : 20 per page       
        # =============================================================================
        ### liste time article appeared
        times = driver.find_elements_by_xpath("//div[@class='post-list universe-journal']/div/div/time")
        liste_times =[]
        for t in times:
            liste_times.append(t.get_attribute("datetime"))
        
        ### liste_href_articles
        href = driver.find_elements_by_xpath("//div[@class='post-list universe-journal']/div/h3/a")
        liste_href = []
        for h in href:
            liste_href.append(h.get_attribute("href"))
            
        articles = driver.find_elements_by_xpath("//div[@class='post-list universe-journal']/div")
        liste_text = []
        for ar in articles:
            if ar.get_attribute("data-type") == "article":
                liste_text.append(ar.text)

        information = np.array(np.transpose([x for x in [liste_times, liste_href, liste_text] if x != []]))
        
        try:
            assert len(liste_times) == len(liste_href) == len(liste_text)
        except AssertionError:
            pass
        
        return information
            

    def get_max_pages(self, element):
         self.driver.get(element)
         pagination = self.driver.find_element_by_xpath("//ul[@class='pager']")
         last_page = pagination.find_element_by_class_name("pager-last").text
         
         cap_articles = (datetime.now() - self.end_date).days*3
         
         if last_page.isdigit():
             max_pages = min(int(last_page), cap_articles)
         else:
             max_pages = 1
             
         print("max pages to crawl for {0} : {1}".format(element, max_pages))
         #### fill the queue with all possible urls
         for i in range(1, max_pages+1):
             self.queues["urls"].put(element+"?page=%i"%i)


### test unitaire
if __name__ == "__main__":
    lemonde = MediapartScrapping()
    