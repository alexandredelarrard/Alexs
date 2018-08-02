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

class LesechosScrapping(Crawling):
    
    def __init__(self, min_date):
    
        Crawling.__init__(self)
        self.url= "lesechos"
        self.end_date = pd.to_datetime(min_date, format = "%Y-%m-%d")
        self.id_col_date = 0
        self.main_lesechos()


    def main_lesechos(self):
        
        t0 = time.time()
        self.start_threads_and_queues(self.lesechos_article_information)
        self.get_max_pages()
        print('*** Main thread waiting')
        self.queues["urls"].join()
        print('*** Done in {0}'.format(time.time() - t0))
        self.save_results(self.url)

            
    def lesechos_article_information(self, driver):

        # =============================================================================
        #  get all desired infromation of the list of articles : 20 per page       
        # =============================================================================
        ### liste time article appeared
        times = driver.find_elements_by_xpath("//article[@class='liste-article']/div/time")
        liste_times =[]
        for t in times:
            liste_times.append(t.get_attribute("datetime"))
        
        ### liste_href_articles
        href = driver.find_elements_by_xpath("//article[@class='liste-article']/h2/a")
        liste_href = []
        for h in href:
            liste_href.append(h.get_attribute("href"))
            
        articles = driver.find_elements_by_xpath("//article[@class='liste-article']")
        liste_text = []
        for ar in articles:
            liste_text.append(ar.text)

        information = np.array(np.transpose([x for x in [liste_times, liste_href, liste_text] if x != []]))

        return information
            

    def get_max_pages(self):
        
         self.driver.get("http://recherche.lesechos.fr/recherche.php?exec=2&texte=&dans=touttexte&ftype=-1&date1={0}&date2={1}&page=1".format(self.end_date.strftime("%Y-%m-%d"), datetime.now().strftime("%Y-%m-%d")))
         pagination = self.driver.find_element_by_xpath("//div[@class='main-content content-page']/header/div")
         last_page = pagination.text.split("sur")[1].replace("résultats","").strip().replace(" ","")
        
         if last_page.isdigit():
             max_pages = int(int(last_page)/10) # because 10 results per page
         else:
             max_pages = 1
             
         print("max pages to crawl for {0} : {1}".format(self.url, max_pages))
         #### fill the queue with all possible urls
         for i in range(1, max_pages+1):
             self.queues["urls"].put(self.url+ "&date1={0}&date2={1}&page={2}".format(self.end_date.strftime("%Y-%m-%d"), datetime.now().strftime("%Y-%m-%d"), i))


### test unitaire
if __name__ == "__main__":
    lemonde = LesechosScrapping()
    