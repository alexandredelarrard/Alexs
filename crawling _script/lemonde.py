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

class LemondeScrapping(Crawling):
    
    def __init__(self, min_date, queues, driver):
    
        Crawling.__init__(self)
        self.url= "https://www.lemonde.fr/"
        self.end_date = pd.to_datetime(min_date, format = "%Y-%m-%d")
        self.id_col_date = 0
        self.driver = driver
        self.queues = queues
        self.main_lemonde()


    def main_lemonde(self):
        """
        Main function initializing threads and the list of root urls to crawl
        Once one root url has been crawled, all drivers are closed and then reopened
        The queue_url element has to be empty in order to move to another root url
        """
        
        print("_"*40 + "\n\n" + "*"*15 + "  Le monde  " + "*"*15 + "\n"+ "_"*40 )
        
        self.driver.get(self.url)
        liste_menu_href = self.get_lis_from_nav("id","navigation-generale")
        liste_menu_href = [x for x in liste_menu_href if x not in [self.url, 
                                                                   'https://www.lemonde.fr/campus/',
                                                                   'https://www.lemonde.fr/m-le-mag/',
                                                                  'https://www.lemonde.fr/pixels/',
                                                                  'https://www.lemonde.fr/teaser/presentation.html#xtor=CS1-263[BOUTONS_LMFR]-[BARRE_DE_NAV]-5-[Home]',
                                                                  'https://www.lemonde.fr/grands-formats/',
                                                                  'https://www.lemonde.fr/les-decodeurs/',
                                                                  'https://www.lemonde.fr/videos/',
                                                                  'https://www.lemonde.fr/data/',
                                                                  'https://www.lemonde.fr/guides-d-achat/']]   
        for element in liste_menu_href:
            
            try:
                self.start_threads_and_queues(self.lemonde_article_information)
                t0 = time.time()
                self.get_max_pages(element)
                print('*** Main thread waiting')
                self.queues["urls"].join()
                print('*** Done in {0}'.format(time.time() - t0))
                self.save_results(element)
                
            except Exception as e:
                print(e)
                print(element)
  
    
    def lemonde_article_information(self, driver):
        """
        function specific to each media
        This function crawl all important information per url.
        output :        - Date
                        - url full article
                        - Description in text such as title, small desc, autor, category
        """
        # =============================================================================
        #  get all desired infromation of the list of articles : 20 per page       
        # =============================================================================
        ### liste time article appeared
        times = driver.find_elements_by_xpath("//time[@class='grid_1 alpha']")
        liste_times =[]
        for t in times:
            liste_times.append(t.get_attribute("datetime"))
        
        ### liste_href_articles
        href = driver.find_elements_by_xpath("//div[@class='grid_11 conteneur_fleuve omega']/div/h3/a")
        liste_href =[]
        for h in href:
            liste_href.append(h.get_attribute("href"))
        
        ### text in each item article
        nbr = driver.find_elements_by_tag_name("article")
        liste_text = []
        for comment in nbr:
            liste_text.append(comment.text)
             
        information = np.array(np.transpose([x for x in [liste_times, liste_href, liste_text] if x != []]))
        
        try:
            assert len(liste_times) == len(liste_href) == len(liste_text)
        except AssertionError:
            pass
        
        return information
            
        
    def get_max_pages(self, element):
         """
         Fill in the queue of urls based on the maximum number of pages with same url root 
         Depending on the number of days to crawl, the max_number of pages to crawl is capped
         """
         
         self.driver.get(element+"1.html")
         pagination = self.driver.find_element_by_xpath("//div[@class='conteneur_pagination']")
         last_page = pagination.find_element_by_class_name("adroite").text
         
         cap_articles = (datetime.now() - self.end_date).days*3
          
         if last_page.isdigit():
             max_pages = min(int(last_page), cap_articles)
         else:
             max_pages = 500
             
         print("max pages to crawl for {0} : {1}".format(element, max_pages))
         #### fill the queue with all possible urls
         for i in range(1, max_pages+1):
             self.queues["urls"].put(element+"{0}.html".format(i))


### test unitaire
if __name__ == "__main__":
    lemonde = LemondeScrapping()
    