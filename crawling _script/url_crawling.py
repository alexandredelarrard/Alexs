# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 16:26:58 2018

@author: JARD
"""

import time
import numpy as np
import pandas as pd 
from datetime import datetime, timedelta
import unidecode
import locale
locale.setlocale(locale.LC_ALL, 'fr_FR')
import dateparser


try:
    from crawling import Crawling
except Exception:
    import sys
    import os
    sys.path.append(os.environ["DIR_PATH"] + "/script/crawling_script")
    from crawling import Crawling


class URLCrawling(Crawling):
    
    def __init__(self,  queues, driver):
    
        Crawling.__init__(self)
        self.url= queues["carac"]["url_crawl"]["url"]
        self.end_date = pd.to_datetime("01-01-2000", format = "%d-%m-%Y")
        self.driver = driver 
        self.queues = queues
        self.journal = queues["carac"]["journal"]
        self.main_url()


    def main_url(self):
        """
        Main function initializing threads and the list of root urls to crawl
        Once one root url has been crawled, all drivers are closed and then reopened
        The queue_url element has to be empty in order to move to another root url
        """
        
        print("_"*40 + "\n\n" + "*"*15 + "  %s  "%self.journal + "*"*15 + "\n"+ "_"*40 )
        liste_menu_href = self.queues["carac"]["url_crawl"]["in_liste"]
    
        for element in liste_menu_href:

            self.start_threads_and_queues(self.article_information)
            t0 = time.time()
            self.fill_queue_url(element)
            print('*** Main thread waiting')
            self.queues["urls"].join()
            print('*** Done in {0}'.format(time.time() - t0))
            self.save_results(self.journal)
                
    
    def article_information(self, driver, queue):
        """
        function specific to each media
        This function crawl all important information per url.
        output :        - Date
                        - url full article
                        - Description in text such as title, small desc, autor, category
        """
        if len(queue["carac"]["url_crawl"]["href_element"]) > 0:
            href = []
            for string in queue["carac"]["url_crawl"]["href_element"]:
               href += driver.find_elements_by_xpath("//"+string)
            liste_href =[]
            for h in href:
                ref = h.get_attribute("href")
                if ref != "":
                    liste_href.append(ref)
        else:
             liste_href = eval("self.%s_href_element(driver)"%queue["carac"]["journal"])
        
        if len(queue["carac"]["url_crawl"]["time_element"]) == 4:
           
            liste_times =[]
            filters = queue["carac"]["url_crawl"]["time_element"]
            for link in liste_href:
                try:
                    date= pd.to_datetime("-".join(link.split(filters[3])[filters[0]:filters[1]]), format = filters[2])
                except Exception:
                    date = datetime.now()
                    
                liste_times.append(date)
        else:
            liste_times = eval("self.%s_time_element(driver)"%queue["carac"]["journal"])

        
        if len(queue["carac"]["url_crawl"]["article_element"]) > 0:
            articles = []
            for string in queue["carac"]["url_crawl"]["article_element"]:
                articles += driver.find_elements_by_xpath("//"+string)
            liste_text = []
            for ar in articles:
                texte = ar.text
                if texte != "":
                    liste_text.append(texte)
        else:
             liste_text = eval("self.%s_article_element(driver)"%queue["carac"]["journal"])
        
        information = np.array(np.transpose([x for x in [liste_times, liste_href, liste_text] if x != []]))
        driver.delete_all_cookies()

        return information
    
    
    def fill_queue_url(self, element):
         """
         Fill in the queue of urls based on the maximum number of pages with same url root 
         Depending on the number of days to crawl, the max_number of pages to crawl is capped
         The assumption is that there is 1 page of articles per day
         """
             
         cap_articles = (datetime.now() - self.end_date).days*3
         
         #### fill the queue with all possible urls
         if len(self.queues["carac"]["url_crawl"]["fill_queue"]) == 2:
             print("max pages to crawl for {0} : {1}".format(element, cap_articles))
             
             if self.journal == "lesechos":
                 for i in range(cap_articles*7 + 1, 1, -1):
                     new_url = element+self.queues["carac"]["url_crawl"]["fill_queue"][0].format(i)
                     self.queues["urls"].put(new_url)
             else:
                 for i in range(self.queues["carac"]["url_crawl"]["fill_queue"][1], cap_articles + 1):
                     new_url = element+self.queues["carac"]["url_crawl"]["fill_queue"][0].format(i)
                     self.queues["urls"].put(new_url)
         else:
             exec("self.fill_queue_url_%s(element)"%self.journal)
        
# =============================================================================
#  Fill in url queue with sepecific cases
# =============================================================================
    def fill_queue_url_lefigaro(self, element):
        self.driver = self.handle_timeout(self.driver, element)
        delta = datetime.now() - self.end_date
         
        print("max pages to crawl for {0} : {1}".format(element, delta.days))
        #### fill the queue with all possible urls
        for i in range(delta.days + 1):
            new_date = self.end_date + timedelta(i)
            new_month = str(new_date.month) if len(str(new_date.month)) ==2 else "0" +  str(new_date.month)
            new_day = str(new_date.day) if len(str(new_date.day)) ==2 else "0" +  str(new_date.day)
            self.queues["urls"].put(self.url + "{0}/{1}".format(str(new_date.year) + new_month, new_day))
             
             
    def fill_queue_url_latribune(self, element):
        self.driver = self.handle_timeout(self.driver, element+ "page-1")
        delta_liste = [unidecode.unidecode(d.strftime('%B-%Y')) for d in pd.date_range(self.end_date, datetime.now() + timedelta(30), freq='M')]
         
        print("pages to crawl for {0} : {1}".format(element, delta_liste))
        #### fill the queue with all possible urls
        for date in delta_liste:
            self.driver.get(element + date + "/page-1")
            time.sleep(2)
            try:
                ul = self.driver.find_element_by_xpath("//ul[@class='pagination-archive pages']")
                nbr_pages = int(ul.find_elements_by_tag_name("li")[-1].text)
                for i in range(1, nbr_pages+1):
                    self.queues["urls"].put(element + date +"/page-%i"%i)
            except Exception:
                pass
        
# =============================================================================
# specific element crawling into time, url text of article        
# =============================================================================      
    #### la tribune
    def latribune_time_element(self, driver):
        date = driver.current_url.split("/")[-2].replace("aout", "août").replace("fevrier", "février").replace("decembre", "décembre")
        date = pd.to_datetime(date, format = "%B-%Y").strftime("%Y-%m-%d")
        href = driver.find_elements_by_xpath("//article[@class='article-wrapper row clearfix ']")
        liste_times =[]
        for h in href:
            liste_times.append(date)
        return liste_times
    
    #### lesechos
    def lesechos_time_element(self, driver):
        times = driver.find_elements_by_xpath("//article[@class='liste-article']/div/time")
        liste_times =[]
        for t in times:
            liste_times.append(t.get_attribute("datetime"))
        return liste_times
    
    #### humanite
    def humanite_time_element(self, driver):
        times = driver.find_elements_by_xpath("//div[@class='group-ft-description field-group-div']/div[4]/div/div/div/span")
        liste_times =[]
        for t in times:
            liste_times.append(dateparser.parse(t.text).strftime("%Y-%m-%d"))
        return liste_times

    #### lexpress
    def lexpress_time_element(self, driver):
        href = driver.find_elements_by_xpath("//div[@class='groups']/div/a")
        
        ### say we have 3 articles on average per day  per category at max, will then stop when number is reached
        delta = (datetime.now() - self.end_date).days
        if delta < self.queues["results"].qsize():
            date = self.end_date - timedelta(10)
        else:
            date = datetime.now()
            
        liste_times = []
        for i in range(len(href)): 
            liste_times.append(date)
        return liste_times
    
    #### liberation
    def liberation_href_element(self, driver):
        lis = driver.find_elements_by_xpath("//ul[@class='live-items']/li/div/p/a")
        liste_href =[]
        for h in lis:
            liste_href.append(h.get_attribute("href"))
            
        lis = driver.find_elements_by_xpath("//ul[@class='live-items']/li/a")
        for h in lis:
            if h.get_attribute("class") == "tag-first-item-link":
                liste_href.append(h.get_attribute("href"))
            
        return liste_href

