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
import dateparser

try:
    from crawling import Crawling
except Exception:
    import sys
    import os
    sys.path.append(os.environ["DIR_PATH"] + "/script/crawling_script")
    from crawling import Crawling


class URLCrawling(Crawling):
    
    def __init__(self, min_date, queues, driver):
    
        Crawling.__init__(self)
        self.url= queues["carac"]["url_crawl"]["url"]
        self.end_date = pd.to_datetime(min_date, format = "%Y-%m-%d")
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
        
        if queue["carac"]["url_crawl"]["time_element"] != []:
            times = []
            for string in queue["carac"]["url_crawl"]["time_element"]:
                times += driver.find_elements_by_xpath("//"+string)
            liste_times =[]
            for t in times:
                liste_times.append(t.get_attribute("datetime"))
        else:
            liste_times = eval("self.%s_time_element(driver)"%queue["carac"]["journal"])

        if queue["carac"]["url_crawl"]["href_element"] != []:
            href = []
            for string in queue["carac"]["url_crawl"]["href_element"]:
               href += driver.find_elements_by_xpath("//"+string)
            liste_href =[]
            for h in href:
                liste_href.append(h.get_attribute("href"))
        else:
           liste_href =  eval("self.%s_href_element(driver)"%queue["carac"]["journal"])
        
        if queue["carac"]["url_crawl"]["article_element"] != []:
            articles = []
            for string in queue["carac"]["url_crawl"]["article_element"]:
                articles += driver.find_elements_by_xpath("//"+string)
            liste_text = []
            for ar in articles:
                liste_text.append(ar.text)
        else:
             liste_text = eval("self.%s_article_element(driver)"%queue["carac"]["journal"])
        
        information = np.array(np.transpose([x for x in [liste_times, liste_href, liste_text] if x != []]))
        print(information.shape)
        return information
    
    
    def fill_queue_url(self, element):
         """
         Fill in the queue of urls based on the maximum number of pages with same url root 
         Depending on the number of days to crawl, the max_number of pages to crawl is capped
         """
         
         if len(self.queues["carac"]["url_crawl"]["fill_queue"]) ==3:
             self.driver = self.handle_timeout(self.driver, element)
             try:
                 pagination = self.driver.find_element_by_xpath("//"+self.queues["carac"]["url_crawl"]["fill_queue"][0])
                 last_page = pagination.find_element_by_class_name(self.queues["carac"]["url_crawl"]["fill_queue"][1]).text
             except Exception:
                 last_page = '100'
                 
             cap_articles = (datetime.now() - self.end_date).days*3
              
             if last_page.isdigit():
                 max_pages = min(int(last_page), cap_articles)
             else:
                 max_pages = 100
                             
             #### fill the queue with all possible urls
             print("max pages to crawl for {0} : {1}".format(element, max_pages))
             for i in range(1,max_pages+1):
                 self.queues["urls"].put(element+self.queues["carac"]["url_crawl"]["fill_queue"][2].format(i))
         else:
            exec("self.fill_queue_url_%s(element)"%self.queues["carac"]["journal"])
        
# =============================================================================
#  Fill in url queue with sepecific cases
# =============================================================================
    def fill_queue_url_lesechos(self, element):
         url = "http://recherche.lesechos.fr/recherche.php?exec=2&texte=&dans=touttexte&ftype=-1&"
         self.driver = self.handle_timeout(self.driver, url + "date1={0}&date2={1}&page=1".format(self.end_date.strftime("%Y-%m-%d"), datetime.now().strftime("%Y-%m-%d")))
         pagination = self.driver.find_element_by_xpath("//div[@class='main-content content-page']/header/div")
         last_page = pagination.find_element_by_tag_name("strong").text.replace("résultats","").strip().replace(" ","")
        
         if last_page.isdigit():
             max_pages = int(int(last_page)/10) # because 10 results per page
         else:
             max_pages = 1
             
         print("max pages to crawl for {0} : {1}".format(self.url, max_pages))
         #### fill the queue with all possible urls
         for i in range(1, max_pages+1):
             self.queues["urls"].put(url + "date1={0}&date2={1}&page={2}".format(self.end_date.strftime("%Y-%m-%d"), datetime.now().strftime("%Y-%m-%d"), i))
             
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
        locale.setlocale(locale.LC_ALL, 'fr_FR')
        self.driver = self.handle_timeout(self.driver, element+ "page-1")
        delta_liste = [unidecode.unidecode(d.strftime('%B-%Y')) for d in pd.date_range(self.end_date, datetime.now(), freq='M')]
         
        print("max pages to crawl for {0} : {1}".format(element,len(delta_liste)*5))
        #### fill the queue with all possible urls
        for date in delta_liste:
            self.driver.get(element + date + "/page-1")
            nbr_pages = len(self.driver.find_element_by_xpath("//ul[@class='pagination-archive pages']").find_elements_by_tag_name("li"))
            for i in range(1, nbr_pages+1):
                self.queues["urls"].put(element + date +"/page-%i"%i)
        locale.setlocale(locale.LC_ALL, 'C')

    def fill_queue_url_leparisien(self, element):
        self.driver = self.handle_timeout(self.driver, element)
        pagination = self.driver.find_element_by_xpath("//a[@class='pagination__last']")
        max_pages = int(pagination.get_attribute("href").split("/")[-1].split("-")[1])
         
        print("max pages to crawl for {0} : {1}".format(element, max_pages))
        #### fill the queue with all possible urls
        for i in range(2, max_pages+1): 
            self.queues["urls"].put(element + "page-%i"%i)
            
    def fill_queue_url_lexpress(self, element):
        self.driver = self.handle_timeout(self.driver, element)
        pagination = self.driver.find_element_by_xpath("//div[@class='paginate paginate_list']")
        last_ul = pagination.find_elements_by_tag_name("ul")[-1]
        max_pages = int(last_ul.find_elements_by_tag_name("li")[-1].text)
        
        print("max pages to crawl for {0} : {1}".format(element, max_pages))
        #### fill the queue with all possible urls
        for i in range(1, max_pages+1): 
            self.queues["urls"].put(element + "?p=%i"%i)

# =============================================================================
# specific element crawling into time, url text of article        
# =============================================================================
    #### lemonde
    def lemonde_article_element(self, driver):
        nbr = driver.find_elements_by_tag_name("article")
        liste_text = []
        for comment in nbr:
            liste_text.append(comment.text)
        return liste_text
        
    #### mediapart
    def mediapart_article_element(self, driver):
        articles = driver.find_elements_by_xpath("//div[@class='post-list universe-journal']/div")
        liste_text = []
        for ar in articles:
            if ar.get_attribute("data-type") == "article":
                liste_text.append(ar.text)
        return liste_text
    
    #### le figaro
    def lefigaro_time_element(self, driver):
        href = driver.find_elements_by_xpath("//div[@class='SiteMap']/a")
        liste_times =[]
        for h in href:
            link = h.get_attribute("href")
            integers = [x for x in link.replace("http://www.lefigaro.fr/", "").split("/") if x.isdigit()]
            time = "-".join(integers)
            liste_times.append(time)
        return liste_times
    
    #### la tribune
    def latribune_time_element(self, driver):
        locale.setlocale(locale.LC_ALL, 'fr_FR')
        date = driver.current_url.split("/")[-2]
        date = pd.to_datetime(date, format = "%B-%Y").strftime("%Y-%m-%d")
        href = driver.find_elements_by_xpath("//article[@class='article-wrapper row clearfix ']")
        liste_times =[]
        for h in href:
            liste_times.append(date)
        locale.setlocale(locale.LC_ALL, 'C')
        return liste_times
    
    #### le parisien
    def leparisien_time_element(self, driver):
        articles = []
        for string in self.queues["carac"]["url_crawl"]["article_element"]:
            articles += driver.find_elements_by_xpath("//"+string)
            
        times = driver.find_elements_by_xpath("//div[@class='article__list-alt-date']")
        for new_time in times:
            date =  dateparser.parse(new_time.text).strftime("%Y-%m-%d")
            
        if len(times) == 0:
            date = True
            i = 0
            while date and i < len(articles):
                i +=1
                try:
                    date = dateparser.parse(articles[-i].find_element_by_xpath("div[@class='article__list-infos']/div/div[2]").text)
                except Exception:
                    pass
            if date:
                date = datetime.now()
                
        liste_times = []
        for i in range(len(articles)): 
            liste_times.append(date)
        return liste_times
    
    #### lexpress
    def lexpress_time_element(self, driver):
        href = []
        for string in self.queues["carac"]["url_crawl"]["href_element"]:
             href += driver.find_elements_by_xpath("//"+string)
             
        date = True
        i = 0
        while date and i < len(href):
            i +=1
            try:
                url = href[-i].get_attribute("href")
                driver.get(url)
                date = driver.find_element_by_xpath("//div[@class='signature']/time").get_attribute("datetime")
            except Exception:
                pass
        if date:
            date = datetime.now()  
        
        liste_times = []
        for i in range(len(href)): 
            liste_times.append(date)
        
        return liste_times
    