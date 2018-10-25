# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 10:55:26 2018

@author: JARD
"""


import time
from queue import Queue
from production.crawling import Crawling


class ArticleCrawling(Crawling):
    
    def __init__(self, queues, urls):
        """
        """
        Crawling.__init__(self)
        self.queues = queues
        self.queues["results"] = Queue()
        self.liste_urls = urls 

        for i in range(self.cores):
             self.queues["drivers"].put(self.initialize_driver())

    def main_article_crawling(self):
        
        print("_"*40)
        print("|" + " "*10 + "Article crawling" + " "*10 + "|")
        print("_"*40)
         
        self.start_threads_and_queues(self.crawl_article)
        t0 = time.time()
        for item in self.liste_urls.to_dict(orient='records'):
            self.queues["urls"].put(item)
        print('*** Main thread waiting')
        self.queues["urls"].join()
        print('*** Done in {0}'.format(time.time() - t0))
        articles = self.save_results()
        return articles
             
        
    def crawl_article(self, driver, queues, date):
        
        url = driver.current_url
        journal = self.deduce_journal(url)
        queue = queues[journal]
        
        if len([1 for x in queue["not_to_crawl"] if x in url]) == 0:

            # =============================================================================
            #         Is article restricted
            # =============================================================================
            restricted = 0
            for string in queue["restricted"]:
                 if len(driver.find_elements_by_xpath("//" + string)) >0:
                    restricted = 1
                
            # =============================================================================
            #             Article Title
            # =============================================================================
            title = ""
            for string in queue["title"]:
                if len(driver.find_elements_by_xpath("//" + string)) >0:
                        title += driver.find_element_by_xpath("//" + string).text
                        break
                    
            # =============================================================================
            #             Article Categorie
            # =============================================================================
            categorie = ""
            for string in queue["categorie"]:
                if len(driver.find_elements_by_xpath("//" + string)) >0:
                    categorie += driver.find_element_by_xpath("//" + string).text 

            
            # =============================================================================
            #             Article Categorie
            # =============================================================================
            description_article = ""
            for string in queue["description_article"]:
                if len(driver.find_elements_by_xpath("//" + string)) >0:
                    description_article += driver.find_element_by_xpath("//" + string).text 
            
            # =============================================================================
            #             Article author
            # =============================================================================
            author = ""
            for string in queue["author"]:
                if len(driver.find_elements_by_xpath("//" + string)) >0:
                    author += driver.find_element_by_xpath("//" + string).text 

            # =============================================================================
            #             Article content
            # =============================================================================
            article = ""
            for string in queue["article"]:
                if len(driver.find_elements_by_xpath("//" + string)) >0:
                    article += driver.find_element_by_xpath("//" + string).text
                    break

            information = [date, journal, driver.current_url, restricted, str(title), str(author), str(article), str(categorie), str(description_article)]
        else:
            information = [date, journal, driver.current_url, '', '', '', '', '', '']
            
        return information
    
    
    def deduce_journal(self, url):
        if ".lexpress." in url:
            return "lexpress"
        elif ".leparisien" in url:
            return "leparisien"
        elif ".lefigaro" in url:
            return "lefigaro"
        else:
            x  = url.replace("http://www.", "").replace("https://www.","").replace(".fr","").split("/",1)[0]
            return x