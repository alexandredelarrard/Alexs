# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 16:26:58 2018

@author: JARD
"""

import pandas as pd
import numpy as np
import tqdm 
import time
import re

try:
    from crawling import Crawling
except Exception:
    import sys
    import os
    sys.path.append(os.environ["DIR_PATH"] + "/script/crawling _script")
    from crawling import Crawling

class LemondeScrapping(Crawling):
    
    def __init__(self):
    
        Crawling.__init__(self)
        self.url= "https://www.lemonde.fr/"
        self.main_lemonde()

    def main_lemonde(self):
    
        self.driver.get(self.url)
        liste_menu_href = self.get_lis_from_div("nav-section")
        liste_menu_href = [x for x in liste_menu_href if x not in ['https://www.lemonde.fr/', 
                                                                   'https://www.lemonde.fr/pixels',
                                                                  'https://www.lemonde.fr/teaser/presentation.html#xtor=CS1-263[BOUTONS_LMFR]-[BARRE_DE_NAV]-5-[Home]',
                                                                  'https://www.lemonde.fr/grands-formats/',
                                                                  'https://www.lemonde.fr/les-decodeurs/',
                                                                  'https://www.lemonde.fr/videos/',
                                                                  'https://www.lemonde.fr/data/',
                                                                  'https://www.lemonde.fr/guides-d-achat/']]   

        for element in liste_menu_href:
            max_pages = self.get_max_pages(element)
            for page in tqdm.tqdm(range(1, max_pages + 1)):
                self.driver.get(element+ "{0}.html".format(page))
                self.crawl_url_articles()
                
                ### check if articles have been written more recently than self.end_date
                if self.check_in_date(element):
                    break
                
        
    def crawl_url_articles(self, wait = 0):

        # =============================================================================
        #  get all desired infromation of the list of articles : 20 per page       
        # =============================================================================
        
        ### liste time article appeared
        times = self.driver.find_elements_by_xpath("//time[@class='grid_1 alpha']")
        liste_times =[]
        for t in times:
            liste_times.append(t.get_attribute("datetime"))
        time.sleep(wait)
        
        ### liste_href_articles
        href = self.driver.find_elements_by_xpath("//a[@class='grid_3 obf']")
        liste_href =[]
        for h in href:
            liste_href.append(h.get_attribute("href"))
        time.sleep(wait)
        
        ### liste title
        titles = self.driver.find_elements_by_xpath("//div[@class='grid_8 omega']/h3/a")
        liste_titles =[]
        for title in titles:
            liste_titles.append(title.text)
        time.sleep(wait)
        
        ### liste small desc
        ps = self.driver.find_elements_by_xpath("//p[@class='txt3']")
        liste_description =[]
        for p in ps:
            liste_description.append(p.text)
        time.sleep(wait)
            
        ### autor when available, before suppress information between parenthesis , e.g:Brice Pedroletti (Taipei, envoyé spécial) -> Brice Pedroletti
        nbr = self.driver.find_elements_by_tag_name("article")
        liste_autors =[]
        for comment in nbr:
            text = comment.text.split("\n")
            if len(re.sub(r'\([^)]*\)', '', text[-2]).split(" ")) < 5:
                liste_autors.append(text[-2])
            else:
                liste_autors.append("")
        
        information = np.transpose([x for x in [liste_times, liste_href, liste_titles, liste_description, liste_autors] if x != []])
        
        try:
            if information.shape[1] != 5:
                print("reexecuting the page")
                self.crawl_url_articles(2)
            else:
                overall_information = pd.DataFrame(np.array(information), columns = ["Date","url","Title","Description","Autor"])
                overall_information["Date"] = pd.to_datetime(overall_information["Date"])
                self.article_bdd = pd.concat([self.article_bdd, overall_information], axis=0)
                
        except Exception as e:
            print(e)
            print(information)
            pass
        
    def get_max_pages(self, element):
         self.driver.get(element+"1.html")
         pagination = self.driver.find_element_by_xpath("//div[@class='conteneur_pagination']")
         last_page = pagination.find_element_by_class_name("adroite").text
         if last_page.isdigit():
             return int(last_page)
         else:
             return 1

### test unitaire
if __name__ == "__main__":
    lemonde = LemondeScrapping()
