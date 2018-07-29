# -*- coding: utf-8 -*-
"""
Created on Mon May 28 21:49:49 2018

@author: User
"""

import os
import tqdm

from utils_crawling import start, multiprocess_crawling

def parcourir_liste(driver, set_to_see, set_seen):
    """
    Extraire les liens de tous les articles de la liste
    """

    for article in set_to_see.copy():
        driver.get(article)
        article_links = extract_links(driver)
        set_to_see = set_to_see.union(article_links) - set_seen
        set_seen = set_seen.union(set([article]))
        print("to see : {0} seen : {1}".format(len(set_to_see), len(set_seen)))
        print(set_to_see)
        
    return set_to_see, set_seen
    
    
def extract_links(driver):
    """
    Extract the JULY 18 article links from the URL
    """
    liste_articles = set()
    html = driver.find_elements_by_xpath("//a[@href]")
    for a in tqdm.tqdm(html):
        try:
            b = a.get_attribute('href') 
            if '2018/07' in b:
                liste_articles.add(b)
        except Exception:
            pass
                
    return liste_articles 


def extract_url_articles(liste_urls):
    """
    Iterate over all urls to crawl and call parcourir list function
     - Input : Liste of websites url to crawl
     - Output: Dictionnary shape : {website : [URLS], ...}
    """
    
    dico_links = {}
    for url in liste_urls:
        
        ### init liste and driver for each url to crawl
        driver = start()
        set_to_see=set([url])
        set_seen= set()
        
        ### crawl urls and append them to final list
        while len(set_to_see)>0:
            set_to_see, set_seen = parcourir_liste(driver, set_to_see, set_seen)
            
        dico_links[url] = list(set_seen)
        driver.close()
        
    return dico_links

if __name__ == "__main__":
    os.environ["DIR_PATH"] = r"C:\Users\User\Documents\Alexs"
    liste_urls = ["https://www.lemonde.fr/"]
    extracted_links = extract_url_articles(liste_urls)
    