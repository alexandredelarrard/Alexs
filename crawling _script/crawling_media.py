# -*- coding: utf-8 -*-
"""
Created on Mon May 28 21:49:49 2018

@author: User
"""

import os
import tqdm
import pandas as pd

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
        
    return set_to_see, set_seen
    
    
def extract_links(driver):
    """
    Extract the JULY 18 article links from the URL
    """
    liste_articles = set()
    html = driver.find_elements_by_xpath("//a[@href]")
    for a in html:#tqdm.tqdm(html):
        try:
            b = a.get_attribute('href') 
            if '2018/07' in b:
                liste_articles.add(b)
        except Exception:
            pass
                
    return liste_articles 

def while_per_url(liste_url, additionnal_path):
    """
    Iterate over all urls to crawl and call parcourir list function
     - Input : Liste of websites url to crawl
     - Output: Dictionnary shape : {website : [URLS], ...}
    """
    
    driver = start()
    for url in liste_url:
        
        ### init liste and driver for each url to crawl
        set_to_see=set([url])
        set_seen= set()
        
        ### crawl urls and append them to final list
        while len(set_to_see)>0:
            set_to_see, set_seen = parcourir_liste(driver, set_to_see, set_seen)
        pd.DataFrame(list(set_seen)).to_csv(additionnal_path+ "/url_{0}.csv".format(url.replace("https://","").replace("/","")))
    driver.close()


if __name__ == "__main__":
    os.environ["DIR_PATH"] = r"C:\Users\User\Documents\Alexs"
    liste_urls = ["https://www.lemonde.fr/", "http://www.lefigaro.fr/", "https://www.lesechos.fr/",
                  "https://www.mediapart.fr/", "http://www.liberation.fr/", "https://www.latribune.fr/"]
    
    save_path = os.environ["DIR_PATH"] + "/data"
    multiprocess_crawling(while_per_url, liste_urls, save_path, ncore=6)
