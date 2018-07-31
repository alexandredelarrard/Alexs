# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 13:26:20 2018

@author: JARD
"""

from selenium import webdriver
import pandas as pd
import os
import multiprocessing

class Crawling(object):
    
    def __init__(self):
        
        self.end_date = pd.to_datetime("2018-07-01", format = "%Y-%m-%d")
        self.driver = self.initialize_driver()
        self.article_bdd = pd.DataFrame([], columns = ["Date","url","Title","Description","Autor"])
        
        
    def initialize_driver(self):
        """
        Initialize the web driver with Firefox driver as principal driver geckodriver
        parameters are here to not load images and keep the default css --> make page loading faster
        """
        firefox_profile = webdriver.FirefoxProfile()
        firefox_profile.set_preference('permissions.default.stylesheet', 2)
        firefox_profile.set_preference('permissions.default.image', 2)
        firefox_profile.set_preference('dom.ipc.plugins.enabled.libflashplayer.so', 'false')
        firefox_profile.set_preference('disk-cache-size', 4096)
        firefox_profile.set_preference("http.response.timeout", 10)
    
        driver = webdriver.Firefox(firefox_profile=firefox_profile, log_path= os.environ["DIR_PATH"] + "/webdriver/geckodriver.log")#firefox_options=options, 
        driver.delete_all_cookies()
        driver.set_page_load_timeout(100)     
        return driver
    
    
    def click_input(self, web_element, category):
        """
        click on input when there is a value to set.
        Useful for checking captcha
        """
        first_div2 = web_element.find_elements_by_tag_name("input")
        for inp in first_div2:
             if inp.get_attribute("value").lower() == category.lower():
                 inp.click()
                 
                 
    def get_lis_from_div(self, id_ul):
        """
        click on input when there is a value to set.
        Useful for checking captcha
        """
        nav = self.driver.find_element_by_xpath("//div[@id='{0}']".format(id_ul))
        liste_href = nav.find_element_by_tag_name("ul")
        liste = []
        for li in liste_href.find_elements_by_tag_name("a"):
            liste.append(li.get_attribute("href"))
        return liste   


    def check_in_date(self, url):
        if min(self.article_bdd["Date"]) < self.end_date:
            if not os.path.isdir(os.environ["DIR_PATH"] + "/data/{0}".format(url.replace("https://www.", "").replace(".fr","").split("/")[0])):
                os.mkdir(os.environ["DIR_PATH"] + "/data/{0}".format(url.replace("https://www.", "").replace(".fr","").split("/")[0]))
            
            self.article_bdd.to_csv(os.environ["DIR_PATH"] + "/data/{0}.csv".format(url.replace("https://www.", "").replace(".fr","")[:-1]), index=False)
            self.article_bdd = pd.DataFrame([], columns = ["Date","url","Title","Description","Autor"])
            return True
        return False
    
    
    def multiprocess_crawling(self, function, data_liste_matches, additionnal_path, ncore):
    
        jobs = []
        nbr_core = ncore
    
        for i in range(nbr_core):
            
            sub_liste_refs = data_liste_matches[int(i*len(data_liste_matches)/nbr_core): int((i+1)*len(data_liste_matches)/nbr_core)]
            if i == nbr_core -1:
                sub_liste_refs = data_liste_matches[int(i*len(data_liste_matches)/nbr_core): ]
                
            p = multiprocessing.Process(target=function, args=(sub_liste_refs, additionnal_path,))
            jobs.append(p)
            p.start()