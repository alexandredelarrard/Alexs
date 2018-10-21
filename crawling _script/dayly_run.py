# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 10:40:00 2018

@author: User
"""

import os
import pandas as pd
import configparser
import platform
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")

from crawling import Crawling
from utils.sitemaps_history_crawling import parse_xml


def environment_variables():
    configParser = configparser.RawConfigParser() 
    if platform.system() in  ["Darwin", "Linux"]: # Mac or Linux
        configFilePath = os.environ["PROFILE"] + '/config_alexs.txt' # to check if PROFILE in os environ for Mac
    else:
        configFilePath = os.environ["USERPROFILE"] + '/config_alexs.txt'
    configParser.read(configFilePath)
    os.environ["DIR_PATH"] = configParser.get("config-Alexs", "project_path")
    
    
class DaylyRun(Crawling):
    
    def __init__(self):
        Crawling.__init__(self)
        
        self.today =  datetime.utcnow()
        self.path_url     = os.environ["DIR_PATH"] + "/data/continuous_run/url"
        self.path_article = os.environ["DIR_PATH"] + "/data/continuous_run/article"
        
    def get_urls(self):
    
        sitemaps_url = {"lemonde" : "https://www.lemonde.fr/sitemap_news.xml",# 2 jours
                    "lesechos": "https://www.lesechos.fr/sitemap_1.xml",# several years
                    "mediapart":"https://www.mediapart.fr/news_sitemap.xml",# 1 semaine
                    "lefigaro": "http://articles.lefigaro.fr",# par jour 
                    "liberation" :"https://www.liberation.fr/liberation/daily-sitemap.xml",# par jour 
                    "express": "https://www.lexpress.fr/sitemap_actu_1.xml",# par jour 
                    "humanite": "https://www.humanite.fr/sitemap.xml",# several years
                    "parisien": "http://www.leparisien.fr/sitemap_news_1.xml",# 1 semaine
                    } 
        return sitemaps_url
    
    def get_all_urls(self):
        sitemaps = self.get_urls()
        
        total_urls = pd.DataFrame(columns= ["date","url"])
        for key, sitemap in sitemaps.items():
            print(key)
            total_urls = pd.concat([total_urls, eval("create_{0}_urls(sitemap)".format(key))], axis =0)
        total_urls.to_csv(self.path_url + "/{0}.csv".format(self.today.strftime("%Y-%m-%d")))
            
        return total_urls
    
    def create_lemonde_urls(sitemap):
        return parse_xml(sitemap)
    
    def create_mediapart_urls(sitemap):
        return parse_xml(sitemap)
    
    def create_liberation_urls( sitemap):
        return parse_xml(sitemap)
    
    def create_express_urls( sitemap):
        return parse_xml(sitemap)
    
    def create_parisien_urls( sitemap):
        return parse_xml(sitemap)
    
    def create_humanite_urls( sitemap):
        urls = []
        for page in range (1,13):
            urls.append(parse_xml(sitemap + "?page=%i"%page))
        return pd.concat(urls, axis=0)
    
    def create_lesechos_urls( sitemap):
        urls = parse_xml(sitemap)
        index = urls["url"].apply(lambda x : True if x[-1] != "/" else False)
        return urls.loc[index]
    
    def create_lefigaro_urls(self, sitemap):
        
        new_month = str(self.today.month) if len(str(self.today.month)) ==2 else "0" + str(self.today.month)
        driver  = self.initialize_driver()
        
        urls = pd.DataFrame()
        for i in range(self.today.day - 1, self.today.day + 1):
            url = pd.DataFrame()
            new_day = str(i) if len(str(i)) ==2 else "0" +  str(i) 
            driver.get("/".join([sitemap, str(self.today.year) + new_month, new_day, ""]))
            parent = driver.find_element_by_xpath("//ul[@class='list-group']")
            liste_href = [x.get_attribute('href') for x in parent.find_elements_by_css_selector('a')]
            url["url"] = liste_href
            url["date"] = pd.to_datetime("/".join([str(self.today.year) , new_month, new_day]))
            urls = pd.concat([urls,url],axis=0)
        driver.quit()
        
        return urls
    
    