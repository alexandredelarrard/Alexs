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
from datetime import datetime, timedelta
import xml.etree.cElementTree as ET
import requests
warnings.filterwarnings("ignore")

from production.crawling.crawling import Crawling

def environment_variables():
    configParser = configparser.RawConfigParser() 
    if platform.system() in  ["Darwin", "Linux"]: # Mac or Linux
        configFilePath = os.environ["PROFILE"] + '/config_alexs.txt' # to check if PROFILE in os environ for Mac
    else:
        configFilePath = os.environ["USERPROFILE"] + '/config_alexs.txt'
    configParser.read(configFilePath)
    os.environ["DIR_PATH"] = configParser.get("config-Alexs", "project_path")
    
    
class UrlCrawling(Crawling):
    
    def __init__(self, queues):
        Crawling.__init__(self)
        
        self.queues = queues
        self.today =  datetime.now()#utc
        self.path_url     = os.environ["DIR_PATH"] + "/data/continuous_run/url"
        self.path_article = os.environ["DIR_PATH"] + "/data/continuous_run/article"
        
    def get_urls(self):
        sitemaps_url = {"lemonde" : "https://www.lemonde.fr/sitemap_news.xml",# 2 jours
                    "lesechos": "https://www.lesechos.fr/sitemap_1.xml",# several years
                    "mediapart":"https://www.mediapart.fr/news_sitemap.xml",# 1 semaine
                    "lefigaro": "http://articles.lefigaro.fr",# par jour 
                    "liberation" :"https://www.liberation.fr/liberation/daily-sitemap.xml",# par jour 
                    "lexpress": "https://www.lexpress.fr/sitemap_actu_1.xml",# par jour 
                    "humanite": "https://www.humanite.fr/sitemap.xml",# several years
#                    "parisien": "http://www.leparisien.fr/sitemap_news_1.xml", # 1 semaine
                    "latribune": "https://www.latribune.fr/toute-l-actualite/toute-l-actualite.html"}
        
        return sitemaps_url
    
    def main_url_crawling(self):
        
        sitemaps = self.get_urls()
        total_urls = pd.DataFrame(columns= ["date","url"])
        
        print("_"*40)
        print("|" + " "*10 + "URL crawling" + " "*10 + "|")
        print("_"*40)
        
        for key, sitemap in sitemaps.items():
            print(key)
            total_urls = pd.concat([total_urls, eval("self.create_{0}_urls(sitemap)".format(key))], axis =0)
            
        ### suppress all urls crawled last day
        try:
            previous_urls = pd.read_csv(self.path_url+ "/{0}.csv".format((self.today - timedelta(days = 1)).strftime("%Y-%m-%d")))["url"].tolist()
            total_urls = total_urls.loc[~total_urls["url"].isin(previous_urls)]   
        except Exception:
            pass
        
        total_urls = total_urls.drop_duplicates("url")
        
        total_urls.to_csv(self.path_url + "/{0}.csv".format(self.today.strftime("%Y-%m-%d")), index = False)
        
        return total_urls
    
    def create_lemonde_urls(self, sitemap):
        return self.parse_xml(sitemap)
    
    def create_mediapart_urls(self, sitemap):
        return self.parse_xml(sitemap)
    
    def create_liberation_urls(self, sitemap):
        return self.parse_xml(sitemap)
    
    def create_lexpress_urls(self, sitemap):
        return self.parse_xml(sitemap)
    
    def create_parisien_urls(self, sitemap):
        return self.parse_xml(sitemap)
    
    def create_humanite_urls(self, sitemap):
        urls = []
        for page in range (1,13):
            urls.append(self.parse_xml(sitemap + "?page=%i"%page))
        return pd.concat(urls, axis=0)
    
    def create_lesechos_urls(self, sitemap):
        urls = self.parse_xml(sitemap)
        index = urls["url"].apply(lambda x : True if x[-1] != "/" else False)
        return urls.loc[index]
    
    def create_lefigaro_urls(self, sitemap):
        
        driver = self.initialize_driver()
        
        urls = pd.DataFrame()
        yesterday = self.today - timedelta(days=1)
        
        for day, month in [(yesterday.day, yesterday.month), (self.today.day, self.today.month)]:
            try:
                url = pd.DataFrame()
                new_day = str(day) if len(str(day)) ==2 else "0" +  str(day) 
                new_month = str(month) if len(str(month)) ==2 else "0" + str(month)
                
                driver.get("/".join([sitemap, str(self.today.year) + new_month, new_day, ""]))
                parent = driver.find_element_by_xpath("//ul[@class='list-group']")
                liste_href = [x.get_attribute('href') for x in parent.find_elements_by_css_selector('a')]
                url["url"] = liste_href
                url["date"] = pd.to_datetime("/".join([str(self.today.year) , new_month, new_day]))
                urls = pd.concat([urls,url],axis=0)
            except Exception:
                pass
        driver.quit()
        
        return urls
    
    def create_latribune_urls(self, sitemap):
        
        driver = self.initialize_driver()
        urls = pd.DataFrame()
        driver.get(sitemap)
        
        liste_href = [x.get_attribute('href') for x in driver.find_elements_by_xpath("//article[@class='article-wrapper row clearfix ']/div/a")]
        urls["url"] = liste_href
        urls["date"] = pd.to_datetime(datetime.utcnow())
        driver.quit()
        return urls
    
    
    def xml2df(self, xml_data):
        root = ET.XML(xml_data) # element tree
        all_records = []
        for i, child in enumerate(root):
            record = {}
            for subchild in child:
                record[subchild.tag] = subchild.text
                for sub_sub in subchild:
                    record[sub_sub.tag] = sub_sub.text
            all_records.append(record)
        df = pd.DataFrame(all_records)
        return df

    def fetch_sitemap(self, url):
        get_url = requests.get(url)
        if get_url.status_code == 200:
            return get_url.text
        else:
            print('Unable to fetch sitemap: %s.'%url) 
            return "" 
        
    def parse_xml(self, sitemap):
        today = datetime.utcnow()
        xml_data = self.fetch_sitemap(sitemap)
        urls = self.xml2df(xml_data)
        
        urls.rename(columns = {'{http://www.sitemaps.org/schemas/sitemap/0.9}loc': "url",
                               '{http://www.sitemaps.org/schemas/sitemap/0.9}lastmod' : "date",
                               '{http://www.google.com/schemas/sitemap-news/0.9}publication_date' :"date2",
                               '{http://www.google.com/schemas/sitemap/0.84}loc' : "url",
                               '{http://www.google.com/schemas/sitemap/0.84}lastmod' : "date"
                               }, inplace = True)
            
        cols = urls.columns
        if "date" not in cols and "date2" in cols:
            urls.rename(columns ={"date2" :"date"}, inplace= True)
            
        urls = urls[["date", "url"]]
        urls["date"] = pd.to_datetime(urls["date"])
        urls = urls.loc[urls["date"]>= today - timedelta(days= 1, hours = today.hour, minutes = today.minute , seconds = today.second) - timedelta(minutes = 5)]
        return urls
    
    