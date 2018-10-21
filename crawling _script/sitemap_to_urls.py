# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 15:08:08 2018

@author: User
"""

import pandas as pd
import xml.etree.cElementTree as ET
import requests
from datetime import datetime, timedelta

def xml2df(xml_data):
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

def fetch_sitemap(url):
    get_url = requests.get(url)
    if get_url.status_code == 200:
        return get_url.text
    else:
        print('Unable to fetch sitemap: %s.'%url) 
        return "" 
    
def parse_xml(sitemap):
    today = datetime.utcnow()
    xml_data = fetch_sitemap(sitemap)
    urls = xml2df(xml_data)
    
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