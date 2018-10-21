# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 15:08:08 2018

@author: User
"""

import pandas as pd
import glob
import xml.etree.cElementTree as ET
import requests
import os
import gzip
import urllib


def create_history_from_csv(path):
    liste_files = glob.glob(path + "/*.csv")
    
    for i, f in enumerate(liste_files):
        print(f)
        try:
            if i == 0:
                total = pd.read_csv(f)
            else:
                total = pd.concat([total, pd.read_csv(f)], axis = 0)
        except Exception as e:
            print("error for file {0}: {1}".format(f, e))
    
    total = total.sort_values("0").reset_index(drop=True)
    return total


def xml2df(xml_data):
    root = ET.XML(xml_data) # element tree
    all_records = []
    for i, child in enumerate(root):
        record = {}
        for subchild in child:
            record[subchild.tag] = subchild.text
        all_records.append(record)
    df = pd.DataFrame(all_records)
    return df


def parse_xml_files(path):
    liste_files = glob.glob(path + "/*.xml")
    for i, f in enumerate(liste_files):
        print(f)
        xml_data = open(f).read()
        try:
            if i == 0:
                total = xml2df(xml_data)
                total.columns = ["date", "url"]
            else:
                addi = xml2df(xml_data)
                addi.columns = ["date", "url"]
                total = pd.concat([total, addi], axis = 0)
        except Exception as e:
            print("error for file {0}: {1}".format(f, e))
            
    return total


def fetch_sitemap(url):
    get_url = requests.get(url)
    if get_url.status_code == 200:
        return get_url.text
    else:
        print('Unable to fetch sitemap: %s.'%url) 
        return ""
    

def read_tar_gz(url, path, filename):
    f = gzip.open(r"C:\Users\User\Documents\Alexs\data\lemonde/{0}.xml.gz".format(filename), 'rb')
    file_content = f.read()
    f.close()
    return file_content
   
# =============================================================================
# Les echos
# =============================================================================
def create_lesechos_history():
    
    save_path = r"C:\Users\User\Documents\Alexs\data\clean_data\history"
    lesechos =  r"C:\Users\User\Documents\Alexs\data\lesechos\url\*"

    total = create_history_from_csv(lesechos, save_path)
    xml_total = parse_xml_files(r"C:\Users\User\Documents\Alexs\data\lesechos\url")
    xml_total["date"] = pd.to_datetime(xml_total["date"], format = "%Y-%m-%d")
    xml_total["description"] = xml_total["url"]
    
    total = total[:-1]
    total["date"] = pd.to_datetime(total["date"], format = "%Y-%m-%d")
    total = total.loc[total["date"]< min(xml_total["date"])]
    
    big_total = pd.concat([total, xml_total], axis= 0).reset_index(drop=True)
    big_total.to_csv(save_path + "/lesechos_history_08-08-2018.csv", index = False)
    
# =============================================================================
# Liberation
# =============================================================================
def create_liberation_history(save_path):
    
    xml_data = fetch_sitemap("http://www.liberation.fr/liberation/sitemap.xml")
    urls = xml2df(xml_data)
    index = urls[urls.columns[0]].apply(lambda x : True if "archives-sitemap.xml" in x else False)
    urls = urls[index]
    
    for i, url in enumerate(urls[urls.columns[0]]):
        print(url)
        xml_data = fetch_sitemap(url)
        if xml_data !="":
            try:
                if i == 0:
                    total = xml2df(xml_data)
                else:
                    addi = xml2df(xml_data)
                    total = pd.concat([total, addi], axis = 0)
            except Exception as e:
                print("error for file {0}: {1}".format(url, e))
                
    total.columns = ["frequence", "date", "url", "priority"]
    total = total.sort_values("date").reset_index(drop = True)
    del total["priority"]
    total.to_csv(save_path  + "/liberation_history_03-08-2018.csv", index = False)
    
# =============================================================================
#     Le Monde
# =============================================================================
def create_lemonde_history(save_path):
    
    xml_data = fetch_sitemap("https://www.lemonde.fr/sitemap_index.xml")
    urls = xml2df(xml_data)
    index = urls[urls.columns[1]].apply(lambda x : True if "https://www.lemonde.fr/sitemaps/articles/" in x else False)
    urls = urls[index]
    
    for i, url in enumerate(urls[urls.columns[1]]):
        print(i,url)
        xml_data = read_tar_gz(url, save_path, url.split("/")[-1:])
        if xml_data !="":
            try:
                if i == 0:
                    total = xml2df(xml_data)
                else:
                    addi = xml2df(xml_data)
                    total = pd.concat([total, addi], axis = 0)
            except Exception as e:
                print("error for file {0}: {1}".format(url, e))
    
    a = total[['{http://www.google.com/schemas/sitemap-image/1.1}image',
       '{http://www.sitemaps.org/schemas/sitemap/0.9}lastmod',
       '{http://www.sitemaps.org/schemas/sitemap/0.9}loc',
       '{http://www.w3.org/1999/xhtml}link']]   
    a = a.loc[~pd.isnull(a['{http://www.sitemaps.org/schemas/sitemap/0.9}lastmod'])]         
    b = total[['{https://www.google.com/schemas/sitemap-image/1.1}image',
       '{https://www.sitemaps.org/schemas/sitemap/0.9}lastmod',
       '{https://www.sitemaps.org/schemas/sitemap/0.9}loc',
       '{https://www.w3.org/1999/xhtml}link']] 
    b = b.loc[~pd.isnull(b['{https://www.sitemaps.org/schemas/sitemap/0.9}lastmod'])]   
    b.columns = [x.replace("https","http") for x in b.columns]  
    
    new_total = pd.concat([a,b],axis = 0)[["{http://www.sitemaps.org/schemas/sitemap/0.9}lastmod",
                                           "{http://www.sitemaps.org/schemas/sitemap/0.9}loc"]]
    
    new_total.columns = ["date", "url"]
    new_total = new_total.drop_duplicates("url").sort_values("date").reset_index(drop = True)
    new_total.to_csv(save_path  + "/lemonde_history_31-08-2018.csv", index = False)    
    
    
def create_lhumanite_history(save_path):
    
    xml_data = fetch_sitemap("https://www.humanite.fr/sitemap.xml")
    urls = xml2df(xml_data)
    
    for i, url in enumerate(urls[urls.columns[1]]):
        print(url)
        xml_data = fetch_sitemap(url)
        if xml_data !="":
            try:
                if i == 0:
                    total = xml2df(xml_data)
                else:
                    addi = xml2df(xml_data)
                    total = pd.concat([total, addi], axis = 0)
            except Exception as e:
                print("error for file {0}: {1}".format(url, e))
                
    total.columns = ["frequence", "date", "url", "priority"]
    total = total.sort_values("date").reset_index(drop = True)
    total = total.loc[~pd.isnull(total["date"])].drop_duplicates("url")
    total.to_csv(save_path  + "/humanite_history_31-08-2018.csv", index = False)   
    
    
def create_lexpress_history(save_path):
    
    xml_data = fetch_sitemap("https://www.lemonde.fr/sitemap_index.xml")
    urls = xml2df(xml_data)
    index = urls[urls.columns[1]].apply(lambda x : True if "https://www.lemonde.fr/sitemaps/articles/" in x else False)
    urls = urls[index]
    
    for i, url in enumerate(urls[urls.columns[1]]):
        print(url)
        xml_data = fetch_sitemap(url)
        if xml_data !="":
            try:
                if i == 0:
                    total = xml2df(xml_data)
                else:
                    addi = xml2df(xml_data)
                    total = pd.concat([total, addi], axis = 0)
            except Exception as e:
                print("error for file {0}: {1}".format(url, e))
                
    total.columns = ["frequence", "date", "url", "priority"]
    total = total.sort_values("date").reset_index(drop = True)
    del total["priority"]
    total.to_csv(save_path  + "/liberation_history_03-08-2018.csv", index = False)    
    
    
if __name__ == "__main__":
    save_path = r"C:\Users\User\Documents\Alexs\data\clean_data\history"
    lesechos =  r"C:\Users\User\Documents\Alexs\data\lesechos\url\*"

    create_lemonde_history(save_path)
