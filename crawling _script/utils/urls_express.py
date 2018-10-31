# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 19:11:11 2018

@author: User
"""

from datetime import datetime, timedelta
import xml.etree.cElementTree as ET
import requests
import pandas as pd
import glob

def xml2df( xml_data):
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


def parse_xml_files(path):
    liste_files = glob.glob(path + "/*.xml")
    for i, f in enumerate(liste_files):
        print(f)
        xml_data = open(f).read().replace("ï»¿","")
        try:
            if i == 0:
                total = xml2df(xml_data)
#                total.columns = ["date", "url"]
            else:
                addi = xml2df(xml_data)
#                addi.columns = ["date", "url"]
                total = pd.concat([total, addi], axis = 0)
        except Exception as e:
            print("error for file {0}: {1}".format(f, e))
            
    return total


urls = parse_xml_files(r"C:\Users\User\Documents\Alexs\data\history\url\history\express_history.csv")
urls.columns = ["freq", "date", "url", "priority "]
urls = urls.drop_duplicates("url")
urls[["date", "url"]].to_csv(r"C:\Users\User\Documents\Alexs\data\history\url\history\express_history.csv", index= False)
