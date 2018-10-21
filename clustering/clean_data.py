# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 20:59:25 2018

@author: User
"""

import pandas as pd
import numpy as np
import re
import gc

def clean_date(x):
    
    if pd.isnull(x[0]) and not pd.isnull(x[1]):
        return x[1]
    elif not pd.isnull(x[0]) and pd.isnull(x[1]):
        return x[0]
    elif not pd.isnull(x[0]) and not pd.isnull(x[1]):
        return x[1]
    else:
        return np.nan
    
mediapart = pd.read_csv(r"C:\Users\User\Documents\Alexs\data\clean_data\articles\mediapart\articles_mediapart.csv")
mediapart["article"] =  mediapart["Titre"] + "\r\n"  + mediapart["head"] + "\r\n"  + mediapart["article"].fillna("") 

liberation = pd.read_csv(r"C:\Users\User\Documents\Alexs\data\clean_data\articles\liberation\articles.csv")
lesechos = pd.read_csv(r"C:\Users\User\Documents\Alexs\data\clean_data\articles\lesechos\articles.csv")
urls = pd.read_csv(r"C:\Users\User\Documents\Alexs\data\clean_data\url\history\lesechos_history.csv")
urls = urls[["url", "date"]]
lesechos = pd.merge(lesechos, urls, on ="url", how = "left")
lesechos["date"] = lesechos[["date_x", "date_y"]].apply(lambda x : clean_date(x),axis=1)
lesechos = lesechos.drop(["date_x", "date_y"], axis=1)
lesechos = lesechos.loc[~pd.isnull(lesechos["article"])]
lesechos["article"] = lesechos["Titre"] + "\r\n"  + lesechos["article"]

lefigaro = pd.read_csv(r"C:\Users\User\Documents\Alexs\data\clean_data\articles\lefigaro\articles.csv")
lefigaro["article"] = lefigaro["article"].apply(lambda x : re.sub(r'(\nPar  )([A-Z].*)(\r)', '', str(x)))

lemonde = pd.read_csv(r"C:\Users\User\Documents\Alexs\data\clean_data\articles\lemonde\articles.csv")
urls = pd.read_csv(r"C:\Users\User\Documents\Alexs\data\clean_data\url\history\lemonde_history.csv")
urls = urls[["url", "date"]]
lemonde = pd.merge(lemonde, urls, on ="url", how = "left")
lemonde = lemonde.loc[~pd.isnull(lemonde["article"])]

humanite = pd.read_csv(r"C:\Users\User\Documents\Alexs\data\clean_data\articles\humanite\articles_humanite.csv")
humanite["article"] = humanite["Titre"] + "\r\n"  + humanite["article"]
urls = pd.read_csv(r"C:\Users\User\Documents\Alexs\data\clean_data\url\history\humanite_history.csv")
urls = urls[["url", "date"]]
humanite = pd.merge(humanite, urls, on ="url", how = "left")

mediapart = mediapart[["url", "date", "article"]]
mediapart["journal"] = "mediapart"
liberation = liberation[["url", "date", "article"]]
liberation["journal"] = "liberation"
lesechos = lesechos[["url", "date", "article"]]
lesechos["journal"] = "lesechos"

lemonde = lemonde[["url", "date", "article"]]
lemonde["journal"] = "lemonde"
humanite = humanite[["url", "date", "article"]]
humanite["journal"] = "humanite"
lefigaro = lefigaro[["url", "date", "article"]]
lefigaro["journal"] = "lefigaro"

gc.collect()

full = pd.concat([mediapart, liberation, lesechos, lemonde, humanite, lefigaro], axis= 0)
full = full.drop_duplicates("url")
full = full.loc[~pd.isnull(full["article"])]
del full["date"]
del full["journal"]
full.to_csv(r"D:\data\articles_journaux\all_articles.csv", index= False)
