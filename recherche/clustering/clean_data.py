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
    
lesechos = pd.read_csv(r"C:\Users\User\Documents\Alexs\data\history\articles\lesechos\articles.csv")
urls = pd.read_csv(r"C:\Users\User\Documents\Alexs\data\history\url\history\lesechos_history.csv")
urls = urls[["url", "date"]]
lesechos = pd.merge(lesechos, urls, on ="url", how = "left")
lesechos["date"] = lesechos[["date_x", "date_y"]].apply(lambda x : clean_date(x),axis=1)
lesechos = lesechos.drop(["date_x", "date_y"], axis=1)
lesechos = lesechos.loc[~pd.isnull(lesechos["article"])]
lesechos["article"] = lesechos["Titre"] + "\r\n"  + lesechos["article"]

lefigaro = pd.read_csv(r"C:\Users\User\Documents\Alexs\data\history\articles\lefigaro\articles.csv")
lefigaro["article"] = lefigaro["article"].apply(lambda x : re.sub(r'(\nPar  )([A-Z].*)(\r)', '', str(x)))
lefigaro = lefigaro[["url", "date", "article"]]
lefigaro["journal"] = "lefigaro"

lemonde = pd.read_csv(r"C:\Users\User\Documents\Alexs\data\history\articles\lemonde\articles.csv")
urls = pd.read_csv(r"C:\Users\User\Documents\Alexs\data\history\url\history\lemonde_history.csv")
urls = urls[["url", "date"]]
lemonde = pd.merge(lemonde, urls, on ="url", how = "left")
lemonde = lemonde.loc[~pd.isnull(lemonde["article"])]
lemonde = lemonde[["url", "date", "article"]]
lemonde["journal"] = "lemonde"

humanite = pd.read_csv(r"C:\Users\User\Documents\Alexs\data\history\articles\humanite\articles_humanite.csv")
humanite["article"] = humanite["Titre"] + "\r\n"  + humanite["article"]
urls = pd.read_csv(r"C:\Users\User\Documents\Alexs\data\history\url\history\humanite_history.csv")
urls = urls[["url", "date"]]
humanite = pd.merge(humanite, urls, on ="url", how = "left")
humanite = humanite[["url", "date", "article"]]
humanite["journal"] = "humanite"

mediapart = pd.read_csv(r"C:\Users\User\Documents\Alexs\data\history\articles\mediapart\articles_mediapart.csv")
mediapart["article"] =  mediapart["Titre"] + "\r\n"  + mediapart["head"] + "\r\n"  + mediapart["article"].fillna("") 
mediapart = mediapart[["url", "date", "article"]]
mediapart["journal"] = "mediapart"

liberation = pd.read_csv(r"C:\Users\User\Documents\Alexs\data\history\articles\liberation\articles.csv")
liberation = liberation[["url", "date", "article"]]
liberation["journal"] = "liberation"
lesechos = lesechos[["url", "date", "article"]]
lesechos["journal"] = "lesechos"

latribune = pd.read_csv(r"C:\Users\User\Documents\Alexs\data\history\articles\latribune\articles.csv")
latribune = latribune[["url", "date", "article"]]

lexpress = pd.read_csv(r"C:\Users\User\Documents\Alexs\data\history\articles\lexpress\articles.csv")
lexpress = lexpress[["url", "date", "article"]]

gc.collect()

full = pd.concat([mediapart, liberation, lesechos, lemonde, humanite, lefigaro, latribune, lexpress], axis= 0)
full = full.drop_duplicates("url")
full = full.loc[~pd.isnull(full["article"])]
a = full["article"].apply(lambda x : len(x))
full = full.loc[a>100]
full.to_csv(r"D:\data\articles_journaux\all_articles.csv", index= False)
