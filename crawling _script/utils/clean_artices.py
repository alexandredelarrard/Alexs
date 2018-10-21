# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 10:56:37 2018

@author: User
"""

import time
import numpy as np
import pandas as pd
import os
import glob
from datetime import datetime
from random import shuffle
import re

def deduce_date(x):
    try:
        sp = x.replace("https://","").replace("http://","").split("/")
        if sp[3].isdigit() and sp[2].isdigit() and sp[1].isdigit():
            date = sp[3] + "/" + sp[2] + "/" + sp[1]
        else:
            date = ""
    except Exception:
        date = ""
    return date


def deduce_categorie(x):
    try:
        sp = x.replace("https://www.","").replace("http://www.","").split("/")
        return sp[1]
    except Exception: 
        pass
    return ""

def deduce_cat_2(c, x):
    if x[0] in c:
        return x[0]
    else:
        return ""
    
def liste_head(x):
    y = [a for a in x[0] if a not in x[1]]
    return y


def deduce_title(x):
    if x[1] == 6:
        return [x[0][1], x[0][2]]
    elif x[1] in [2, 4]:
        return [x[0][0], x[0][1]]
    else:
        return x
    
def categorie_lesechos(x):
    
    if len(x) > 1:
        if x[0].isupper():
            return x[0]
        else:
            return ""
    else:
        return ""
    
    
def clean_article(x):
    
    x = x.split("LIRE AUSSI")[0].split("\r\nSUR LE MÊME SUJET")[0]
    x = x.replace("\\'", "'").split("\r\nMots clés")[0]
    x = x.split("\r\r\nA lire aussi")[0].split("\r\r\nVoir aussi :")[0]
    x = x.split("Accédez à la recommandation d'Investir")[0]
    
    
    try:
        y = x.split("|")
        if y[0].isupper():
            x = x.split("|", 1)[1]
            
    except Exception:
        pass
    
    return x
        

def auteur(x):
    try: 
        x = re.search(r'(\nPar )([A-Z].*)(\n)', str(x)).group(2).strip()
        x = re.sub(r'\([^)]*\)', '', x).strip().split(" - ")[0]
        if len(x) < 100:
            return x 
        else:
            return ""
    except Exception: 
         return ""


def make_title_echo(x):
    x = x.split("/")[-1]
    y = x.split("ECH_",1)
    
    if len(y)>1:
        x = y[1]
        
    if ".php" in x :
        x = x.split("-")[:-1]
    else:
        x = x.replace(".html","").replace(".htm","").split("-")

    return " ".join(x).strip()


def clean_titles_lemonde(x, liste):
    y = x.split("\r\n")[0]
    if y in liste:
        return "\r\n".join(x.split("\r\n")[1:])
    else:
        return x

def get_auteur(x):
    spl = x.split("\r\n")
    if len(spl)>1:
        if "Par " in spl[1]:
            return spl[1].split("Mis à jour")[0].replace("Par  ", "").strip()
    return ""
    
def clean_article_figaro(x):
    y = x.split("\r\nMis à jour le ")
    if len(y)>1:
        if len(x.split("\r\nMis à jour le "))>= 2:
            return x.split("\r\nMis à jour le ",1)[1].split("\r\n",1)[1]
    
    y = x.split("Publié le ")
    if len(y)>1:
        if len(x.split("Publié le "))>= 2:
            return x.split("Publié le ",1)[1].split("\r\n",1)[1]    
    
    return x

def get_list_articles(path, media):
    
     files = glob.glob("/".join([path, "article", "*", "extraction_*.csv"]))
     not_open = []
     if len(files)>0:
        for i, f in enumerate(files):
            try:
                if i ==0:
                    total = pd.read_csv(f, error_bad_lines=False, quotechar="\"")
                else:
                    total = pd.concat([total, pd.read_csv(f, error_bad_lines=False, quotechar="\"")], axis=0)
            except Exception:
                print(f)
                not_open.append(f)
                
     total.columns = ["url", "restricted", "count paragraphs", "count h1", "count h2", "count_h3", "head", "article"]
     total = total.loc[~pd.isnull(total["article"])]
     total = total.loc[~pd.isnull(total["url"])]
     
     if media == "liberation":
         total["date"] = total["url"].apply(lambda x:deduce_date(x))
         total["date"] = pd.to_datetime(total["date"])
         total = total.loc[~pd.isnull(total["date"])]
         total = total.drop_duplicates("url")
         total["categorie"] = total["url"].apply(lambda x:deduce_categorie(x))
         
         ### title cleaning
         a = total["head"].apply(lambda x: x.split("\r\n"))
         liste_cat2 = [x for x in pd.DataFrame(list(list(zip(*a))[0]))[0].value_counts().index if x.isupper()]
         category_2 = a.apply(lambda x : deduce_cat_2(liste_cat2,x))
         total["categorie_2"] = category_2
         total["liste_head"] = a
         total["liste_head"] = total[["liste_head", "categorie_2"]].apply(lambda x : liste_head(x), axis = 1)
         total["len_head"] =    total["liste_head"].apply(lambda x : len(x)) 
         total = total.loc[total["len_head"].isin([4,6])]
         c = total[["liste_head", "len_head"]].apply(lambda x : deduce_title(x), axis = 1)
         total["Titre"] = list(list(zip(*c))[0])
         d= list(list(zip(*c))[1])
         e = pd.DataFrame(d)[0].str.split("—")
         total["auteur"] = list(list(zip(*e))[0])
         total["auteur"] = total["auteur"].apply(lambda x : x.replace("Par").strip())
         total["date_title"] = list(list(zip(*e))[1])
         total["date_title"] = total["date_title"].apply(lambda x : re.sub(r'\([^)]*\)', '', x).strip())
    
         ### article cleaning
         total["article"] =  total["article"].apply(lambda x : x.split("\r\nPARTAGER\r\nTWEETER\r\nPARTAGER\r\nTWEETER")[0])
         
         ### important columns
         final = total[['url', 'restricted', 'count paragraphs', 'date', 'date_title', 'categorie', 'categorie_2', 'Titre',  'auteur', 'article']]
         
     if media == "lesechos":
         
         total = total.loc[total["restricted"].isin(['0',0,'0.0'])]
         ### article cleaning
         total["date"] = total["url"].apply(lambda x:deduce_date(x))
         
         a= total["article"].str.split("|")
         total["liste_article"] = a
         total["len_liste"] = total["liste_article"].apply(lambda x: len(x))
         total["categorie"] = total["liste_article"].apply(lambda x: categorie_lesechos(x))
         
         b = total["article"].apply(lambda x: clean_article(x))
         total["article"] = b
         total["Titre"] = total["url"].apply(lambda x : make_title_echo(x))
         final = total[['url', 'restricted', 'count paragraphs', 'date', 'categorie', 'Titre', 'article']]
         final = final.drop_duplicates("url")
         
     if media == "lemonde":
         
         total["categorie"] = total["url"].apply(lambda x: deduce_categorie(x))
         total = total.loc[total["categorie"] != ""]
    
         test = total["article"].apply(lambda x :  re.sub(r'(\r\nLE MONDE )(.*)(|)', '', (re.sub(r'(\r\nLe Monde)(.*)(|)', '', x))))
         test = test.apply(lambda x :  re.sub(r'(\nPar )([A-Z].*)(\n)', '', str(x)))
         liste_fake_title = test.apply(lambda x : x.split("\r\n",1)[0]).value_counts()
         liste_fake_title = liste_fake_title.loc[liste_fake_title >10].index
         test = test.apply(lambda x : clean_titles_lemonde(x, liste_fake_title))
         auteurs_col =  total["article"].apply(lambda x :  auteur(x))
         
         total["article"] = test
         total["auteur"] = auteurs_col
         final = total[['url', 'restricted', 'count_paragraphs', 'Title', 'article',  'categorie', 'auteur']]
         final = final.drop_duplicates("url")
         
     if media == "mediapart":
         to_replace = 'Vous êtes abonné(e)\r\nIdentifiez-vous\r\nIDENTIFIANT\r\nMOT DE PASSE\r\nMot de passe oublié ?\r\nPas encore abonné(e) ?\r\nRejoignez-nous\r\nChoisissez votre formule et créez votre compte pour accéder à tout Mediapart.\r\nABONNEZ-VOUS'
         total["article"] = total["article"].apply(lambda x : x.replace(to_replace, ""))
         total["Titre"] = total["head"].apply(lambda x : x.split("\r\n")[1])
         total["categorie"] = total["url"].apply(lambda x : x.replace("https://www.mediapart.fr/", "").split("/")[1])
         total["categorie_2"] = total["head"].apply(lambda x : x.split("\r\n")[0])
         total["auteur"] = total["head"].apply(lambda x : x.split("\r\n")[2].split(" PAR ")[1])
         total["date"] = total["head"].apply(lambda x : x.split("\r\n")[2].split(" PAR ")[0])
         total["head"] = total["head"].apply(lambda x : x.split("\r\n",3)[3])
         total = total[['url', 'restricted', 'count paragraphs', 'date', 'categorie', 'categorie_2', 'Titre',  'auteur', 'article', "head"]]
         final = total.drop_duplicates("url")
         
     if media == "humanite":
         total["Titre"] = total["article"].apply(lambda x : x.split("\r\n")[0])
         total["head"] = total["article"].apply(lambda x : x.split("\r\n")[2] if len(x.split("\r\n"))> 2 else "")
         total["article"] = total["article"].apply(lambda x : x.split("\r\n",2)[2] if len(x.split("\r\n"))> 2 else x)
         
         total = total.loc[~total["Titre"].isin(["*", "M6", "FRANCE 2", "FRANCE 3", "20.40", "20.45", "20.55", "20.35", "CANAL+", "CANAL +", "21.00", "20.50", "20.30", "20.25","RENDEZ-VOUS", "AGENDA", "LES LECTEURS EN DIRECT", "CARNET", "PAR ICI LES SORTIES", "TF1",
                           "LES PROGRAMMES DE LUNDI", "TF1 6","LES PROGRAMMES DE JEUDI","LES PROGRAMMES DE MARDI","LES PROGRAMMES DE SAMEDI",])]
         
         urls = pd.read_csv(r"C:\Users\User\Documents\Alexs\data\clean_data\url\history\humanite_history.csv")
         urls = urls[["url", "date"]]
         total = pd.merge(total, urls, on ="url", how = "left")
         
         total = total.loc[~pd.isnull(total["date"])]
         total = total[["url", 'restricted', 'count paragraphs', 'Titre', 'article', "head"]]
         final = total.drop_duplicates("url")
     
     if media == "lefigaro":
         total = total.drop_duplicates("url")
         total["Titre"] = total["article"].apply(lambda x : x.split("\r\n")[0])
         urls = pd.read_csv(r"C:\Users\User\Documents\Alexs\data\clean_data\url\history\lefigaro_history.csv")
         urls = urls[["url", "date"]]
         total = pd.merge(total, urls, on ="url", how = "left")
         total["article"] = total["article"].apply(lambda x : x.split("\r\nCet article est réservé aux abonnés")[0])
         total["auteur"] = total["article"].apply(lambda x : get_auteur(x))
         total = total.loc[~pd.isnull(total["date"])]
         total["paragraphs"] = total["article"].apply(lambda x :  len(x.split("\r\n")))
         total = total.loc[total["paragraphs"]> 2]
         total["categorie"] =  total["url"].apply(lambda x : x.replace("http://www.lefigaro.fr/", "").split("/")[0])
         total["categorie_2"] =  total["url"].apply(lambda x : x.replace("http://www.lefigaro.fr/", "").split("/")[1] if not x.replace("http://www.lefigaro.fr/", "").split("/")[1].isdigit() else "")
         total["article"] = total["article"].apply(lambda x : clean_article_figaro(x))
         total = total[['url', 'restricted', 'count paragraphs', 'date', 'categorie', 'categorie_2', 'Titre',  'auteur', 'article']]
         final = total.drop_duplicates("url")
    
     data = pd.read_csv(r"C:\Users\User\Documents\Alexs\data\clean_data\url\history\%s_history.csv"%media)
     index = data["url"].apply(lambda x  : True if  "/www.dailymotion.com/video/" not in x else False)
     data= data[index]
     
     remaining_urls = list(set(data["url"].apply(lambda x: x.replace("http://www.","https://www."))) - set(final["url"]))
     
     print(" {0} : articles crawled : {1} / {2}, ({3}%)".format(media, total.shape[0], data.shape[0], total.shape[0]/data.shape[0]))
     
     final.to_csv(r"C:\Users\User\Documents\Alexs\data\clean_data\articles\%s\articles.csv"%media, index = False)
     pd.DataFrame(remaining_urls).to_csv(r"C:\Users\User\Documents\Alexs\data\clean_data\residus\%s\missing_urls.csv"%media, index = False)

if __name__ == "__main__":
    media = "lefigaro"
    path = r"C:\Users\User\Documents\Alexs\data\extracted\%s"%media