# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 22:22:06 2018

@author: User
"""

from stanfordcorenlp import StanfordCoreNLP
import os
import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize
import glob
import tqdm
from random import shuffle
java_path = r"C:\Program Files\Java\jre1.8.0_171\bin\java.exe"
os.environ['JAVAHOME'] = java_path

#java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000


def pre_clean(text):
    text = text.replace("\r\r\r\r\n", ". ").replace("\r\r\r\n", ". ").replace("\r\r\n", ". ")
    text = text.replace(". .", ". ").replace("...", ".").replace("..", ".")
    return text

def load_articles(path):
    articles = pd.read_csv(path + "/all_articles.csv")
    return articles["article"]


def article_to_sentence(article, start_id = 0):
    phrases = sent_tokenize(pre_clean(article))
    phrases_id = []
    b = []
    for i in range(len(phrases)):
        c = nlp.ner(phrases[i])
        phrases_id += [start_id + i]*len(c)
        b +=c
    return b, phrases_id
   
    
def create_dataset(path):
    files = glob.glob(path + "/*.h5")
    total = pd.DataFrame()
    for i, f in enumerate(files):
        if i > 0:
            total= pd.concat([total, article_to_sentence(f, total["phrase_id"].iloc[-1] + 1)], axis = 0).reset_index(drop=True)
        else:
            total = article_to_sentence(f)
    
    tags_dict = {"O":"O", "PERSON": "PERS", "NUMBER":"NUM", "LOCATION": "LOC", "ORGANIZATION":"ORG", "DATE":"DATE", "MONEY":"MON",
                 "PERCENT":"PERC", "TIME" : "O", "ORDINAL" : "O", "DURATION":"DUR", "MISC": "PERS"}
    total["Tag"] = total["Tag"].map(tags_dict)     
    
    return total


def NER_articles(articles, base, min_sent_id):
    
    b = []
    phrases_id = []
    mini = min_sent_id
    for i, art in tqdm.tqdm(enumerate(articles)):
        b_, phrases_id_ = article_to_sentence(art, mini)
        b += b_
        phrases_id +=phrases_id_
        mini = max(phrases_id)
        if i%300 == 0 and i> 0:
            data = pd.DataFrame(np.transpose(list(zip(*b))))
            data["phrase_id"] =  phrases_id
            data.to_csv(path + "/ner_stanford/{0}.csv".format(i+base))
            b= []
            phrases_id = [] 
            

if __name__ == "__main__":
    
    nlp = StanfordCoreNLP(path_or_host=r'D:\articles_journaux\recherche\tagger\stanford-corenlp-full-2018-10-05',lang='fr')
    path = r"D:\articles_journaux\data"
    articles = load_articles(path)
    shuffle(articles)
#    text_files = glob.glob(path + "/tagger/NER_tagged/*.h5")
#    total = pd.DataFrame()
#    for f in text_files:
#        total = pd.concat([total, pd.read_hdf(f)], axis =0)
    
    base = 0
    articles = articles[base:]
    NER_articles(articles, base, min_sent_id = 0)
    nlp.close()
    
