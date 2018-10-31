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
java_path = r"C:\Program Files\Java\jre1.8.0_171\bin\java.exe"
os.environ['JAVAHOME'] = java_path

#java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
nlp = StanfordCoreNLP(path_or_host=r'C:\Users\User\Documents\Alexs\data\tagger\stanford-corenlp-full-2018-10-05',lang='fr')


def load_articles(path):
    articles = pd.read_csv(path + "/all_articles.csv")
    return articles["article"]


def article_to_sentence(article, start_id = 0):
    phrases = sent_tokenize(article)
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


def NER_articles(articles):
    
    b = []
    phrases_id = []
    mini = 0
    for i, art in tqdm.tqdm(enumerate(articles)):
        b_, phrases_id_ = article_to_sentence(art, mini)
        b += b_
        phrases_id +=phrases_id_
        mini = max(phrases_id)
        if i%300 == 0 and i> 0:
            data = pd.DataFrame(np.transpose(list(zip(*b))))
            data["phrase_id"] =  phrases_id
            data.to_hdf(path + "/tagger/NER_tagged/{0}-{0}.h5".format(i), key='data', mode='w')
            b= []
            phrases_id = [] 
            

if __name__ == "__main__":
    path = r"D:\data\articles_journaux"
    articles = load_articles(path)
    
    articles = articles[3601:].tolist()
    NER_articles(articles)
   
    nlp.close()
    
#    total = pd.DataFrame()
#    for f in text_files:
#        total = pd.concat([total, pd.read_hdf(f)], axis =0)
#        