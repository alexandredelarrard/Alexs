# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 10:56:14 2018

@author: User
"""

import pandas as pd
import numpy as np
import os
import tqdm
from utils.clean_articles import tokenize, from_output_to_classe
import pickle
from keras.preprocessing import sequence
import json 


class ClusteringArticles(object):
    
    def __init__(self, articles):
        self.articles = articles
        self.mode_path = os.environ["DIR_PATH"] + "/data/models/classification"
        with open(self.mode_path + "/parameters_classification.json", 'r') as fp:
            self.params = json.load(fp)
        
        
    def main_classification_sujets(self):
        sentences, tok  = self.clean_articles()
        self.classification_sujets(sentences, tok)
        return self.articles
    
    
    def clean_articles(self):
      
        self.articles = self.articles.loc[~pd.isnull(self.articles["article"])]
        self.articles = self.articles.loc[~pd.isnull(self.articles["restricted"])]
        a = self.articles["article"].apply(lambda x : len(x))
        self.articles = self.articles.loc[a>100]
        
        sentences = []
        for art in tqdm.tqdm(self.articles["article"].tolist()):
            sentences.append(tokenize(art))
        return sentences
    
    
    def classification_sujets(self, sentences):
        
        with open(self.mode_path + '/tokenizer.pickle', 'rb') as handle:
            tok = pickle.load(handle)
            
        model = np.load(self.mode_path + '/model_weights_0.pny')
        new_phrases = tok.texts_to_sequences(sentences)
        new_phrases = sequence.pad_sequences(new_phrases, maxlen=self.params['maxlen'])
        y = model.predict(new_phrases, batch_size = self.params["batch_size"]*2)
        self.articles["sujets"] = from_output_to_classe(y, self.params["classes"])
        
        
    