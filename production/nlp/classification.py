# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 10:56:14 2018

@author: User
"""

import pandas as pd
import os
import tqdm
from utils.clean_articles import tokenize, from_output_to_classe
import pickle
from keras.preprocessing import sequence
from keras.models import model_from_json
import json 
from utils.layers import Attention
import numpy as np

class ClassificationSujet(object):
    
    def __init__(self, articles):
        self.articles = articles
        self.mode_path = os.environ["DIR_PATH"] + "/data/models/classification"
        with open(self.mode_path + "/parameters_classification.json", 'r') as fp:
            self.params = json.load(fp)
        
        
    def main_classification_sujets(self):
        sentences  = self.clean_articles()
        self.classification_sujets(sentences)
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
            
        # load json and create model
        json_file = open(self.mode_path + '/model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json, custom_objects={'Attention': Attention})
        loaded_model.load_weights(self.mode_path + '/model_weights_0.h5')
            
        new_phrases = tok.texts_to_sequences(sentences)
        new_phrases = sequence.pad_sequences(new_phrases, maxlen=self.params['maxlen'])
        y = loaded_model.predict(new_phrases, batch_size = self.params["batch_size"])
        self.articles["sujets"] = from_output_to_classe(y, self.params["classes"])
#        self.clean_classification_by_cluster()
        
    def clean_classification_by_cluster(self):
        
        for cluster in self.articles["cluster"].unique():
            if cluster != -1:
                sub_data = self.articles.loc[self.articles["cluster"] == cluster]
                best_classe = sub_data["sujets"].astype(str).value_counts().index[0]
                if pd.notnull(best_classe) and best_classe not in [None, "None", ""]:
                    self.articles.loc[self.articles["cluster"] == cluster, "sujets"] = best_classe
 
if __name__ == "__main__":
    articles = pd.read_csv(r"C:\Users\User\Documents\Alexs\data\continuous_run\article\extraction_2018-11-18.csv", sep = "#")
    art = ClassificationSujet(articles).main_classification_sujets()
    print(art["sujets"].astype(str).value_counts())
    
    