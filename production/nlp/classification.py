# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 10:56:14 2018

@author: User
"""

import pandas as pd
import os
import tqdm
import numpy as np
from keras.preprocessing import sequence

from utils.clean_articles import classification_tokenize, from_output_to_classe
from utils.layers import load_information

class ClassificationSujet(object):
    
    def __init__(self, articles, clusters):
        self.articles = articles
        self.clusters = clusters
        self.mode_path = os.environ["DIR_PATH"] + "/data/models/classification"
        
        
    def main_classification_sujets(self):
        y = self.classification_sujets()
        self.classification_per_cluster(y)
        return self.articles, self.clusters
    
    
    def classification_sujets(self):
        
        sentences  = self.clean_articles(self.articles["article"].tolist())
        self.tok, self.loaded_model, self.params = load_information(self.mode_path)
        new_phrases = self.tok.texts_to_sequences(sentences)
        new_phrases = sequence.pad_sequences(new_phrases, maxlen=self.params['maxlen'])
        y = self.loaded_model.predict(new_phrases, batch_size = self.params["batch_size"])
        self.articles["sujets"] = from_output_to_classe(y, self.params["classes"])
    
        return y
        
        
    def classification_per_cluster(self, y):
        
        cluster_big_article = []
        for cluster in self.clusters.keys():
            index_y = self.articles.loc[self.articles["cluster"] == cluster].index
            yy = y[index_y,:].mean(axis=0)
            cluster_big_article.append(yy)

        sujets = from_output_to_classe(np.array(cluster_big_article), self.params["classes"])
        for i, cluster in enumerate(self.clusters.keys()):
            self.clusters[cluster]["sujets"] = sujets[i]
        
        
    def clean_articles(self, articles):
        sentences = []
        for art in tqdm.tqdm(articles):
            sentences.append(classification_tokenize(art))
        return sentences
    
    
if __name__ == "__main__":
    articles = pd.read_csv(r"C:\Users\User\Documents\Alexs\data\continuous_run\article\extraction_2018-11-18.csv", sep = "#")
    art = ClassificationSujet(articles).main_classification_sujets()
    print(art["sujets"].astype(str).value_counts())
    
    