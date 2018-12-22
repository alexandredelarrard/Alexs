# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 10:56:14 2018

@author: User
"""

import pandas as pd
import os
import tqdm
from utils.clean_articles import classification_tokenize, from_output_to_classe
from keras.preprocessing import sequence
from utils.layers import load_information

class ClassificationSujet(object):
    
    def __init__(self, articles, clusters):
        self.articles = articles
        self.clusters = clusters
        self.mode_path = os.environ["DIR_PATH"] + "/data/models/classification"
        
        
    def main_classification_sujets(self):
        self.classification_sujets()
        self.classification_per_cluster()
        return self.articles, self.clusters
    
    
    def classification_sujets(self):
        
        sentences  = self.clean_articles(self.articles["article"].tolist())
        self.tok, self.loaded_model, self.params = load_information(self.mode_path)
        new_phrases = self.tok.texts_to_sequences(sentences)
        new_phrases = sequence.pad_sequences(new_phrases, maxlen=self.params['maxlen'])
        y = self.loaded_model.predict(new_phrases, batch_size = self.params["batch_size"])
        self.articles["sujets"] = from_output_to_classe(y, self.params["classes"])
        
        
    def classification_per_cluster(self):
        
        cluster_big_article = []
        for cluster in self.clusters.keys():
            sub_data = self.articles.loc[self.articles["cluster"] == cluster]
            art = ""
            for k in range(sub_data.shape[0]):
                art += sub_data.iloc[k]["article"] + " "
            cluster_big_article.append(art)
        
        sentences  = self.clean_articles(cluster_big_article)
        new_phrases = self.tok.texts_to_sequences(sentences)
        new_phrases = sequence.pad_sequences(new_phrases, maxlen=self.params['maxlen'])
        y = self.loaded_model.predict(new_phrases, batch_size = self.params["batch_size"])
        sujets = from_output_to_classe(y, self.params["classes"])
        
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
    
    